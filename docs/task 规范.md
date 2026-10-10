# task 规范

## 1. 适用范围

本规范定义正式 task 文件的组织方式及 `daisy/task/` 的实现约定。task 描述训练、评估或分析的实验配置；`task_id` 默认取文件名（不含扩展名），用于关联输出、日志和报表。

## 2. task 系统架构

- `base.py`：定义 Pydantic 配置基类 `BaseConfig`、`BaseMetaConfig`、`BaseOutputConfig` 与 `BaseTaskConfig`。`BaseTaskConfig` 继承 `BaseConfig`，由字段直接定义模型结构，统一管理 `task_type`、`meta`、`task_file` 与 `task_id`；
- `protocol.py`：任务协议 `Task[C, R]`（只读 `config`、`run(ctx) -> R`）、运行上下文 `RunContext`（`output_dir`、`device`、`run(key, task)` 运行子任务）、根任务类协议 `TaskType[C]`（`config_type`、以配置构造任务、`open_run(run_dir, config, device) -> TaskRun`）与已完成运行的分析视图协议 `TaskRun`（只读 `run_dir`、`config`）；
- `runner.py`：提供通用任务调度函数 `load_config(path) -> BaseTaskConfig` 与 `run_task(config, device) -> Path`，解析 `extends`、`imports` 与 `task_type`，以配置构造根任务节点并在输出目录中运行；
- `runs.py`：`load_run_config(run_dir)` 与 `open_run(run_dir, run_type, device=None)`，读取已完成运行（见 2.4）；
- `imports.py`：`import_files(paths)` 导入 task 文件 `imports` 字段声明的外部 Python 文件，触发其中的 task / feeder 注册；
- `tasks/__init__.py`：维护任务字典 `TASKS: dict[str, TaskType]`，将任务类型映射为根任务类，提供 `get_task(task_type) -> TaskType` 与类装饰器 `register_task(name)`；
- `feed.py`：`feed(config, seed=...)` 调用 feeder 构建数据划分并执行通用检查，`assert_label_range` / `assert_multitask_label_range` 校验标签范围；
- `runtime.py`：输出路径渲染、运行头部与完成信息、任务快照与 JSON 保存，供 runner 与任务使用；
- `provenance.py`：把运行涉及的 git 仓库工作区记录为隐藏 commit（源码快照），并核对已加载模块与运行期间的源码改动；
- `serialization.py`：提供 `build_example_config`、`export_example_config` 与 `save_config`，负责任务配置序列化与示例导出；
- `shared/`：共享配置模型（含 `FeederConfig` / `DatasetConfig`）与 transform；
- `tasks/<task_name>/`：任务专属配置与执行逻辑。每个任务模块包含 `config.py` 与实现 `Task` 协议的任务类及其分析视图所在的 `runner.py`。

### 2.1 新增任务

任务是实现 `Task` 协议的类：唯一输入为构造时传入的结构化配置（pydantic 模型，保存为 `config` 属性），`run(ctx)` 在 `ctx.output_dir` 中执行并把结果返回给调用方。由 TOML 配置启动的根任务还实现 `TaskType` 协议：

```python
class ExampleTask:
	config_type: ClassVar = ExampleConfig

	def __init__(self, config: ExampleConfig) -> None:
		self.config = config

	def run(self, ctx: RunContext) -> None: ...

	@classmethod
	def open_run(cls, run_dir: Path, config: ExampleConfig, device: torch.device) -> ExampleRun:
		return ExampleRun(run_dir, config, device)
```

- `config_type` 须标注为 `ClassVar`，类对象才满足 `TaskType` 协议；
- 根任务不渲染输出路径、不保存快照，seed 由任务按自身配置设置（见 2.2）；
- 子任务同样是 `Task`：配置继承 `BaseConfig`，不注册、不需要 `config_type` 与 `open_run`。父任务在 `run` 中构造子任务配置与子任务，经 `ctx.run(key, task)` 运行：创建 `output_dir / key`（`key` 可含 `/` 以嵌套），写入与根任务格式相同的 `task_snapshot.json`（见 2.2），返回子任务的结果；
- 子任务需要的其他产物（如另一个子任务保存的模型）以路径写入其配置，使快照足以重建该子任务；嵌套方式、结果汇总、是否支持中断续跑均由父任务决定；
- 父任务的分析视图以 `load_task_config(task_dir, config_type)` 读取子任务快照（见 2.4）；
- 例：`proj/mae_v5` 的评分交叉验证由根任务逐折构造训练子任务（`fold{k}/`）与加载该折模型的新评分者适应子任务（`fold{k}/adapt/`）。

内置任务：

1. 在 `tasks/<task_name>/` 中创建 `config.py` 与 `runner.py`：
   - `config.py` 定义继承 `BaseTaskConfig` 的具体配置类，声明字段 `task_type: Annotated[Literal['<task_name>'], tyro.conf.Suppress] = '<task_name>'`；
   - `runner.py` 实现任务类与分析视图类（见 2.4）；
2. 在 `tasks/__init__.py` 的静态字典 `TASKS` 中显式添加任务条目 `'<task_name>': ExampleTask`；
3. 跨任务复用的配置放入 `shared/`；
4. 公开对象在 `daisy/task/__init__.py` 显式导出，外部代码统一从 `daisy.task` 导入。

外部任务（项目目录中的 Python 文件）：

1. 配置类同上，定义在模块顶层；
2. 根任务类使用 `@register_task('<task_name>')` 装饰，注册类本身；`name` 须与 `config_type` 的 `task_type` 默认值一致，且未被注册；
3. task 文件通过顶层 `imports` 字段导入该文件，统一以 `python -m daisy run <task.toml>` 运行，外部文件不再提供命令行入口。

### 2.2 运行生命周期

`run_tasks` 对每个 run 依次：

1. 创建渲染后的输出目录，打印运行头部（`print_task_header`）；
2. 以配置构造根任务，`save_task_snapshot` 保存 `task_snapshot.json`：`task` 为任务类的完整名称，`runtime` 记录设备，其余为 `model_dump(mode='json', exclude_none=True)` 得到的配置；子任务的快照由 `RunContext.run` 以同一函数保存；
3. 在 `RunContext(output_dir, device)` 中调用根任务的 `run`；
4. 打印完成信息（`print_task_completed`），核对源码并写入 `provenance.json`（见 2.3）。

seed 与其他运行期信息由任务自行处理：任务在 `run` 开始时按自身配置调用 `set_global_seed`，需要记录的派生量（如实际学习率）写入输出目录中的独立文件。

### 2.3 源码快照

做法与 DVC experiments 相同，不改动分支、暂存区与工作区：

1. 确定仓库：daisy 包、各 task 文件及其 `imports` 文件所属的 git 仓库（`git rev-parse --show-toplevel`），不在仓库中时报错；
2. 记录快照：对每个仓库，在 index 的临时副本上执行 `git add --all` 与 `git write-tree`，得到工作区（含未提交改动与未被忽略的未跟踪文件）的 tree，再以 HEAD 为父节点 `git commit-tree`，挂在同名引用 `refs/daisy/runs/<时间>-<进程号>` 下防止被垃圾回收；一次 `daisy run` 的全部 run 共用同一快照；
3. 模块核对：快照后与每个 run 结束后，已加载模块的源文件须位于某个仓库的快照 tree 中，或属于 Python 安装（`sys.prefix`、`sys.base_prefix` 等目录下的标准库与 site-packages），否则报错。被 `.gitignore` 忽略的目录（如 `tmp/`）中的代码不能用于正式运行；
4. 改动检测：每个 run 结束后重新计算各仓库工作区的 tree，与快照不同的仓库记为运行期间源码被修改，打印警告，不中止运行；
5. 每个 run 的输出目录写入 `provenance.json`：`ref`、各仓库的 `root`、`head`、`commit`（快照）、`dirty`（工作区与 HEAD 不同），以及 `changed_during_run`。

复现某次运行的代码：`git worktree add <目录> <commit>`（各仓库分别执行）。快照引用可按前缀清理：`git for-each-ref --format='%(refname)' refs/daisy/runs/` 配合 `git update-ref -d`。

函数内导入在运行中途才加载模块，可能读到快照之后被修改的源码；项目代码应在启动时全部加载（proj 通过 ruff 的 `PLC0415` 禁止 `src/` 中的函数内导入）。

### 2.4 已完成运行的分析

离线统计分析通过运行目录复用配置、产物与模型，不恢复训练状态：

```python
import_files([Path('src/mae_v5/classification/task.py')])  # 外部任务先导入注册
run = open_run(run_dir, MultiTaskClassificationRun, device=device)
run.test_predictions  # 只读产物时不加载模型与数据
run.predict('test')  # 首次访问时构建模型、加载最佳权重并重建数据划分
```

- `load_run_config(run_dir)`：读取 `task_snapshot.json`，去掉 `task` 与 `runtime`，按 `task_type` 选择已注册根任务的 `config_type` 严格校验；不导入快照记录的 `imports`（历史路径可能已失效），外部任务须由调用方预先导入；配置字段已删除的历史快照校验失败并报错；
- `load_task_config(task_dir, config_type)`：同上，但由调用方给出配置类，根任务与子任务通用；
- `open_run(run_dir, run_type, device=None)`：调用该任务的 `open_run` 构建分析视图，返回值须为 `run_type` 实例，否则违约；`device` 默认自动选择 CUDA/CPU；
- 分析视图为普通 dataclass，字段为 `run_dir`、`config`、`device`；模型、数据划分与产物以 `functools.cached_property` 在首次访问时加载，模型严格加载并处于 eval 模式；需要参数的操作（如 `loader(split, ...)`）为普通方法；
- 数据划分、评估变换与推理等功能直接定义在运行对象中，不另设以配置为参数的模块级函数；任务的 `run` 以输出目录构建同一运行对象复用这些功能，训练中的模型直接赋给其 `model`；
- 内置视图：`ClassificationRun`（`model` 读取 `best_model.pth`、`trainer`、`datasets`、`loader`）、`MAEPretrainRun`（`model` 读取 `checkpoint_latest.pth`、`trainer`、`datasets`、`loader`、`loss`）。

## 3. CLI 参数与执行机制

CLI 使用 tyro 绑定 Pydantic 配置模型。`--` 前为命令选项，后为任务配置覆盖项。参数优先级遵循 **CLI > run 条目 > TOML > 模型默认值**；task 文件含 `run` 数组（见 4.4）时，CLI 覆盖项作用于每个 run。

`--cpu-affinity` 接收十进制正整数位掩码，第 i 位对应 CPU i。也可通过环境变量 `DAISY_CPU_AFFINITY` 设置相同的值，显式命令行参数优先。CLI 在加载任务前通过 `psutil.Process().cpu_affinity()` 设置当前进程的亲和度，后续创建的子进程（含 DataLoader worker）继承该设置；两者均未设置时保留当前亲和度。解析或设置失败时立即抛出异常。`4095` 等于 `0xFFF`，选择 CPU 0–11。

执行流程包含以下阶段：

1. `load_run_configs(path)` 使用 `tomllib` 读取 TOML 文件，沿 `extends` 链合并配置，并按父级在前、去重的顺序收集各层 `imports`（相对声明它的文件解析）后调用 `import_files` 导入。导入以文件名为顶层模块名，文件所在目录加入 `sys.path`；同名不同路径的文件视为冲突并报错。文件须显式声明 `task_type` 字段：
   - 缺失 `task_type` 时，抛出 `ValueError("Missing 'task_type' in task file: <path>. Please specify a task type.")`；
   - `task_type` 为空或非字符串类型时，抛出 `ValueError("Invalid 'task_type' in task file: <path>. Expected non-empty string.")`；
   - `task_type` 未在 `TASKS` 中注册时，由 `get_task` 抛出 `ValueError("Unknown task type: '<task_type>'. Available types: [...]")`；
2. 通过 `config_cls.model_validate(data)` 校验并实例化具体任务配置对象；存在 `run` 数组时，先校验不含任何条目的基础配置，再对每个条目合并后分别校验，得到各 run 的配置；
3. 按 `--runs` 选择 run（保持 task 文件中的顺序）：`default` 表示所有默认运行（`default = true`）的 run，可与其他 run id 组合；不指定 `--runs` 时等价于 `--runs default`；未知 id、task 文件无 `run` 却给出 run id、或最终没有选中任何 run 时报错；
4. 使用 `tyro.cli` 将命令行 `--` 后的覆盖参数绑定至每个 run 的配置实例，完成最终校验；
5. `run_tasks(configs, device)` 检查各 run 渲染后的输出路径互不相同，记录源码快照并核对已加载模块（见 2.3），再在同一进程中顺序调用各任务的执行函数，每个 run 结束后写入 `provenance.json`；任一 run 报错立即终止，此前完成的 run 保留。`run_task(config, device)` 等价于只含一个配置的 `run_tasks`。

`load_config(path)` 返回忽略 `run` 数组的基础配置，供 `daisy list` 等只需单个配置的场景使用。

常用命令格式：

- 命令帮助：`python -m daisy run --help`；
- 任务字段帮助：`python -m daisy run task.toml -- --help`；
- 指定设备与学习率：`python -m daisy run task.toml -d cpu -- --training.lr 0.02`；
- 指定 CPU 亲和度：`python -m daisy run task.toml --cpu-affinity 4095`；
- 通过环境变量指定 CPU 亲和度（PowerShell）：`$env:DAISY_CPU_AFFINITY = '4095'`，随后执行 `python -m daisy run task.toml`；
- 只运行部分 run，并对其统一覆盖：`python -m daisy run ablation.toml --runs imagenet_mae random -- --training.epochs 1`；
- 默认运行的 run 加上一个非默认 run：`python -m daisy run ablation.toml --runs default extra`；
- 导出示例配置：`python -m daisy example <task_type>`；
- 列出任务文件：`python -m daisy list`。

## 4. task 文件格式与序列化

task 采用 TOML 格式，显式声明 `task_type`、`[meta]`、`[output]` 以及实验所需的数据、模型、训练或分析参数。

```toml
task_type = "classification"

[meta]
title = "ResNet34 baseline"
description = "Internal baseline for smile classification"
creator = "your_name"
created_at = "2026-03-09"

[output]
save_path = "outputs/{task_id}"
```

`[meta]` 包含 `title`（标题）、`description`（实验目标或变量）、`creator`（创建人）和 `created_at`（日期）。源码版本由运行时记录在 `provenance.json` 中（见 2.3）。

### 4.1 输出与数据集模式规范

各任务配置采用明确的输出模型与数据集字段定义：

- 输出模型模式：
  - `classification` 采用 `BaseOutputConfig`，包含 `save_path`、`keep_count`、`save_best` 与 `log`；
  - `mae_pretrain` 采用 `LogOutputConfig`，包含 `save_path` 与 `log`；
- 数据集配置 `[dataset]` 只包含 `feeder`（已注册的 feeder 标识符）与 `[dataset.params]`（由该 feeder 的参数模型校验），详见 4.2；
  - `dataset` 标注 `tyro.conf.Suppress`，数据协议只由 task 文件决定，不接受 CLI 覆盖；
  - `classification` 在 `model.num_classes` 声明类别数，`mae_pretrain` 默认使用内置 `image_folder`。

### 4.2 feeder 规范

feeder 负责数据的定位、组装与划分，框架不介入数据布局。

- 签名：`(params: P, *, seed: int | None) -> dict[str, DiskDataset]`，使用 `daisy.feeder.register_feeder('<id>')` 注册；`P` 为模块顶层定义、`model_config = ConfigDict(extra='forbid')` 的 pydantic 模型，由首参注解推导；
- 返回的划分名由 feeder 决定，task 执行器自行检查所需划分（如 `train`、`val`）；`seed` 为 task 的训练 seed，feeder 可自行声明独立的划分 seed 参数；
- `feed()` 统一执行与布局无关的检查：返回非空字典、各划分为非空 `DiskDataset`、各划分文件路径互斥；标签语义（类别数、多任务头）由 task 配置声明，执行器校验 feeder 产出的标签范围；
- 数据划分、采样等可能造成泄露的逻辑全部在外部 feeder 中显式编写；`daisy.feeder` 提供 `read_sheet`、`parse_multitask_table`、`parse_column_split_indices`、`make_disk_dataset` 等工具函数；
- 内置 feeder 仅有无标签 `image_folder`（`root`、`extensions`、`sample_ratio`、`sample_seed`），返回单一 `train` 划分；
- 示例见 `tasks/feeders/example_feeders.py` 与 `tasks/example.toml`：

```toml
task_type = "classification"
imports = ["feeders/example_feeders.py"]

[dataset]
feeder = "example_sheet_ratio"

[dataset.params]
root = "path/to/images"
sheet = "path/to/labels.xlsx"
num_classes = 3
```

### 4.3 配置序列化与示例生成

- 示例构建：`build_example_config(task_type)` 通过 `get_task(task_type).config_type` 获取具体的配置类，直接实例化配置对象填充默认值，设置 `meta` 预设字段；`dataset` 为必填的任务写入占位 `feeder = '<feeder_id>'`；外部任务通过 `python -m daisy example <task_type> -i <file.py>` 先导入再导出；
- 配置持久化：`save_config(config, path)` 通过 `config.model_dump(exclude={'task_file', 'task_id', 'imports', 'run_id', 'run_overrides'}, exclude_none=True)`（`run_default` 在配置模型中声明 `exclude=True`，不参与序列化） 导出字典，使用 `tomli_w.dump` 写入 TOML 文件；`export_example_config(task_type, path)` 负责输出路径处理与文件写入。

### 4.4 run 数组

消融可以在一个 task 文件中用 `run` 数组描述：每个条目是一次参数重载，等价于一次带 CLI 覆盖的运行；对训练集比例、seed 等的网格调参用 `grid`（见 4.6）。

```toml
extends = "base.toml"

[output]
save_path = "outputs/{task_id}/{run_id}"

[[run]]
id = "imagenet_mae"
_remove = ["model.checkpoint_path"]
model.name = "vit_base_patch16_224.mae"

[[run]]
id = "imagenet_mae_ffhq"
```

- `run` 与 `extends`、`imports` 同为顶层保留键，值为非空的表数组；
- 每个条目必须有 `id`：非空，只含字母、数字、`.`、`_`、`-`，同一 task 文件内不重复，且不能为保留字 `default`。`id` 不进入配置，用于输出路径模板变量 `{run_id}`、`--runs` 选择、运行时打印与快照；
- 条目可选 `default`（布尔，默认 `true`）：为 `false` 的 run 只在 `--runs` 显式给出其 id 时运行，适合只在部分设置下才需要的变体（如只在低训练集比例下补充的长训练）。`default` 不进入配置；
- 条目先按其 `_remove`（见 4.5）从基础配置中删除键，再将其余内容按 `extends` 的语义深度合并到基础配置上（表递归合并，标量与列表整体替换），最后由配置模型校验，拼错的字段名同样报错；
- `run` 随 `extends` 继承：子文件未声明 `run` 时沿用父级的 `run`，声明时整体替换，`_remove = ["run"]` 则去掉继承来的 `run`；
- 基础配置（不含任何条目）须能单独通过校验；
- 各 run 的输出路径必须互不相同，通常在 `save_path` 中使用 `{run_id}`；
- `task_snapshot.json` 记录 `run_id` 与 `run_overrides`（条目去掉 `id` 后的原文，含 `_remove` 与 `default`）；无 `run` 时 `run_id` 为空字符串，`run_overrides` 不出现。

### 4.5 `_remove`：删除继承的值

TOML 没有 null，无法用覆盖把字段设回 `None` 或模型默认值。`_remove` 是点分隔键路径的数组，表示删除这些键，使字段回到上一层的值或配置模型默认值：

- 在 task 文件顶层：先从 `extends` 加载的父级配置中删除这些键，再合并当前文件；没有 `extends` 的文件使用 `_remove` 报错；
- 在 `[[run]]` 条目中：先从基础配置中删除这些键，再合并该条目；
- 要删除的键在继承来的配置中不存在时报错，避免拼写错误被静默忽略；
- 例：父级设置了 `model.checkpoint_path`，而模型默认值为 `None`，则 `_remove = ["model.checkpoint_path"]` 使其为 `None`。

### 4.6 `grid`：网格调参

`grid` 是一个表，叶子为非空的候选值数组，键路径即要替换的配置字段；运行时取各参数轴的笛卡尔积，每个组合用候选值整体替换对应字段：

```toml
extends = "base.toml"

[output]
save_path = "outputs/{run_id}_train{train_ratio}_seed{training.seed}"

[grid]
train_ratio = [0.1, 1.0]
training.seed = [42, 43, 44]

[[run]]
id = "ce"

[[run]]
id = "ce-ft100"
training.epochs = 100
grid.train_ratio = [0.1]  # 覆盖顶层的同名参数轴
```

- `grid` 可写在顶层与 `[[run]]` 条目中，与 `run` 等同为保留键，不进入配置；
- 顶层 `grid` 随 `extends` 深度合并继承，可被顶层 `_remove`（如 `_remove = ["grid.train_ratio"]`）删除；条目的 `grid` 与顶层 `grid` 按 `extends` 的语义深度合并，同名参数轴由条目覆盖，条目的 `_remove` 同样可以删除顶层参数轴；
- 展开在 `_remove` 与重载合并之后进行，候选值整体替换字段（候选为表时不与原值合并），展开后的每个配置由配置模型校验；
- 参数轴按 TOML 插入顺序排列，最右侧参数轴变化最快；同一条目展开的所有配置共享该条目的 `id` 与 `default`，`--runs` 按 `id` 选中其全部组合；
- 无 `run` 时顶层 `grid` 直接展开基础配置；`load_config` 忽略 `grid`；
- 输出路径须区分各组合，在 `save_path` 中引用参数轴字段（如 `{training.seed}`）；CLI 覆盖作用于展开后的每个配置，覆盖参数轴字段会使各组合输出路径相同而报错；

`grid._zip` 将多个字段作为一个参数轴，支持显式列出配对候选：

```toml
[run.grid]
_zip = [
    { protocol.fold_seed = 0, training.seed = 42 },
    { protocol.fold_seed = 1, training.seed = 43 },
    { protocol.fold_seed = 2, training.seed = 44 },
]
training.blr = [0.001, 0.0005]
```

也可以用表提供等长数组，按位置配对：

```toml
[run.grid]
training.blr = [0.001, 0.0005]

[run.grid._zip]
protocol.fold_seed = [0, 1, 2]
training.seed = [42, 43, 44]
```

- 两种形式均生成三个 seed 配对，再与 `training.blr` 取笛卡尔积，共六次运行；运行顺序仍取决于轴的声明顺序；
- `_zip` 可写在顶层 `[grid]` 或条目的 `[run.grid]` 中；数组形式随继承整体覆盖，表形式按现有规则深度合并，同名字段的数组整体覆盖；
- `_zip` 及其候选表必须非空；表形式中的候选数组必须非空且等长，不满足时立即报错；
- 同一字段不能同时出现在普通轴与 `_zip` 中。

## 5. 任务行为与产物

### 5.1 分类

使用 `ClassificationMethod` 和 `prepare_classification_training`，由 Trainer Plugins 处理调度、保存与日志。

- `training.use_scheduler`：启用时使用 CosineAnnealing 与 warmup，关闭时保持初始学习率；
- `training.clip_grad`：启用时按 `max_norm` 裁剪梯度范数，适用于 AMP 和普通训练；
- `training.cmp_obj`：选择 `BestModel` 与 `EarlyStop` 的监控指标，支持 `acc`、`f1`、`recall`、`prec`（映射为 `precision`）；
- `output.save_best`：保存最优权重 `best_model.pth`；`output.keep_count > 0` 独立控制滚动检查点 `checkpoint_NNNN.pth`；
- `output.log`：生成 `logs/log_YYYYMMDD_HHMMSS.csv`，记录 `train_loss` 及验证集 `loss`、`acc`、`f1`、`precision`、`recall`、`auroc`。

### 5.2 MAE 预训练

预训练使用 `MAEPretrainMethod` 与 `make_dataloader`，配合 Trainer 插件执行。

- 学习率：`eff_batch_size = training.batch_size * training.accum_iter`，`actual_lr = training.blr * eff_batch_size / 256.0`。使用 cosine 调度，实际学习率与训练 transform 写入 `runtime.json`（`resolved_lr`、`train_transform`）；
- 预训练恢复：`training.resume` 加载 `model`、`optimizer`、`scaler`，从零基编号 `checkpoint['epoch'] + 1` 继续至目标 `epochs`，优先于 `model.timm_pretrained` 初始化；
- 预训练保存：每轮更新完整训练状态 `checkpoint_latest.pth`，最终轮保存 `checkpoint_{epoch:04d}.pth`；`training.save_freq > 0` 时额外按该间隔保存编号检查点；
- 输出：`output.save_path` 指定目录，`output.log` 启用 `logs/` 下的 CSV 日志。预训练日志为 `mae_log_YYYYMMDD_HHMMSS.csv`，记录训练损失与学习率。

## 6. 文件组织与命名

- 仓库级示例或通用任务放在根目录 `tasks/`，项目任务放在对应项目的 `tasks/`；
- CLI 默认扫描顶层 `*.toml`，正式实例平铺存放；模板可放在 `templates/` 子目录；
- 通用文件名使用 `yyyymmdd-id[-comment].toml`，项目专属命名规则在项目文档中声明；
- `proj/mae/tasks/` 使用 `p{phase}-{topic}-{variant}.toml`，研究阶段 `P0` 至 `P8` 对应文件名前缀 `p0` 至 `p8`，例如 `p1-baseline-resnet34-scratch.toml`、`p2-mae-mask040.toml`。

## 7. 实验配置与追踪

- 正式实验通过 task 或 batch 入口运行，显式记录实验使用的路径、模型、划分方式、超参数与 seed；机器相关差异通过配置字段表达；
- 固定索引表、sheet、病例分组、采样清单与外部评估数据作为实验协议记录；
- 输出目录与 `task_id` 对应，默认可用 `outputs/{task_id}`；`proj/mae` 使用 `outputs/mae/{phase}/{task_id}`；
- 结果表、checkpoint、预测与统计产物通过 `task_id` 关联 task 文件、源码快照（`provenance.json`）和关键配置；
- 已产生正式输出的 task 保留原配置，后续调整创建新 task；项目命名、输出目录与阶段映射的变更同步更新项目文档。
