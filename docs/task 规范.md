# task 规范

## 1. 适用范围

本规范定义正式 task 文件的组织方式及 `daisy/task/` 的实现约定。task 描述训练、评估或分析的实验配置；`task_id` 默认取文件名（不含扩展名），用于关联输出、日志和报表。

## 2. task 系统架构

- `base.py`：定义 Pydantic 配置基类 `BaseConfig`、`BaseMetaConfig`、`BaseOutputConfig` 与 `BaseTaskConfig`。`BaseTaskConfig` 继承 `BaseConfig`，由字段直接定义模型结构，统一管理 `task_type`、`meta`、`task_file` 与 `task_id`；
- `runner.py`：提供通用任务调度函数 `load_config(path) -> BaseTaskConfig` 与 `run_task(config, device) -> Path`，解析 `extends`、`imports` 与 `task_type` 并分发至任务执行函数；
- `imports.py`：`import_files(paths)` 导入 task 文件 `imports` 字段声明的外部 Python 文件，触发其中的 task / feeder 注册；
- `tasks/__init__.py`：维护任务字典 `TASKS: dict[str, TaskEntry]`，将任务类型映射为 `(ConcreteConfig, run)` 元组，提供 `get_task(task_type) -> TaskEntry` 与装饰器 `register_task(name)`；
- `feed.py`：`feed(config, seed=...)` 调用 feeder 构建数据划分并执行通用检查，`assert_label_range` / `assert_multitask_label_range` 校验标签范围；
- `runtime.py`：统一处理运行头部、输出目录、运行快照和完成信息；
- `serialization.py`：提供 `build_example_config`、`export_example_config` 与 `save_config`，负责任务配置序列化与示例导出；
- `shared/`：共享配置模型（含 `FeederConfig` / `DatasetConfig`）与 transform；
- `tasks/<task_name>/`：任务专属配置与执行逻辑。每个任务模块包含 `config.py` 与实现 `run(config: ConcreteConfig, device: torch.device) -> Path` 的 `runner.py`。

### 2.1 新增任务

内置任务：

1. 在 `tasks/<task_name>/` 中创建 `config.py` 与 `runner.py`：
   - `config.py` 定义继承 `BaseTaskConfig` 的具体配置类，声明字段 `task_type: Annotated[Literal['<task_name>'], tyro.conf.Suppress] = '<task_name>'`；
   - `runner.py` 实现常规函数 `run(config: ConcreteConfig, device: torch.device) -> Path` 执行任务逻辑；
2. 在 `tasks/__init__.py` 的静态字典 `TASKS` 中显式添加任务条目 `'<task_name>': (ConcreteConfig, run_<task_name>)`；
3. 跨任务复用的配置放入 `shared/`；
4. 公开对象在 `daisy/task/__init__.py` 显式导出，外部代码统一从 `daisy.task` 导入。

外部任务（项目目录中的 Python 文件）：

1. 配置类同上，定义在模块顶层；
2. 执行函数使用 `@register_task('<task_name>')` 装饰，配置类由首参注解推导，且须与配置类 `task_type` 默认值一致；
3. task 文件通过顶层 `imports` 字段导入该文件，统一以 `python -m daisy run <task.toml>` 运行，外部文件不再提供命令行入口。

### 2.2 运行生命周期

runner 复用 `runtime.py`：

- `prepare_task_run()`：统一 seed、banner、commit 与输出目录；
- `save_run_snapshot()`：保存 `task_snapshot.json`，额外运行信息通过 `extra` 追加；存在 `imports` 时将外部文件复制到输出目录 `imports/`，并在 `runtime.imports` 记录各文件 sha256；
- `print_task_completed()`：统一任务完成输出。

## 3. CLI 参数与执行机制

CLI 使用 tyro 绑定 Pydantic 配置模型。`--` 前为命令选项，后为任务配置覆盖项。参数优先级遵循 **CLI > TOML > 模型默认值**。

执行流程包含以下阶段：

1. `load_config(path)` 使用 `tomllib` 读取 TOML 文件，沿 `extends` 链合并配置，并按父级在前、去重的顺序收集各层 `imports`（相对声明它的文件解析）后调用 `import_files` 导入。导入以文件名为顶层模块名，文件所在目录加入 `sys.path`；同名不同路径的文件视为冲突并报错。文件须显式声明 `task_type` 字段：
   - 缺失 `task_type` 时，抛出 `ValueError("Missing 'task_type' in task file: <path>. Please specify a task type.")`；
   - `task_type` 为空或非字符串类型时，抛出 `ValueError("Invalid 'task_type' in task file: <path>. Expected non-empty string.")`；
   - `task_type` 未在 `TASKS` 中注册时，由 `get_task` 抛出 `ValueError("Unknown task type: '<task_type>'. Available types: [...]")`；
2. 通过 `config_cls.model_validate(data)` 校验并实例化具体任务配置对象；
3. 使用 `tyro.cli` 将命令行 `--` 后的覆盖参数绑定至配置实例，完成最终校验；
4. 将校验后的配置对象传入 `run_task(config, device)` 分发执行。

常用命令格式：

- 命令帮助：`python -m daisy run --help`；
- 任务字段帮助：`python -m daisy run task.toml -- --help`；
- 指定设备与学习率：`python -m daisy run task.toml -d cpu -- --training.lr 0.02`；
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
commit = "auto"

[output]
save_path = "outputs/{task_id}"
```

`[meta]` 包含 `title`（标题）、`description`（实验目标或变量）、`creator`（创建人）、`created_at`（日期）和 `commit`（提交标识，`"auto"` 由运行时回填）。

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

- 示例构建：`build_example_config(task_type)` 通过 `get_task(task_type)` 获取具体的配置类，直接实例化配置对象填充默认值，设置 `meta` 预设字段；`dataset` 为必填的任务写入占位 `feeder = '<feeder_id>'`；外部任务通过 `python -m daisy example <task_type> -i <file.py>` 先导入再导出；
- 配置持久化：`save_config(config, path)` 通过 `config.model_dump(exclude={'task_file', 'task_id', 'imports'}, exclude_none=True)` 导出字典，使用 `tomli_w.dump` 写入 TOML 文件；`export_example_config(task_type, path)` 负责输出路径处理与文件写入。

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

- 学习率：`eff_batch_size = training.batch_size * training.accum_iter`，`actual_lr = training.blr * eff_batch_size / 256.0`。使用 cosine 调度，最终学习率写入快照的 `runtime.resolved_lr`；
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
- 结果表、checkpoint、预测与统计产物通过 `task_id` 关联 task 文件、commit 和关键配置；
- 已产生正式输出的 task 保留原配置，后续调整创建新 task；项目命名、输出目录与阶段映射的变更同步更新项目文档。
