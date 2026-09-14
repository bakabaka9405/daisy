# task 规范

## 1. 适用范围

本规范定义正式 task 文件的组织方式及 `daisy/task/` 的实现约定。task 描述训练、评估或分析的实验配置；`task_id` 默认取文件名（不含扩展名），用于关联输出、日志和报表。

## 2. task 系统架构

- `runner.py`：加载 TOML、解析 `task_type`、触发任务发现并分发到对应 runner；
- `registry.py`：维护 `task_type -> config/runner` 注册表；
- `runtime.py`：统一处理运行头部、输出目录、运行快照和完成信息；
- `serialization.py`：统一保存 task 配置；
- `shared/`：共享配置模型与 transform；
- `data/`：共享数据加载、split 选择与协议快照逻辑；
- `tasks/<task_name>/`：任务专属配置与执行逻辑。

### 2.1 新增任务

1. 在 `tasks/<task_name>/` 中创建 `config.py` 和 `runner.py`；
2. 在 runner 上使用 `@TaskRegistry.register`，保持 `get_task_type()`、注册键和配置类的 `Literal[...]` 值一致；
3. 跨任务复用的配置与数据流程分别放入 `shared/` 和 `data/`。

`tasks/__init__.py` 中的 `discover_tasks()` 自动扫描并导入任务目录。公开对象由 `daisy/task/__init__.py` 显式导出，外部代码从 `daisy.task` 导入。

### 2.2 运行生命周期

runner 复用 `runtime.py`：

- `prepare_task_run()`：统一 seed、banner、commit 与输出目录；
- `save_run_snapshot()`：保存 `task_snapshot.json`，额外运行信息通过 `extra` 追加；
- `print_task_completed()`：统一任务完成输出。

## 3. CLI 参数

CLI 使用 tyro 绑定 Pydantic 配置模型。`--` 前为命令选项，后为任务配置覆盖项。优先级为 **CLI > TOML > 模型默认值**。

TOML 先由 `tomllib` 解析并经 Pydantic 校验，再应用 CLI 覆盖并校验最终配置。配置字段具备默认值，TOML 可省略采用默认值的参数。

- 命令帮助：`python -m daisy run --help`；
- 任务字段帮助：`python -m daisy run task.toml -- --help`；
- 指定设备与学习率：`python -m daisy run task.toml -d cpu -- --training.lr 0.02`。

## 4. task 文件格式

task 使用 TOML，显式声明 `task_type`、`[meta]`、`[output]` 及实验所需的数据、模型、训练或分析参数。

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

`[meta]` 包含 `title`（标题）、`description`（实验目标或变量）、`created_at`（日期）和 `commit`（提交标识，`"auto"` 由运行时回填），建议填写 `creator`（创建人）。

## 5. 任务行为与产物

### 5.1 分类

使用 `ClassificationMethod` 和 `prepare_classification_training`，由 Trainer Plugins 处理调度、保存与日志。

- `training.use_scheduler`：启用时使用 CosineAnnealing 与 warmup，关闭时保持初始学习率；
- `training.clip_grad`：启用时按 `max_norm` 裁剪梯度范数，适用于 AMP 和普通训练；
- `training.cmp_obj`：选择 `BestModel` 与 `EarlyStop` 的监控指标，支持 `acc`、`f1`、`recall`、`prec`（映射为 `precision`）；
- `output.save_best`：保存最优权重 `best_model.pth`；`output.keep_count > 0` 独立控制滚动检查点 `checkpoint_NNNN.pth`；
- `output.log`：生成 `logs/log_YYYYMMDD_HHMMSS.csv`，记录 `train_loss` 及验证集 `loss`、`acc`、`f1`、`precision`、`recall`、`auroc`。

### 5.2 MAE

预训练使用 `MAEPretrainMethod` 与 `make_dataloader`，微调使用 `MAEFinetuneMethod` 与 `prepare_classification_training`，配合 Trainer 插件执行。

- 学习率：`eff_batch_size = training.batch_size * training.accum_iter`，`actual_lr = training.blr * eff_batch_size / 256.0`。微调的 `blr` 默认 `5e-4`；直接指定 `lr` 时设置 `blr=None`，例如 `--training.blr None --training.lr 0.001`。两种方式均使用 cosine 调度，最终学习率写入快照的 `runtime.resolved_lr`；
- 预训练恢复：`training.resume` 加载 `model`、`optimizer`、`scaler`，从零基编号 `checkpoint['epoch'] + 1` 继续至目标 `epochs`，优先于 `model.timm_pretrained` 初始化；
- 预训练保存：每轮更新完整训练状态 `checkpoint_latest.pth`，最终轮保存 `checkpoint_{epoch:04d}.pth`；`training.save_freq > 0` 时额外按该间隔保存编号检查点；
- 微调保存：按验证集 `f1` 保存最优权重 `best_model.pth`；`training.save_freq > 0` 时按间隔保存纯权重 `checkpoint_{epoch:04d}.pth`；
- 输出：`output.save_path` 指定目录，`output.log` 启用 `logs/` 下的 CSV 日志。预训练日志为 `mae_log_YYYYMMDD_HHMMSS.csv`，记录训练损失与学习率；微调日志记录各轮验证指标。

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
