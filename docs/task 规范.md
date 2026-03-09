# task 规范

## 1. 适用范围

本规范定义仓库内正式 task 配置文件的通用约束。task 是可审查、可复现的实验入口，负责描述一次训练、评估或分析所需的显式配置，不承载临时脚本逻辑。

`task_id` 默认取自 task 文件名（不含扩展名），因此文件名必须稳定、可读，并能被输出目录、日志和报表直接引用。

## 2. 文件格式

task 统一使用 `TOML` 表示，不再使用 YAML。

一个正式 task 至少应包含：

1. 顶层 `task_type`；
2. `[meta]` 元信息；
3. 与任务类型对应的数据、模型、训练或分析配置；
4. `[output]` 输出配置。

最小示意如下：

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

## 3. 文件组织与命名

### 3.1 通用规则

- 根目录 `tasks/` 用于仓库级示例任务或 UI 创建的通用任务；
- 项目级 task 应放在各自的项目目录中，例如 `proj/mae/tasks/`；
- 当前 CLI/UI 默认通过顶层 `*.toml` 扫描任务，不依赖递归搜索，因此项目级正式 task 实例应采用扁平文件名，不要把正式实例继续嵌套到多级子目录中；
- 若需要模板，可单独放在 `templates/` 子目录，由项目级 batch 或人工复制生成实例。

### 3.2 文件名规则

- 通用任务或 UI 自动创建任务可继续使用 `yyyymmdd-id[-comment].toml`；
- 项目级任务可以在此基础上定义更严格的命名约束，但必须在项目文档中写明；
- `proj/mae/tasks/` 已冻结为阶段化命名：`p{phase}-{topic}-{variant}.toml`。

`proj/mae` 示例：

```text
p1-baseline-resnet34-scratch.toml
p1-baseline-vit-base-imagenet.toml
p2-mae-mask040.toml
p3-mae-unlabel-050.toml
```

## 4. 元信息要求

正式 task 的 `[meta]` 应至少显式包含：

1. `title`：人类可读标题；
2. `description`：本次实验的目标、变量或用途；
3. `creator`：创建人；
4. `created_at`：创建日期；
5. `commit`：运行时使用的 commit hash，允许先写 `"auto"` 由运行时回填。

其中，`title`、`description`、`created_at`、`commit` 是项目级正式实验的硬要求；`creator` 建议始终填写，便于追踪来源。

## 5. 输出与追踪

- 输出目录应与 `task_id` 一一对应，默认可使用 `outputs/{task_id}`；
- 项目级实验可以在此基础上定义更严格的输出规则；`proj/mae` 的正式输出目录约定为 `outputs/mae/{phase}/{task_id}`；
- 运行产物至少应能回链到 task 文件、commit 和关键配置；
- 若任务涉及固定划分、采样清单或外部评估数据，应同时记录输入数据清单、manifest、seed 或其他关键协议参数；
- 结果表、checkpoint、预测导出与统计分析应尽量复用同一个 `task_id`。

## 6. 数据与配置要求

- 路径、模型、划分方式、超参数等必须显式写入 task，不依赖隐藏的本地默认值；
- 若使用随机划分、随机采样或多 seed，必须在 task 中显式记录相关参数；
- 若使用固定 manifest、索引表或病例级分组信息，应把这些输入视为正式协议的一部分；
- 避免在 task 中写只适用于单台机器的临时说明，机器相关差异应通过可替换字段表达。

## 7. 变更约束

1. task 一旦运行并产生正式输出，不再原地修改；后续调整应创建新 task；
2. 项目级命名、输出目录和阶段映射一旦冻结，不应在未更新项目文档的前提下私自变更；
3. 若某个项目有额外规则，应以该项目文档为准，但不能与本规范冲突。

## 8. `proj/mae` 补充约定

- 迁移顺序以 `proj/mae/docs/README.md` 和各阶段文档为准；
- 研究阶段使用 `P0` 到 `P8`，项目 task 文件名使用对应的小写 `p0` 到 `p8` 前缀；
- `proj/mae/` 只保留 docs、tasks、batch 和归档资产，不承载可复用计算实现；
- 正式实验必须通过可复现的 task 或 batch 入口运行，不再依赖根目录 `tmp/` 的一次性脚本。
