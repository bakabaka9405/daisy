# task 规范

## 1. 适用范围

本规范同时定义两类约束：

1. 仓库内正式 `task` 文件的内容与组织方式；
2. `daisy/task/` 模块的实现边界、扩展方式与复用规则。

`task` 是可审查、可复现的实验入口，负责描述一次训练、评估或分析所需的显式配置，不承载临时脚本逻辑。

`task_id` 默认取自 task 文件名（不含扩展名），因此文件名必须稳定、可读，并能被输出目录、日志和报表直接引用。

## 2. task 系统架构

当前 `daisy/task/` 的主结构如下：

- `runner.py`：加载 TOML、解析 `task_type`、触发任务发现并分发到对应 runner；
- `registry.py`：维护 `task_type -> config/runner` 注册表；
- `runtime.py`：统一处理运行头部、输出目录、运行快照和完成信息；
- `serialization.py`：统一保存 task 配置；
- `shared/`：放共享配置模型、共享 transform 等可跨任务复用的定义；
- `data/`：放带标签数据加载、split 选择等共享数据流；
- `tasks/<task_name>/`：任务专属配置与执行逻辑；
- `ui_builder.py` 与 `ui_config.py`：从配置模型和 runner UI 配置生成界面。

### 2.1 任务发现

- 任务发现由 `daisy/task/tasks/__init__.py` 中的 `discover_tasks()` 负责；
- 新 task 目录会被自动扫描并导入，不再维护手工 import 列表；
- 新增任务类型时，不应再通过编辑 `tasks/__init__.py` 注册模块。

### 2.2 共享层分工

- 任务专属配置放在 `daisy/task/tasks/<task_name>/config.py`；
- 多个任务共享的配置模型应下沉到 `daisy/task/shared/`；
- 多个任务共享的数据选择、split 和协议快照逻辑应下沉到 `daisy/task/data/`；
- 不要通过 `task A` 的 `config.py` 或 `runner.py` 让 `task B` 反向依赖其内部实现。

### 2.3 统一运行生命周期

runner 应优先复用 `daisy/task/runtime.py` 中的共享流程：

- `prepare_task_run()`：统一 seed、banner、commit 与输出目录；
- `save_run_snapshot()`：统一保存 `task_snapshot.json`；
- `print_task_completed()`：统一任务完成输出。

若某个 task 需要额外运行信息，应在 `save_run_snapshot(..., extra=...)` 中追加，而不是重新拼装整套运行时元数据。

### 2.4 UI 元数据单一来源

- UI 字段配置统一由 `TaskRunner.get_ui_field_overrides()` 返回；
- 字段配置类型统一使用 `daisy/task/ui_config.py` 中的 `UIFieldConfig`；
- 不再使用 `json_schema_extra` 或历史 `ui_schema.py` 携带 UI 元数据；
- 自定义 UI 仍可通过 `build_custom_ui()` 提供，但自动 UI 的字段控制应只走 `UIFieldConfig`。

## 3. 新增 task 类型的实现规范

新增 task 时应遵循以下流程：

1. 在 `daisy/task/tasks/<task_name>/` 下创建 `config.py` 和 `runner.py`；
2. 在 runner 上使用 `@TaskRegistry.register`；
3. 保持 `get_task_type()`、注册键和配置类中的 `Literal[...]` 值一致；
4. 共享配置优先放入 `shared/`，共享数据流程优先放入 `data/`；
5. UI 自动表单字段配置通过 `get_ui_field_overrides()` 返回 `UIFieldConfig`；
6. 运行时输出、快照和完成提示优先复用 `runtime.py` 的统一辅助函数。

## 4. task 文件格式

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

### 4.1 `task_type` 规则

- 新 task 文件必须显式声明 `task_type`；
- 运行时只对历史遗留的 `classification` 风格配置保留缺省兼容；
- 其他类型若缺失 `task_type`，应视为配置错误并显式报错。

## 5. 文件组织与命名

### 5.1 通用规则

- 根目录 `tasks/` 用于仓库级示例任务或 UI 创建的通用任务；
- 项目级 task 应放在各自的项目目录中，例如 `proj/mae/tasks/`；
- 当前 CLI/UI 默认通过顶层 `*.toml` 扫描任务，不依赖递归搜索，因此项目级正式 task 实例应采用扁平文件名，不要把正式实例继续嵌套到多级子目录中；
- 若需要模板，可单独放在 `templates/` 子目录，由项目级 batch 或人工复制生成实例。

### 5.2 文件名规则

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

## 6. 元信息要求

正式 task 的 `[meta]` 应至少显式包含：

1. `title`：人类可读标题；
2. `description`：本次实验的目标、变量或用途；
3. `creator`：创建人；
4. `created_at`：创建日期；
5. `commit`：运行时使用的 commit hash，允许先写 `'auto'` 由运行时回填。

其中，`title`、`description`、`created_at`、`commit` 是项目级正式实验的硬要求；`creator` 建议始终填写，便于追踪来源。

## 7. 输出与追踪

- 输出目录应与 `task_id` 一一对应，默认可使用 `outputs/{task_id}`；
- 项目级实验可以在此基础上定义更严格的输出规则；`proj/mae` 的正式输出目录约定为 `outputs/mae/{phase}/{task_id}`；
- 运行产物至少应能回链到 task 文件、commit 和关键配置；
- 若任务涉及固定划分、采样清单或外部评估数据，应同时记录输入数据清单、划分方式、seed 或其他关键参数；
- 结果表、checkpoint、预测导出与统计分析应尽量复用同一个 `task_id`。

## 8. 数据与配置要求

- 路径、模型、划分方式、超参数等必须显式写入 task，不依赖隐藏的本地默认值；
- 若使用随机划分、随机采样或多 seed，必须在 task 中显式记录相关参数；
- 若使用固定索引表、sheet 或病例级分组信息，应把这些输入视为正式协议的一部分；
- 避免在 task 中写只适用于单台机器的临时说明，机器相关差异应通过可替换字段表达。

## 9. 导入与公开 API 规则

- 外部代码应统一从 `daisy.task` 导入公开对象；
- `daisy/task/config.py` 和 `daisy/task/compat.py` 已移除，不应重新引入旧兼容入口；
- 新增共享配置时，若需要对外公开，应通过 `daisy/task/__init__.py` 显式导出；
- 不要再新增 `TaskConfig`、`MetaConfig`、`OutputConfig` 这类历史别名。

## 10. 变更约束

1. task 一旦运行并产生正式输出，不再原地修改；后续调整应创建新 task；
2. 项目级命名、输出目录和阶段映射一旦冻结，不应在未更新项目文档的前提下私自变更；
3. 若某个项目有额外规则，应以该项目文档为准，但不能与本规范冲突。

## 11. `proj/mae` 补充约定

- 迁移顺序以 `proj/mae/docs/README.md` 和各阶段文档为准；
- 研究阶段使用 `P0` 到 `P8`，项目 task 文件名使用对应的小写 `p0` 到 `p8` 前缀；
- `proj/mae/` 只保留 docs、tasks、batch 和归档资产，不承载可复用计算实现；
- 正式实验必须通过可复现的 task 或 batch 入口运行，不再依赖根目录 `tmp/` 的一次性脚本。
