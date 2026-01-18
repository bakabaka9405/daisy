"""Task 模块

提供任务配置、执行、注册等功能。

基本用法:
    from daisy.task import load_config, run_task

    # 从 TOML 文件运行任务
    run_task('tasks/example.toml')

    # 加载配置后运行
    config = load_config('tasks/example.toml')
    run_task(config)

扩展任务类型:
    from daisy.task import TaskRegistry, BaseTaskConfig, TaskRunner

    @TaskRegistry.register
    class MyRunner(TaskRunner):
        ...
"""

# 基础类
from .base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig, TaskRunner

# 注册表
from .registry import TaskRegistry

# 主要接口
from .runner import load_config, run_task
from .config import save_config

# 导入任务模块以确保注册
from . import tasks  # noqa: F401

# 为向后兼容，导出分类任务相关类
from .tasks.classification.config import (
	ClassificationConfig,
	DatasetConfig,
	DatasetSplitConfig,
	ModelConfig,
	TrainingConfig,
	TransformConfig,
)

# 向后兼容：TaskConfig 作为 ClassificationConfig 的别名
TaskConfig = ClassificationConfig

# 为向后兼容，从分类任务配置导出 MetaConfig 和 OutputConfig
# 注意：这些现在是 BaseMetaConfig 和 BaseOutputConfig 的别名
MetaConfig = BaseMetaConfig
OutputConfig = BaseOutputConfig

__all__ = [
	# 基础类
	'BaseMetaConfig',
	'BaseOutputConfig',
	'BaseTaskConfig',
	'TaskRunner',
	# 注册表
	'TaskRegistry',
	# 主要接口
	'load_config',
	'run_task',
	'save_config',
	# 分类任务（向后兼容）
	'TaskConfig',
	'ClassificationConfig',
	'DatasetConfig',
	'DatasetSplitConfig',
	'ModelConfig',
	'TrainingConfig',
	'TransformConfig',
	'MetaConfig',
	'OutputConfig',
]
