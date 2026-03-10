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

from .serialization import save_config
from .shared import DatasetConfig, DatasetSplitConfig, InferenceModelConfig, InferenceRuntimeConfig
from .tasks.classification.config import (
	ClassificationConfig,
	ModelConfig,
	TrainingConfig,
	TransformConfig,
)
from .tasks.eval_checkpoint.config import EvalCheckpointConfig
from .tasks.mae_finetune.config import MAEFinetuneConfig
from .tasks.mae_pretrain.config import MAEPretrainConfig
from .tasks.moco_lincls.config import MoCoLinclsConfig
from .tasks.moco_pretrain.config import MoCoPretrainConfig
from .tasks.predict_export.config import PredictExportConfig

# 导入任务模块以确保注册
from . import tasks  # noqa: F401

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
	# 共享配置
	'DatasetConfig',
	'DatasetSplitConfig',
	'InferenceModelConfig',
	'InferenceRuntimeConfig',
	# 任务配置
	'ClassificationConfig',
	'MAEFinetuneConfig',
	'MAEPretrainConfig',
	'EvalCheckpointConfig',
	'PredictExportConfig',
	'MoCoLinclsConfig',
	'MoCoPretrainConfig',
	# 分类任务细粒度配置
	'ModelConfig',
	'TrainingConfig',
	'TransformConfig',
]
