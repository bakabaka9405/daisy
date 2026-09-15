"""Task 模块

提供任务配置、执行与序列化等功能。

基本用法:
    from daisy.task import load_config, run_task

    # 从 TOML 文件运行任务
    run_task('tasks/example.toml')

    # 加载配置后运行
    config = load_config('tasks/example.toml')
    run_task(config)
"""

from __future__ import annotations

# 基础类
from .base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig

# 主要接口
from .runner import load_config, run_task
from .serialization import build_example_config, export_example_config, save_config
from .shared import DatasetConfig, DatasetSplitConfig, InferenceModelConfig, InferenceRuntimeConfig, LogOutputConfig
from .tasks import TASKS, get_task
from .tasks.classification.config import (
	ClassificationConfig,
	ModelConfig,
	TrainingConfig,
	TransformConfig,
)
from .tasks.mae_pretrain.config import MAEPretrainConfig

__all__ = [
	# 基础类
	'BaseMetaConfig',
	'BaseOutputConfig',
	'BaseTaskConfig',
	# 任务字典与查找
	'TASKS',
	'get_task',
	# 主要接口
	'load_config',
	'run_task',
	'build_example_config',
	'export_example_config',
	'save_config',
	# 共享配置
	'DatasetConfig',
	'DatasetSplitConfig',
	'InferenceModelConfig',
	'InferenceRuntimeConfig',
	'LogOutputConfig',
	# 任务配置
	'ClassificationConfig',
	'MAEPretrainConfig',
	# 分类任务细粒度配置
	'ModelConfig',
	'TrainingConfig',
	'TransformConfig',
]
