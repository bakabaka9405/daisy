"""Task 模块

提供任务配置、执行与序列化等功能。

基本用法:
    from pathlib import Path
    from daisy.task import load_config, run_task

    # 从 TOML 文件运行任务
    run_task(Path('tasks/example.toml'))

    # 加载配置后运行
    config = load_config(Path('tasks/example.toml'))
    run_task(config)

    # 含 run 数组的任务：逐个 run 的配置，顺序运行
    run_tasks(load_run_configs(Path('tasks/ablation.toml')))
"""

from __future__ import annotations

# 基础类
from .base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig

# 主要接口
from .feed import assert_label_range, assert_multitask_label_range, feed
from .imports import import_files
from .runner import load_config, load_run_configs, run_task, run_tasks, select_runs
from .serialization import build_example_config, export_example_config, save_config
from .shared import (
	DatasetConfig,
	FeederConfig,
	InferenceModelConfig,
	InferenceRuntimeConfig,
	LogOutputConfig,
)
from .tasks import TASKS, get_task, register_task
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
	'register_task',
	# 主要接口
	'load_config',
	'load_run_configs',
	'run_task',
	'run_tasks',
	'select_runs',
	'build_example_config',
	'export_example_config',
	'save_config',
	'import_files',
	# 数据供给
	'feed',
	'assert_label_range',
	'assert_multitask_label_range',
	# 共享配置
	'DatasetConfig',
	'FeederConfig',
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
