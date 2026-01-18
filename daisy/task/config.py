"""TOML 任务配置解析模块 - 兼容层

此模块为向后兼容保留，新代码应从 daisy.task 导入。
"""

from pathlib import Path

import tomli_w

# 从新位置导入配置类
from .base import BaseMetaConfig as MetaConfig
from .base import BaseOutputConfig as OutputConfig
from .tasks.classification.config import (
	ClassificationConfig as TaskConfig,
	DatasetConfig,
	DatasetSplitConfig,
	ModelConfig,
	TrainingConfig,
	TransformConfig,
)
from .runner import load_config

__all__ = [
	'MetaConfig',
	'DatasetSplitConfig',
	'DatasetConfig',
	'ModelConfig',
	'TransformConfig',
	'TrainingConfig',
	'OutputConfig',
	'TaskConfig',
	'load_config',
	'save_config',
]


def save_config(config: TaskConfig, path: str | Path):
	"""保存任务配置到 TOML 文件"""
	path = Path(path)
	data = config.model_dump(exclude={'task_file', 'task_id'}, exclude_none=True)

	with open(path, 'wb') as f:
		tomli_w.dump(data, f)
