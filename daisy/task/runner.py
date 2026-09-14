"""任务执行引擎"""

from __future__ import annotations

from pathlib import Path
import tomllib

import torch

from .base import BaseTaskConfig
from .tasks import get_task


def load_config(path: str | Path) -> BaseTaskConfig:
	"""从 TOML 文件加载任务配置

	根据 TOML 中的 task_type 字段自动选择正确的配置类。

	Args:
		path: TOML 文件路径

	Returns:
		对应任务类型的配置对象
	"""
	path = Path(path)

	with open(path, 'rb') as f:
		data = tomllib.load(f)

	task_type = data.get('task_type')
	if task_type is None:
		raise ValueError(f"Missing 'task_type' in task file: {path}. Please specify a task type.")
	if not isinstance(task_type, str) or not task_type:
		raise ValueError(f"Invalid 'task_type' in task file: {path}. Expected non-empty string.")

	config_cls, _ = get_task(task_type)
	config = config_cls.model_validate(data)

	config.task_file = path
	config.task_id = path.stem

	return config


def run_task(
	config: BaseTaskConfig | str | Path,
	device: torch.device | str | None = None,
) -> Path:
	"""运行任务

	Args:
		config: BaseTaskConfig 对象或 TOML 文件路径
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		输出路径
	"""
	if isinstance(config, (str, Path)):
		config = load_config(config)

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	_, run_fn = get_task(config.task_type)
	return run_fn(config, device)
