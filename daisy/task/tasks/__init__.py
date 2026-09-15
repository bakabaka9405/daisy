"""任务类型模块与任务注册表。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..base import BaseTaskConfig
from .classification.config import ClassificationConfig
from .classification.runner import run as run_classification
from .mae_pretrain.config import MAEPretrainConfig
from .mae_pretrain.runner import run as run_mae_pretrain

type TaskEntry = tuple[type[BaseTaskConfig], Callable[[Any, torch.device], Path]]

TASKS: dict[str, TaskEntry] = {
	'classification': (ClassificationConfig, run_classification),
	'mae_pretrain': (MAEPretrainConfig, run_mae_pretrain),
}


def get_task(task_type: str) -> TaskEntry:
	"""根据任务类型获取 (配置类, 执行函数) 元组。"""
	task = TASKS.get(task_type)
	if task is None:
		raise ValueError(f"Unknown task type: '{task_type}'. Available types: {list(TASKS.keys())}")
	return task


__all__ = [
	'TASKS',
	'get_task',
	'TaskEntry',
]
