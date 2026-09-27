"""任务类型模块与任务注册表。"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
from typing import Any, get_type_hints

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


def _resolve_config_cls(run_fn: Callable[..., Path]) -> type[BaseTaskConfig]:
	"""从执行函数首参注解推导任务配置类"""
	qualname = getattr(run_fn, '__qualname__', repr(run_fn))
	parameters = list(inspect.signature(run_fn).parameters.values())
	if not parameters:
		raise TypeError(f'Task runner {qualname} must take the task config as its first parameter')

	hints = get_type_hints(run_fn)
	config_name = parameters[0].name
	if config_name not in hints:
		raise TypeError(f'Task runner {qualname} must annotate parameter {config_name!r} with a BaseTaskConfig subclass')
	config_cls = hints[config_name]
	if not (isinstance(config_cls, type) and issubclass(config_cls, BaseTaskConfig)):
		raise TypeError(f'Task runner {qualname} config annotation must be a BaseTaskConfig subclass, got {config_cls!r}')
	return config_cls


def register_task[F: Callable[..., Path]](name: str) -> Callable[[F], F]:
	"""注册任务执行函数的装饰器，配置类由执行函数首参注解推导。

	Args:
		name: 任务类型唯一标识符，须与配置类 task_type 默认值一致
	"""

	def decorator(run_fn: F) -> F:
		if name in TASKS:
			raise ValueError(f"Task '{name}' is already registered.")
		config_cls = _resolve_config_cls(run_fn)
		task_type = config_cls.model_fields['task_type'].default
		if task_type != name:
			raise ValueError(f"Task '{name}' config {config_cls.__name__} declares task_type {task_type!r}")
		TASKS[name] = (config_cls, run_fn)
		return run_fn

	return decorator


__all__ = [
	'TASKS',
	'get_task',
	'register_task',
	'TaskEntry',
]
