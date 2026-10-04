"""任务类型模块与任务注册表。"""

from __future__ import annotations

from collections.abc import Callable
import inspect
from pathlib import Path
from typing import Any, get_type_hints

import icontract
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


@icontract.require(lambda task_type: task_type in TASKS, 'Unknown task type')
def get_task(task_type: str) -> TaskEntry:
	"""根据任务类型获取 (配置类, 执行函数) 元组。"""
	return TASKS[task_type]


def _first_param_name(run_fn: Callable[..., Path]) -> str:
	"""执行函数首个参数名"""
	return next(iter(inspect.signature(run_fn).parameters))


def _annotates_task_config(run_fn: Callable[..., Path]) -> bool:
	"""执行函数首参注解是否为 BaseTaskConfig 子类"""
	config_cls = get_type_hints(run_fn)[_first_param_name(run_fn)]
	return isinstance(config_cls, type) and issubclass(config_cls, BaseTaskConfig)


@icontract.require(lambda run_fn: _annotates_task_config(run_fn), 'Task runner config annotation must be a BaseTaskConfig subclass')
@icontract.require(
	lambda run_fn: _first_param_name(run_fn) in get_type_hints(run_fn),
	'Task runner must annotate its first parameter with a BaseTaskConfig subclass',
)
@icontract.require(
	lambda run_fn: len(inspect.signature(run_fn).parameters) > 0, 'Task runner must take the task config as its first parameter'
)
def _resolve_config_cls(run_fn: Callable[..., Path]) -> type[BaseTaskConfig]:
	"""从执行函数首参注解推导任务配置类"""
	return get_type_hints(run_fn)[_first_param_name(run_fn)]


def register_task[F: Callable[..., Path]](name: str) -> Callable[[F], F]:
	"""注册任务执行函数的装饰器，配置类由执行函数首参注解推导。

	Args:
		name: 任务类型唯一标识符，须与配置类 task_type 默认值一致
	"""

	@icontract.require(
		lambda run_fn: _resolve_config_cls(run_fn).model_fields['task_type'].default == name,
		'Task config declares a different task_type default',
	)
	@icontract.require(lambda run_fn: name not in TASKS, 'Task is already registered.')
	@icontract.ensure(lambda run_fn: TASKS[name][1] is run_fn)
	def decorator(run_fn: F) -> F:
		TASKS[name] = (_resolve_config_cls(run_fn), run_fn)
		return run_fn

	return decorator


__all__ = [
	'TASKS',
	'get_task',
	'register_task',
	'TaskEntry',
]
