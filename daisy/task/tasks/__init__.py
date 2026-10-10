"""任务类型模块与任务注册表。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import icontract

from ..protocol import TaskType
from .classification.runner import ClassificationTask
from .mae_pretrain.runner import MAEPretrainTask

TASKS: dict[str, TaskType[Any]] = {
	'classification': ClassificationTask,
	'mae_pretrain': MAEPretrainTask,
}


@icontract.require(lambda task_type: task_type in TASKS, 'Unknown task type')
def get_task(task_type: str) -> TaskType[Any]:
	"""根据任务类型获取根任务类。"""
	return TASKS[task_type]


def register_task[T: TaskType[Any]](name: str) -> Callable[[T], T]:
	"""注册根任务类的装饰器，将类本身登记到 TASKS。

	Args:
		name: 任务类型唯一标识符，须与 config_type 的 task_type 默认值一致
	"""

	@icontract.require(lambda cls: name not in TASKS, 'Task is already registered.')
	@icontract.ensure(lambda cls: TASKS[name] is cls)
	def decorator(cls: T) -> T:
		declared = cls.config_type.model_fields['task_type'].default
		if declared != name:
			raise ValueError(f"Task '{name}' config declares task_type default {declared!r}")
		TASKS[name] = cls
		return cls

	return decorator


__all__ = [
	'TASKS',
	'get_task',
	'register_task',
]
