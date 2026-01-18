"""任务类型注册表"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .base import BaseTaskConfig, TaskRunner


class TaskRegistry:
	"""任务类型注册表，用于管理不同任务类型的配置和执行器"""

	_runners: dict[str, type[TaskRunner]] = {}
	_configs: dict[str, type[BaseTaskConfig]] = {}

	@classmethod
	def register(cls, runner_cls: type[TaskRunner]) -> type[TaskRunner]:
		"""注册任务执行器

		可作为装饰器使用:
			@TaskRegistry.register
			class MyRunner(TaskRunner):
				...
		"""
		task_type = runner_cls.get_task_type()
		cls._runners[task_type] = runner_cls
		cls._configs[task_type] = runner_cls.get_config_class()
		return runner_cls

	@classmethod
	def get_runner(cls, task_type: str) -> type[TaskRunner]:
		"""获取任务执行器类"""
		if task_type not in cls._runners:
			raise ValueError(
				f'Unknown task type: {task_type}. '
				f'Available types: {list(cls._runners.keys())}'
			)
		return cls._runners[task_type]

	@classmethod
	def get_config_class(cls, task_type: str) -> type[BaseTaskConfig]:
		"""获取任务配置类"""
		if task_type not in cls._configs:
			raise ValueError(
				f'Unknown task type: {task_type}. '
				f'Available types: {list(cls._configs.keys())}'
			)
		return cls._configs[task_type]

	@classmethod
	def list_task_types(cls) -> list[str]:
		"""列出所有已注册的任务类型"""
		return list(cls._runners.keys())

	@classmethod
	def is_registered(cls, task_type: str) -> bool:
		"""检查任务类型是否已注册"""
		return task_type in cls._runners
