"""任务协议：由结构化配置构造的任务、按 task_type 注册的根任务类与已完成运行的分析视图"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from .base import BaseConfig, BaseTaskConfig
from .runtime import save_task_snapshot


class Task[C: BaseConfig, R](Protocol):
	"""任务：唯一输入为构造时传入的结构化配置，run 在上下文给定的输出目录中执行并把结果交给调用方

	根任务由 runner 以 TOML 配置构造；子任务由父任务在 run 中构造配置与任务，经 RunContext.run 运行。
	"""

	@property
	def config(self) -> C: ...

	def run(self, ctx: RunContext) -> R: ...


@dataclass(slots=True, frozen=True)
class RunContext:
	"""任务的运行上下文；根任务的由 runner 构建，子任务的由 run 派生"""

	output_dir: Path
	device: torch.device

	def run[R](self, key: str, task: Task[Any, R]) -> R:
		"""在 output_dir / key 中运行子任务：创建目录、保存配置快照，返回子任务结果"""
		output_dir = self.output_dir / key
		output_dir.mkdir(parents=True, exist_ok=True)
		save_task_snapshot(output_dir, task, self.device)
		return task.run(RunContext(output_dir, self.device))


class TaskRun(Protocol):
	"""运行对象，由运行目录与配置构成；任务执行与离线分析共用其中的数据与推理功能"""

	@property
	def run_dir(self) -> Path: ...

	@property
	def config(self) -> BaseTaskConfig: ...


class TaskType[C: BaseTaskConfig](Protocol):
	"""根任务类：以 TOML 配置构造任务，按 task_type 注册

	config_type: 任务配置类，其 task_type 默认值即任务类型
	__call__: 即类的构造函数，接收配置返回根任务；输出目录、快照与溯源由 runner 处理
	open_run: 由运行目录与快照配置构建分析视图；不恢复训练状态，模型与数据在首次访问时加载
	"""

	@property
	def config_type(self) -> type[C]: ...

	def __call__(self, config: C) -> Task[C, object]: ...

	def open_run(self, run_dir: Path, config: C, device: torch.device) -> TaskRun: ...
