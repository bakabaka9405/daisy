"""任务协议：可嵌套的任务节点、由 TOML 配置构造的根任务与已完成运行的分析视图"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from .base import BaseTaskConfig


@dataclass(slots=True, frozen=True)
class RunContext:
	"""任务节点的运行上下文；根节点的由 runner 构建，子节点的由父节点派生"""

	output_dir: Path
	device: torch.device

	def child(self, key: str) -> RunContext:
		"""子节点上下文，输出目录为 output_dir / key（立即创建）"""
		output_dir = self.output_dir / key
		output_dir.mkdir(parents=True, exist_ok=True)
		return RunContext(output_dir, self.device)


class Task[R](Protocol):
	"""任务节点：输入在构造时绑定，run 在上下文给定的输出目录中执行并把结果交给调用方

	子任务即父任务在 run 中构造并调用的普通节点，嵌套方式与结果的使用由父任务决定。
	"""

	def run(self, ctx: RunContext) -> R: ...


class TaskRun(Protocol):
	"""运行对象，由运行目录与配置构成；任务执行与离线分析共用其中的数据与推理功能"""

	@property
	def run_dir(self) -> Path: ...

	@property
	def config(self) -> BaseTaskConfig: ...


class TaskType[C: BaseTaskConfig](Protocol):
	"""根任务类：以 TOML 配置构造任务节点，按 task_type 注册

	config_type: 任务配置类，其 task_type 默认值即任务类型
	__call__: 即类的构造函数，接收配置返回根节点；输出目录、快照与溯源由 runner 处理
	open_run: 由运行目录与快照配置构建分析视图；不恢复训练状态，模型与数据在首次访问时加载
	"""

	@property
	def config_type(self) -> type[C]: ...

	def __call__(self, config: C) -> Task[object]: ...

	def open_run(self, run_dir: Path, config: C, device: torch.device) -> TaskRun: ...
