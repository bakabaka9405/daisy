"""任务协议：配置类型、训练执行与已完成运行的分析视图"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .base import BaseTaskConfig


class TaskRun(Protocol):
	"""运行对象，由运行目录与配置构成；任务执行与离线分析共用其中的数据与推理功能"""

	@property
	def run_dir(self) -> Path: ...

	@property
	def config(self) -> BaseTaskConfig: ...


class Task[C: BaseTaskConfig](Protocol):
	"""任务协议

	config_type: 任务配置类，其 task_type 默认值即任务类型
	run: 执行任务并返回输出目录
	open_run: 由运行目录与快照配置构建分析视图；不恢复训练状态，模型与数据在首次访问时加载
	"""

	@property
	def config_type(self) -> type[C]: ...

	def run(self, config: C, device: torch.device) -> Path: ...

	def open_run(self, run_dir: Path, config: C, device: torch.device) -> TaskRun: ...
