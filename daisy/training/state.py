from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import icontract
import torch
import torch.nn as nn


@dataclass
class TrainState:
	"""训练循环状态；Trainer 负责更新。"""

	model: nn.Module = field(repr=False)
	optimizer: torch.optim.Optimizer | None = field(repr=False)
	device: torch.device = field(default_factory=lambda: torch.device('cpu'))

	epoch: int = 0  # 从零开始计数
	total_epochs: int = 0
	batch_idx: int = 0
	total_batches: int = 0
	global_step: int = 0  # 优化器更新次数

	batch_loss: torch.Tensor = field(default_factory=lambda: torch.zeros(()))
	stop_training: bool = False
	record: dict[str, Any] = field(default_factory=dict)

	@property
	@icontract.require(lambda self: self.optimizer is not None, 'current_lr 需要 optimizer。')
	def current_lr(self) -> float:
		"""当前第一个参数组的学习率。"""
		assert self.optimizer is not None
		return self.optimizer.param_groups[0]['lr']
