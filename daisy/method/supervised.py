"""完整模型与损失、解码函数的监督训练适配。"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import icontract
import torch.nn as nn
from torch import Tensor

from daisy.method.base import Method


class SupervisedMethod[Output, Targets](Method[nn.Module]):
	"""接收完整模型，支持单任务或多任务的原生输出与标签结构。"""

	def __init__(
		self,
		model: nn.Module,
		criterion: Callable[[Output, Targets], Tensor],
		decode: Callable[[Output], Tensor | dict[str, Tensor]],
		*,
		lr: float,
		weight_decay: float = 0.05,
	) -> None:
		super().__init__(model, lr=lr, weight_decay=weight_decay)
		self.criterion = criterion
		self.decode = decode

	@icontract.ensure(lambda result: result.ndim == 0)
	def process(self, batch: tuple[Tensor, Targets]) -> Tensor:
		"""执行一次前向并计算训练损失。"""
		images, targets = batch
		return self.criterion(self.inference_step(images), targets)

	@override
	def inference_step(self, X: Tensor) -> Output:
		return self.model(X)

	@override
	def predict_step(self, X: Tensor) -> Tensor | dict[str, Tensor]:
		return self.decode(self.inference_step(X))
