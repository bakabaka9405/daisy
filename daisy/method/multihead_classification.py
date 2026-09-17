"""共享 backbone 的多任务分类方法实现。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from daisy.method.base import Method
from daisy.model.multihead import MultiHeadModel

ClassificationLoss = Callable[[Tensor, Tensor], Tensor]
OptimizerFactory = Callable[[MultiHeadModel, float, float], Optimizer]


class MultiHeadClassificationMethod(Method[MultiHeadModel]):
	"""共享 backbone、每个任务一个分类头的多任务分类方法。"""

	@override
	def __init__(
		self,
		model: MultiHeadModel,
		criterion: ClassificationLoss,
		*,
		device: torch.device,
		lr: float,
		weight_decay: float = 0.05,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
		optimizer: Optimizer | None = None,
		optimizer_factory: OptimizerFactory | None = None,
	) -> None:
		# 设置当前算法特有的模型、损失函数与优化器工厂
		self.criterion: ClassificationLoss = criterion
		self._custom_optimizer_factory: OptimizerFactory | None = optimizer_factory

		super().__init__(
			model,
			device=device,
			lr=lr,
			weight_decay=weight_decay,
			use_amp=use_amp,
			accum_iter=accum_iter,
			clip_grad=clip_grad,
			optimizer=optimizer,
		)

	@override
	def build_model(self, backbone: nn.Module) -> MultiHeadModel:
		"""校验并返回 MultiHeadModel 实例。"""
		if not isinstance(backbone, MultiHeadModel):
			raise TypeError(f'Expected MultiHeadModel, got {type(backbone).__name__}')
		return backbone

	@override
	def build_optimizer(self, model: MultiHeadModel, *, lr: float, weight_decay: float) -> Optimizer:
		"""使用自定义工厂或默认基类构建优化器。"""
		if self._custom_optimizer_factory is not None:
			return self._custom_optimizer_factory(model, lr, weight_decay)
		return super().build_optimizer(model, lr=lr, weight_decay=weight_decay)

	def process(self, batch: tuple[Tensor, Sequence[Tensor]]) -> torch.Tensor:
		"""执行单批计算并按任务头平均损失。"""
		images, targets = batch
		logits_list = self.model(images)
		losses = [self.criterion(logits, target) for logits, target in zip(logits_list, targets, strict=True)]

		return torch.stack(losses).mean()
