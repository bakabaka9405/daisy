"""共享 backbone 的多任务分类方法实现。"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn
from torch import Tensor

from daisy.method.base import Method
from daisy.model.multihead import MultiHeadModel

ClassificationLoss = Callable[[Tensor, Tensor], Tensor]


class MultiHeadClassificationMethod(Method[MultiHeadModel]):
	"""共享 backbone、每个任务一个分类头的多任务分类方法。"""

	@override
	def __init__(
		self,
		model: MultiHeadModel,
		criterion: ClassificationLoss,
		*,
		lr: float,
		weight_decay: float = 0.05,
	) -> None:
		self.criterion: ClassificationLoss = criterion

		super().__init__(
			model,
			lr=lr,
			weight_decay=weight_decay,
		)

	@override
	def build_model(self, backbone: nn.Module) -> MultiHeadModel:
		"""校验并返回 MultiHeadModel 实例。"""
		if not isinstance(backbone, MultiHeadModel):
			raise TypeError(f'Expected MultiHeadModel, got {type(backbone).__name__}')
		return backbone

	def process(self, batch: tuple[Tensor, dict[str, Tensor]]) -> torch.Tensor:
		"""执行单批计算并按任务头名称计算平均损失。"""
		images, targets = batch
		logits_dict = self.model(images)
		if targets.keys() != logits_dict.keys():
			raise ValueError(
				f'Target task keys {set(targets.keys())} do not match model output task keys {set(logits_dict.keys())}'
			)

		losses = [self.criterion(logits_dict[task_name], targets[task_name]) for task_name in logits_dict]
		return torch.stack(losses).mean()
