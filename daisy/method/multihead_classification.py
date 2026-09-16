"""共享 backbone 的多任务分类方法实现。"""

from __future__ import annotations

from typing import Any, override

import torch
import torch.nn as nn

from daisy.method.base import Method
from daisy.method.classification import create_criterion
from daisy.model.multihead import MultiHeadModel


class MultiHeadClassificationMethod(Method[MultiHeadModel]):
	"""共享 backbone、每个任务一个分类头的多任务分类方法。

	batch 约定为 ``(images, targets)``，其中 ``targets`` 是长度为任务数的标签序列，
	第 ``i`` 个元素对应该任务头。每个任务使用标准分类损失，按任务平均。
	"""

	@override
	def __init__(
		self,
		backbone: nn.Module,
		num_classes_list: list[int],
		*,
		device: torch.device,
		lr: float,
		weight_decay: float = 0.05,
		smoothing: float = 0.0,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
		optimizer: torch.optim.Optimizer | None = None,
	) -> None:
		self.num_classes_list = list(num_classes_list)
		self.criterion = create_criterion(False, smoothing)
		super().__init__(
			backbone,
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
		return MultiHeadModel(backbone, self.num_classes_list)

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		logits_list = self.model(images)
		losses = [self.criterion(logits, target) for logits, target in zip(logits_list, targets)]
		return torch.stack(losses).mean()
