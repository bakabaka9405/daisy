"""通用多分类算法方法实现。"""

from __future__ import annotations

from typing import Any, override

import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from torch import Tensor

from daisy.classifier_trainer import create_criterion
from daisy.method.base import Method


class ClassificationMethod(Method[nn.Module]):
	"""通用多分类算法方法。"""

	@override
	def __init__(
		self,
		model: nn.Module,
		num_classes: int,
		*,
		device: torch.device,
		lr: float,
		weight_decay: float = 0.05,
		mixup: float = 0.0,
		cutmix: float = 0.0,
		smoothing: float = 0.0,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
		optimizer: torch.optim.Optimizer | None = None,
	) -> None:
		self.num_classes = num_classes
		mixup_enabled = mixup > 0.0 or cutmix > 0.0
		self.criterion = create_criterion(mixup_enabled, smoothing)
		self.mixup_fn: Mixup | None = (
			Mixup(
				mixup_alpha=mixup,
				cutmix_alpha=cutmix,
				prob=1.0,
				switch_prob=0.5,
				mode='batch',
				label_smoothing=smoothing,
				num_classes=num_classes,
			)
			if mixup_enabled
			else None
		)
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
	def build_model(self, backbone: nn.Module) -> nn.Module:
		"""直接采用传入的完整分类模型。"""
		return backbone

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		if self.mixup_fn is not None:
			images, targets = self.mixup_fn(images, targets)
		return self.criterion(self.model(images), targets)

	@override
	def predict_step(self, X: Tensor) -> torch.Tensor:
		return self.model(X)
