"""通用多分类算法方法实现。"""

from __future__ import annotations

from typing import Any, override

import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from torch import Tensor
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from daisy.method.base import Method


def create_criterion(mixup_enabled: bool, smoothing: float) -> nn.Module:
	"""根据配置创建损失函数"""
	if mixup_enabled:
		# Mixup/CutMix 需要 soft target
		return SoftTargetCrossEntropy()
	elif smoothing > 0:
		return LabelSmoothingCrossEntropy(smoothing=smoothing)
	else:
		return nn.CrossEntropyLoss()


class ClassificationMethod(Method[nn.Module]):
	"""通用多分类算法方法。"""

	@override
	def __init__(
		self,
		model: nn.Module,
		num_classes: int,
		*,
		lr: float,
		weight_decay: float = 0.05,
		mixup: float = 0.0,
		cutmix: float = 0.0,
		smoothing: float = 0.0,
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
			lr=lr,
			weight_decay=weight_decay,
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
