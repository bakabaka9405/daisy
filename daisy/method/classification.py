"""通用多分类算法方法实现。"""

from __future__ import annotations

from typing import override

import icontract
import numpy as np
import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from torch import Tensor
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from daisy.method.base import Method
from daisy.metrics import EvalMetrics, auroc_score, fast_calc_metrics
from daisy.training.trainer import Trainer
from daisy.typing import GenericDataLoader as DataLoader


@icontract.require(lambda smoothing: 0 <= smoothing < 1)
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
	@icontract.require(lambda smoothing: 0 <= smoothing < 1)
	@icontract.require(lambda mixup, cutmix: mixup >= 0 and cutmix >= 0)
	@icontract.require(lambda num_classes: num_classes >= 2)
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

	@icontract.require(lambda self, batch: self.mixup_fn is None or batch[0].shape[0] % 2 == 0, 'Mixup 要求批大小为偶数')
	@icontract.require(lambda batch: batch[0].shape[0] == batch[1].shape[0])
	@icontract.ensure(lambda result: result.ndim == 0)
	def process(self, batch: tuple[Tensor, Tensor] | list[Tensor]) -> torch.Tensor:
		images, targets = batch
		if self.mixup_fn is not None:
			images, targets = self.mixup_fn(images, targets)
		return self.criterion(self.model(images), targets)

	@override
	@icontract.ensure(lambda self, X, result: result.shape == (X.shape[0], self.num_classes))
	def inference_step(self, X: Tensor) -> torch.Tensor:
		"""执行单批推理，返回原始未归一化的分类 logits。"""
		return self.model(X)

	@override
	@icontract.ensure(lambda X, result: result.shape == (X.shape[0],))
	def predict_step(self, X: Tensor) -> torch.Tensor:
		"""执行单批预测，调用 inference_step 并返回解码后的类别索引。"""
		return self.inference_step(X).argmax(dim=1)

	def evaluate(self, trainer: Trainer, data_loader: DataLoader) -> EvalMetrics:
		"""使用 Trainer 执行推理并计算分类评估指标。"""
		scores, targets = trainer.inference(
			data_loader,
			lambda batch: (self.inference_step(batch[0]), batch[1]),
		)
		scores = scores.float()
		loss = nn.functional.cross_entropy(scores, targets)
		y_pred = torch.argmax(scores, dim=1).numpy()
		y_true = targets.numpy()

		metrics = fast_calc_metrics(y_true, y_pred, num_classes=self.num_classes)

		y_outputs = scores.numpy()
		auroc = auroc_score(
			y_true.astype(np.int64, copy=False),
			y_outputs.astype(np.float64, copy=False),
			mode='macro',
		)

		assert isinstance(auroc, np.float64)

		return EvalMetrics(
			loss=loss.item(),
			acc=metrics.acc,
			precision=metrics.precision,
			recall=metrics.recall,
			f1=metrics.f1,
			auroc=auroc,
			confusion_matrix=metrics.confusion_matrix,
		)
