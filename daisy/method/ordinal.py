"""有序分类与回归算法实现。"""

from __future__ import annotations

from abc import abstractmethod
import math
from typing import Any, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from daisy.method.base import Method
from daisy.model.ordinal import (
	ClassificationModel,
	CLMModel,
	ContrastiveClassificationModel,
	CORALModel,
	CORNModel,
	BackboneModel,
	ScalarModel,
)


def _ranking_term(score: torch.Tensor, y: torch.Tensor, *, hinge: bool) -> torch.Tensor:
	"""批次内有序对排序项；无有效对时返回零并保持梯度图。"""
	diff = score.unsqueeze(1) - score.unsqueeze(0)
	valid = y.unsqueeze(1) > y.unsqueeze(0)
	if not bool(valid.any()):
		return score.sum() * 0.0
	margin = diff[valid]
	return F.relu(1.0 - margin).mean() if hinge else F.softplus(-margin).mean()


def _contrastive_term(emb: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	"""批次内距离对比项：欧氏距离拟合标签差绝对值。"""
	batch = emb.shape[0]
	if batch < 2:
		return emb.sum() * 0.0
	dist = torch.cdist(emb, emb)
	label_diff = (y.unsqueeze(1) - y.unsqueeze(0)).abs().to(emb.dtype)
	rows, cols = torch.triu_indices(batch, batch, offset=1, device=emb.device)
	return (dist[rows, cols] - label_diff[rows, cols]).pow(2).mean()


def _cumulative_bce(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> torch.Tensor:
	"""累积 BCE 损失。"""
	target = (y.unsqueeze(1) > torch.arange(num_classes - 1, device=y.device)).to(logits.dtype)
	bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
	return bce.sum(dim=1).mean()


def _clm_log_probabilities(model: CLMModel, score: torch.Tensor) -> torch.Tensor:
	"""计算 CLM 模型的各类对数概率。"""
	threshold = model.threshold
	if model.num_classes == 2:
		first = F.logsigmoid(threshold - score)
		last = F.logsigmoid(score - threshold)
		return torch.stack([first, last], dim=1)

	delta = model.delta
	assert delta is not None
	gaps = F.softplus(delta)
	offsets = torch.cat([gaps.new_zeros(1), torch.cumsum(gaps, dim=0)])
	tau = threshold + offsets

	first = F.logsigmoid(tau[0] - score)
	last = F.logsigmoid(score - tau[-1])
	mid = (
		F.logsigmoid(tau[1:].unsqueeze(0) - score.unsqueeze(1))
		+ F.logsigmoid(score.unsqueeze(1) - tau[:-1].unsqueeze(0))
		+ F.logsigmoid(delta).unsqueeze(0)
	)
	return torch.cat([first.unsqueeze(1), mid, last.unsqueeze(1)], dim=1)


class OrdinalBaseMethod[M: BackboneModel](Method[M]):
	"""有序分类与回归方法基类，负责类别数校验与预测流程。"""

	@override
	def __init__(
		self,
		backbone: nn.Module,
		num_classes: int,
		*,
		lr: float,
		weight_decay: float = 0.05,
	) -> None:
		if num_classes < 2:
			raise ValueError(f'num_classes must be >= 2, got {num_classes}')
		self.num_classes = num_classes
		super().__init__(
			backbone,
			lr=lr,
			weight_decay=weight_decay,
		)

	@abstractmethod
	def predict_from_features(self, feat: Tensor) -> Tensor:
		"""由特征解码得到一维 long 类别预测。"""
		...

	@override
	def predict_step(self, X: Tensor) -> torch.Tensor:
		feat = self.model.forward_features(X)
		return self.predict_from_features(feat.float())


class ClassificationBaseMethod(OrdinalBaseMethod[ClassificationModel]):
	"""使用多分类线性头的有序方法基类。"""

	@override
	def build_model(self, backbone: nn.Module) -> ClassificationModel:
		return ClassificationModel(backbone, self.num_classes)

	@override
	def predict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
		return self.model.fc(feat).argmax(dim=1)


class CEMethod(ClassificationBaseMethod):
	"""标准交叉熵分类方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		return F.cross_entropy(self.model.fc(feat), targets.long())


class LabelSmoothingCEMethod(ClassificationBaseMethod):
	"""标签平滑交叉熵分类方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		return F.cross_entropy(self.model.fc(feat), targets.long(), label_smoothing=0.1)


def sord_loss(logits: Tensor, targets: Tensor, temperature: float = 1.0) -> Tensor:
	"""计算软序回归（SORD）损失。

	使用温度系数缩放类别距离，调节软标签目标概率分布的集中程度后计算交叉熵损失。
	"""
	if not math.isfinite(temperature) or temperature <= 0.0:
		raise ValueError(f'Temperature must be a finite positive number, got {temperature}')
	num_classes = logits.shape[1]
	targets = targets.to(device=logits.device)
	classes = torch.arange(num_classes, device=logits.device, dtype=logits.dtype)
	target_dist = torch.softmax(-(targets.view(-1, 1) - classes).abs() / temperature, dim=1)
	return F.cross_entropy(logits, target_dist)


class SORDMethod(ClassificationBaseMethod):
	"""软序回归方法（Soft Ordinal Regression）。

	使用温度系数缩放类别距离，调节软标签目标概率分布的集中程度。
	"""

	@override
	def __init__(
		self,
		backbone: nn.Module,
		num_classes: int,
		*,
		lr: float,
		weight_decay: float = 0.05,
		temperature: float = 1.0,
	) -> None:
		if not math.isfinite(temperature) or temperature <= 0.0:
			raise ValueError(f'Temperature must be a finite positive number, got {temperature}')
		super().__init__(
			backbone,
			num_classes,
			lr=lr,
			weight_decay=weight_decay,
		)
		self.temperature: float = float(temperature)

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		logits = self.model.fc(feat)
		return sord_loss(logits, targets, self.temperature)


class RPSMethod(ClassificationBaseMethod):
	"""连续概率排位分数（Ranked Probability Score）方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat)
		probs = torch.softmax(logits, dim=1)
		onehot = F.one_hot(targets.long(), self.num_classes).to(feat.dtype)
		diff = (probs - onehot).cumsum(dim=1)[:, :-1]
		return diff.pow(2).sum(dim=1).mean()


class WassersteinMethod(ClassificationBaseMethod):
	"""Wasserstein 距离有序方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat)
		probs = torch.softmax(logits, dim=1)
		onehot = F.one_hot(targets.long(), self.num_classes).to(feat.dtype)
		diff = (probs - onehot).cumsum(dim=1)[:, :-1]
		return diff.abs().sum(dim=1).mean()


class DistanceCEMethod(ClassificationBaseMethod):
	"""距离加权交叉熵有序方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat)
		ce = F.cross_entropy(logits, targets.long())
		probs = torch.softmax(logits, dim=1)
		classes = torch.arange(self.num_classes, device=feat.device, dtype=feat.dtype)
		dist2 = (classes.unsqueeze(0) - targets.unsqueeze(1).to(feat.dtype)).pow(2)
		return ce + (probs * dist2).sum(dim=1).mean()


class UnimodalMethod(ClassificationBaseMethod):
	"""单峰正则化有序方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat)
		ce = F.cross_entropy(logits, targets.long())
		probs = torch.softmax(logits, dim=1)
		prefix = torch.cummax(probs, dim=1).values
		suffix = torch.flip(torch.cummax(torch.flip(probs, dims=[1]), dim=1).values, dims=[1])
		left = prefix[:, :-2]
		right = suffix[:, 2:]
		mid = probs[:, 1:-1]
		penalty = (F.relu(left - mid) * F.relu(right - mid)).sum(dim=1).mean()
		return ce + penalty


class MSERankBaseMethod(OrdinalBaseMethod[ScalarModel]):
	"""MSE 锚定排序方法基类。"""

	@override
	def build_model(self, backbone: nn.Module) -> ScalarModel:
		return ScalarModel(backbone, bias=True)

	@override
	def predict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
		return self.model.fc(feat).squeeze(-1).round().clamp(0, self.num_classes - 1).long()


class MSERankNetMethod(MSERankBaseMethod):
	"""MSE 与批次内对排序 Logistic 损失混合方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		score = self.model.fc(feat).squeeze(-1)
		mse = F.mse_loss(score, targets.float())
		return mse + _ranking_term(score, targets, hinge=False)


class MSERankHingeMethod(MSERankBaseMethod):
	"""MSE 与批次内对排序 Hinge 损失混合方法。"""

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		score = self.model.fc(feat).squeeze(-1)
		mse = F.mse_loss(score, targets.float())
		return mse + _ranking_term(score, targets, hinge=True)


class CLMMethod(OrdinalBaseMethod[CLMModel]):
	"""累积链接模型（Cumulative Link Model）方法。"""

	@override
	def build_model(self, backbone: nn.Module) -> CLMModel:
		return CLMModel(backbone, self.num_classes)

	@override
	def predict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
		score = self.model.fc(feat).squeeze(-1)
		return _clm_log_probabilities(self.model, score).argmax(dim=1)

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		score = self.model.fc(feat).squeeze(-1)
		return F.nll_loss(_clm_log_probabilities(self.model, score), targets.long())


class CORALMethod(OrdinalBaseMethod[CORALModel]):
	"""CORAL 有序回归方法。"""

	@override
	def build_model(self, backbone: nn.Module) -> CORALModel:
		return CORALModel(backbone, self.num_classes)

	@override
	def predict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
		return ((self.model.fc(feat) + self.model.coral_bias) > 0).sum(dim=1).long()

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat) + self.model.coral_bias
		return _cumulative_bce(logits, targets, self.num_classes)


class CORNMethod(OrdinalBaseMethod[CORNModel]):
	"""CORN 条件有序回归方法。"""

	@override
	def build_model(self, backbone: nn.Module) -> CORNModel:
		return CORNModel(backbone, self.num_classes)

	@override
	def predict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
		cum = torch.sigmoid(self.model.fc(feat)).cumprod(dim=1)
		return (cum > 0.5).sum(dim=1).long()

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat)
		thresholds = torch.arange(self.num_classes - 1, device=feat.device)
		mask = targets.unsqueeze(1) >= thresholds
		target = (targets.unsqueeze(1) > thresholds).to(feat.dtype)
		bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
		return (bce * mask).sum() / mask.sum().clamp(min=1)


class OrdinalContrastiveMethod(OrdinalBaseMethod[ContrastiveClassificationModel]):
	"""有序对比学习方法。"""

	@override
	def build_model(self, backbone: nn.Module) -> ContrastiveClassificationModel:
		return ContrastiveClassificationModel(backbone, self.num_classes)

	@override
	def predict_from_features(self, feat: torch.Tensor) -> torch.Tensor:
		return self.model.fc(feat).argmax(dim=1)

	def process(self, batch: Any) -> torch.Tensor:
		images, targets = batch
		feat = self.model.forward_features(images).float()
		targets = targets.to(feat.device)
		logits = self.model.fc(feat)
		ce = F.cross_entropy(logits, targets.long())
		proj = self.model.proj(feat)
		return ce + _contrastive_term(proj, targets)


METHODS: dict[str, type[OrdinalBaseMethod[Any]]] = {
	'ce': CEMethod,
	'label_smoothing_ce': LabelSmoothingCEMethod,
	'clm': CLMMethod,
	'coral': CORALMethod,
	'corn': CORNMethod,
	'sord': SORDMethod,
	'rps': RPSMethod,
	'wasserstein': WassersteinMethod,
	'distance_ce': DistanceCEMethod,
	'unimodal': UnimodalMethod,
	'mse_ranknet': MSERankNetMethod,
	'mse_rank_hinge': MSERankHingeMethod,
	'ordinal_contrastive': OrdinalContrastiveMethod,
}
