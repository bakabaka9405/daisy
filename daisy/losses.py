from __future__ import annotations

import math

import icontract
import torch
import torch.nn.functional as F
from torch import Tensor


@icontract.require(lambda logits, targets: targets.shape == (logits.shape[0],))
@icontract.require(lambda logits: logits.ndim == 2)
@icontract.require(lambda temperature: math.isfinite(temperature) and temperature > 0.0)
@icontract.ensure(lambda result: result.ndim == 0)
def sord_loss(logits: Tensor, targets: Tensor, temperature: float = 1.0) -> Tensor:
	"""按类别距离构造软标签，计算软序回归交叉熵。"""
	classes = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
	target_dist = torch.softmax(-(targets.to(logits.device).view(-1, 1) - classes).abs() / temperature, dim=1)
	return F.cross_entropy(logits, target_dist)


@icontract.require(lambda logits, targets: targets.shape == (logits.shape[0],))
@icontract.require(lambda logits: logits.ndim == 2)
def _cdf_difference(logits: Tensor, targets: Tensor) -> Tensor:
	"""预测分布与标签分布的累积概率差。"""
	onehot = F.one_hot(targets.long(), logits.shape[1]).to(logits.dtype)
	return (logits.softmax(dim=1) - onehot).cumsum(dim=1)[:, :-1]


@icontract.ensure(lambda result: result.ndim == 0)
def rps_loss(logits: Tensor, targets: Tensor) -> Tensor:
	"""连续概率排位分数，按阈值求和后取批次均值。"""
	return _cdf_difference(logits, targets).square().sum(dim=1).mean()


@icontract.ensure(lambda result: result.ndim == 0)
def wasserstein_loss(logits: Tensor, targets: Tensor) -> Tensor:
	"""有序类别分布的一阶 Wasserstein 距离。"""
	return _cdf_difference(logits, targets).abs().sum(dim=1).mean()


@icontract.require(lambda logits, targets: targets.shape == (logits.shape[0],))
@icontract.require(lambda logits: logits.ndim == 2)
@icontract.ensure(lambda result: result.ndim == 0)
def distance_ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
	"""交叉熵与预测类别平方距离的期望。"""
	classes = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
	dist2 = (classes.unsqueeze(0) - targets.unsqueeze(1).to(logits.dtype)).square()
	return F.cross_entropy(logits, targets.long()) + (logits.softmax(dim=1) * dist2).sum(dim=1).mean()


@icontract.require(lambda logits, targets: targets.shape == (logits.shape[0],))
@icontract.require(lambda logits: logits.ndim == 2)
@icontract.ensure(lambda result: result.ndim == 0)
def unimodal_loss(logits: Tensor, targets: Tensor) -> Tensor:
	"""交叉熵与单峰正则项。"""
	probs = logits.softmax(dim=1)
	prefix = torch.cummax(probs, dim=1).values
	suffix = torch.flip(torch.cummax(torch.flip(probs, dims=[1]), dim=1).values, dims=[1])
	mid = probs[:, 1:-1]
	penalty = (F.relu(prefix[:, :-2] - mid) * F.relu(suffix[:, 2:] - mid)).sum(dim=1).mean()
	return F.cross_entropy(logits, targets.long()) + penalty


@icontract.require(lambda score, targets: targets.shape == score.shape)
@icontract.require(lambda score: score.ndim == 1)
@icontract.ensure(lambda result: result.ndim == 0)
def mse_rank_loss(score: Tensor, targets: Tensor, *, hinge: bool = False) -> Tensor:
	"""均方误差与批次内排序项；默认使用 Logistic 排序损失。"""
	diff = score.unsqueeze(1) - score.unsqueeze(0)
	valid = targets.unsqueeze(1) > targets.unsqueeze(0)
	ranking = score.sum() * 0.0
	if bool(valid.any()):
		margin = diff[valid]
		ranking = F.relu(1.0 - margin).mean() if hinge else F.softplus(-margin).mean()
	return F.mse_loss(score, targets.float()) + ranking


@icontract.require(lambda logits, targets: targets.shape == (logits.shape[0],))
@icontract.require(lambda logits: logits.ndim == 2)
@icontract.ensure(lambda result: result.ndim == 0)
def cumulative_bce_loss(logits: Tensor, targets: Tensor) -> Tensor:
	"""累积二分类损失，按阈值求和后取批次均值。"""
	target = (targets.unsqueeze(1) > torch.arange(logits.shape[1], device=logits.device)).to(logits.dtype)
	return F.binary_cross_entropy_with_logits(logits, target, reduction='none').sum(dim=1).mean()


@icontract.require(lambda logits, targets: targets.shape == (logits.shape[0],))
@icontract.require(lambda logits: logits.ndim == 2)
@icontract.ensure(lambda result: result.ndim == 0)
def corn_loss(logits: Tensor, targets: Tensor) -> Tensor:
	"""按条件风险集计算有序二分类损失。"""
	thresholds = torch.arange(logits.shape[1], device=logits.device)
	mask = targets.unsqueeze(1) >= thresholds
	target = (targets.unsqueeze(1) > thresholds).to(logits.dtype)
	bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
	return (bce * mask).sum() / mask.sum().clamp(min=1)


@icontract.require(lambda embedding, targets: targets.shape == (embedding.shape[0],))
@icontract.require(lambda embedding: embedding.ndim == 2)
@icontract.ensure(lambda result: result.ndim == 0)
def ordinal_contrastive_loss(embedding: Tensor, targets: Tensor) -> Tensor:
	"""批次内欧氏距离拟合标签差绝对值。"""
	batch = embedding.shape[0]
	if batch < 2:
		return embedding.sum() * 0.0
	dist = torch.cdist(embedding, embedding)
	label_diff = (targets.unsqueeze(1) - targets.unsqueeze(0)).abs().to(embedding.dtype)
	rows, cols = torch.triu_indices(batch, batch, offset=1, device=embedding.device)
	return (dist[rows, cols] - label_diff[rows, cols]).square().mean()
