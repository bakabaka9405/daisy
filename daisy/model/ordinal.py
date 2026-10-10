"""各类有序预测头：前向计算、损失与类别解码。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, override

import icontract
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from daisy.losses import (
	corn_loss,
	cumulative_bce_loss,
	distance_ce_loss,
	mse_rank_loss,
	ordinal_contrastive_loss,
	rps_loss,
	sord_loss,
	unimodal_loss,
	wasserstein_loss,
)


class OrdinalHead[Output](nn.Module, ABC):
	"""预测头接口；输入为 (B, in_features)，损失与解码复用前向输出。"""

	@icontract.require(lambda in_features: in_features > 0)
	@icontract.require(lambda num_classes: num_classes >= 2)
	def __init__(self, in_features: int, num_classes: int) -> None:
		super().__init__()
		self.in_features = in_features
		self.num_classes = num_classes

	@abstractmethod
	def forward(self, features: Tensor) -> Output:
		"""计算预测头的原始输出。"""
		...

	@abstractmethod
	def loss(self, output: Output, targets: Tensor) -> Tensor:
		"""由前向输出计算标量损失。"""
		...

	@abstractmethod
	def decode(self, output: Output) -> Tensor:
		"""解码为形状 (B,) 的 long 类别索引。"""
		...

	def high_lr_parameters(self) -> tuple[nn.Parameter, ...]:
		"""返回需要较大学习率的参数。"""
		return ()


class ClassificationHead(OrdinalHead[Tensor]):
	"""线性分类头，共享 argmax 解码并由 criterion 指定训练目标。"""

	def __init__(self, in_features: int, num_classes: int, *, criterion: Callable[[Tensor, Tensor], Tensor] = F.cross_entropy) -> None:
		super().__init__(in_features, num_classes)
		self.fc = nn.Linear(in_features, num_classes)
		self.criterion = criterion

	@override
	@icontract.require(lambda self, features: features.shape[1] == self.in_features)
	@icontract.require(lambda features: features.ndim == 2)
	def forward(self, features: Tensor) -> Tensor:
		return self.fc(features)

	@override
	@icontract.ensure(lambda result: result.ndim == 0)
	def loss(self, output: Tensor, targets: Tensor) -> Tensor:
		return self.criterion(output, targets)

	@override
	def decode(self, output: Tensor) -> Tensor:
		return output.argmax(dim=1)

	def probabilities(self, output: Tensor) -> Tensor:
		return output.softmax(dim=1)


class ScalarHead(OrdinalHead[Tensor]):
	"""标量回归头，取整并限制到类别范围内。"""

	def __init__(self, in_features: int, num_classes: int, *, criterion: Callable[[Tensor, Tensor], Tensor] = mse_rank_loss) -> None:
		super().__init__(in_features, num_classes)
		self.fc = nn.Linear(in_features, 1)
		self.criterion = criterion

	@override
	@icontract.require(lambda self, features: features.shape[1] == self.in_features)
	@icontract.require(lambda features: features.ndim == 2)
	def forward(self, features: Tensor) -> Tensor:
		return self.fc(features).squeeze(-1)

	@override
	@icontract.ensure(lambda result: result.ndim == 0)
	def loss(self, output: Tensor, targets: Tensor) -> Tensor:
		return self.criterion(output, targets)

	@override
	def decode(self, output: Tensor) -> Tensor:
		return output.round().clamp(0, self.num_classes - 1).long()


@icontract.require(lambda delta: delta.ndim == 1)
@icontract.require(lambda threshold: threshold.ndim == 0)
@icontract.require(lambda score: score.ndim == 1)
@icontract.ensure(lambda score, delta, result: result.shape == (score.shape[0], delta.shape[0] + 2))
def clm_log_probabilities(score: Tensor, threshold: Tensor, delta: Tensor) -> Tensor:
	"""累积链接模型的对数概率；空 delta 对应二分类，在 float32 下计算。"""
	score = score.float()
	gaps = F.softplus(delta.float())
	tau = threshold.float() + torch.cat([gaps.new_zeros(1), gaps.cumsum(dim=0)])
	first = F.logsigmoid(tau[0] - score)
	last = F.logsigmoid(score - tau[-1])
	mid = F.logsigmoid(tau[1:] - score[:, None]) + F.logsigmoid(score[:, None] - tau[:-1]) + F.logsigmoid(delta.float())
	return torch.cat([first[:, None], mid, last[:, None]], dim=1)


class CLMHead(OrdinalHead[Tensor]):
	"""累积链接模型头，输出各类别的对数概率。"""

	def __init__(self, in_features: int, num_classes: int) -> None:
		super().__init__(in_features, num_classes)
		self.fc = nn.Linear(in_features, 1, bias=False)
		self.threshold = nn.Parameter(torch.zeros(()))
		self.delta = nn.Parameter(torch.zeros(num_classes - 2))

	def get_ordered_thresholds(self) -> Tensor:
		gaps = F.softplus(self.delta)
		return self.threshold + torch.cat([gaps.new_zeros(1), gaps.cumsum(dim=0)])

	@torch.no_grad()
	@icontract.require(lambda counts: bool((counts > 0).all()))
	@icontract.require(lambda self, counts: counts.shape == (self.num_classes,))
	def init_from_counts(self, counts: Tensor) -> None:
		"""根据平滑后的正类别计数初始化阈值。"""
		tau = torch.logit(counts.cumsum(dim=0)[:-1] / counts.sum())
		self.threshold.copy_(tau[0])
		self.delta.copy_(torch.log(torch.expm1(tau.diff())))

	@override
	def high_lr_parameters(self) -> tuple[nn.Parameter, ...]:
		return self.threshold, self.delta

	@override
	@icontract.require(lambda self, features: features.shape[1] == self.in_features)
	@icontract.require(lambda features: features.ndim == 2)
	def forward(self, features: Tensor) -> Tensor:
		return clm_log_probabilities(self.fc(features).squeeze(-1), self.threshold, self.delta)

	@override
	def loss(self, output: Tensor, targets: Tensor) -> Tensor:
		return F.nll_loss(output, targets.long())

	@override
	def decode(self, output: Tensor) -> Tensor:
		return output.argmax(dim=1)

	def probabilities(self, output: Tensor) -> Tensor:
		return output.exp()


class CORALHead(OrdinalHead[Tensor]):
	"""共享线性权重和独立阈值偏置，输出累积阈值 logits。"""

	def __init__(self, in_features: int, num_classes: int) -> None:
		super().__init__(in_features, num_classes)
		self.fc = nn.Linear(in_features, 1, bias=False)
		self.coral_bias = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))

	@torch.no_grad()
	@icontract.require(lambda counts: bool((counts > 0).all()))
	@icontract.require(lambda self, counts: counts.shape == (self.num_classes,))
	def init_from_counts(self, counts: Tensor) -> None:
		"""根据平滑后的正类别计数初始化阈值偏置。"""
		tau = torch.logit(counts.cumsum(dim=0)[:-1] / counts.sum())
		self.coral_bias.copy_(-tau)

	@override
	def high_lr_parameters(self) -> tuple[nn.Parameter, ...]:
		return (self.coral_bias,)

	@override
	@icontract.require(lambda self, features: features.shape[1] == self.in_features)
	@icontract.require(lambda features: features.ndim == 2)
	def forward(self, features: Tensor) -> Tensor:
		return self.fc(features) + self.coral_bias

	@override
	def loss(self, output: Tensor, targets: Tensor) -> Tensor:
		return cumulative_bce_loss(output, targets)

	@override
	def decode(self, output: Tensor) -> Tensor:
		return (output > 0).sum(dim=1).long()

	def probabilities(self, output: Tensor) -> Tensor:
		cumulative = output.sigmoid()
		boundaries = torch.cat([torch.ones_like(cumulative[:, :1]), cumulative, torch.zeros_like(cumulative[:, :1])], dim=1)
		return boundaries[:, :-1] - boundaries[:, 1:]


class CORNHead(OrdinalHead[Tensor]):
	"""条件有序回归头，输出各条件阈值 logits。"""

	def __init__(self, in_features: int, num_classes: int) -> None:
		super().__init__(in_features, num_classes)
		self.fc = nn.Linear(in_features, num_classes - 1)

	@torch.no_grad()
	@icontract.require(lambda counts: bool((counts > 0).all()))
	@icontract.require(lambda self, counts: counts.shape == (self.num_classes,))
	def init_from_counts(self, counts: Tensor) -> None:
		"""根据平滑后的正类别计数初始化条件阈值偏置。"""
		tail_counts = counts.flip(0).cumsum(dim=0).flip(0)
		assert self.fc.bias is not None
		self.fc.bias.copy_(torch.logit(tail_counts[1:] / tail_counts[:-1]))

	@override
	@icontract.require(lambda self, features: features.shape[1] == self.in_features)
	@icontract.require(lambda features: features.ndim == 2)
	def forward(self, features: Tensor) -> Tensor:
		return self.fc(features)

	@override
	def loss(self, output: Tensor, targets: Tensor) -> Tensor:
		return corn_loss(output, targets)

	@override
	def decode(self, output: Tensor) -> Tensor:
		return (output.sigmoid().cumprod(dim=1) > 0.5).sum(dim=1).long()

	def probabilities(self, output: Tensor) -> Tensor:
		cumulative = output.sigmoid().cumprod(dim=1)
		boundaries = torch.cat([torch.ones_like(cumulative[:, :1]), cumulative, torch.zeros_like(cumulative[:, :1])], dim=1)
		return boundaries[:, :-1] - boundaries[:, 1:]


class ContrastiveClassificationHead(OrdinalHead[tuple[Tensor, Tensor]]):
	"""分类与有序对比分支，前向返回 (logits, embedding)。"""

	@icontract.require(lambda projection_dim: projection_dim > 0)
	def __init__(self, in_features: int, num_classes: int, *, projection_dim: int = 16) -> None:
		super().__init__(in_features, num_classes)
		self.fc = nn.Linear(in_features, num_classes)
		self.proj = nn.Linear(in_features, projection_dim, bias=False)

	@override
	@icontract.require(lambda self, features: features.shape[1] == self.in_features)
	@icontract.require(lambda features: features.ndim == 2)
	def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
		return self.fc(features), self.proj(features)

	@override
	def loss(self, output: tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
		logits, embedding = output
		return F.cross_entropy(logits, targets.long()) + ordinal_contrastive_loss(embedding, targets)

	@override
	def decode(self, output: tuple[Tensor, Tensor]) -> Tensor:
		return output[0].argmax(dim=1)

	def probabilities(self, output: tuple[Tensor, Tensor]) -> Tensor:
		return output[0].softmax(dim=1)


HEADS: dict[str, Callable[..., OrdinalHead[Any]]] = {
	'ce': ClassificationHead,
	'label_smoothing_ce': partial(ClassificationHead, criterion=partial(F.cross_entropy, label_smoothing=0.1)),
	'clm': CLMHead,
	'coral': CORALHead,
	'corn': CORNHead,
	'sord': partial(ClassificationHead, criterion=sord_loss),
	'rps': partial(ClassificationHead, criterion=rps_loss),
	'wasserstein': partial(ClassificationHead, criterion=wasserstein_loss),
	'distance_ce': partial(ClassificationHead, criterion=distance_ce_loss),
	'unimodal': partial(ClassificationHead, criterion=unimodal_loss),
	'mse_ranknet': ScalarHead,
	'mse_rank_hinge': partial(ScalarHead, criterion=partial(mse_rank_loss, hinge=True)),
	'ordinal_contrastive': ContrastiveClassificationHead,
}
