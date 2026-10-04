"""共享 backbone 的多任务分类模型组件。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

import icontract
import torch
import torch.nn as nn


class MultiHeadModel(nn.Module):
	"""共享 backbone 与多个命名独立分类头的模型，每个任务对应一个头。"""

	@override
	@icontract.require(lambda num_classes: all(count >= 2 for count in num_classes.values()), 'num_classes for each task must be at least 2')
	@icontract.require(
		lambda num_classes: all(isinstance(name, str) and len(name) > 0 for name in num_classes), 'task name must be a nonempty string'
	)
	@icontract.require(lambda num_classes: len(num_classes) > 0, 'num_classes must not be empty')
	@icontract.require(lambda feature_dim: feature_dim > 0, 'feature_dim must be positive')
	def __init__(
		self,
		backbone: nn.Module,
		num_classes: Mapping[str, int],
		*,
		feature_dim: int,
	) -> None:
		super().__init__()
		self.backbone: nn.Module = backbone
		self.num_classes: dict[str, int] = dict(num_classes)
		# 保留 ModuleList 结构，以确保已有的 heads.0, heads.1 ... 检查点权重键名严格兼容
		self.heads: nn.ModuleList = nn.ModuleList([nn.Linear(feature_dim, self.num_classes[name]) for name in self.num_classes])

	@override
	@icontract.ensure(lambda self, images, result: all(v.shape == (images.shape[0], self.num_classes[k]) for k, v in result.items()))
	@icontract.ensure(lambda self, result: result.keys() == self.num_classes.keys())
	def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
		"""返回每个任务头的 logits 字典，键为任务名称。"""
		features = self.backbone(images)
		return {name: head(features) for name, head in zip(self.num_classes, self.heads, strict=True)}
