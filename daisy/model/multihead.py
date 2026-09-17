"""共享 backbone 的多任务分类模型组件。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
import torch.nn as nn


class MultiHeadModel(nn.Module):
	"""共享 backbone 与多个独立分类头的模型，每个任务对应一个头。"""

	@override
	def __init__(
		self,
		backbone: nn.Module,
		num_classes_list: Sequence[int],
		*,
		feature_dim: int,
	) -> None:
		super().__init__()
		# 校验输入特征维度与各任务分类类别数
		if feature_dim <= 0:
			raise ValueError(f'feature_dim must be positive, got {feature_dim}')

		classes_list = list(num_classes_list)
		if not classes_list:
			raise ValueError('num_classes_list must not be empty')
		for num_classes in classes_list:
			if num_classes < 2:
				raise ValueError(f'num_classes must be at least 2, got {num_classes}')

		self.backbone: nn.Module = backbone
		self.num_classes_list: list[int] = classes_list
		self.heads: nn.ModuleList = nn.ModuleList([nn.Linear(feature_dim, num_classes) for num_classes in self.num_classes_list])

	@override
	def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
		"""返回每个任务头的 logits，顺序与 ``num_classes_list`` 一致。"""
		features = self.backbone(images)
		return [head(features) for head in self.heads]
