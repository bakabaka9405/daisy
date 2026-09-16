"""共享 backbone 的多任务分类模型组件。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
import torch.nn as nn

from daisy.model.ordinal import BackboneModel


class MultiHeadModel(BackboneModel):
	"""共享 backbone 与多个独立分类头的模型，每个任务对应一个头。"""

	@override
	def __init__(self, backbone: nn.Module, num_classes_list: Sequence[int]) -> None:
		super().__init__(backbone)
		self.num_classes_list: list[int] = list(num_classes_list)
		self.heads = nn.ModuleList([nn.Linear(self.width, num_classes) for num_classes in self.num_classes_list])

	@override
	def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
		"""返回每个任务头的 logits，顺序与 ``num_classes_list`` 一致。"""
		features = self.forward_features(images)
		return [head(features) for head in self.heads]
