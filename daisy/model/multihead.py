"""共享 backbone 的多任务分类模型组件。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

import torch
import torch.nn as nn


class MultiHeadModel(nn.Module):
	"""共享 backbone 与多个命名独立分类头的模型，每个任务对应一个头。"""

	@override
	def __init__(
		self,
		backbone: nn.Module,
		num_classes: Mapping[str, int],
		*,
		feature_dim: int,
	) -> None:
		super().__init__()
		# 校验输入特征维度与各任务分类类别数
		if feature_dim <= 0:
			raise ValueError(f'feature_dim must be positive, got {feature_dim}')

		tasks_dict = dict(num_classes)
		if not tasks_dict:
			raise ValueError('num_classes must not be empty')
		for task_name, count in tasks_dict.items():
			if not isinstance(task_name, str) or not task_name:
				raise ValueError(f'task name must be a nonempty string, got {task_name!r}')
			if count < 2:
				raise ValueError(f'num_classes for task {task_name!r} must be at least 2, got {count}')

		self.backbone: nn.Module = backbone
		self.num_classes: dict[str, int] = tasks_dict
		# 保留 ModuleList 结构，以确保已有的 heads.0, heads.1 ... 检查点权重键名严格兼容
		self.heads: nn.ModuleList = nn.ModuleList([nn.Linear(feature_dim, self.num_classes[name]) for name in self.num_classes])

	@override
	def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
		"""返回每个任务头的 logits 字典，键为任务名称。"""
		features = self.backbone(images)
		return {name: head(features) for name, head in zip(self.num_classes.keys(), self.heads, strict=True)}
