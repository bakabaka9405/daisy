"""有序回归与分类模型组件。"""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

_PROJ_DIM = 16


class BackboneModel(nn.Module):
	"""包含特征提取 backbone 的基础模型组件。"""

	@override
	def __init__(self, backbone: nn.Module) -> None:
		super().__init__()
		self.backbone = backbone
		self.width: int = getattr(backbone, 'num_features')

	def forward_features(self, images: torch.Tensor) -> torch.Tensor:
		return self.backbone(images)


class ClassificationModel(BackboneModel):
	"""带标准多分类线性头的模型。"""

	@override
	def __init__(self, backbone: nn.Module, num_classes: int) -> None:
		super().__init__(backbone)
		self.num_classes = num_classes
		self.fc = nn.Linear(self.width, num_classes)

	@override
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		return self.fc(self.backbone(images))


class ScalarModel(BackboneModel):
	"""标量回归与排序头模型。"""

	@override
	def __init__(self, backbone: nn.Module, *, bias: bool = True) -> None:
		super().__init__(backbone)
		self.fc = nn.Linear(self.width, 1, bias=bias)

	@override
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		return self.fc(self.backbone(images)).squeeze(-1)


class CLMModel(BackboneModel):
	"""累积链接模型（CLM）：单输出无偏置线性头与有序阈值参数。"""

	@override
	def __init__(self, backbone: nn.Module, num_classes: int) -> None:
		super().__init__(backbone)
		self.num_classes = num_classes
		self.fc = nn.Linear(self.width, 1, bias=False)
		self.threshold = nn.Parameter(torch.zeros(()))
		self.delta: nn.Parameter | None = nn.Parameter(torch.zeros(num_classes - 2)) if num_classes >= 3 else None

	@override
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		return self.fc(self.backbone(images)).squeeze(-1)


class CORALModel(BackboneModel):
	"""CORAL 模型：单输出无偏置线性头与共享权重递减偏置向量。"""

	@override
	def __init__(self, backbone: nn.Module, num_classes: int) -> None:
		super().__init__(backbone)
		self.num_classes = num_classes
		self.fc = nn.Linear(self.width, 1, bias=False)
		self.coral_bias = nn.Parameter(torch.linspace(1.0, -1.0, num_classes - 1))

	@override
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		return self.fc(self.backbone(images)).squeeze(-1)


class CORNModel(BackboneModel):
	"""CORN 模型：输出 K-1 个二分类条件阈值 logit。"""

	@override
	def __init__(self, backbone: nn.Module, num_classes: int) -> None:
		super().__init__(backbone)
		self.num_classes = num_classes
		self.fc = nn.Linear(self.width, num_classes - 1)

	@override
	def forward(self, images: torch.Tensor) -> torch.Tensor:
		return self.fc(self.backbone(images))


class ContrastiveClassificationModel(ClassificationModel):
	"""包含分类头与对比投影头的模型。"""

	@override
	def __init__(self, backbone: nn.Module, num_classes: int) -> None:
		super().__init__(backbone, num_classes)
		self.proj = nn.Linear(self.width, _PROJ_DIM, bias=False)
