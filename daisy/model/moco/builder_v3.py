"""MoCo v3 模型 (queue-free, symmetrized, 单卡)

来自 MoCo v3: https://arxiv.org/abs/2104.02057
- 移除 DDP (concat_all_gather)
- base_encoder + momentum_encoder
- 对称 loss: contrastive_loss(q1, k2) + contrastive_loss(q2, k1)
"""

import icontract
import torch
import torch.nn as nn

from daisy.util import get_model_classifier


def _has_classifier_head(model: nn.Module) -> bool:
	"""模型是否带有可替换的分类头（fc / head / classifier）。"""
	return (
		(hasattr(model, 'fc') and isinstance(model.fc, (nn.Linear, nn.Sequential)))
		or hasattr(model, 'head')
		or hasattr(model, 'classifier')
	)


def _has_linear_head(model: nn.Module) -> bool:
	"""模型的 head 或 fc 是否为 nn.Linear。"""
	return (hasattr(model, 'head') and isinstance(model.head, nn.Linear)) or (hasattr(model, 'fc') and isinstance(model.fc, nn.Linear))


def _has_replaceable_head(model: nn.Module) -> bool:
	"""模型的 head 或 fc 是否为可替换的 nn.Linear / nn.Sequential。"""
	return (hasattr(model, 'head') and isinstance(model.head, (nn.Linear, nn.Sequential))) or (
		hasattr(model, 'fc') and isinstance(model.fc, (nn.Linear, nn.Sequential))
	)


class MoCoV3(nn.Module):
	"""单卡 MoCo v3: base_encoder + momentum_encoder, 无 queue"""

	@icontract.require(lambda T: T > 0)
	@icontract.require(lambda dim, mlp_dim: dim >= 1 and mlp_dim >= 1)
	def __init__(self, base_encoder, dim: int = 256, mlp_dim: int = 4096, T: float = 1.0):
		"""
		Args:
			base_encoder: callable，返回 encoder 模型 (num_classes=mlp_dim)
			dim: feature dimension (default: 256)
			mlp_dim: hidden dimension in MLPs (default: 4096)
			T: softmax temperature (default: 1.0)
		"""
		super().__init__()

		self.T = T

		# build encoders
		self.base_encoder: nn.Module = base_encoder(num_classes=mlp_dim)
		self.momentum_encoder: nn.Module = base_encoder(num_classes=mlp_dim)

		self._build_projector_and_predictor_mlps(dim, mlp_dim)

		# initialize momentum encoder with base encoder weights
		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
			param_m.data.copy_(param_b.data)
			param_m.requires_grad = False

	@icontract.require(lambda num_layers: num_layers >= 1)
	def _build_mlp(self, num_layers: int, input_dim: int, mlp_dim: int, output_dim: int, last_bn: bool = True) -> nn.Sequential:
		mlp: list[nn.Module] = []
		for l in range(num_layers):
			dim1 = input_dim if l == 0 else mlp_dim
			dim2 = output_dim if l == num_layers - 1 else mlp_dim

			mlp.append(nn.Linear(dim1, dim2, bias=False))

			if l < num_layers - 1:
				mlp.append(nn.BatchNorm1d(dim2))
				mlp.append(nn.ReLU(inplace=True))
			elif last_bn:
				# follow SimCLR's design
				mlp.append(nn.BatchNorm1d(dim2, affine=False))

		return nn.Sequential(*mlp)

	def _build_projector_and_predictor_mlps(self, dim: int, mlp_dim: int):
		# 子类必须设置 self.predictor
		self.predictor: nn.Module = nn.Identity()

	@torch.no_grad()
	@icontract.require(lambda m: 0 <= m <= 1)
	def _update_momentum_encoder(self, m: float):
		"""Momentum update of the momentum encoder"""
		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
			param_m.data = param_m.data * m + param_b.data * (1.0 - m)

	@icontract.require(lambda q, k: k.shape == q.shape)
	@icontract.require(lambda q: q.ndim == 2)
	@icontract.ensure(lambda result: result.ndim == 0)
	def contrastive_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
		# normalize
		q = nn.functional.normalize(q, dim=1)
		k = nn.functional.normalize(k, dim=1)
		# Einstein sum is more intuitive
		logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
		N = logits.shape[0]
		labels = torch.arange(N, dtype=torch.long, device=logits.device)
		return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

	@icontract.require(lambda m: 0 <= m <= 1)
	@icontract.require(lambda x1, x2: x2.shape == x1.shape)
	@icontract.require(lambda x1: x1.ndim == 4)
	@icontract.ensure(lambda result: result.ndim == 0)
	def forward(self, x1: torch.Tensor, x2: torch.Tensor, m: float) -> torch.Tensor:
		"""
		Args:
			x1: first views of images
			x2: second views of images
			m: moco momentum (from training loop, supports cosine schedule)
		Returns:
			loss
		"""
		# compute features
		q1 = self.predictor(self.base_encoder(x1))
		q2 = self.predictor(self.base_encoder(x2))

		with torch.no_grad():
			self._update_momentum_encoder(m)
			k1 = self.momentum_encoder(x1)
			k2 = self.momentum_encoder(x2)

		return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCoV3_ResNet(MoCoV3):
	"""ResNet backbone: 2 层 projector + 2 层 predictor"""

	def _build_projector_and_predictor_mlps(self, dim: int, mlp_dim: int):
		hidden_dim = self._get_classifier_dim(self.base_encoder)
		self._replace_classifier(self.base_encoder, self._build_mlp(2, hidden_dim, mlp_dim, dim))
		self._replace_classifier(self.momentum_encoder, self._build_mlp(2, hidden_dim, mlp_dim, dim))
		self.predictor = self._build_mlp(2, dim, mlp_dim, dim, last_bn=False)

	@staticmethod
	def _get_classifier_dim(model: nn.Module) -> int:
		classifier = get_model_classifier(model)
		assert isinstance(classifier, nn.Linear)
		return classifier.weight.shape[1]

	@staticmethod
	@icontract.require(lambda model: _has_classifier_head(model), 'Cannot find classifier head to replace')
	def _replace_classifier(model: nn.Module, new_head: nn.Module):
		if hasattr(model, 'fc') and isinstance(model.fc, (nn.Linear, nn.Sequential)):
			model.fc = new_head  # type: ignore[assignment]
		elif hasattr(model, 'head'):
			if hasattr(model.head, 'fc'):
				model.head.fc = new_head  # type: ignore[union-attr]
			else:
				model.head = new_head  # type: ignore[assignment]
		else:
			model.classifier = new_head  # type: ignore[assignment]


class MoCoV3_ViT(MoCoV3):
	"""ViT backbone: 3 层 projector + 2 层 predictor"""

	def _build_projector_and_predictor_mlps(self, dim: int, mlp_dim: int):
		hidden_dim = self._get_head_dim(self.base_encoder)
		self._replace_head(self.base_encoder, self._build_mlp(3, hidden_dim, mlp_dim, dim))
		self._replace_head(self.momentum_encoder, self._build_mlp(3, hidden_dim, mlp_dim, dim))
		self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

	@staticmethod
	@icontract.require(lambda model: _has_linear_head(model), 'Cannot find head dimension')
	def _get_head_dim(model: nn.Module) -> int:
		if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
			return model.head.weight.shape[1]
		fc = model.fc
		assert isinstance(fc, nn.Linear)
		return fc.weight.shape[1]

	@staticmethod
	@icontract.require(lambda model: _has_replaceable_head(model), 'Cannot find head to replace')
	def _replace_head(model: nn.Module, new_head: nn.Module):
		if hasattr(model, 'head') and isinstance(model.head, (nn.Linear, nn.Sequential)):
			model.head = new_head  # type: ignore[assignment]
		else:
			model.fc = new_head  # type: ignore[assignment]
