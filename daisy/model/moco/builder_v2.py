"""MoCo v2 模型 (queue-based, 单卡)

来自 MoCo v1/v2: https://arxiv.org/abs/1911.05722, https://arxiv.org/abs/2003.04297
- 移除 DDP (batch shuffle/unshuffle, concat_all_gather)
- 使用 timm 创建 encoder
- 始终使用 v2 的 2 层 MLP head
"""

import torch
import torch.nn as nn

from daisy.util import get_model_classifier


class MoCoV2(nn.Module):
	"""单卡 MoCo v2: query encoder + key encoder + queue"""

	queue: torch.Tensor
	queue_ptr: torch.Tensor

	def __init__(
		self,
		base_encoder,
		dim: int = 128,
		K: int = 65536,
		m: float = 0.999,
		T: float = 0.07,
	):
		"""
		Args:
			base_encoder: callable，返回 encoder 模型 (num_classes=dim)
			dim: feature dimension (default: 128)
			K: queue size; number of negative keys (default: 65536)
			m: moco momentum of updating key encoder (default: 0.999)
			T: softmax temperature (default: 0.07)
		"""
		super().__init__()

		self.K = K
		self.m = m
		self.T = T

		# create the encoders
		self.encoder_q: nn.Module = base_encoder(num_classes=dim)
		self.encoder_k: nn.Module = base_encoder(num_classes=dim)

		# MoCo v2: 2-layer MLP head
		classifier_q = get_model_classifier(self.encoder_q)
		assert isinstance(classifier_q, nn.Linear)
		dim_mlp = classifier_q.weight.shape[1]
		self._replace_classifier(self.encoder_q, dim_mlp, dim)
		self._replace_classifier(self.encoder_k, dim_mlp, dim)

		# initialize key encoder with query encoder weights
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)
			param_k.requires_grad = False

		# create the queue
		self.register_buffer('queue', torch.randn(dim, K))
		self.queue = nn.functional.normalize(self.queue, dim=0)
		self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

	@staticmethod
	def _replace_classifier(model: nn.Module, hidden_dim: int, out_dim: int):
		"""替换模型分类头为 2 层 MLP (Linear -> ReLU -> Linear)"""
		mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, out_dim),
		)
		if hasattr(model, 'fc') and isinstance(model.fc, (nn.Linear, nn.Sequential)):
			model.fc = mlp  # type: ignore[assignment]
		elif hasattr(model, 'head'):
			if hasattr(model.head, 'fc'):
				model.head.fc = mlp  # type: ignore[union-attr]
			else:
				model.head = mlp  # type: ignore[assignment]
		elif hasattr(model, 'classifier'):
			model.classifier = mlp  # type: ignore[assignment]
		else:
			raise ValueError('Cannot find classifier head to replace')

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		"""Momentum update of the key encoder"""
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys: torch.Tensor):
		batch_size = keys.shape[0]
		ptr = int(self.queue_ptr.item())

		# 如果 batch_size 不能整除 K，用取模方式处理
		if ptr + batch_size > self.K:
			remaining = self.K - ptr
			self.queue[:, ptr:] = keys[:remaining].T
			overflow = batch_size - remaining
			self.queue[:, :overflow] = keys[remaining:].T
		else:
			self.queue[:, ptr:ptr + batch_size] = keys.T

		ptr = (ptr + batch_size) % self.K
		self.queue_ptr[0] = ptr

	def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			im_q: query images [N, C, H, W]
			im_k: key images [N, C, H, W]
		Returns:
			logits [N, 1+K], labels [N]
		"""
		# compute query features
		q = self.encoder_q(im_q)  # NxC
		q = nn.functional.normalize(q, dim=1)

		# compute key features
		with torch.no_grad():
			self._momentum_update_key_encoder()
			k = self.encoder_k(im_k)  # NxC
			k = nn.functional.normalize(k, dim=1)

		# positive logits: Nx1
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
		# negative logits: NxK
		l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

		# logits: Nx(1+K)
		logits = torch.cat([l_pos, l_neg], dim=1)
		logits /= self.T

		# labels: positive key indicators (index 0)
		labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

		# dequeue and enqueue
		self._dequeue_and_enqueue(k)

		return logits, labels
