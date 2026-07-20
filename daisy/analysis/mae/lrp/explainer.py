"""为 avg-pool Vision Transformer 生成与目标输出相关的 attention relevance 热力图。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from timm.models.vision_transformer import Block, VisionTransformer

from .vit import _TimmRelpropEngine


def _rollout_patch_relevance(matrices: tuple[torch.Tensor, ...], prefix_tokens: int, start_layer: int) -> torch.Tensor:
	"""累乘带残差的逐层 relevance，去除恒等路径，再汇总所有参与平均池化的 patch query。"""
	if not matrices or not 0 <= start_layer < len(matrices):
		raise ValueError('start_layer 必须满足 0 <= start_layer < block 数。')
	tokens = matrices[0].shape[-1]
	identity = torch.eye(tokens, device=matrices[0].device, dtype=matrices[0].dtype).unsqueeze(0)
	joint = identity
	for matrix in matrices[start_layer:]:
		joint = (identity + matrix) @ joint
	return (joint - identity)[:, prefix_tokens:, prefix_tokens:].sum(dim=1)


class ViTAttentionRelevanceExplainer:
	def __init__(self, model: VisionTransformer) -> None:
		self.model = model
		self._engine = _TimmRelpropEngine(model)

	def generate(self, image: torch.Tensor, target_index: int, start_layer: int = 0) -> torch.Tensor:
		if image.ndim == 3:
			image = image.unsqueeze(0)
		if image.ndim != 4 or image.shape[0] != 1:
			raise ValueError('仅支持单张 CHW 图像。')

		backbone = self.model
		layers = len(cast(Sequence[Block], backbone.blocks))
		if not 0 <= start_layer < layers:
			raise ValueError('start_layer 必须满足 0 <= start_layer < block 数。')
		image = image.to(next(self.model.parameters()).device)
		result = self._engine.run(image, target_index)
		matrices = tuple((relevance * gradient).clamp_min(0).mean(dim=1).detach() for relevance, gradient in zip(result.attention_relevance, result.attention_gradients, strict=True))
		raw = _rollout_patch_relevance(matrices, backbone.num_prefix_tokens, start_layer)[0].reshape(backbone.patch_embed.grid_size)
		positive = raw.clamp_min(0)
		maximum = positive.max()
		heatmap = positive / maximum if maximum.item() > 0 else torch.zeros_like(positive)
		return heatmap.detach()
