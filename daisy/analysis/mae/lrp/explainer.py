"""为 avg-pool Vision Transformer 生成与目标输出相关的 attention relevance 热力图。"""

from __future__ import annotations

import torch
from timm.models.vision_transformer import VisionTransformer

from .vit import _TimmRelpropEngine


def _rollout_patch_relevance(matrices: tuple[torch.Tensor, ...], prefix_tokens: int, start_layer: int) -> torch.Tensor:
	"""累乘带残差的逐层 relevance，利用 S = J - I 递推去除恒等路径，汇总 patch query。"""
	s = torch.zeros_like(matrices[0])
	for matrix in matrices[start_layer:]:
		s = s + matrix + matrix @ s
	return s[:, prefix_tokens:, prefix_tokens:].sum(dim=1)


class ViTAttentionRelevanceExplainer:
	def __init__(self, model: VisionTransformer) -> None:
		self.model = model
		self._engine = _TimmRelpropEngine(model)

	def generate(self, image: torch.Tensor, target_index: int, start_layer: int = 0) -> torch.Tensor:
		if image.ndim == 3:
			image = image.unsqueeze(0)

		backbone = self.model
		image = image.to(next(self.model.parameters()).device)
		result = self._engine.run(image, target_index)
		matrices = tuple(
			(relevance * gradient).clamp_min(0).mean(dim=1).detach()
			for relevance, gradient in zip(result.attention_relevance, result.attention_gradients)
		)
		raw = _rollout_patch_relevance(matrices, backbone.num_prefix_tokens, start_layer)[0].reshape(backbone.patch_embed.grid_size)
		positive = raw.clamp_min(0)
		maximum = positive.max()
		heatmap = positive / maximum if maximum.item() > 0 else torch.zeros_like(positive)
		return heatmap.detach()
