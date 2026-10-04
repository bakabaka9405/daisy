"""为 avg-pool Vision Transformer 生成与目标输出相关的 attention relevance 热力图。"""

from __future__ import annotations

import icontract
import torch
from timm.models.vision_transformer import VisionTransformer

from .vit import _TimmRelpropEngine


@icontract.require(lambda matrices, prefix_tokens: 0 <= prefix_tokens < matrices[0].shape[-1])
@icontract.require(lambda matrices: all(m.shape == matrices[0].shape for m in matrices))
@icontract.require(lambda matrices, start_layer: 0 <= start_layer < len(matrices))
@icontract.require(lambda matrices: len(matrices) > 0)
@icontract.ensure(lambda matrices, prefix_tokens, result: result.shape == (matrices[0].shape[0], matrices[0].shape[-1] - prefix_tokens))
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

	@icontract.require(lambda self, start_layer: 0 <= start_layer < len(self.model.blocks))
	@icontract.require(lambda target_index: target_index >= 0)
	@icontract.require(lambda image: image.ndim == 3 or (image.ndim == 4 and image.shape[0] == 1), 'image must be (C, H, W) or (1, C, H, W)')
	@icontract.ensure(lambda result: bool(((result >= 0) & (result <= 1)).all()))
	@icontract.ensure(lambda self, result: result.shape == tuple(self.model.patch_embed.grid_size))
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
