"""Avg-pool Chefer transformer-attribution rollout。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from .vit import LRPVisionTransformer, _Block


@dataclass(frozen=True, slots=True)
class AttentionRelevanceResult:
	mirror_logits: torch.Tensor
	raw_patch_relevance: torch.Tensor
	heatmap: torch.Tensor
	layer_matrices: tuple[torch.Tensor, ...]


def _rollout_patch_relevance(matrices: tuple[torch.Tensor, ...], prefix_tokens: int, start_layer: int) -> torch.Tensor:
	"""计算 u^T[(I+C_L)...(I+C_s)-I] 的 patch-key 边缘量。"""
	if not matrices or not 0 <= start_layer < len(matrices):
		raise ValueError('start_layer 必须满足 0 <= start_layer < block 数。')
	tokens = matrices[0].shape[-1]
	identity = torch.eye(tokens, device=matrices[0].device, dtype=matrices[0].dtype).unsqueeze(0)
	joint = identity
	for matrix in matrices[start_layer:]:
		joint = (identity + matrix) @ joint
	return (joint - identity)[:, prefix_tokens:, prefix_tokens:].sum(dim=1)


class ViTAttentionRelevanceExplainer:
	def __init__(self, model: LRPVisionTransformer) -> None:
		self.model = model.eval()
		for parameter in self.model.parameters(): parameter.requires_grad_(False)

	@classmethod
	def from_timm(cls, model: nn.Module) -> ViTAttentionRelevanceExplainer:
		return cls(LRPVisionTransformer.from_timm(model))

	def generate(self, image: torch.Tensor, target_index: int, start_layer: int = 0) -> AttentionRelevanceResult:
		if image.ndim == 3: image = image.unsqueeze(0)
		if image.ndim != 4 or image.shape[0] != 1: raise ValueError('仅支持单张 CHW 图像。')
		layers = len(self.model.blocks)
		if not 0 <= start_layer < layers: raise ValueError('start_layer 必须满足 0 <= start_layer < block 数。')
		image = image.to(next(self.model.parameters()).device)
		try:
			with torch.enable_grad():
				logits = self.model(image)
				if not 0 <= target_index < logits.shape[-1]: raise ValueError('target_index 超出 logits 范围。')
				seed = torch.zeros_like(logits); seed[0, target_index] = 1
				logits.backward(seed)
				self.model.relprop_to_block0(seed)
				matrices = []
				for block in self.model.blocks:
					attention = cast(_Block, block).attn
					if attention.attention_cam is None or attention.attention_gradient is None: raise RuntimeError('attention relevance 或 gradient 缺失。')
					matrices.append((attention.attention_cam * attention.attention_gradient).clamp_min(0).mean(dim=1).detach())
				prefix = self.model.num_prefix_tokens
				raw = _rollout_patch_relevance(tuple(matrices), prefix, start_layer)[0].reshape(self.model.patch_embed.grid_size)
				positive = raw.clamp_min(0); maximum = positive.max(); heatmap = positive / maximum if maximum.item() > 0 else torch.zeros_like(positive)
				return AttentionRelevanceResult(logits.detach(), raw.detach(), heatmap.detach(), tuple(matrices))
		finally:
			self.model.clear_cache()
