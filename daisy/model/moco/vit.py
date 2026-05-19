"""MoCo 专用 ViT

来自 MoCo v3: https://arxiv.org/abs/2104.02057
- VisionTransformerMoCo: 固定 2D sin-cos 位置编码 + 自定义初始化
- ConvStem: 卷积 patch embedding
"""

import math
from collections.abc import Callable
from functools import partial, reduce
from operator import mul
from typing import Any

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers.helpers import to_2tuple
from timm.layers.patch_embed import PatchEmbed

__all__ = [
	'VisionTransformerMoCo',
	'ConvStem',
	'vit_small',
	'vit_base',
	'vit_conv_small',
	'vit_conv_base',
	'MOCO_VIT_MODELS',
]


class VisionTransformerMoCo(VisionTransformer):
	def __init__(self, stop_grad_conv1: bool = False, **kwargs: Any):
		super().__init__(**kwargs)
		# Use fixed 2D sin-cos position embedding
		self.build_2d_sincos_position_embedding()

		# weight initialization
		for name, m in self.named_modules():
			if isinstance(m, nn.Linear):
				if 'qkv' in name:
					# treat the weights of Q, K, V separately
					val = math.sqrt(6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
					nn.init.uniform_(m.weight, -val, val)
				else:
					nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)
		if self.cls_token is not None:
			nn.init.normal_(self.cls_token, std=1e-6)

		if isinstance(self.patch_embed, PatchEmbed):
			# xavier_uniform initialization
			val = math.sqrt(6.0 / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
			nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
			if self.patch_embed.proj.bias is not None:
				nn.init.zeros_(self.patch_embed.proj.bias)

			if stop_grad_conv1:
				self.patch_embed.proj.weight.requires_grad = False
				if self.patch_embed.proj.bias is not None:
					self.patch_embed.proj.bias.requires_grad = False

	def build_2d_sincos_position_embedding(self, temperature: float = 10000.0):
		h, w = self.patch_embed.grid_size
		grid_w = torch.arange(w, dtype=torch.float32)
		grid_h = torch.arange(h, dtype=torch.float32)
		grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
		assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
		pos_dim = self.embed_dim // 4
		omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
		omega = 1.0 / (temperature**omega)
		out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
		out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
		pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

		# num_prefix_tokens 是 timm 新版 API (替代旧版 num_tokens)
		num_prefix = getattr(self, 'num_prefix_tokens', getattr(self, 'num_tokens', 1))
		assert num_prefix == 1, 'Assuming one and only one token, [cls]'
		pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
		self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
		self.pos_embed.requires_grad = False


class ConvStem(nn.Module):
	"""ConvStem, from Early Convolutions Help Transformers See Better, Tete et al.
	https://arxiv.org/abs/2106.14881
	"""

	def __init__(
		self,
		img_size: int = 224,
		patch_size: int = 16,
		in_chans: int = 3,
		embed_dim: int = 768,
		norm_layer: Callable[..., nn.Module] | None = None,
		flatten: bool = True,
	):
		super().__init__()

		assert patch_size == 16, 'ConvStem only supports patch size of 16'
		assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

		img_size_t: tuple[int, int] = to_2tuple(img_size)  # type: ignore[assignment]
		patch_size_t: tuple[int, int] = to_2tuple(patch_size)  # type: ignore[assignment]
		self.img_size = img_size_t
		self.patch_size = patch_size_t
		self.grid_size = (img_size_t[0] // patch_size_t[0], img_size_t[1] // patch_size_t[1])
		self.num_patches = self.grid_size[0] * self.grid_size[1]
		self.flatten = flatten

		# build stem, similar to the design in https://arxiv.org/abs/2106.14881
		stem: list[nn.Module] = []
		input_dim, output_dim = 3, embed_dim // 8
		for l in range(4):
			stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
			stem.append(nn.BatchNorm2d(output_dim))
			stem.append(nn.ReLU(inplace=True))
			input_dim = output_dim
			output_dim *= 2
		stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
		self.proj = nn.Sequential(*stem)

		self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, C, H, W = x.shape
		assert H == self.img_size[0] and W == self.img_size[1], (
			f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
		)
		x = self.proj(x)
		if self.flatten:
			x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
		x = self.norm(x)
		return x


def vit_small(**kwargs: Any) -> VisionTransformerMoCo:
	model = VisionTransformerMoCo(
		patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
	)
	return model


def vit_base(**kwargs: Any) -> VisionTransformerMoCo:
	model = VisionTransformerMoCo(
		patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
	)
	return model


def vit_conv_small(**kwargs: Any) -> VisionTransformerMoCo:
	# minus one ViT block
	model = VisionTransformerMoCo(
		patch_size=16,
		embed_dim=384,
		depth=11,
		num_heads=12,
		mlp_ratio=4,
		qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		embed_layer=ConvStem,
		**kwargs,
	)
	return model


def vit_conv_base(**kwargs: Any) -> VisionTransformerMoCo:
	# minus one ViT block
	model = VisionTransformerMoCo(
		patch_size=16,
		embed_dim=768,
		depth=11,
		num_heads=12,
		mlp_ratio=4,
		qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		embed_layer=ConvStem,
		**kwargs,
	)
	return model


MOCO_VIT_MODELS: dict[str, Callable[..., VisionTransformerMoCo]] = {
	'vit_small': vit_small,
	'vit_base': vit_base,
	'vit_conv_small': vit_conv_small,
	'vit_conv_base': vit_conv_base,
}
