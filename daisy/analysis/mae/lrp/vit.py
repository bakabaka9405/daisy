"""timm avg ViT 的最小显式 attention-relevance mirror。"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F

from .layers import Add, Clone, Dropout, GELU, LayerNorm, Linear, MatMul, MeanPool, RelProp, Sequential, Softmax


@dataclass(frozen=True, slots=True)
class ParameterMappingReport:
	mapped: tuple[tuple[str, str], ...]
	missing: tuple[str, ...]
	unexpected: tuple[str, ...]


class _PatchEmbed(nn.Module):
	def __init__(self, source: Any) -> None:
		super().__init__()
		self.weight = nn.Parameter(source.proj.weight.detach().clone(), requires_grad=False)
		self.bias = nn.Parameter(source.proj.bias.detach().clone(), requires_grad=False) if source.proj.bias is not None else None
		self.stride = source.proj.stride
		self.padding = source.proj.padding
		self.grid_size = source.grid_size

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return F.conv2d(x, self.weight, self.bias, self.stride, self.padding).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
	def __init__(self, source: Any) -> None:
		super().__init__()
		self.num_heads, self.head_dim, self.scale = source.num_heads, source.head_dim, source.scale
		self.qkv, self.proj = Linear(source.qkv), Linear(source.proj)
		self.softmax, self.attn_drop, self.proj_drop = Softmax(), Dropout(source.attn_drop), Dropout(source.proj_drop)
		self.matmul_qk, self.matmul_av = MatMul(), MatMul()
		self.attention: torch.Tensor | None = None
		self.attention_gradient: torch.Tensor | None = None
		self.attention_cam: torch.Tensor | None = None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch, tokens, _ = x.shape
		qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
		query, key, value = qkv.unbind(0)
		attention = self.attn_drop(self.softmax(self.matmul_qk((query, key.transpose(-2, -1))) * self.scale)).detach().requires_grad_(True)
		self.attention = attention
		attention.register_hook(self._save_attention_gradient)
		x = self.matmul_av((attention, value)).transpose(1, 2).reshape(batch, tokens, -1)
		return self.proj_drop(self.proj(x))

	def _save_attention_gradient(self, gradient: torch.Tensor) -> None:
		self.attention_gradient = gradient.detach()

	def relprop(self, relevance: torch.Tensor) -> torch.Tensor:
		relevance = self.proj_drop.relprop(relevance)
		relevance = self.proj.relprop(relevance)
		batch, tokens, _ = relevance.shape
		relevance = relevance.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
		relevance_attention, relevance_value = self.matmul_av.relprop(relevance)
		relevance_attention, relevance_value = relevance_attention / 2, relevance_value / 2
		self.attention_cam = relevance_attention.detach()
		relevance_query, relevance_key = self.matmul_qk.relprop(self.softmax.relprop(self.attn_drop.relprop(relevance_attention)))
		relevance_qkv = torch.cat((relevance_query.transpose(1, 2).reshape(batch, tokens, -1) / 2, relevance_key.transpose(-2, -1).transpose(1, 2).reshape(batch, tokens, -1) / 2, relevance_value.transpose(1, 2).reshape(batch, tokens, -1)), dim=-1)
		return self.qkv.relprop(relevance_qkv)

	def clear_cache(self) -> None:
		self.attention = self.attention_gradient = self.attention_cam = None
		for module in self.modules():
			if isinstance(module, RelProp): module.clear_cache()


class _Mlp(nn.Module):
	def __init__(self, source: Any) -> None:
		super().__init__()
		self.fc1, self.act, self.drop1 = Linear(source.fc1), GELU(), Dropout(source.drop1)
		self.fc2, self.drop2 = Linear(source.fc2), Dropout(source.drop2)
	def forward(self, x: torch.Tensor) -> torch.Tensor: return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
	def relprop(self, relevance: torch.Tensor) -> torch.Tensor: return self.fc1.relprop(self.act.relprop(self.drop1.relprop(self.fc2.relprop(self.drop2.relprop(relevance)))))


class _Block(nn.Module):
	def __init__(self, source: Any) -> None:
		super().__init__()
		self.norm1, self.attn, self.norm2, self.mlp = LayerNorm(source.norm1), _Attention(source.attn), LayerNorm(source.norm2), _Mlp(source.mlp)
		self.clone1, self.add1, self.clone2, self.add2 = Clone(), Add(), Clone(), Add()
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		left, right = self.clone1(x, 2); x = self.add1((left, self.attn(self.norm1(right))))
		left, right = self.clone2(x, 2); return self.add2((left, self.mlp(self.norm2(right))))
	def relprop(self, relevance: torch.Tensor) -> torch.Tensor:
		left, right = self.add2.relprop(relevance); relevance = self.clone2.relprop((left, self.norm2.relprop(self.mlp.relprop(right))))
		left, right = self.add1.relprop(relevance); return self.clone1.relprop((left, self.norm1.relprop(self.attn.relprop(right))))


def _head(module: nn.Module) -> RelProp:
	if isinstance(module, nn.Linear): return Linear(module)
	if isinstance(module, nn.GELU): return GELU()
	if isinstance(module, nn.Dropout): return Dropout(module)
	raise ValueError(f'不支持 head 模块 {type(module).__name__}。')


class LRPVisionTransformer(nn.Module):
	def __init__(self, source: Any, head_modules: Sequence[nn.Module]) -> None:
		super().__init__()
		self.num_prefix_tokens = source.num_prefix_tokens
		self.patch_embed = _PatchEmbed(source.patch_embed)
		self.cls_token = nn.Parameter(source.cls_token.detach().clone(), requires_grad=False)
		self.pos_embed = nn.Parameter(source.pos_embed.detach().clone(), requires_grad=False)
		self.pos_drop, self.blocks, self.norm = Dropout(source.pos_drop), nn.ModuleList(_Block(block) for block in source.blocks), LayerNorm(source.norm)
		self.mean_pool, self.fc_norm, self.head_drop = MeanPool(self.num_prefix_tokens), LayerNorm(source.fc_norm), Dropout(source.head_drop)
		converted = [_head(module) for module in head_modules]; self.head: RelProp | Sequential = converted[0] if len(converted) == 1 else Sequential(converted)
		self.parameter_mapping = ParameterMappingReport((), (), ())

	@classmethod
	def from_timm(cls, model: nn.Module) -> LRPVisionTransformer:
		if model.training: raise ValueError('仅支持 eval 模型。')
		if isinstance(model, nn.Sequential):
			children = list(model.children());
			if len(children) < 2: raise ValueError('Sequential 必须包含 backbone 与 head。')
			backbone: Any = children[0]; heads = children[1:]; prefix, head_prefixes = '0.', tuple(f'{index}.' for index in range(1, len(children)))
		else:
			backbone = model; source_head = cast(nn.Module, backbone.head); heads = list(source_head.children()) if isinstance(source_head, nn.Sequential) else [source_head]
			prefix, head_prefixes = '', tuple(f'head.{index}.' for index in range(len(heads))) if isinstance(source_head, nn.Sequential) else ('head.',)
		if backbone.global_pool != 'avg' or backbone.num_prefix_tokens != 1 or backbone.pool_include_prefix is not False or backbone.no_embed_class is not False: raise ValueError('仅支持单 CLS、global_pool=avg 的标准配置。')
		if backbone.cls_token is None or backbone.pos_embed is None or not isinstance(backbone.patch_embed.norm, nn.Identity): raise ValueError('需要 cls/pos embedding 与 Identity patch norm。')
		if backbone.patch_drop.__class__ is not nn.Identity or backbone.norm_pre.__class__ is not nn.Identity: raise ValueError('不支持 patch_drop 或 norm_pre。')
		for block in backbone.blocks:
			if not isinstance(block.ls1, nn.Identity) or not isinstance(block.ls2, nn.Identity) or not isinstance(block.attn.q_norm, nn.Identity) or not isinstance(block.attn.k_norm, nn.Identity) or not isinstance(block.attn.norm, nn.Identity) or not isinstance(block.mlp.norm, nn.Identity) or not isinstance(block.mlp.act, nn.GELU): raise ValueError('不支持当前 ViT 结构变体。')
		mirror = cls(backbone, heads).eval(); mirror._map(model, backbone, prefix, heads, head_prefixes); return mirror

	def _map(self, source: nn.Module, backbone: nn.Module, prefix: str, heads: Sequence[nn.Module], head_prefixes: Sequence[str]) -> None:
		pairs = [(f'{prefix}{name}', name.replace('patch_embed.proj.', 'patch_embed.')) for name, _ in backbone.named_parameters() if not name.startswith('head.')]
		for index, (module, source_prefix) in enumerate(zip(heads, head_prefixes, strict=True)):
			for name, _ in module.named_parameters(): pairs.append((f'{source_prefix}{name}', f'head.{name}' if len(heads) == 1 else f'head.modules_list.{index}.{name}'))
		source_parameters, mirror_parameters = dict(source.named_parameters()), dict(self.named_parameters())
		mapped_source, mapped_mirror = {a for a, _ in pairs}, {b for _, b in pairs}
		self.parameter_mapping = ParameterMappingReport(tuple(pairs), tuple(name for name in source_parameters if name not in mapped_source), tuple(name for name in mirror_parameters if name not in mapped_mirror))
		if self.parameter_mapping.missing or self.parameter_mapping.unexpected: raise RuntimeError(f'参数映射不完整: {self.parameter_mapping}')
		for source_name, mirror_name in pairs:
			if source_parameters[source_name].shape != mirror_parameters[mirror_name].shape or not torch.equal(source_parameters[source_name], mirror_parameters[mirror_name]): raise ValueError(f'参数映射不一致: {source_name}->{mirror_name}')

	def forward_with_stages(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
		x = self.patch_embed(image); stages = {'patch_embed': x}; x = self.pos_drop(torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1) + self.pos_embed); stages['position'] = x
		for index, block in enumerate(self.blocks):
			x = block(x)
			if index == 0: stages['block0'] = x
		stages['last_block'] = x; x = self.norm(x); x = self.mean_pool(x); stages['mean_pool'] = x; x = self.fc_norm(x); stages['fc_norm'] = x; stages['logits'] = self.head(self.head_drop(x)); return stages
	def forward(self, image: torch.Tensor) -> torch.Tensor: return self.forward_with_stages(image)['logits']
	def relprop_to_block0(self, relevance: torch.Tensor) -> torch.Tensor:
		relevance = self.head.relprop(relevance); relevance = self.head_drop.relprop(relevance); relevance = self.fc_norm.relprop(relevance); relevance = self.mean_pool.relprop(relevance); relevance = self.norm.relprop(relevance)
		for block in reversed(self.blocks): relevance = cast(_Block, block).relprop(relevance)
		return relevance
	def clear_cache(self) -> None:
		for module in self.blocks: cast(_Block, module).attn.clear_cache()
