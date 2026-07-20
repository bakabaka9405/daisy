"""在 timm VisionTransformer 上缓存中间量并传播 attention relevance。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, VisionTransformer

from .layers import add_relprop, clone_relprop, linear_relprop, matmul_relprop, mean_pool_relprop


@dataclass(frozen=True, slots=True)
class _AttentionCache:
	qkv_input: torch.Tensor
	qkv_output: torch.Tensor
	attention: torch.Tensor
	proj_input: torch.Tensor
	output: torch.Tensor


@dataclass(frozen=True, slots=True)
class _MlpCache:
	fc1_input: torch.Tensor
	fc2_input: torch.Tensor
	output: torch.Tensor


@dataclass(frozen=True, slots=True)
class _BlockCache:
	x0: torch.Tensor
	x1: torch.Tensor
	attention: _AttentionCache
	mlp: _MlpCache


@dataclass(frozen=True, slots=True)
class _TimmForwardCache:
	blocks: tuple[_BlockCache, ...]
	tokens: torch.Tensor
	head_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _TimmForwardResult:
	logits: torch.Tensor
	cache: _TimmForwardCache
	attention_gradients: tuple[torch.Tensor, ...]


def _timm_forward_capture(model: VisionTransformer, image: torch.Tensor, target_index: int) -> _TimmForwardResult:
	"""捕获兼容 fixed-size avg-pool timm VisionTransformer 的前向量。

	调用者须提供沿标准 blocks、avg pool、head 数据流执行的模型；attention 与 MLP 须提供 qkv、proj、fc1、fc2 及 dropout 节点。
	解释时模型应处于 eval、float32 且关闭 gradient checkpointing；输入为位于模型设备上的单张固定尺寸 float32 图像。
	LayerScale、q/k normalization、attention pooling 和其他改变该数据流的结构未实现。
	head 必须可由其 Linear 层逆序传播，中间仅允许不改变 relevance 路径的 GELU 或 Dropout。
	"""
	backbone = model
	blocks = cast(Sequence[Block], backbone.blocks)
	attention_modules = tuple(cast(Attention, block.attn) for block in blocks)
	mlps = tuple(cast(Mlp, block.mlp) for block in blocks)
	norm2s = tuple(cast(nn.LayerNorm, block.norm2) for block in blocks)
	handles: list[RemovableHandle] = []
	fused_backup: list[tuple[Attention, bool]] = []
	block_inputs: list[torch.Tensor | None] = [None] * len(blocks)
	norm2_inputs: list[torch.Tensor | None] = [None] * len(blocks)
	qkv_inputs: list[torch.Tensor | None] = [None] * len(blocks)
	qkv_outputs: list[torch.Tensor | None] = [None] * len(blocks)
	attentions: list[torch.Tensor | None] = [None] * len(blocks)
	proj_inputs: list[torch.Tensor | None] = [None] * len(blocks)
	attention_outputs: list[torch.Tensor | None] = [None] * len(blocks)
	fc1_inputs: list[torch.Tensor | None] = [None] * len(blocks)
	fc2_inputs: list[torch.Tensor | None] = [None] * len(blocks)
	mlp_outputs: list[torch.Tensor | None] = [None] * len(blocks)
	tokens: torch.Tensor | None = None
	head_inputs: list[torch.Tensor] = []
	expected_head_inputs = sum(isinstance(module, nn.Linear) for module in backbone.head.modules())

	try:
		for attention in attention_modules:
			fused_backup.append((attention, attention.fused_attn))
			attention.fused_attn = False  # type: ignore[misc]

		for index, (block, norm2, attention, mlp) in enumerate(zip(blocks, norm2s, attention_modules, mlps, strict=True)):
			handles.extend(
				(
					block.register_forward_pre_hook(lambda _module, inputs, *, index=index: block_inputs.__setitem__(index, inputs[0].detach())),
					norm2.register_forward_pre_hook(lambda _module, inputs, *, index=index: norm2_inputs.__setitem__(index, inputs[0].detach())),
					attention.qkv.register_forward_pre_hook(
						lambda _module, inputs, *, index=index: qkv_inputs.__setitem__(index, inputs[0].detach())
					),
					attention.qkv.register_forward_hook(
						lambda _module, _inputs, output, *, index=index: qkv_outputs.__setitem__(index, output.detach())
					),
					attention.proj.register_forward_pre_hook(
						lambda _module, inputs, *, index=index: proj_inputs.__setitem__(index, inputs[0].detach())
					),
					attention.register_forward_hook(
						lambda _module, _inputs, output, *, index=index: attention_outputs.__setitem__(index, output.detach())
					),
					mlp.fc1.register_forward_pre_hook(lambda _module, inputs, *, index=index: fc1_inputs.__setitem__(index, inputs[0].detach())),
					mlp.fc2.register_forward_pre_hook(lambda _module, inputs, *, index=index: fc2_inputs.__setitem__(index, inputs[0].detach())),
					mlp.register_forward_hook(lambda _module, _inputs, output, *, index=index: mlp_outputs.__setitem__(index, output.detach())),
				)
			)

			def capture_attention(_module, _inputs, output: torch.Tensor, *, index: int = index) -> torch.Tensor:
				leaf = output.detach().requires_grad_(True)
				attentions[index] = leaf
				return leaf

			handles.append(attention.attn_drop.register_forward_hook(capture_attention))

		def capture_tokens(_module, _inputs, output: torch.Tensor) -> None:
			nonlocal tokens
			tokens = output.detach()

		handles.append(backbone.norm.register_forward_hook(capture_tokens))

		for module in backbone.head.modules():
			if isinstance(module, nn.Linear):
				handles.append(module.register_forward_pre_hook(lambda _module, inputs: head_inputs.append(inputs[0].detach())))

		logits = model(image)
		if logits.ndim != 2 or target_index >= logits.shape[1]:
			raise ValueError('target_index 超出 logits 范围。')
		expected_tokens = backbone.num_prefix_tokens + backbone.patch_embed.num_patches
		if tokens is None or tuple(tokens.shape) != (image.shape[0], expected_tokens, backbone.embed_dim):
			raise RuntimeError('前向 token 数量不匹配。')
		if any(value is None for value in attentions):
			raise RuntimeError('缺少 attention leaf。')
		attention_leaves = tuple(cast(torch.Tensor, value) for value in attentions)
		gradients = torch.autograd.grad(logits[0, target_index], attention_leaves)
		all_cache_values = (
			*block_inputs,
			*norm2_inputs,
			*qkv_inputs,
			*qkv_outputs,
			*proj_inputs,
			*attention_outputs,
			*fc1_inputs,
			*fc2_inputs,
			*mlp_outputs,
			tokens,
		)
		if any(value is None for value in all_cache_values):
			raise RuntimeError('前向 cache 不完整。')
		if len(head_inputs) != expected_head_inputs:
			raise RuntimeError('head 前向 cache 不完整。')
		cache = _TimmForwardCache(
			blocks=tuple(
				_BlockCache(
					x0=cast(torch.Tensor, block_inputs[index]),
					x1=cast(torch.Tensor, norm2_inputs[index]),
					attention=_AttentionCache(
						cast(torch.Tensor, qkv_inputs[index]),
						cast(torch.Tensor, qkv_outputs[index]),
						attention_leaves[index],
						cast(torch.Tensor, proj_inputs[index]),
						cast(torch.Tensor, attention_outputs[index]),
					),
					mlp=_MlpCache(
						cast(torch.Tensor, fc1_inputs[index]), cast(torch.Tensor, fc2_inputs[index]), cast(torch.Tensor, mlp_outputs[index])
					),
				)
				for index in range(len(blocks))
			),
			tokens=cast(torch.Tensor, tokens),
			head_inputs=tuple(head_inputs),
		)
		return _TimmForwardResult(logits.detach(), cache, tuple(gradient.detach() for gradient in gradients))
	finally:
		for handle in reversed(handles):
			handle.remove()
		for attention, fused_attn in fused_backup:
			attention.fused_attn = fused_attn  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class _TimmRelpropResult:
	attention_relevance: tuple[torch.Tensor, ...]
	attention_gradients: tuple[torch.Tensor, ...]


class _TimmRelpropEngine:
	def __init__(self, model: VisionTransformer) -> None:
		self.model = model

	@staticmethod
	def _activation(value: torch.Tensor) -> torch.Tensor:
		return value.detach().requires_grad_(True)

	def _head_relprop(self, cache: _TimmForwardCache, relevance: torch.Tensor) -> torch.Tensor:
		backbone = self.model
		linears = tuple(module for module in backbone.head.modules() if isinstance(module, nn.Linear))
		if len(linears) != len(cache.head_inputs):
			raise RuntimeError('head cache 与 Linear 数量不一致。')
		for module, inputs in zip(reversed(linears), reversed(cache.head_inputs), strict=True):
			relevance = linear_relprop(self._activation(inputs), module.weight.detach(), relevance)
		return relevance

	def _attention_relprop(self, block: Block, cache: _AttentionCache, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		attention_module = cast(Attention, block.attn)
		relevance = linear_relprop(self._activation(cache.proj_input), attention_module.proj.weight.detach(), relevance)
		batch, tokens, _ = relevance.shape
		relevance = relevance.reshape(batch, tokens, attention_module.num_heads, attention_module.head_dim).transpose(1, 2)
		qkv = cache.qkv_output.detach().reshape(batch, tokens, 3, attention_module.num_heads, attention_module.head_dim).permute(2, 0, 3, 1, 4)
		query, key, value = (self._activation(item) for item in qkv.unbind(0))
		attention, relevance_value = matmul_relprop(self._activation(cache.attention), value, relevance)
		attention, relevance_value = attention / 2, relevance_value / 2
		relevance_query, relevance_key = matmul_relprop(query, key.transpose(-2, -1), attention)
		relevance_qkv = torch.cat(
			(
				relevance_query.transpose(1, 2).reshape(batch, tokens, -1) / 2,
				relevance_key.transpose(-2, -1).transpose(1, 2).reshape(batch, tokens, -1) / 2,
				relevance_value.transpose(1, 2).reshape(batch, tokens, -1),
			),
			dim=-1,
		)
		return linear_relprop(self._activation(cache.qkv_input), attention_module.qkv.weight.detach(), relevance_qkv), attention.detach()

	def _block_relprop(self, block: Block, cache: _BlockCache, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		mlp = cast(Mlp, block.mlp)
		left, right = add_relprop((self._activation(cache.x1), self._activation(cache.mlp.output)), relevance)
		right = linear_relprop(self._activation(cache.mlp.fc2_input), mlp.fc2.weight.detach(), right)
		right = linear_relprop(self._activation(cache.mlp.fc1_input), mlp.fc1.weight.detach(), right)
		relevance = clone_relprop(self._activation(cache.x1), (left, right))
		left, right = add_relprop((self._activation(cache.x0), self._activation(cache.attention.output)), relevance)
		right, attention = self._attention_relprop(block, cache.attention, right)
		return clone_relprop(self._activation(cache.x0), (left, right)), attention

	def run(self, image: torch.Tensor, target_index: int) -> _TimmRelpropResult:
		with torch.enable_grad():
			forward = _timm_forward_capture(self.model, image, target_index)
			seed = torch.zeros_like(forward.logits)
			seed[0, target_index] = 1
			relevance = mean_pool_relprop(
				self._activation(forward.cache.tokens), self._head_relprop(forward.cache, seed), self.model.num_prefix_tokens
			)
			blocks = cast(Sequence[Block], self.model.blocks)
			attention_relevance: list[torch.Tensor | None] = [None] * len(blocks)
			for index in reversed(range(len(blocks))):
				relevance, attention_relevance[index] = self._block_relprop(blocks[index], forward.cache.blocks[index], relevance)
			if any(value is None for value in attention_relevance):
				raise RuntimeError('attention relevance 不完整。')
			return _TimmRelpropResult(tuple(cast(torch.Tensor, value) for value in attention_relevance), forward.attention_gradients)
