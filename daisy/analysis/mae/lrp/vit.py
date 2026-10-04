"""在 timm VisionTransformer 上缓存中间量并传播 attention relevance。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import cast

import icontract
import torch
from timm.layers.attention import Attention
from timm.layers.mlp import Mlp
from timm.models.vision_transformer import Block, VisionTransformer
from torch import nn

from .layers import add_relprop, clone_relprop, layer_norm_relprop, linear_relprop, matmul_relprop, mean_pool_relprop


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
	readout_inputs: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _TimmForwardResult:
	logits: torch.Tensor
	cache: _TimmForwardCache
	attention_gradients: tuple[torch.Tensor, ...]


@dataclass(slots=True)
class _TensorCapture:
	value: torch.Tensor = field(init=False)

	def on_input(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
		self.value = inputs[0].detach()

	def on_output(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
		self.value = output.detach()


@dataclass(slots=True)
class _BlockCapture:
	x0: _TensorCapture = field(default_factory=_TensorCapture)
	x1: _TensorCapture = field(default_factory=_TensorCapture)
	qkv_input: _TensorCapture = field(default_factory=_TensorCapture)
	qkv_output: _TensorCapture = field(default_factory=_TensorCapture)
	proj_input: _TensorCapture = field(default_factory=_TensorCapture)
	attention_output: _TensorCapture = field(default_factory=_TensorCapture)
	fc1_input: _TensorCapture = field(default_factory=_TensorCapture)
	fc2_input: _TensorCapture = field(default_factory=_TensorCapture)
	mlp_output: _TensorCapture = field(default_factory=_TensorCapture)
	attention: torch.Tensor = field(init=False)

	def capture_attention(self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
		self.attention = output

	@icontract.require(lambda self: _is_capture_complete(self), '前向过程中存在未被捕获的中间量')
	def finish(self) -> _BlockCache:
		return _BlockCache(
			x0=self.x0.value,
			x1=self.x1.value,
			attention=_AttentionCache(
				qkv_input=self.qkv_input.value,
				qkv_output=self.qkv_output.value,
				attention=self.attention.detach(),
				proj_input=self.proj_input.value,
				output=self.attention_output.value,
			),
			mlp=_MlpCache(
				fc1_input=self.fc1_input.value,
				fc2_input=self.fc2_input.value,
				output=self.mlp_output.value,
			),
		)


def _is_capture_complete(capture: _BlockCapture) -> bool:
	tensor_captures = (
		capture.x0,
		capture.x1,
		capture.qkv_input,
		capture.qkv_output,
		capture.proj_input,
		capture.attention_output,
		capture.fc1_input,
		capture.fc2_input,
		capture.mlp_output,
	)
	return hasattr(capture, 'attention') and all(hasattr(tensor_capture, 'value') for tensor_capture in tensor_captures)


def _is_supported_vit(model: VisionTransformer) -> bool:
	"""avg pool、无 attention pooling、无 gradient checkpointing，且各 block 无 LayerScale、q/k normalization 与注意力输出 norm。"""
	blocks = cast(Sequence[Block], model.blocks)
	return (
		model.global_pool == 'avg'
		and model.attn_pool is None
		and not model.grad_checkpointing
		and all(
			isinstance(block.attn, Attention)
			and isinstance(block.mlp, Mlp)
			and all(isinstance(module, nn.Identity) for module in (block.ls1, block.ls2, block.attn.q_norm, block.attn.k_norm, block.attn.norm))
			for block in blocks
		)
	)


def _set_fused_attention(attention: Attention, enabled: bool) -> None:
	"""集中修改 timm Attention 实例的 Final 属性 fused_attn。"""
	setattr(attention, 'fused_attn', enabled)


def _register_pre_hook(
	stack: ExitStack,
	module: nn.Module,
	hook: Callable[[nn.Module, tuple[torch.Tensor, ...]], None],
) -> None:
	handle = module.register_forward_pre_hook(hook)
	stack.callback(handle.remove)


def _register_hook(
	stack: ExitStack,
	module: nn.Module,
	hook: Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None],
) -> None:
	handle = module.register_forward_hook(hook)
	stack.callback(handle.remove)


def _register_block_capture(
	stack: ExitStack,
	block: Block,
	capture: _BlockCapture,
) -> None:
	attention = cast(Attention, block.attn)
	mlp = cast(Mlp, block.mlp)

	stack.callback(_set_fused_attention, attention, attention.fused_attn)
	_set_fused_attention(attention, False)

	pre_hook_bindings = (
		(block, capture.x0.on_input),
		(block.norm2, capture.x1.on_input),
		(attention.qkv, capture.qkv_input.on_input),
		(attention.proj, capture.proj_input.on_input),
		(mlp.fc1, capture.fc1_input.on_input),
		(mlp.fc2, capture.fc2_input.on_input),
	)
	for module, hook in pre_hook_bindings:
		_register_pre_hook(stack, module, hook)

	post_hook_bindings = (
		(attention.qkv, capture.qkv_output.on_output),
		(attention.attn_drop, capture.capture_attention),
		(attention, capture.attention_output.on_output),
		(mlp, capture.mlp_output.on_output),
	)
	for module, hook in post_hook_bindings:
		_register_hook(stack, module, hook)


def _readout_modules(model: VisionTransformer) -> tuple[nn.Linear | nn.LayerNorm, ...]:
	return tuple(module for module in (model.fc_norm, *model.head.modules()) if isinstance(module, (nn.Linear, nn.LayerNorm)))


@icontract.require(lambda target_index: target_index >= 0)
@icontract.require(lambda model, image: image.device == next(model.parameters()).device)
@icontract.require(lambda image: image.dtype == torch.float32)
@icontract.require(lambda image: image.ndim == 4 and image.shape[0] == 1, 'image must have shape (1, C, H, W)')
@icontract.require(lambda model: not model.training, 'model must be in eval mode')
@icontract.ensure(lambda model, result: len(result.cache.readout_inputs) == len(_readout_modules(model)))
@icontract.ensure(lambda model, result: len(result.cache.blocks) == len(result.attention_gradients) == len(model.blocks))
def _timm_forward_capture(model: VisionTransformer, image: torch.Tensor, target_index: int) -> _TimmForwardResult:
	"""捕获兼容 fixed-size avg-pool timm VisionTransformer 的前向量。

	调用者须提供沿标准 blocks、avg pool 及 readout (fc_norm / head) 数据流执行的模型；attention 与 MLP 须提供 qkv、proj、fc1、fc2 及 dropout 节点。
	解释时模型应处于 eval、float32 且关闭 gradient checkpointing；输入为位于模型设备上的单张固定尺寸 float32 图像。
	LayerScale、q/k normalization、attention pooling 和其他改变该数据流的结构未实现。
	readout 必须可由其 Linear 与 LayerNorm 层逆序传播，中间仅允许不改变 relevance 路径的 GELU 或 Dropout。
	"""
	blocks = cast(Sequence[Block], model.blocks)

	with ExitStack() as stack:
		block_captures: list[_BlockCapture] = []
		for block in blocks:
			capture = _BlockCapture()
			_register_block_capture(stack, block, capture)
			block_captures.append(capture)

		token_capture = _TensorCapture()
		_register_hook(stack, model.norm, token_capture.on_output)

		readout_captures: list[_TensorCapture] = []
		for module in _readout_modules(model):
			capture = _TensorCapture()
			_register_pre_hook(stack, module, capture.on_input)
			readout_captures.append(capture)

		# 以独立可求导输入作为图根节点，确保模型参数完全冻结时仍保留中间计算图
		input_image = image.detach().requires_grad_(True)
		logits = model(input_image)
		attentions = tuple(capture.attention for capture in block_captures)
		gradients = torch.autograd.grad(logits[0, target_index], attentions)

		cache = _TimmForwardCache(
			blocks=tuple(capture.finish() for capture in block_captures),
			tokens=token_capture.value,
			readout_inputs=tuple(capture.value for capture in readout_captures),
		)
		return _TimmForwardResult(logits.detach(), cache, tuple(gradient.detach() for gradient in gradients))


@dataclass(frozen=True, slots=True)
class _TimmRelpropResult:
	attention_relevance: tuple[torch.Tensor, ...]
	attention_gradients: tuple[torch.Tensor, ...]


class _TimmRelpropEngine:
	@icontract.require(lambda model: _is_supported_vit(model), '仅支持 avg-pool、无 LayerScale / q-k norm / attention pooling / gradient checkpointing 的 timm ViT')
	def __init__(self, model: VisionTransformer) -> None:
		self.model = model

	@icontract.require(lambda self, cache: len(cache.readout_inputs) == len(_readout_modules(self.model)))
	@icontract.ensure(lambda cache, result: len(cache.readout_inputs) == 0 or result.shape == cache.readout_inputs[0].shape)
	def _readout_relprop(self, cache: _TimmForwardCache, relevance: torch.Tensor) -> torch.Tensor:
		modules = _readout_modules(self.model)
		for module, inputs in zip(reversed(modules), reversed(cache.readout_inputs)):
			if isinstance(module, nn.Linear):
				relevance = linear_relprop(inputs, module.weight.detach(), relevance)
			else:
				relevance = layer_norm_relprop(inputs, module, relevance)
		return relevance

	@icontract.require(lambda cache, relevance: relevance.shape == cache.output.shape)
	@icontract.ensure(lambda cache, result: result[0].shape == cache.qkv_input.shape and result[1].shape == cache.attention.shape)
	def _attention_relprop(self, block: Block, cache: _AttentionCache, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		attention_module = cast(Attention, block.attn)
		relevance = linear_relprop(cache.proj_input, attention_module.proj.weight.detach(), relevance)
		batch, tokens, _ = relevance.shape
		relevance = relevance.reshape(batch, tokens, attention_module.num_heads, attention_module.head_dim).transpose(1, 2)

		qkv = cache.qkv_output.reshape(batch, tokens, 3, attention_module.num_heads, attention_module.head_dim).permute(2, 0, 3, 1, 4)
		query, key, value = qkv.unbind(0)
		attention, relevance_value = matmul_relprop(cache.attention, value, relevance)
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

		return linear_relprop(cache.qkv_input, attention_module.qkv.weight.detach(), relevance_qkv), attention

	@icontract.require(lambda cache, relevance: relevance.shape == cache.x0.shape)
	@icontract.ensure(lambda cache, result: result[0].shape == cache.x0.shape)
	def _block_relprop(self, block: Block, cache: _BlockCache, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		mlp = cast(Mlp, block.mlp)
		left, right = add_relprop((cache.x1, cache.mlp.output), relevance)
		right = linear_relprop(cache.mlp.fc2_input, mlp.fc2.weight.detach(), right)
		right = linear_relprop(cache.mlp.fc1_input, mlp.fc1.weight.detach(), right)
		relevance = clone_relprop(cache.x1, (left, right))

		left, right = add_relprop((cache.x0, cache.attention.output), relevance)
		right, attention = self._attention_relprop(block, cache.attention, right)
		return clone_relprop(cache.x0, (left, right)), attention

	@icontract.ensure(lambda self, result: len(result.attention_relevance) == len(result.attention_gradients) == len(self.model.blocks))
	def run(self, image: torch.Tensor, target_index: int) -> _TimmRelpropResult:
		with torch.enable_grad():
			forward = _timm_forward_capture(self.model, image, target_index)

		with torch.no_grad():
			seed = torch.zeros_like(forward.logits)
			seed[0, target_index] = 1

			relevance = mean_pool_relprop(forward.cache.tokens, self._readout_relprop(forward.cache, seed), self.model.num_prefix_tokens)

			blocks = cast(Sequence[Block], self.model.blocks)
			attention_relevances: list[torch.Tensor] = []
			for block, block_cache in zip(reversed(blocks), reversed(forward.cache.blocks)):
				relevance, block_attention_relevance = self._block_relprop(block, block_cache, relevance)
				attention_relevances.append(block_attention_relevance)

			return _TimmRelpropResult(tuple(reversed(attention_relevances)), forward.attention_gradients)
