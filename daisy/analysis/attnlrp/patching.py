# Copyright 2024, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. &
# the authors: Reduan Achtibat, Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Aakriti Jain,
# Thomas Wiegand, Sebastian Lapuschkin, Wojciech Samek. All rights reserved.
#
# This code is based on the following work:
#
#   'AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers. ICML 2024.'
#
# The copyright in this software is being made available under the Clear BSD License.
# No patent rights, trademark rights and/or other Intellectual Property Rights other than
# the copyrights concerning the Software are granted under this license.
# You may obtain a full copy of the License at
#
#   https://github.com/rachtibat/LRP-eXplains-Transformers/blob/main/LICENSE
#

"""针对 Attention、LayerNorm 与 GELU 的实例级前向函数替换模块。"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from typing import Any

from timm.layers.attention import Attention as TimmAttention, maybe_add_mask, resolve_self_attn_mask
from torch import Tensor, nn

from .rules import divide_gradient, identity_rule_implicit


def make_attention_forward(mod: nn.Module) -> Callable[..., Tensor]:
	"""为 timm eager Attention 构造满足梯度缩减规则的替换前向函数。"""
	attn_mod: Any = mod

	def forward(
		x: Tensor,
		attn_mask: Tensor | None = None,
		is_causal: bool = False,
	) -> Tensor:
		b, n, _ = x.shape
		num_heads: int = attn_mod.num_heads
		head_dim: int = attn_mod.head_dim
		attn_dim: int = attn_mod.attn_dim
		scale: float = attn_mod.scale

		qkv: Tensor = attn_mod.qkv(x).reshape(b, n, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
		q, k, v = qkv.unbind(0)
		q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)

		# 施加 Q/4, K/4, V/2 梯度缩减规则
		q = divide_gradient(q, 4.0)
		k = divide_gradient(k, 4.0)
		v = divide_gradient(v, 2.0)

		q = q * scale
		attn = q @ k.transpose(-2, -1)
		attn_bias = resolve_self_attn_mask(n, attn, attn_mask, is_causal)
		attn = maybe_add_mask(attn, attn_bias)
		attn = attn.softmax(dim=-1)
		attn = attn_mod.attn_drop(attn)

		x_out = attn @ v
		x_out = x_out.transpose(1, 2).reshape(b, n, attn_dim)
		x_out = attn_mod.norm(x_out)
		x_out = attn_mod.proj(x_out)
		x_out = attn_mod.proj_drop(x_out)

		return x_out

	return forward


def make_layernorm_forward(mod: nn.LayerNorm) -> Callable[[Tensor], Tensor]:
	"""为 LayerNorm 构造可微均值中心化并分离标准差梯度的替换前向函数。"""

	def forward(x: Tensor) -> Tensor:
		dims = tuple(range(-len(mod.normalized_shape), 0))
		mean = x.mean(dim=dims, keepdim=True)
		var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
		std = (var + mod.eps).sqrt()

		# 可微均值中心化并分离标准差梯度
		y = (x - mean) / std.detach()

		if mod.weight is not None:
			y = y * mod.weight
		if mod.bias is not None:
			y = y + mod.bias

		return y

	return forward


def make_gelu_forward(mod: nn.Module) -> Callable[[Tensor], Tensor]:
	"""为 GELU 激活函数构造隐式恒等规则的替换前向函数。"""
	orig_forward: Callable[[Tensor], Tensor] = getattr(mod, 'forward')

	def forward(x: Tensor) -> Tensor:
		return identity_rule_implicit(orig_forward, x, 1e-10)

	return forward


def patch_module_forward(mod: nn.Module, new_forward: Callable[..., Any], stack: ExitStack) -> None:
	"""在实例属性上临时替换 forward 函数，并在 ExitStack 中注册精确还原回调。"""
	had_forward = 'forward' in mod.__dict__
	orig_forward = getattr(mod, 'forward')

	mod.forward = new_forward  # type: ignore

	def restore() -> None:
		if had_forward:
			mod.forward = orig_forward  # type: ignore
		else:
			del mod.forward

	stack.callback(restore)


def patch_model_for_attnlrp(model: nn.Module, stack: ExitStack) -> None:
	"""遍历模型并临时替换 Attention、LayerNorm 与 GELU 的实例前向函数。"""
	for mod in model.modules():
		if isinstance(mod, TimmAttention):
			patch_module_forward(mod, make_attention_forward(mod), stack)
		elif isinstance(mod, nn.LayerNorm):
			patch_module_forward(mod, make_layernorm_forward(mod), stack)
		elif isinstance(mod, nn.GELU):
			patch_module_forward(mod, make_gelu_forward(mod), stack)
