"""最小 layers_ours 移植；来源见 THIRD_PARTY_NOTICES.md。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn
from torch.nn import functional as F


def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	denominator = b.clamp(min=1e-9) + b.clamp(max=1e-9)
	denominator = denominator + denominator.eq(0).to(denominator.dtype) * 1e-9
	return a / denominator * b.ne(0).to(b.dtype)


class RelProp(nn.Module):
	X: torch.Tensor | tuple[torch.Tensor, ...] | None

	def __init__(self) -> None:
		super().__init__()
		self.X = None

	def _capture(self, value: torch.Tensor | Sequence[torch.Tensor]) -> None:
		if torch.is_tensor(value):
			self.X = value.detach().requires_grad_(True)
		else:
			self.X = tuple(item.detach().requires_grad_(True) for item in value)

	def clear_cache(self) -> None:
		self.X = None

	def relprop(self, relevance: Any, alpha: float = 1.0) -> Any:
		return relevance


class RelPropSimple(RelProp):
	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
		if self.X is None:
			raise RuntimeError('缺少前向 activation。')
		inputs = self.X
		if torch.is_tensor(inputs):
			output = self.forward(inputs)
			gradient = torch.autograd.grad(output, inputs, safe_divide(relevance, output), retain_graph=True)[0]
			return inputs * gradient
		output = self.forward(inputs)
		gradients = torch.autograd.grad(output, inputs, safe_divide(relevance, output), retain_graph=True)
		return inputs[0] * gradients[0], inputs[1] * gradients[1]


class Linear(RelProp):
	def __init__(self, source: nn.Linear) -> None:
		super().__init__()
		self.weight = nn.Parameter(source.weight.detach().clone(), requires_grad=False)
		self.bias = nn.Parameter(source.bias.detach().clone(), requires_grad=False) if source.bias is not None else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._capture(x)
		return F.linear(x, self.weight, self.bias)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		if not isinstance(self.X, torch.Tensor):
			raise RuntimeError('Linear 缺少前向 activation。')
		beta = alpha - 1
		positive_weight, negative_weight = self.weight.clamp_min(0), self.weight.clamp_max(0)
		positive_input, negative_input = self.X.clamp_min(0), self.X.clamp_max(0)

		def propagate(first_weight: torch.Tensor, second_weight: torch.Tensor) -> torch.Tensor:
			first = F.linear(positive_input, first_weight)
			second = F.linear(negative_input, second_weight)
			scale = safe_divide(relevance, first + second)
			return (
				positive_input * torch.autograd.grad(first, positive_input, scale, retain_graph=True)[0]
				+ negative_input * torch.autograd.grad(second, negative_input, scale, retain_graph=True)[0]
			)

		return alpha * propagate(positive_weight, negative_weight) - beta * propagate(negative_weight, positive_weight)


class Add(RelProp):
	def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
		self._capture(inputs)
		return torch.add(*inputs)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
		if not isinstance(self.X, tuple):
			raise RuntimeError('Add 缺少前向 activation。')
		inputs = self.X
		output = inputs[0] + inputs[1]
		left_gradient, right_gradient = torch.autograd.grad(output, inputs, safe_divide(relevance, output), retain_graph=True)
		left, right = inputs[0] * left_gradient, inputs[1] * right_gradient
		left_factor = safe_divide(left.sum().abs(), left.sum().abs() + right.sum().abs()) * relevance.sum()
		right_factor = safe_divide(right.sum().abs(), left.sum().abs() + right.sum().abs()) * relevance.sum()
		return left * safe_divide(left_factor, left.sum()), right * safe_divide(right_factor, right.sum())


class Clone(RelProp):
	def forward(self, x: torch.Tensor, count: int) -> tuple[torch.Tensor, ...]:
		self._capture(x)
		self.count = count
		return tuple(x for _ in range(count))

	def relprop(self, relevance: Sequence[torch.Tensor], alpha: float = 1.0) -> torch.Tensor:
		if not isinstance(self.X, torch.Tensor):
			raise RuntimeError('Clone 缺少前向 activation。')
		outputs = tuple(self.X for _ in range(self.count))
		gradients = torch.autograd.grad(
			outputs, self.X, tuple(safe_divide(item, output) for item, output in zip(relevance, outputs, strict=True)), retain_graph=True
		)[0]
		return self.X * gradients


class MatMul(RelPropSimple):
	def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
		self._capture(inputs)
		return inputs[0] @ inputs[1]


class Einsum(RelPropSimple):
	def __init__(self, equation: str) -> None:
		super().__init__()
		self.equation = equation

	def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
		self._capture(inputs)
		return torch.einsum(self.equation, *inputs)


class LayerNorm(RelProp):
	normalized_shape: tuple[int, ...]
	eps: float
	weight: nn.Parameter | None
	bias: nn.Parameter | None

	def __init__(self, source: nn.LayerNorm | nn.Identity) -> None:
		super().__init__()
		self.is_identity = isinstance(source, nn.Identity)
		if self.is_identity:
			self.normalized_shape, self.eps, self.weight, self.bias = (), 0.0, None, None
			return
		layer_norm = cast(nn.LayerNorm, source)
		self.normalized_shape = tuple(layer_norm.normalized_shape)
		self.eps = layer_norm.eps
		self.weight = nn.Parameter(layer_norm.weight.detach().clone(), requires_grad=False) if layer_norm.elementwise_affine else None
		self.bias = nn.Parameter(layer_norm.bias.detach().clone(), requires_grad=False) if layer_norm.elementwise_affine else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._capture(x)
		if self.is_identity:
			return x
		return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		return relevance


class GELU(RelProp):
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._capture(x)
		return F.gelu(x)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		return relevance


class Dropout(RelProp):
	def __init__(self, source: nn.Dropout) -> None:
		super().__init__()
		self.p = source.p

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._capture(x)
		return F.dropout(x, self.p, training=False)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		return relevance


class Softmax(RelProp):
	def __init__(self, dim: int = -1) -> None:
		super().__init__()
		self.dim = dim

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._capture(x)
		return x.softmax(self.dim)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		return relevance


class Sequential(nn.Module):
	def __init__(self, modules: Sequence[RelProp]) -> None:
		super().__init__()
		self.modules_list = nn.ModuleList(modules)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		for module in self.modules_list:
			x = module(x)
		return x

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		for module in reversed(self.modules_list):
			relevance = cast(RelProp, module).relprop(relevance, alpha)
			if not torch.is_tensor(relevance):
				raise RuntimeError('Sequential relevance 类型错误。')
		return relevance


class MeanPool(RelProp):
	def __init__(self, prefix_tokens: int) -> None:
		super().__init__()
		self.prefix_tokens = prefix_tokens

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._capture(x)
		return x[:, self.prefix_tokens :].mean(dim=1)

	def relprop(self, relevance: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
		if not isinstance(self.X, torch.Tensor):
			raise RuntimeError('MeanPool 缺少前向 activation。')
		inputs = self.X
		output = inputs[:, self.prefix_tokens :].mean(dim=1)
		gradient = torch.autograd.grad(output, inputs, safe_divide(relevance, output), retain_graph=True)[0]
		return torch.cat((torch.zeros_like(inputs[:, : self.prefix_tokens]), (inputs * gradient)[:, self.prefix_tokens :]), dim=1)
