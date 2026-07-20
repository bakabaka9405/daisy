"""无状态的 Layer-wise Relevance Propagation 运算。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""稳定计算 a / b，并将分母为零的位置置零。"""
	denominator = b.clamp(min=1e-9) + b.clamp(max=1e-9)
	denominator = denominator + denominator.eq(0).to(denominator.dtype) * 1e-9
	return a / denominator * b.ne(0).to(b.dtype)


def linear_relprop(x: torch.Tensor, weight: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
	"""仅沿输入与权重同号的正向贡献传播 relevance，bias 不参与分配。"""
	positive_weight, negative_weight = weight.clamp_min(0), weight.clamp_max(0)
	positive_input, negative_input = x.clamp_min(0), x.clamp_max(0)
	first = F.linear(positive_input, positive_weight)
	second = F.linear(negative_input, negative_weight)
	scale = safe_divide(relevance, first + second)
	return (
		positive_input * torch.autograd.grad(first, positive_input, scale, retain_graph=True)[0]
		+ negative_input * torch.autograd.grad(second, negative_input, scale, retain_graph=True)[0]
	)


def add_relprop(inputs: tuple[torch.Tensor, torch.Tensor], relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""按两个加数的局部贡献传播 relevance，再按分支绝对总量保持全局占比。"""
	left_input, right_input = inputs
	output = left_input + right_input
	left_gradient, right_gradient = torch.autograd.grad(output, inputs, safe_divide(relevance, output), retain_graph=True)
	left, right = left_input * left_gradient, right_input * right_gradient
	denominator = left.sum().abs() + right.sum().abs()
	left_factor = safe_divide(left.sum().abs(), denominator) * relevance.sum()
	right_factor = safe_divide(right.sum().abs(), denominator) * relevance.sum()
	return left * safe_divide(left_factor, left.sum()), right * safe_divide(right_factor, right.sum())


def clone_relprop(x: torch.Tensor, branch_relevances: Sequence[torch.Tensor]) -> torch.Tensor:
	"""将多个计算分支的 relevance 按各分支相对输入的贡献汇回同一输入。"""
	outputs = tuple(x for _ in branch_relevances)
	gradient = torch.autograd.grad(
		outputs, x, tuple(safe_divide(relevance, output) for relevance, output in zip(branch_relevances, outputs, strict=True)), retain_graph=True
	)[0]
	return x * gradient


def matmul_relprop(a: torch.Tensor, b: torch.Tensor, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""按矩阵乘法中各输入对输出的局部贡献向左右输入分配 relevance。"""
	output = a @ b
	left_gradient, right_gradient = torch.autograd.grad(output, (a, b), safe_divide(relevance, output), retain_graph=True)
	return a * left_gradient, b * right_gradient


def mean_pool_relprop(x: torch.Tensor, relevance: torch.Tensor, num_prefix_tokens: int) -> torch.Tensor:
	"""按 activation 贡献将池化后的 relevance 传播给 patch token，prefix token 置零。"""
	output = x[:, num_prefix_tokens:].mean(dim=1)
	gradient = torch.autograd.grad(output, x, safe_divide(relevance, output), retain_graph=True)[0]
	return torch.cat((torch.zeros_like(x[:, :num_prefix_tokens]), (x * gradient)[:, num_prefix_tokens:]), dim=1)
