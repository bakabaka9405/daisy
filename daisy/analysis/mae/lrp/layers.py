"""无状态的 Layer-wise Relevance Propagation 运算。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	"""将非有限分配权重置零；有限的非零分母保持原有比值。"""
	return torch.nan_to_num(a / b, nan=0.0, posinf=0.0, neginf=0.0)


def linear_relprop(x: torch.Tensor, weight: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
	"""仅沿输入与权重同号的正向贡献传播 relevance，bias 不参与分配。"""
	positive_weight, negative_weight = weight.clamp_min(0), weight.clamp_max(0)
	positive_input, negative_input = x.clamp_min(0), x.clamp_max(0)
	first = F.linear(positive_input, positive_weight)
	second = F.linear(negative_input, negative_weight)
	scale = safe_divide(relevance, first + second)
	return positive_input * (scale @ positive_weight) + negative_input * (scale @ negative_weight)


def layer_norm_relprop(x: torch.Tensor, norm: nn.LayerNorm, relevance: torch.Tensor) -> torch.Tensor:
	"""固定标准差，沿 LayerNorm 的中心化与仿射变换传播 relevance。"""
	axes = tuple(range(-len(norm.normalized_shape), 0))
	mean = x.mean(dim=axes, keepdim=True)
	std = torch.sqrt(torch.mean((x - mean) ** 2, dim=axes, keepdim=True) + norm.eps)
	y = F.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)
	scale = safe_divide(relevance, y)
	h = scale * norm.weight if norm.weight is not None else scale
	return (x / std) * (h - h.mean(dim=axes, keepdim=True))


def add_relprop(inputs: tuple[torch.Tensor, torch.Tensor], relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""按两个加数的局部贡献传播 relevance，再按分支绝对总量保持全局占比。"""
	left_input, right_input = inputs
	output = left_input + right_input
	scale = safe_divide(relevance, output)
	left = left_input * scale.sum_to_size(left_input.shape)
	right = right_input * scale.sum_to_size(right_input.shape)
	denominator = left.sum().abs() + right.sum().abs()
	left_factor = safe_divide(left.sum().abs(), denominator) * relevance.sum()
	right_factor = safe_divide(right.sum().abs(), denominator) * relevance.sum()
	return left * safe_divide(left_factor, left.sum()), right * safe_divide(right_factor, right.sum())


def clone_relprop(x: torch.Tensor, branch_relevances: Sequence[torch.Tensor]) -> torch.Tensor:
	"""将多个计算分支的 relevance 按各分支相对输入的贡献汇回同一输入。"""
	total_scale = torch.zeros_like(x)
	for relevance in branch_relevances:
		total_scale = total_scale + safe_divide(relevance, x)
	return x * total_scale


def matmul_relprop(a: torch.Tensor, b: torch.Tensor, relevance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""按矩阵乘法中各输入对输出的局部贡献向左右输入分配 relevance。"""
	scale = safe_divide(relevance, a @ b)
	left = scale @ b.transpose(-2, -1)
	right = a.transpose(-2, -1) @ scale
	return a * left.sum_to_size(a.shape), b * right.sum_to_size(b.shape)


def mean_pool_relprop(x: torch.Tensor, relevance: torch.Tensor, num_prefix_tokens: int) -> torch.Tensor:
	"""按 activation 贡献将池化后的 relevance 传播给 patch token，prefix token 置零。"""
	patches = x[:, num_prefix_tokens:]
	mean = patches.mean(dim=1)
	scale = safe_divide(relevance, mean)
	patch_relevance = patches * (scale / patches.shape[1]).unsqueeze(1)
	return torch.cat((torch.zeros_like(x[:, :num_prefix_tokens]), patch_relevance), dim=1)
