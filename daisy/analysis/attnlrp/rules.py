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

"""基于 Gradient*Input 框架的隐式 AttnLRP 传播规则与 Zennit 适配钩子。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.autograd import Function
from zennit.core import BasicHook, ParamMod, stabilize
from zennit.rules import Epsilon, Gamma


class DivideGradientFn(Function):
	"""在梯度反向传播路径上除以指定常数因子的 Autograd 函数。"""

	@staticmethod
	def forward(ctx: Any, input: Tensor, factor: float = 2.0) -> Tensor:
		ctx.factor = factor
		return input

	@staticmethod
	def backward(ctx: Any, *grad_outputs: Any) -> tuple[Tensor, None]:
		grad_output: Tensor = grad_outputs[0]
		return grad_output / ctx.factor, None


def divide_gradient(input: Tensor, factor: float = 2.0) -> Tensor:
	"""将张量的反向梯度除以指定因子，用于注意力模块的均匀分配规则。"""
	return DivideGradientFn.apply(input, factor)  # type: ignore


class IdentityRuleImplicitFn(Function):
	"""非线性激活函数的隐式恒等传播规则 Autograd 函数。"""

	@staticmethod
	def forward(ctx: Any, fn: Callable[[Tensor], Tensor], input: Tensor, epsilon: float = 1e-10) -> Tensor:
		output = fn(input)
		if input.requires_grad:
			ctx.save_for_backward(output / stabilize(input, epsilon=epsilon))
		return output

	@staticmethod
	def backward(ctx: Any, *grad_outputs: Any) -> tuple[None, Tensor, None]:
		grad_output: Tensor = grad_outputs[0]
		gradient = ctx.saved_tensors[0] * grad_output
		return None, gradient, None


def identity_rule_implicit(fn: Callable[[Tensor], Tensor], input: Tensor, epsilon: float = 1e-10) -> Tensor:
	"""在元素级非线性函数上应用隐式恒等传播规则。"""
	return IdentityRuleImplicitFn.apply(fn, input, epsilon)  # type: ignore


class GradientInputBasicHook(BasicHook):
	"""梯度与输入相乘 (Gradient*Input) 兼容易用基础钩子。"""

	def forward(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Tensor) -> None:
		super().forward(module, args, kwargs, output)
		self.stored_tensors['output'] = output

	def backward(
		self,
		module: nn.Module,
		grad_input: tuple[Tensor | None, ...],
		grad_output: tuple[Tensor, ...],
	) -> tuple[Tensor | None, ...]:
		# 将输出端梯度转换至标准 LRP 相关性分数
		output: Tensor = self.stored_tensors['output']
		grad_out = grad_output[0] * output

		original_input: Tensor
		original_input, *original_args = self.stored_tensors['input']
		original_input = original_input.clone()
		original_kwargs: dict[str, Any] = self.stored_tensors.get('kwargs', {})

		inputs: list[Tensor] = []
		outputs: list[Tensor] = []
		in_mods = self.input_modifiers or ()
		param_mods = self.param_modifiers or ()
		out_mods = self.output_modifiers or ()
		for in_mod, param_mod, out_mod in zip(in_mods, param_mods, out_mods):
			inp: Tensor = in_mod(original_input).requires_grad_()
			with ParamMod.ensure(param_mod)(module) as modified, torch.autograd.enable_grad():
				out: Tensor = modified.forward(inp, *original_args, **original_kwargs)
				out = out_mod(out)
			inputs.append(inp)
			outputs.append(out)

		grad_outputs = self.gradient_mapper(grad_out, outputs)
		gradients = torch.autograd.grad(
			outputs,
			inputs,
			grad_outputs=grad_outputs,
			create_graph=grad_output[0].requires_grad,
		)
		relevance: Tensor = self.reducer(inputs, gradients)  # type: ignore

		# 除以输入张量以还原至 Gradient*Input 框架
		relevance = relevance / stabilize(original_input, epsilon=1e-10)

		return tuple(relevance if original is not None and original.shape == relevance.shape else None for original in grad_input)


class GradientInputGamma(Gamma, GradientInputBasicHook):
	"""梯度输入兼容的 Gamma 规则钩子。"""


class GradientInputEpsilon(Epsilon, GradientInputBasicHook):
	"""梯度输入兼容的 Epsilon 规则钩子。"""


def gamma_rule(gamma: float = 0.25) -> BasicHook:
	"""创建梯度输入兼容的 Gamma 规则钩子。"""
	return GradientInputGamma(gamma=gamma)


def epsilon_rule(epsilon: float = 1e-6) -> BasicHook:
	"""创建梯度输入兼容的 Epsilon 规则钩子。"""
	return GradientInputEpsilon(epsilon=epsilon)
