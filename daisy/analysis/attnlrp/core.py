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

"""隐式 AttnLRP 解释器核心类。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from typing import cast

import torch
from torch import Tensor, nn
from zennit.core import BasicHook

from .patching import patch_model_for_attnlrp


class AttnLRP:
	"""隐式 AttnLRP 归因解释器。"""

	def __init__(self, model: nn.Module, rules: Mapping[nn.Module, BasicHook]) -> None:
		self.model = model
		self.rules = rules

	def generate(
		self,
		images: Tensor,
		target_index: int,
		*,
		task_name: str | None = None,
	) -> Tensor:
		"""为输入图像批次生成指定目标类别的 AttnLRP 归因热力图。"""
		leaf = images.detach().requires_grad_(True)

		with ExitStack() as stack:
			for module, hook in self.rules.items():
				handle = hook.register(module)
				stack.callback(handle.remove)

			patch_model_for_attnlrp(self.model, stack)

			with torch.enable_grad():
				outputs = self.model(leaf)
				if task_name is None:
					logits = cast(Tensor, outputs)
				else:
					outputs_mapping = cast(Mapping[str, Tensor], outputs)
					logits = outputs_mapping[task_name]

				selected = logits[:, target_index]
				grad = torch.autograd.grad(selected.sum(), leaf)[0]
				relevance = leaf * grad

		return relevance
