"""MoCo 预训练算法方法实现。"""

from __future__ import annotations

import math
from typing import Literal, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from daisy.method.base import Method
from daisy.training.state import TrainState


class MoCoPretrainMethod(Method[nn.Module]):
	"""MoCo (v2/v3) 自监督对比学习预训练方法。"""

	@override
	def __init__(
		self,
		model: nn.Module,
		*,
		device: torch.device,
		engine: Literal['v2', 'v3'] = 'v3',
		lr: float = 0.03,
		weight_decay: float = 1e-4,
		momentum: float = 0.9,
		optimizer_type: Literal['sgd', 'adamw', 'lars'] = 'sgd',
		moco_m: float = 0.999,
		moco_m_cos: bool = False,
		use_amp: bool = True,
		clip_grad: float | None = None,
		optimizer: torch.optim.Optimizer | None = None,
	) -> None:
		self.engine = engine
		self.optimizer_type = optimizer_type
		self.momentum = momentum
		self.moco_m = moco_m
		self.moco_m_cos = moco_m_cos
		self.current_m: float = moco_m

		super().__init__(
			model,
			device=device,
			lr=lr,
			weight_decay=weight_decay,
			use_amp=use_amp,
			accum_iter=1,
			clip_grad=clip_grad,
			optimizer=optimizer,
		)

		self.trainer.on_batch_start(self._update_momentum)

	@override
	def build_model(self, backbone: nn.Module) -> nn.Module:
		"""直接复用完整模型实例，确保权重键名与检查点结构严格对齐。"""
		return backbone

	@override
	def build_optimizer(
		self,
		model: nn.Module,
		*,
		lr: float,
		weight_decay: float,
	) -> torch.optim.Optimizer:
		"""根据指定优化算法组装参数优化器，保持参数原始拓扑与遍历顺序。"""
		if self.optimizer_type == 'lars':
			from daisy.model.moco.lars import LARS

			return LARS(
				model.parameters(),
				lr=lr,
				weight_decay=weight_decay,
				momentum=self.momentum,
			)
		if self.optimizer_type == 'adamw':
			return torch.optim.AdamW(
				model.parameters(),
				lr=lr,
				weight_decay=weight_decay,
			)
		return torch.optim.SGD(
			model.parameters(),
			lr=lr,
			momentum=self.momentum,
			weight_decay=weight_decay,
		)

	def _update_momentum(self, state: TrainState) -> None:
		"""在各批次开始时依进度坐标计算并写入当前动量编码器更新系数。"""
		if self.engine == 'v3':
			if self.moco_m_cos:
				t = state.epoch + state.batch_idx / state.total_batches
				self.current_m = 1.0 - (1.0 - self.moco_m) * (1.0 + math.cos(math.pi * t / state.total_epochs)) / 2.0
			else:
				self.current_m = self.moco_m
		else:
			self.current_m = float(getattr(self.model, 'm', self.moco_m))

		state.extras['moco_m'] = self.current_m

	def process(self, batch: tuple[Tensor, Tensor] | list[Tensor]) -> Tensor:
		"""解包双视图张量，分发 v2 与 v3 对比损失并阻断非有限数值的反向传播。"""
		im_1, im_2 = batch

		if self.engine == 'v2':
			logits, labels = self.model(im_1, im_2)
			loss = F.cross_entropy(logits, labels)
		else:
			loss = self.model(im_1, im_2, self.current_m)

		if not torch.isfinite(loss):
			loss_val = loss.item()
			raise RuntimeError(f'Loss is {loss_val}')

		return loss


__all__ = ['MoCoPretrainMethod']
