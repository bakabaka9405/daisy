"""通用算法方法抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast, override

import torch
import torch.nn as nn
from torch import Tensor

from daisy.training.state import TrainState
from daisy.training.trainer import Trainer
from daisy.typing import GenericDataLoader


class Method[M: nn.Module](ABC):
	"""算法方法基类，统一样本训练 step、推理 step 与 Trainer 生命周期。"""

	@override
	def __init__(
		self,
		backbone: nn.Module,
		*,
		device: torch.device,
		lr: float,
		weight_decay: float = 0.05,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
		optimizer: torch.optim.Optimizer | None = None,
	) -> None:
		# 构建完整模型并移至目标设备
		model = self.build_model(backbone).to(device)
		if optimizer is None:
			optimizer = self.build_optimizer(model, lr=lr, weight_decay=weight_decay)

		# 组装 Trainer 实例
		self.trainer = Trainer(
			model=model,
			optimizer=optimizer,
			device=device,
			use_amp=use_amp,
			accum_iter=accum_iter,
			clip_grad=clip_grad,
		)

		process_fn = getattr(self, 'process', None)
		if process_fn is not None:
			self.trainer.set_process_fn(process_fn)

	def build_optimizer(self, model: M, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
		"""构建默认优化器（AdamW）。"""
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	@property
	def model(self) -> M:
		"""获取模型实例（只读属性，委托自 trainer.model）。"""
		return cast(M, self.trainer.model)

	@abstractmethod
	def build_model(self, backbone: nn.Module) -> M:
		"""构建算法专属完整模型。"""
		...

	def predict_step(self, X: Tensor) -> torch.Tensor:
		"""执行单批推理，返回预测张量。"""
		raise NotImplementedError('Prediction is not supported by this method.')

	def fit(self, loader: GenericDataLoader, epochs: int, *, start_epoch: int = 0) -> TrainState:
		"""执行 [start_epoch, epochs) 区间训练并返回最终状态。"""
		return self.trainer.fit(loader, epochs=epochs, start_epoch=start_epoch)

	def predict(self, loader: GenericDataLoader) -> torch.Tensor:
		"""按 batch 执行推理并拼接为完整张量。"""
		if type(self).predict_step is Method.predict_step:
			raise NotImplementedError('Prediction is not supported by this method.')
		return self.trainer.inference(loader, lambda batch: self.predict_step(batch[0]))
