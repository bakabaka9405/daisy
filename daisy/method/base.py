"""通用算法方法抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from daisy.training.plugins import Plugin
from daisy.training.trainer import Trainer


class Method[M: nn.Module](ABC, Plugin):
	"""算法方法基类，作为核心插件向 Trainer 提供模型、默认优化器与训练 step。"""

	def __init__(
		self,
		backbone: nn.Module,
		*,
		lr: float,
		weight_decay: float = 0.05,
	) -> None:
		self.backbone = backbone
		self.lr = lr
		self.weight_decay = weight_decay

	@abstractmethod
	def build_model(self, backbone: nn.Module) -> M:
		"""构建算法专属完整模型。"""
		...

	def build_optimizer(self, model: M, *, lr: float, weight_decay: float) -> Optimizer:
		"""构建默认优化器（AdamW）。"""
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

	@override
	def register(self, trainer: Trainer) -> None:
		"""向 Trainer 安装模型、默认优化器与 process 闭包。"""
		model = self.build_model(self.backbone)
		self.model = model
		trainer.model = model
		if trainer.optimizer is None:
			trainer.optimizer = self.build_optimizer(model, lr=self.lr, weight_decay=self.weight_decay)

		process_fn = getattr(self, 'process', None)
		if process_fn is not None:
			trainer.set_process_fn(process_fn)

	def inference_step(self, X: Tensor) -> Tensor | dict[str, Tensor]:
		"""执行单批推理，返回原始未解码的模型输出（例如 logits）。"""
		raise NotImplementedError('Inference is not supported by this method.')

	def predict_step(self, X: Tensor) -> Tensor | dict[str, Tensor]:
		"""执行单批预测，返回解码后的任务预测结果（例如类别标签），非 logits。"""
		raise NotImplementedError('Prediction is not supported by this method.')
