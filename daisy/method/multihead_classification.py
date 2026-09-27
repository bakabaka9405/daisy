"""共享 backbone 的多任务分类方法实现。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import override

import torch
import torch.nn as nn
from torch import Tensor

from daisy.method.base import Method
from daisy.model.multihead import MultiHeadModel
from daisy.multitask_classification import ClassificationMetric, TaskPredictions, calculate_multihead_metrics
from daisy.training.trainer import Trainer
from daisy.typing import GenericDataLoader

ClassificationLoss = Callable[[Tensor, Tensor], Tensor]


class MultiHeadClassificationMethod(Method[MultiHeadModel]):
	"""共享 backbone、每个任务一个分类头的多任务分类方法。"""

	@override
	def __init__(
		self,
		model: MultiHeadModel,
		criterion: ClassificationLoss,
		*,
		lr: float,
		weight_decay: float = 0.05,
	) -> None:
		self.criterion: ClassificationLoss = criterion

		super().__init__(
			model,
			lr=lr,
			weight_decay=weight_decay,
		)

	@override
	def build_model(self, backbone: nn.Module) -> MultiHeadModel:
		"""校验并返回 MultiHeadModel 实例。"""
		if not isinstance(backbone, MultiHeadModel):
			raise TypeError(f'Expected MultiHeadModel, got {type(backbone).__name__}')
		return backbone

	def process(self, batch: tuple[Tensor, dict[str, Tensor]]) -> torch.Tensor:
		"""执行单批计算并按任务头名称计算平均损失。"""
		images, targets = batch
		logits_dict = self.model(images)
		if targets.keys() != logits_dict.keys():
			raise ValueError(
				f'Target task keys {set(targets.keys())} do not match model output task keys {set(logits_dict.keys())}'
			)

		losses = [self.criterion(logits_dict[task_name], targets[task_name]) for task_name in logits_dict]
		return torch.stack(losses).mean()

	@override
	def inference_step(self, X: Tensor) -> dict[str, Tensor]:
		"""执行单批推理，返回各任务头原始未归一化的分类 logits 字典。"""
		return self.model(X)

	@override
	def predict_step(self, X: Tensor) -> dict[str, Tensor]:
		"""执行单批预测，返回各任务解码后的 argmax 类别索引字典。"""
		return {task_name: logits.argmax(dim=1) for task_name, logits in self.inference_step(X).items()}

	def evaluate(
		self,
		trainer: Trainer,
		data_loader: GenericDataLoader,
		metrics: Sequence[ClassificationMetric],
	) -> dict[str, dict[str, float]]:
		"""使用 Trainer 执行推理并计算多任务分类评估指标。"""
		need_probs = 'auroc' in metrics

		def predict_fn(batch: tuple[Tensor, dict[str, Tensor]]) -> dict[str, dict[str, Tensor]]:
			images, targets = batch
			logits_dict = self.inference_step(images)
			if targets.keys() != logits_dict.keys():
				raise ValueError(
					f'Mismatch between target tasks {set(targets.keys())} and model output tasks {set(logits_dict.keys())}'
				)

			outputs: dict[str, dict[str, Tensor]] = {}
			for task in targets:
				leaf_dict: dict[str, Tensor] = {
					'targets': targets[task],
					'predictions': logits_dict[task].argmax(dim=1),
				}
				if need_probs:
					leaf_dict['probabilities'] = torch.softmax(logits_dict[task], dim=1)
				outputs[task] = leaf_dict
			return outputs

		collected = trainer.inference(data_loader, predict_fn)
		predictions = {
			task: TaskPredictions(
				targets=data['targets'],
				predictions=data['predictions'],
				probabilities=data.get('probabilities'),
			)
			for task, data in collected.items()
		}
		return calculate_multihead_metrics(predictions, self.model.num_classes, metrics)
