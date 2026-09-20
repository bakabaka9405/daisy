"""分类训练数据与评估插件编排。

负责数据集划分、变换配置、DataLoader 构建以及为 ClassificationMethod 注册 Eval 评估插件。
"""

from __future__ import annotations

from typing import Any

import daisy
import numpy as np
import torch
import torch.nn as nn
from timm.data.loader import MultiEpochsDataLoader

from daisy.dataset.index_dataset import IndexDataset
from daisy.method.classification import ClassificationMethod
from daisy.metrics import EvalMetrics, auroc_score, fast_calc_metrics
from daisy.training import Eval, Trainer


def evaluate(
	scores: torch.Tensor,
	targets: torch.Tensor,
	criterion: nn.Module,
	num_classes: int,
	compute_metrics: bool = True,
) -> EvalMetrics:
	"""模型评估。

	Args:
		compute_metrics: 是否计算 precision/recall/f1 等扩展指标
	"""
	scores = scores.float()
	loss = criterion(scores, targets)
	y_pred = torch.argmax(scores, dim=1).numpy()
	y_true = targets.numpy()

	if not compute_metrics:
		acc = float(np.mean(y_true == y_pred)) if len(y_true) > 0 else 0.0
		return EvalMetrics(
			loss=loss.item(),
			acc=acc,
		)

	metrics = fast_calc_metrics(y_true, y_pred, num_classes=num_classes)

	# 计算 AUROC
	y_outputs = scores.numpy()
	try:
		auroc_result = auroc_score(
			y_true.astype(np.int64, copy=False),
			y_outputs.astype(np.float64, copy=False),
			mode='macro',
		)
		auroc_val = float(auroc_result['macro'] if isinstance(auroc_result, dict) else auroc_result)
	except Exception:
		auroc_val = 0.0

	return EvalMetrics(
		loss=loss.item(),
		acc=metrics.acc,
		precision=metrics.precision,
		recall=metrics.recall,
		f1=metrics.f1,
		auroc=auroc_val,
		confusion_matrix=metrics.confusion_matrix,
	)


def prepare_classification_training(
	method: ClassificationMethod,
	trainer: Trainer,
	dataset: tuple[IndexDataset, IndexDataset] | IndexDataset,
	*,
	batch_size: int = 64,
	train_transform: Any = None,
	val_transform: Any = None,
	val_ratio: float = 0.1,
	num_workers: int | tuple[int, int] = 4,
	pin_memory: bool = True,
	drop_last: bool = False,
) -> MultiEpochsDataLoader:
	"""配置分类任务的 DataLoader 并向 Trainer 注册 Eval 插件。"""
	if isinstance(dataset, tuple):
		train_dataset, val_dataset = dataset
	else:
		train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(dataset, val_ratio=val_ratio)

	if train_transform is None:
		train_transform = daisy.util.transform.get_rectangle_train_transform()
	if val_transform is None:
		val_transform = daisy.util.transform.get_rectangle_val_transform()

	train_dataset.setTransform(train_transform)
	val_dataset.setTransform(val_transform)

	if isinstance(num_workers, int):
		workers = (num_workers, num_workers)
	else:
		workers = num_workers

	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=workers[0],
		pin_memory=pin_memory,
		drop_last=drop_last or (method.mixup_fn is not None),
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=workers[1],
		pin_memory=pin_memory,
	)

	val_criterion = nn.CrossEntropyLoss()

	def eval_fn(active_trainer: Trainer) -> dict[str, float]:
		scores, targets = active_trainer.inference(
			val_loader,
			lambda batch: (method.predict_step(batch[0]), batch[1]),
		)
		metrics = evaluate(
			scores=scores,
			targets=targets,
			criterion=val_criterion,
			num_classes=method.num_classes,
			compute_metrics=True,
		)
		return {
			'loss': metrics.loss,
			'acc': metrics.acc,
			'f1': metrics.f1,
			'precision': metrics.precision,
			'recall': metrics.recall,
			'auroc': metrics.auroc,
		}

	trainer.use(Eval(eval_fn=eval_fn))

	return train_loader
