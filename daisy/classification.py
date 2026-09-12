"""分类训练数据与评估插件编排。

负责数据集划分、变换配置、DataLoader 构建以及为 ClassificationMethod 注册 Eval 评估插件。
"""

from __future__ import annotations

from typing import Any

import daisy
import torch
import torch.nn as nn
from timm.data.loader import MultiEpochsDataLoader

from daisy.classifier_trainer import evaluate
from daisy.dataset.index_dataset import IndexDataset
from daisy.method.classification import ClassificationMethod
from daisy.training import Eval, Trainer


def prepare_classification_training(
	method: ClassificationMethod,
	dataset: tuple[IndexDataset, IndexDataset] | IndexDataset,
	*,
	batch_size: int = 64,
	train_transform: Any = None,
	val_transform: Any = None,
	val_ratio: float = 0.1,
	num_workers: int | tuple[int, int] = 4,
	pin_memory: bool = True,
	drop_last: bool = False,
) -> tuple[MultiEpochsDataLoader, Eval]:
	"""配置分类任务的 DataLoader 并向方法 Trainer 注册 Eval 插件。"""
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
	val_targets = torch.as_tensor(val_dataset.getRawData()[1], dtype=torch.long)

	def eval_fn(trainer: Trainer) -> dict[str, float]:
		metrics = evaluate(
			scores=method.predict(val_loader),
			targets=val_targets,
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

	evaluator = Eval(eval_fn=eval_fn)
	method.trainer.use(evaluator)

	return train_loader, evaluator
