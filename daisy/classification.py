"""分类训练数据与评估插件编排。

负责数据集划分、变换配置、DataLoader 构建以及为 ClassificationMethod 注册 Eval 评估插件。
"""

from __future__ import annotations

from typing import Any

import daisy
from timm.data.loader import MultiEpochsDataLoader

from daisy.dataset.index_dataset import IndexDataset
from daisy.method.classification import ClassificationMethod
from daisy.training import Eval, Trainer
from daisy.util import to_2tuple


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

	workers = to_2tuple(num_workers)

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

	def eval_fn(active_trainer: Trainer) -> dict[str, float]:
		metrics = method.evaluate(active_trainer, val_loader)
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
