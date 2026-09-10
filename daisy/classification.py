"""通用分类训练组装

组合数据集划分、默认 transform、DataLoader、损失函数与 Trainer/Eval 插件，
构建可直接用于 Trainer.fit 的分类训练组件。
"""

from __future__ import annotations

from typing import Any

import daisy
import torch
import torch.nn as nn
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup

from daisy.classifier_trainer import create_criterion, evaluate
from daisy.dataset.index_dataset import IndexDataset
from daisy.training import Eval, Trainer


def build_classifier_trainer(
	device: torch.device,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	dataset: tuple[IndexDataset, IndexDataset] | IndexDataset,
	num_classes: int,
	*,
	batch_size: int = 64,
	train_transform: Any = None,
	val_transform: Any = None,
	val_ratio: float = 0.1,
	num_workers: int | tuple[int, int] = 4,
	pin_memory: bool = True,
	drop_last: bool = False,
	mixup: float = 0.0,
	cutmix: float = 0.0,
	smoothing: float = 0.0,
	use_amp: bool = True,
	accum_iter: int = 1,
	clip_grad: float | None = None,
) -> tuple[Trainer, MultiEpochsDataLoader, Eval]:
	"""组装通用分类训练的 Trainer、训练 DataLoader 与 Eval 插件。

	调用方必须提供优化器，并可在返回的 Trainer 上继续追加自定义插件；Eval
	已在此处注册，且先于后续注册的插件执行。

	Args:
		device: 训练设备
		model: 分类模型
		optimizer: 优化器
		dataset: (训练集, 验证集) 或单个数据集（按 val_ratio 划分）
		num_classes: 分类数
		batch_size: 批大小
		train_transform: 训练 transform，None 时使用默认矩形 transform
		val_transform: 验证 transform，None 时使用默认矩形 transform
		val_ratio: 单数据集划分时的验证集比例
		num_workers: DataLoader worker 数（int 或 (train, val)）
		pin_memory: 是否固定内存
		drop_last: 训练集是否丢弃尾批（启用 mixup 时强制丢弃）
		mixup: Mixup alpha（0 表示禁用）
		cutmix: CutMix alpha（0 表示禁用）
		smoothing: 标签平滑系数
		use_amp: 是否启用混合精度
		accum_iter: 梯度累积步数
		clip_grad: 梯度裁剪阈值（None 表示禁用）

	Returns:
		(Trainer, 训练 DataLoader, Eval)。DataLoader 为普通实例，设备搬运
		由 Trainer.fit / Trainer.inference 内部的 Prefetcher 完成。
	"""
	# --- 数据集划分与 transform ---
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

	# --- DataLoader ---
	mixup_enabled = mixup > 0 or cutmix > 0

	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=workers[0],
		pin_memory=pin_memory,
		drop_last=drop_last or mixup_enabled,
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=workers[1],
		pin_memory=pin_memory,
	)

	# --- 损失函数与 Mixup ---
	criterion = create_criterion(mixup_enabled, smoothing)
	val_criterion = nn.CrossEntropyLoss()
	val_targets = torch.as_tensor(val_dataset.getRawData()[1], dtype=torch.long)

	mixup_fn: Mixup | None = None
	if mixup_enabled:
		mixup_fn = Mixup(
			mixup_alpha=mixup,
			cutmix_alpha=cutmix,
			prob=1.0,
			switch_prob=0.5,
			mode='batch',
			label_smoothing=smoothing,
			num_classes=num_classes,
		)

	def process_fn(batch: Any) -> torch.Tensor:
		images, targets = batch
		if mixup_fn is not None:
			images, targets = mixup_fn(images, targets)
		return criterion(model(images), targets)

	def predict_fn(batch: Any) -> torch.Tensor:
		return model(batch[0])

	# --- Trainer 与 Eval ---
	trainer = Trainer(
		model=model,
		optimizer=optimizer,
		process_fn=process_fn,
		device=device,
		use_amp=use_amp,
		accum_iter=accum_iter,
		clip_grad=clip_grad,
	)

	def eval_fn(trainer: Trainer) -> dict[str, float]:
		metrics = evaluate(
			scores=trainer.inference(val_loader, predict_fn),
			targets=val_targets,
			criterion=val_criterion,
			num_classes=num_classes,
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
	trainer.use(evaluator)

	return trainer, train_loader, evaluator
