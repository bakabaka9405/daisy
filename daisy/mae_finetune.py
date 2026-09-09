"""MAE Finetune 辅助模块

基于 daisy.training 生态，通过 Plugin 组合完成 MAE 微调训练。

功能:
1. 加载 MAE 预训练 checkpoint
2. 以 MAE 论文推荐配置组装 Trainer + Plugin 栈
3. 提供 mae_finetune() 便捷入口

参考:
- https://github.com/facebookresearch/mae
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from daisy.typing import Replaceable

from daisy.classifier_trainer import evaluate
from daisy.dataset.index_dataset import IndexDataset
from daisy.util import Prefetcher
from daisy.training import (
	BestModel,
	Checkpoint,
	CosineAnnealingLR,
	CSVLog,
	EarlyStop,
	EpochPrint,
	Eval,
	Mixup,
	Trainer,
	TrainState,
)


def load_mae_pretrain_checkpoint(
	model: nn.Module,
	checkpoint_path: str | Path,
	strict: bool = False,
) -> nn.Module:
	"""
	加载 MAE 预训练权重

	Args:
		model: 目标模型 (通常是 ViT)
		checkpoint_path: checkpoint 文件路径
		strict: 是否严格匹配参数名

	Returns:
		加载权重后的模型
	"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu')

	# MAE checkpoint 可能的 key
	if 'model' in checkpoint:
		state_dict = checkpoint['model']
	elif 'state_dict' in checkpoint:
		state_dict = checkpoint['state_dict']
	else:
		state_dict = checkpoint

	# 过滤掉 decoder 相关的权重 (MAE 特有)
	state_dict = {k: v for k, v in state_dict.items() if not k.startswith('decoder')}

	# 加载权重
	msg = model.load_state_dict(state_dict, strict=strict)
	print(f'Loaded MAE checkpoint: {msg}')

	return model


@dataclass(frozen=True, slots=True, kw_only=True)
class MAEFinetuneParams(Replaceable):
	"""MAE 微调超参数。"""

	epochs: int
	batch_size: int = 64
	blr: float | None = 1e-3
	lr: float = 0
	layer_decay: float = 0.75
	weight_decay: float = 0.05
	warmup_epochs: int = 5
	min_lr: float = 1e-6
	mixup: float = 0.8
	cutmix: float = 1.0
	smoothing: float = 0.1
	accum_iter: int = 1
	use_amp: bool = True
	clip_grad: float | None = None
	num_workers: int | tuple[int, int] = 4
	pin_memory: bool = True
	early_stop: bool = False
	early_stop_patience: int = 10
	val_ratio: float = 0.1


def _build(
	device: torch.device,
	model: nn.Module,
	dataset: tuple[IndexDataset, IndexDataset] | IndexDataset,
	num_classes: int,
	params: MAEFinetuneParams,
	save_path: Path | None,
	save_freq: int,
	log_dir: Path | None,
	train_transform: Any,
	val_transform: Any,
	optimizer: torch.optim.Optimizer | None,
) -> tuple[Trainer, Any, Eval | None, BestModel | None, CSVLog | None]:
	"""组装 Trainer + Plugin 栈，返回内部 Plugin 引用。"""
	import daisy
	from timm.data.loader import MultiEpochsDataLoader
	from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

	# --- 数据集处理 ---
	if isinstance(dataset, tuple):
		train_dataset, val_dataset = dataset
	else:
		train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(dataset, val_ratio=params.val_ratio)

	# 默认 transform
	if train_transform is None:
		train_transform = daisy.util.transform.get_rectangle_train_transform()
	if val_transform is None:
		val_transform = daisy.util.transform.get_rectangle_val_transform()

	train_dataset.setTransform(train_transform)
	val_dataset.setTransform(val_transform)

	if isinstance(params.num_workers, int):
		workers = (params.num_workers, params.num_workers)
	else:
		workers = params.num_workers

	# --- DataLoader ---
	train_loader = Prefetcher(
		MultiEpochsDataLoader(
			train_dataset,
			batch_size=params.batch_size,
			shuffle=True,
			num_workers=workers[0],
			pin_memory=params.pin_memory,
			drop_last=True,
		),
		device=device,
	)

	val_loader = Prefetcher(
		MultiEpochsDataLoader(
			val_dataset,
			batch_size=params.batch_size,
			shuffle=False,
			num_workers=workers[1],
			pin_memory=params.pin_memory,
		),
		device=device,
	)

	# --- 学习率计算 ---
	if params.blr is not None:
		eff_batch_size = params.batch_size * params.accum_iter
		actual_lr = params.blr * eff_batch_size / 256
		print(f'Base LR: {params.blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {actual_lr:.2e}')
	else:
		actual_lr = params.lr

	# --- Optimizer ---
	if optimizer is None:
		from daisy.model.mae.lr_decay import param_groups_lrd
		from torch.optim import AdamW

		param_groups = param_groups_lrd(model, weight_decay=params.weight_decay, layer_decay=params.layer_decay)
		optimizer = AdamW(
			param_groups,
			lr=actual_lr,
			betas=(0.9, 0.999),
			fused=True,
		)

	# --- Criterion ---
	mixup_enabled = params.mixup > 0 or params.cutmix > 0
	if mixup_enabled:
		criterion = SoftTargetCrossEntropy()
	elif params.smoothing > 0:
		criterion = LabelSmoothingCrossEntropy(smoothing=params.smoothing)
	else:
		criterion = nn.CrossEntropyLoss()

	val_criterion = nn.CrossEntropyLoss()
	val_targets = torch.as_tensor(val_dataset.getRawData()[1], dtype=torch.long)

	# --- Trainer ---
	trainer = Trainer(
		model=model,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		use_amp=params.use_amp,
		accum_iter=params.accum_iter,
		clip_grad=params.clip_grad,
	)

	# --- Plugins ---
	# LR schedule (per-iteration for MAE style)
	lr_plugin = CosineAnnealingLR(
		lr=actual_lr,
		min_lr=params.min_lr,
		warmup_epochs=params.warmup_epochs,
		per_iteration=True,
	)

	plugins: list[Any] = [lr_plugin]

	# Mixup/CutMix
	if mixup_enabled:
		plugins.append(
			Mixup(
				mixup_alpha=params.mixup,
				cutmix_alpha=params.cutmix,
				smoothing=params.smoothing,
				num_classes=num_classes,
			)
		)

	# Eval
	evaluator: Eval | None = None
	best_model: BestModel | None = None

	def _eval_fn(trainer: Trainer) -> dict[str, float]:
		metrics = evaluate(
			scores=trainer.inference(val_loader),
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

	evaluator = Eval(eval_fn=_eval_fn)
	plugins.append(evaluator)

	# BestModel
	if save_path:
		save_path.mkdir(parents=True, exist_ok=True)
		best_model = BestModel(evaluator, watch_metric='f1', mode='max', save_path=save_path)
		plugins.append(best_model)

		# Checkpoint
		if save_freq > 0:
			plugins.append(Checkpoint(save_dir=save_path, save_freq=save_freq))

	# CSVLog
	csv_log: CSVLog | None = None
	if log_dir:
		log_dir.mkdir(parents=True, exist_ok=True)
		log_path = log_dir / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		csv_log = CSVLog(log_path=log_path, evaluator=evaluator)
		plugins.append(csv_log)

	# EpochPrint
	plugins.append(EpochPrint(evaluator=evaluator))

	# EarlyStop
	if params.early_stop:
		plugins.append(EarlyStop(evaluator, patience=params.early_stop_patience, watch_metric='f1', mode='max'))

	trainer.use(*plugins)

	return trainer, train_loader, evaluator, best_model, csv_log


def _extract_result(
	state: TrainState,
	evaluator: Eval | None,
	best_model: BestModel | None,
	csv_log: CSVLog | None,
) -> TrainState:
	"""将 Plugin 聚合结果写入 state.extras。"""
	if best_model:
		state.extras['best_f1'] = best_model.best_value
		state.extras['best_epoch'] = best_model.best_epoch

	if evaluator:
		state.extras['final_metrics'] = evaluator.metrics

	if csv_log:
		state.extras['history'] = csv_log.history

	return state


def mae_finetune(
	device: torch.device,
	model: nn.Module,
	dataset: tuple[IndexDataset, IndexDataset] | IndexDataset,
	num_classes: int,
	params: MAEFinetuneParams,
	*,
	save_path: Path | str | None = None,
	save_freq: int = 20,
	log_dir: Path | str | None = None,
	train_transform: Any = None,
	val_transform: Any = None,
	optimizer: torch.optim.Optimizer | None = None,
) -> TrainState:
	"""
	MAE 微调训练

	基于 daisy.training.Trainer + Plugin 生态的便捷入口。
	预设 MAE 论文推荐的训练配置。

	Args:
		device: 训练设备
		model: ViT 模型 (已加载 MAE 预训练权重)
		dataset: 数据集 (训练和验证) 或单个数据集 (按 params.val_ratio 拆分)
		num_classes: 分类数
		params: 微调超参数
		save_path: 模型保存路径
		save_freq: 保存频率（每 N 个 epoch）
		log_dir: 日志目录 (CSV 输出)
		train_transform: 训练数据变换
		val_transform: 验证数据变换
		optimizer: 自定义优化器 (None 时内部构建带 layer decay 的 AdamW)

	Returns:
		TrainState，聚合结果存在 state.extras 中:
		- extras['best_f1']: 最佳 F1
		- extras['best_epoch']: 最佳 epoch
		- extras['final_metrics']: 最终验证指标 dict
		- extras['history']: 训练历史 list[dict]
	"""
	# 路径标准化
	_save_path = Path(save_path) if save_path is not None else None
	_log_dir = Path(log_dir) if log_dir is not None else None

	torch.cuda.empty_cache()

	trainer, train_loader, evaluator, best_model, csv_log = _build(
		device=device,
		model=model,
		dataset=dataset,
		num_classes=num_classes,
		params=params,
		save_path=_save_path,
		save_freq=save_freq,
		log_dir=_log_dir,
		train_transform=train_transform,
		val_transform=val_transform,
		optimizer=optimizer,
	)

	print('Ready to train...')
	state = trainer.fit(train_loader, epochs=params.epochs)

	return _extract_result(state, evaluator, best_model, csv_log)
