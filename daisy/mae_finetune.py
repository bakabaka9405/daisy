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
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from daisy.classifier_trainer import evaluate
from daisy.dataset.index_dataset import IndexDataset
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


def _build(
	device: torch.device,
	model: nn.Module,
	dataset: tuple[IndexDataset, IndexDataset] | IndexDataset,
	num_classes: int,
	epochs: int,
	batch_size: int,
	blr: float | None,
	lr: float,
	layer_decay: float,
	weight_decay: float,
	warmup_epochs: int,
	min_lr: float,
	mixup: float,
	cutmix: float,
	smoothing: float,
	accum_iter: int,
	use_amp: bool,
	clip_grad: float | None,
	num_workers: int,
	pin_memory: bool,
	save_path: Path | None,
	save_freq: int,
	log_dir: Path | None,
	early_stop: bool,
	early_stop_patience: int,
	train_transform: Any,
	val_transform: Any,
	val_ratio: float,
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
		train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(
			dataset, val_ratio=val_ratio
		)

	# 默认 transform
	if train_transform is None:
		train_transform = daisy.util.transform.get_rectangle_train_transform()
	if val_transform is None:
		val_transform = daisy.util.transform.get_rectangle_val_transform()

	train_dataset.setTransform(train_transform)
	val_dataset.applyTransform(val_transform)

	# --- DataLoader ---
	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=True,
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
	)

	# --- 学习率计算 ---
	if blr is not None:
		eff_batch_size = batch_size * accum_iter
		actual_lr = blr * eff_batch_size / 256
		print(f'Base LR: {blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {actual_lr:.2e}')
	else:
		actual_lr = lr

	# --- Optimizer ---
	if optimizer is None:
		from daisy.model.mae.lr_decay import param_groups_lrd
		from torch.optim import AdamW

		param_groups = param_groups_lrd(model, weight_decay=weight_decay, layer_decay=layer_decay)
		optimizer = AdamW(param_groups, lr=actual_lr, betas=(0.9, 0.999))

	# --- Criterion ---
	mixup_enabled = mixup > 0 or cutmix > 0
	if mixup_enabled:
		criterion = SoftTargetCrossEntropy()
	elif smoothing > 0:
		criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
	else:
		criterion = nn.CrossEntropyLoss()

	val_criterion = nn.CrossEntropyLoss()

	# --- Trainer ---
	trainer = Trainer(
		model=model,
		optimizer=optimizer,
		criterion=criterion,
		device=device,
		use_amp=use_amp,
		accum_iter=accum_iter,
		clip_grad=clip_grad,
	)

	# --- Plugins ---
	# LR schedule (per-iteration for MAE style)
	lr_plugin = CosineAnnealingLR(
		lr=actual_lr,
		min_lr=min_lr,
		warmup_epochs=warmup_epochs,
		per_iteration=True,
	)

	plugins: list[Any] = [lr_plugin]

	# Mixup/CutMix
	if mixup_enabled:
		plugins.append(Mixup(
			mixup_alpha=mixup,
			cutmix_alpha=cutmix,
			smoothing=smoothing,
			num_classes=num_classes,
		))

	# Eval
	evaluator: Eval | None = None
	best_model: BestModel | None = None

	def _eval_fn(m: nn.Module, d: torch.device) -> dict[str, float]:
		metrics = evaluate(
			model=m,
			data_loader=val_loader,
			criterion=val_criterion,
			device=d,
			num_classes=num_classes,
			compute_metrics=True,
			use_amp=use_amp,
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

	# EarlyStop
	if early_stop:
		plugins.append(EarlyStop(
			evaluator, patience=early_stop_patience, watch_metric='f1', mode='max'
		))

	# CSVLog
	csv_log: CSVLog | None = None
	if log_dir:
		log_dir.mkdir(parents=True, exist_ok=True)
		log_path = log_dir / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		csv_log = CSVLog(log_path=log_path, evaluator=evaluator)
		plugins.append(csv_log)

	# EpochPrint
	plugins.append(EpochPrint(evaluator=evaluator))

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
	epochs: int,
	# MAE 特定默认值
	batch_size: int = 64,
	blr: float | None = 1e-3,
	lr: float = 0,
	layer_decay: float = 0.75,
	weight_decay: float = 0.05,
	warmup_epochs: int = 5,
	min_lr: float = 1e-6,
	mixup: float = 0.8,
	cutmix: float = 1.0,
	smoothing: float = 0.1,
	# 其他
	accum_iter: int = 1,
	use_amp: bool = True,
	clip_grad: float | None = None,
	num_workers: int = 4,
	pin_memory: bool = True,
	save_path: Path | str | None = None,
	save_freq: int = 20,
	log_dir: Path | str | None = None,
	# 额外参数
	early_stop: bool = False,
	early_stop_patience: int = 10,
	# Transform (由 TaskRunner 传入)
	train_transform: Any = None,
	val_transform: Any = None,
	val_ratio: float = 0.1,
	# 高级: 自定义 optimizer
	optimizer: torch.optim.Optimizer | None = None,
) -> TrainState:
	"""
	MAE 微调训练

	基于 daisy.training.Trainer + Plugin 生态的便捷入口。
	预设 MAE 论文推荐的训练配置。

	Args:
		device: 训练设备
		model: ViT 模型 (已加载 MAE 预训练权重)
		dataset: 数据集 (训练和验证) 或单个数据集 (按 val_ratio 拆分)
		num_classes: 分类数
		epochs: 训练轮数
		batch_size: 批大小
		blr: 基础学习率 (实际 lr = blr * batch_size * accum_iter / 256)
		lr: 直接指定学习率 (blr 优先)
		layer_decay: Layer-wise LR decay 系数
		weight_decay: 权重衰减
		warmup_epochs: warmup 轮数
		min_lr: 最小学习率
		mixup: Mixup alpha
		cutmix: CutMix alpha
		smoothing: Label smoothing
		accum_iter: 梯度累积迭代数
		use_amp: 是否使用混合精度
		clip_grad: 梯度裁剪的最大范数 (None 表示不裁剪)
		num_workers: DataLoader workers 数量
		pin_memory: 是否 pin memory
		save_path: 模型保存路径
		save_freq: 保存频率（每 N 个 epoch）
		log_dir: 日志目录 (CSV 输出)
		early_stop: 是否启用早停
		early_stop_patience: 早停耐心值
		train_transform: 训练数据变换
		val_transform: 验证数据变换
		val_ratio: 验证集比例 (仅在 dataset 为单个时使用)
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
		epochs=epochs,
		batch_size=batch_size,
		blr=blr,
		lr=lr,
		layer_decay=layer_decay,
		weight_decay=weight_decay,
		warmup_epochs=warmup_epochs,
		min_lr=min_lr,
		mixup=mixup,
		cutmix=cutmix,
		smoothing=smoothing,
		accum_iter=accum_iter,
		use_amp=use_amp,
		clip_grad=clip_grad,
		num_workers=num_workers,
		pin_memory=pin_memory,
		save_path=_save_path,
		save_freq=save_freq,
		log_dir=_log_dir,
		early_stop=early_stop,
		early_stop_patience=early_stop_patience,
		train_transform=train_transform,
		val_transform=val_transform,
		val_ratio=val_ratio,
		optimizer=optimizer,
	)

	print('Ready to train...')
	state = trainer.fit(train_loader, epochs=epochs)

	return _extract_result(state, evaluator, best_model, csv_log)
