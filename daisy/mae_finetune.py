"""MAE Finetune 辅助模块

仅负责:
1. 加载 MAE 预训练 checkpoint
2. 配置 MAE 特定的训练参数
3. 调用统一训练器

参考:
- https://github.com/facebookresearch/mae
"""

from pathlib import Path

import torch
import torch.nn as nn

from daisy.classfier_trainer import TrainResult, train_classifier
from daisy.dataset.index_dataset import IndexDataset


def load_mae_checkpoint(
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


def mae_finetune(
	device: torch.device,
	model: nn.Module,
	train_dataset: IndexDataset,
	val_dataset: IndexDataset,
	num_classes: int,
	epochs: int,
	# MAE 特定默认值
	batch_size: int = 64,
	blr: float = 1e-3,
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
	**kwargs,
) -> TrainResult:
	"""
	MAE 微调训练

	这是一个便捷函数，预设了 MAE 论文推荐的训练配置。

	Args:
		device: 训练设备
		model: ViT 模型 (已加载 MAE 预训练权重)
		train_dataset: 训练数据集
		val_dataset: 验证数据集
		num_classes: 分类数
		epochs: 训练轮数
		batch_size: 批大小
		blr: 基础学习率 (实际 lr = blr * batch_size / 256)
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
		log_dir: 日志目录
		early_stop: 是否启用早停
		early_stop_patience: 早停耐心值
		**kwargs: 传递给 train_classifier 的其他参数

	Returns:
		TrainResult 包含训练结果
	"""
	return train_classifier(
		device=device,
		model=model,
		num_classes=num_classes,
		epochs=epochs,
		dataset=(train_dataset, val_dataset),
		# MAE 特定配置
		blr=blr,
		layer_decay=layer_decay,
		weight_decay=weight_decay,
		warmup_epochs=warmup_epochs,
		min_lr=min_lr,
		mixup=mixup,
		cutmix=cutmix,
		smoothing=smoothing,
		batch_size=batch_size,
		accum_iter=accum_iter,
		use_amp=use_amp,
		clip_grad=clip_grad,
		num_workers=num_workers,
		pin_memory=pin_memory,
		save_path=save_path,
		save_freq=save_freq,
		log_dir=log_dir,
		drop_last=True,  # MAE finetune 需要 drop_last
		# MAE 默认启用的功能
		compute_metrics=False,  # MAE 只关注 acc
		save_best_metric='acc',
		early_stop=early_stop,
		early_stop_metric='acc',
		early_stop_patience=early_stop_patience,
		**kwargs,
	)
