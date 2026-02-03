"""统一分类训练器

支持:
- 通用分类训练 (原 fast_train_smile 功能)
- MAE Finetune (Layer-wise LR Decay, Mixup/CutMix)
- 多种早停和保存策略
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import daisy
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from timm.data.loader import MultiEpochsDataLoader
from timm.data.mixup import Mixup
from timm.loss.cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.optim import AdamW

from daisy.dataset import IndexDataset


# ============================================================================
# 辅助类
# ============================================================================


class LRScheduler:
	"""学习率调度器，支持 warmup + cosine decay，可选 per-iteration 调整"""

	def __init__(
		self,
		optimizer: torch.optim.Optimizer,
		lr: float,
		min_lr: float,
		warmup_epochs: int,
		total_epochs: int,
		steps_per_epoch: int,
		per_iteration: bool = False,
	):
		self.optimizer = optimizer
		self.lr = lr
		self.min_lr = min_lr
		self.warmup_epochs = warmup_epochs
		self.total_epochs = total_epochs
		self.steps_per_epoch = steps_per_epoch
		self.per_iteration = per_iteration

	def get_lr(self, epoch: int, step: int = 0) -> float:
		"""计算当前学习率"""
		if self.per_iteration:
			current = epoch + step / self.steps_per_epoch
		else:
			current = epoch

		if current < self.warmup_epochs:
			# Linear warmup
			return self.lr * current / max(self.warmup_epochs, 1e-8)
		else:
			# Cosine decay
			progress = (current - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
			return self.min_lr + (self.lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

	def step(self, epoch: int, step: int = 0):
		"""更新优化器学习率"""
		current_lr = self.get_lr(epoch, step)
		for param_group in self.optimizer.param_groups:
			# 支持 layer-wise lr scale
			lr_scale = param_group.get('lr_scale', 1.0)
			param_group['lr'] = current_lr * lr_scale


@dataclass
class EvalMetrics:
	"""评估指标容器"""

	loss: float = 0.0
	acc: float = 0.0  # 准确率
	precision: float = 0.0
	recall: float = 0.0
	f1: float = 0.0
	confusion_matrix: list[list[int]] | None = None

	def get_metric(self, name: str) -> float:
		"""根据名称获取指标值"""
		if name == 'acc' or name == 'acc1':
			return self.acc
		elif name == 'prec':
			return self.precision
		elif name == 'recall':
			return self.recall
		elif name == 'f1':
			return self.f1
		else:
			return 0.0


@dataclass
class TrainResult:
	"""训练结果"""

	best_epoch: int
	best_metrics: EvalMetrics
	final_metrics: EvalMetrics
	history: list[dict] = field(default_factory=list)


# ============================================================================
# 工具函数
# ============================================================================


def calc_confusion_matrix(y_true, y_pred, num_classes: int) -> list[list[int]]:
	"""计算混淆矩阵"""
	count = [[0] * num_classes for _ in range(num_classes)]
	for i, j in zip(y_true, y_pred):
		count[i][j] += 1  # type:ignore
	return count


def create_criterion(mixup_enabled: bool, smoothing: float) -> nn.Module:
	"""根据配置创建损失函数"""
	if mixup_enabled:
		# Mixup/CutMix 需要 soft target
		return SoftTargetCrossEntropy()
	elif smoothing > 0:
		return LabelSmoothingCrossEntropy(smoothing=smoothing)
	else:
		return nn.CrossEntropyLoss()


def build_optimizer(
	model: nn.Module,
	lr: float,
	weight_decay: float,
	layer_decay: float | None = None,
) -> torch.optim.Optimizer:
	"""
	构建优化器

	Args:
		layer_decay: 若提供，使用 layer-wise lr decay (用于 MAE finetune)
	"""
	if layer_decay is not None:
		# 使用 layer-wise lr decay
		from daisy.model.mae.lr_decay import param_groups_lrd

		param_groups = param_groups_lrd(model, weight_decay=weight_decay, layer_decay=layer_decay)
	else:
		# 普通参数组
		param_groups = [{'params': model.parameters(), 'weight_decay': weight_decay}]

	return AdamW(param_groups, lr=lr, betas=(0.9, 0.999))


# ============================================================================
# 评估函数
# ============================================================================


def evaluate(
	model: nn.Module,
	data_loader,
	criterion: nn.Module,
	device: torch.device,
	num_classes: int,
	compute_metrics: bool = True,
	use_amp: bool = True,
) -> EvalMetrics:
	"""
	模型评估

	Args:
		compute_metrics: 是否计算 precision/recall/f1
	"""
	model.eval()
	total_loss = 0.0
	y_pred = []
	y_true = []
	num_batches = len(data_loader)

	with torch.no_grad():
		for images, targets in data_loader:
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)

			with torch.autocast('cuda', enabled=use_amp):
				outputs = model(images)
				loss = criterion(outputs, targets)

			total_loss += loss.item()

			# 收集预测结果
			preds = torch.argmax(outputs, dim=1)
			y_pred.extend(preds.cpu().numpy())
			y_true.extend(targets.cpu().numpy())

	# 计算平均值
	avg_loss = total_loss / num_batches

	# 准确率
	acc = accuracy_score(y_true, y_pred)

	# 计算其他指标
	if compute_metrics:
		prec = precision_score(y_true, y_pred, average='macro', zero_division=0)  # type: ignore[arg-type]
		rec = recall_score(y_true, y_pred, average='macro', zero_division=0)  # type: ignore[arg-type]
		f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)  # type: ignore[arg-type]
		matrix = calc_confusion_matrix(y_true, y_pred, num_classes)
	else:
		prec = rec = f1 = 0.0
		matrix = None

	return EvalMetrics(
		loss=avg_loss,
		acc=float(acc),
		precision=float(prec),
		recall=float(rec),
		f1=float(f1),
		confusion_matrix=matrix,
	)


# ============================================================================
# 训练循环
# ============================================================================


def train_one_epoch(
	model: nn.Module,
	data_loader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	lr_scheduler: LRScheduler,
	device: torch.device,
	epoch: int,
	mixup_fn: Mixup | None = None,
	scaler: torch.GradScaler | None = None,
	accum_iter: int = 1,
	clip_grad: float | None = None,
	print_freq: int = 20,
	compute_train_acc: bool = True,
) -> tuple[float, float]:
	"""
	单个 epoch 训练

	Returns:
		(train_loss, train_acc)
	"""
	model.train()
	total_loss = 0.0
	y_pred, y_true = [], []

	num_batches = len(data_loader)
	for i, (images, targets) in enumerate(data_loader):
		# 1. 学习率调整 (per-iteration if enabled)
		lr_scheduler.step(epoch, i)

		# 2. 数据移动
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		# 3. 保存原始 targets (用于准确率计算)
		if compute_train_acc and mixup_fn is None:
			y_true.extend(targets.cpu().numpy())

		# 4. Mixup/CutMix
		if mixup_fn is not None:
			images, targets = mixup_fn(images, targets)

		# 5. 前向传播
		with torch.autocast('cuda', enabled=(scaler is not None)):
			outputs = model(images)
			loss = criterion(outputs, targets) / accum_iter

		# 6. 记录预测 (仅非 mixup 时)
		if compute_train_acc and mixup_fn is None:
			preds = torch.argmax(outputs, dim=1)
			y_pred.extend(preds.cpu().numpy())

		# 7. 反向传播
		if scaler is not None:
			scaler.scale(loss).backward()
		else:
			loss.backward()

		# 8. 梯度累积 & 更新
		if (i + 1) % accum_iter == 0 or (i + 1) == num_batches:
			if clip_grad is not None and scaler is not None:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

			if scaler is not None:
				scaler.step(optimizer)
				scaler.update()
			else:
				optimizer.step()
			optimizer.zero_grad()

		total_loss += loss.item() * accum_iter

		# 9. 打印进度
		if print_freq > 0 and ((i + 1) % print_freq == 0 or (i + 1) == num_batches):
			current_lr = optimizer.param_groups[0]['lr']
			print(f'[{i + 1}/{num_batches}] Loss: {loss.item() * accum_iter:.4f} LR: {current_lr:.6f}')

	avg_loss = total_loss / num_batches
	train_acc = float(accuracy_score(y_true, y_pred)) if y_pred else 0.0

	return avg_loss, train_acc


# ============================================================================
# 主训练函数
# ============================================================================


def train_classifier(
	device: torch.device,
	model: torch.nn.Module,
	num_classes: int,
	epochs: int,
	dataset: IndexDataset | tuple[IndexDataset, IndexDataset],
	# ========== 学习率配置 ==========
	lr: float = 1e-3,
	blr: float | None = None,  # 基础学习率 (若提供则 lr = blr * batch_size / 256)
	min_lr: float = 1e-6,  # 最小学习率 (cosine decay 终点)
	warmup_epochs: int = 0,
	weight_decay: float = 1e-4,
	# ========== Layer-wise LR Decay (MAE finetune) ==========
	layer_decay: float | None = None,  # 若提供则启用 layer-wise lr decay
	# ========== 数据增强 ==========
	mixup: float = 0.0,  # Mixup alpha (0 表示禁用)
	cutmix: float = 0.0,  # CutMix alpha (0 表示禁用)
	mixup_prob: float = 1.0,  # Mixup/CutMix 应用概率
	mixup_switch_prob: float = 0.5,  # Mixup 和 CutMix 切换概率
	# ========== 损失函数 ==========
	smoothing: float = 0.1,  # Label smoothing
	# ========== 训练配置 ==========
	batch_size: int = 128,
	accum_iter: int = 1,
	use_amp: bool = True,
	clip_grad: float | None = None,  # 梯度裁剪 (None 表示禁用)
	# ========== 数据加载 ==========
	train_transform=None,
	val_transform=None,
	val_ratio: float = 0.1,
	num_workers: int | tuple[int, int] = 4,
	pin_memory: bool = True,
	drop_last: bool = False,
	# ========== 评估配置 ==========
	compute_metrics: bool = True,  # 是否计算 precision/recall/f1
	show_confusion_matrix: bool = False,
	# ========== Early Stopping ==========
	early_stop: bool = False,
	early_stop_patience: int = 5,
	early_stop_metric: Literal['acc', 'acc1', 'prec', 'recall', 'f1'] = 'f1',
	# ========== 模型保存 ==========
	save_path: Path | str | None = None,
	save_freq: int = 0,  # 定期保存频率 (0 表示禁用)
	save_best: bool = True,
	save_best_metric: Literal['acc', 'acc1', 'prec', 'recall', 'f1'] = 'f1',
	keep_recent: int = 0,  # 保留最近 N 个 checkpoint (0 表示禁用)
	# ========== 日志 ==========
	log_dir: Path | str | None = None,
	print_freq: int = 20,  # 训练进度打印频率
) -> TrainResult:
	"""
	统一分类训练器

	Returns:
		TrainResult 包含最佳指标和训练历史
	"""
	# ========== 默认 transform ==========
	if train_transform is None:
		train_transform = daisy.util.transform.get_rectangle_train_transform()
	if val_transform is None:
		val_transform = daisy.util.transform.get_rectangle_val_transform()

	# ========== 路径处理 ==========
	if save_path is not None:
		if isinstance(save_path, str):
			save_path = Path(save_path)
		save_path.mkdir(parents=True, exist_ok=True)

	# ========== 日志处理 ==========
	if log_dir is not None:
		if isinstance(log_dir, str):
			log_dir = Path(log_dir)
		log_dir.mkdir(parents=True, exist_ok=True)
		log_file = log_dir / f"log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
		# 构建 CSV 表头
		header_parts = ['epoch', 'lr', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
		if compute_metrics:
			header_parts.extend(['precision', 'recall', 'f1'])
		header_parts.append('confusion_matrix')
		with open(log_file, 'w', encoding='utf-8') as f:
			f.write(','.join(header_parts) + '\n')
	else:
		log_file = None

	# ========== 数据集划分 ==========
	if isinstance(dataset, tuple):
		train_dataset, val_dataset = dataset
	else:
		train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(dataset, val_ratio=val_ratio)
	torch.cuda.empty_cache()

	train_dataset.setTransform(train_transform)
	val_dataset.applyTransform(val_transform)

	if isinstance(num_workers, int):
		num_workers = (num_workers, num_workers)

	# ========== DataLoader ==========
	print('Loading dataloaders...')
	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers[0],
		pin_memory=pin_memory,
		drop_last=drop_last or (mixup > 0 or cutmix > 0),  # Mixup 需要 drop_last
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers[1],
		pin_memory=pin_memory,
	)

	model.to(device)

	# ========== 计算实际学习率 ==========
	if blr is not None:
		eff_batch_size = batch_size * accum_iter
		lr = blr * eff_batch_size / 256
		print(f'Base LR: {blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

	# ========== 优化器 ==========
	optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, layer_decay=layer_decay)

	# ========== 学习率调度器 ==========
	# 判断是否使用 per-iteration 调整 (MAE 风格)
	per_iteration = layer_decay is not None or mixup > 0 or cutmix > 0
	lr_scheduler = LRScheduler(
		optimizer=optimizer,
		lr=lr,
		min_lr=min_lr,
		warmup_epochs=warmup_epochs,
		total_epochs=epochs,
		steps_per_epoch=len(train_loader),
		per_iteration=per_iteration,
	)

	# ========== Mixup/CutMix ==========
	mixup_fn = None
	if mixup > 0 or cutmix > 0:
		mixup_fn = Mixup(
			mixup_alpha=mixup,
			cutmix_alpha=cutmix,
			prob=mixup_prob,
			switch_prob=mixup_switch_prob,
			mode='batch',
			label_smoothing=smoothing,
			num_classes=num_classes,
		)

	# ========== 损失函数 ==========
	criterion = create_criterion(mixup_fn is not None, smoothing)
	val_criterion = nn.CrossEntropyLoss()  # 验证时不使用 mixup/smoothing

	print('Ready to train...')
	scaler = torch.GradScaler(enabled=use_amp) if use_amp else None

	# ========== 训练状态 ==========
	best_epoch = 0
	best_metrics = EvalMetrics()
	history: list[dict] = []
	val_metrics = None

	for epoch in range(epochs):
		print(f'\n========== Epoch {epoch + 1}/{epochs} ==========')

		# ===== Training =====
		train_loss, train_acc = train_one_epoch(
			model=model,
			data_loader=train_loader,
			criterion=criterion,
			optimizer=optimizer,
			lr_scheduler=lr_scheduler,
			device=device,
			epoch=epoch,
			mixup_fn=mixup_fn,
			scaler=scaler,
			accum_iter=accum_iter,
			clip_grad=clip_grad,
			print_freq=print_freq,
			compute_train_acc=(mixup_fn is None),  # Mixup 时不计算训练准确率
		)

		# 如果不是 per-iteration 调度，在 epoch 结束后更新
		if not per_iteration:
			lr_scheduler.step(epoch + 1)

		# ===== Validation =====
		val_metrics = evaluate(
			model=model,
			data_loader=val_loader,
			criterion=val_criterion,
			device=device,
			num_classes=num_classes,
			compute_metrics=compute_metrics,
			use_amp=use_amp,
		)

		# ===== 打印结果 =====
		current_lr = optimizer.param_groups[0]['lr']
		result_str = (
			f'Epoch {epoch + 1}/{epochs}, LR: {current_lr:.6f}, '
			f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
			f'Val Loss: {val_metrics.loss:.4f}, Val Acc: {val_metrics.acc:.4f}'
		)

		# 其他指标
		if compute_metrics:
			result_str += f', Precision: {val_metrics.precision:.4f}, Recall: {val_metrics.recall:.4f}, F1: {val_metrics.f1:.4f}'

		print(result_str)

		if show_confusion_matrix and val_metrics.confusion_matrix:
			print(val_metrics.confusion_matrix)

		# ===== 记录历史 =====
		epoch_record = {
			'epoch': epoch + 1,
			'lr': current_lr,
			'train_loss': train_loss,
			'train_acc': train_acc,
			'val_loss': val_metrics.loss,
			'val_acc': val_metrics.acc,
			'precision': val_metrics.precision,
			'recall': val_metrics.recall,
			'f1': val_metrics.f1,
		}
		history.append(epoch_record)

		# ===== 写入日志 =====
		if log_file is not None:
			log_parts = [
				str(epoch + 1),
				f'{current_lr:.6f}',
				f'{train_loss:.4f}',
				f'{train_acc:.4f}',
				f'{val_metrics.loss:.4f}',
				f'{val_metrics.acc:.4f}',
			]
			if compute_metrics:
				log_parts.extend([f'{val_metrics.precision:.4f}', f'{val_metrics.recall:.4f}', f'{val_metrics.f1:.4f}'])
			log_parts.append(f'"{val_metrics.confusion_matrix}"' if val_metrics.confusion_matrix else '""')
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(','.join(log_parts) + '\n')

		# ===== 判断是否更好 =====
		current_metric = val_metrics.get_metric(save_best_metric)
		best_metric = best_metrics.get_metric(save_best_metric)
		is_better = current_metric > best_metric

		# ===== 保存模型 =====
		if save_path is not None:
			# 保留最近 N 个 checkpoint
			if keep_recent > 0:
				torch.save(model.state_dict(), save_path / f'model_epoch_{epoch + 1}.pth')
				if epoch >= keep_recent:
					remove_path = save_path / f'model_epoch_{epoch + 1 - keep_recent}.pth'
					if remove_path.exists():
						remove_path.unlink()

			# 定期保存
			if save_freq > 0 and (epoch + 1) % save_freq == 0:
				torch.save(model.state_dict(), save_path / f'model_epoch_{epoch + 1}.pth')
				print(f'Saved checkpoint at epoch {epoch + 1}')

			# 保存最佳模型
			if is_better and save_best:
				torch.save(model.state_dict(), save_path / 'best_model.pth')
				print(f'Saved best model with {save_best_metric}: {current_metric:.4f}')

		# ===== 更新最佳 =====
		if is_better:
			best_metrics = val_metrics
			best_epoch = epoch

		# ===== Early Stopping =====
		early_metric = val_metrics.get_metric(early_stop_metric)
		best_early_metric = best_metrics.get_metric(early_stop_metric)
		if early_stop and epoch - best_epoch >= early_stop_patience:
			print(f'Early stopping at epoch {epoch + 1}')
			if log_file is not None:
				with open(log_file, 'a', encoding='utf-8') as f:
					f.write(f'# Early stopping at epoch {epoch + 1}\n')
			break

	# 获取最终指标
	assert val_metrics is not None
	final_metrics = val_metrics

	print(f'\nTraining completed! Best epoch: {best_epoch + 1}')
	return TrainResult(
		best_epoch=best_epoch,
		best_metrics=best_metrics,
		final_metrics=final_metrics,
		history=history,
	)


# ============================================================================
# 向后兼容别名
# ============================================================================


def fast_train_smile(
	device: torch.device,
	model: torch.nn.Module,
	num_classes: int,
	epochs: int,
	lr: float,
	dataset: IndexDataset | tuple[IndexDataset, IndexDataset],
	batch_size: int = 128,
	accum_iter: int = 1,
	train_transform=None,
	val_transform=None,
	val_ratio: float = 0.1,
	warmup_epochs: int = 0,
	num_workers: int | tuple[int, int] = 10,
	weight_decay: float = 1e-4,
	smoothing: float = 0.1,
	use_amp: bool = True,
	use_scheduler: bool = True,
	clip_grad: bool = False,
	max_norm: float = 1.0,
	pin_memory: bool = True,
	early_stop: bool = False,
	early_stop_epoch: int = 5,
	save_path: Path | str | None = None,
	keep_count: int = 0,
	save_best: bool = True,
	cmp_obj: Literal['acc', 'prec', 'recall', 'f1'] = 'f1',
	show_matrix: bool = True,
	log_dir: Path | str | None = None,
):
	"""向后兼容的训练函数别名"""
	result = train_classifier(
		device=device,
		model=model,
		num_classes=num_classes,
		epochs=epochs,
		lr=lr,
		dataset=dataset,
		batch_size=batch_size,
		accum_iter=accum_iter,
		train_transform=train_transform,
		val_transform=val_transform,
		val_ratio=val_ratio,
		warmup_epochs=warmup_epochs if use_scheduler else 0,
		num_workers=num_workers,
		weight_decay=weight_decay,
		smoothing=smoothing,
		use_amp=use_amp,
		clip_grad=max_norm if clip_grad else None,
		pin_memory=pin_memory,
		early_stop=early_stop,
		early_stop_patience=early_stop_epoch,
		early_stop_metric=cmp_obj,
		save_path=save_path,
		keep_recent=keep_count,
		save_best=save_best,
		save_best_metric=cmp_obj,
		show_confusion_matrix=show_matrix,
		log_dir=log_dir,
		compute_metrics=True,
	)
	return result


# ============================================================================
# 快速评估函数
# ============================================================================


def fast_eval(device, model, dataset, transform, batch_size=1, num_workers=0):
	"""快速评估函数"""
	dataset.setTransform(transform)
	data_loader = MultiEpochsDataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	model.to(device)
	model.eval()
	y_pred = []
	y_true = []
	with torch.no_grad():
		for images, label in data_loader:
			images, label = (
				images.to(device, non_blocking=True),
				label.to(device, non_blocking=True),
			)

			outputs = model(images)
			preds = torch.argmax(outputs, dim=1)

			y_pred.extend(preds.cpu().numpy())
			y_true.extend(label.cpu().numpy())

	return y_true, y_pred


def fast_calc_metrics(y_true, y_pred, num_classes=0):
	"""快速计算评估指标"""
	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
	rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
	f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
	if num_classes == 0:
		matrix = None
	else:
		matrix = calc_confusion_matrix(y_true, y_pred, num_classes)

	return EvalMetrics(
		loss=0.0,
		acc=float(acc),
		precision=float(prec),
		recall=float(rec),
		f1=float(f1),
		confusion_matrix=matrix,
	)
