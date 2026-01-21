"""MAE Finetune 训练器

基于 MAE 预训练权重进行分类任务微调。

参考:
- https://github.com/facebookresearch/mae
- timm.data.mixup
"""

import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.data.loader import MultiEpochsDataLoader
from timm.loss.cross_entropy import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from daisy.dataset.index_dataset import IndexDataset
from daisy.model.mae.lr_decay import param_groups_lrd


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[float]:
	"""计算 top-k 准确率"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.item() * 100.0 / batch_size)
	return res


def mae_finetune(
	device: torch.device,
	model: nn.Module,
	train_dataset: IndexDataset,
	val_dataset: IndexDataset,
	num_classes: int,
	epochs: int,
	batch_size: int = 64,
	blr: float = 1e-3,
	layer_decay: float = 0.75,
	weight_decay: float = 0.05,
	warmup_epochs: int = 5,
	min_lr: float = 1e-6,
	# Mixup/CutMix
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
):
	"""
	MAE Finetune 训练

	Args:
		device: 训练设备
		model: ViT 模型
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
	"""
	if save_path is not None:
		if isinstance(save_path, str):
			save_path = Path(save_path)
		save_path.mkdir(parents=True, exist_ok=True)

	# 处理日志路径
	if log_dir is not None:
		if isinstance(log_dir, str):
			log_dir = Path(log_dir)
		log_dir.mkdir(parents=True, exist_ok=True)
		log_file = log_dir / f'mae_finetune_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		if not log_file.exists():
			with open(log_file, 'w', encoding='utf-8') as f:
				f.write('epoch,lr,train_loss,val_loss,val_acc1,val_acc5\n')
	else:
		log_file = None

	# 计算实际学习率
	eff_batch_size = batch_size * accum_iter
	lr = blr * eff_batch_size / 256
	print(f'Base LR: {blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

	# DataLoader
	print('Loading dataloader...')
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
		drop_last=False,
	)

	model.to(device)

	# Layer-wise LR decay 参数组
	param_groups = param_groups_lrd(
		model,
		weight_decay=weight_decay,
		layer_decay=layer_decay,
	)
	optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))

	# Mixup/CutMix
	mixup_fn = None
	if mixup > 0 or cutmix > 0:
		mixup_fn = Mixup(
			mixup_alpha=mixup,
			cutmix_alpha=cutmix,
			prob=1.0,
			switch_prob=0.5,
			mode='batch',
			label_smoothing=smoothing,
			num_classes=num_classes,
		)

	# 损失函数
	if mixup_fn is not None:
		criterion = SoftTargetCrossEntropy()
	elif smoothing > 0:
		criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)
	else:
		criterion = nn.CrossEntropyLoss()

	# 验证损失函数（不使用 mixup）
	val_criterion = nn.CrossEntropyLoss()

	# 学习率调度: warmup + cosine decay
	def lr_func(epoch: int, iter_in_epoch: float = 0.0) -> float:
		"""计算学习率"""
		current = epoch + iter_in_epoch
		if current < warmup_epochs:
			return lr * current / warmup_epochs
		else:
			# cosine decay to min_lr
			return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (current - warmup_epochs) / (epochs - warmup_epochs)))

	print('Ready to train...')
	scaler = torch.GradScaler(enabled=use_amp)

	best_acc1 = 0.0

	for epoch in range(epochs):
		# ============ Training ============
		model.train()
		train_loss = 0.0
		num_batches = len(train_loader)
		current_lr = 0.0
		for i, (images, targets) in enumerate(train_loader):
			# 调整学习率 (per iteration)
			iter_ratio = i / num_batches
			current_lr = lr_func(epoch, iter_ratio)
			for param_group in optimizer.param_groups:
				# 应用 layer-wise lr scale
				param_group['lr'] = current_lr * param_group.get('lr_scale', 1.0)

			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)

			# Mixup/CutMix
			if mixup_fn is not None:
				images, targets = mixup_fn(images, targets)

			with torch.autocast('cuda', enabled=use_amp):
				outputs = model(images)
				loss = criterion(outputs, targets)

			loss_value = loss.item()

			if not math.isfinite(loss_value):
				print(f'Loss is {loss_value}, stopping training')
				raise RuntimeError(f'Loss is {loss_value}')

			loss = loss / accum_iter
			scaler.scale(loss).backward()

			if (i + 1) % accum_iter == 0 or (i + 1) == num_batches:
				if clip_grad is not None:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()

			train_loss += loss_value

			# 打印进度
			if (i + 1) % 20 == 0 or (i + 1) == num_batches:
				print(f'Epoch [{epoch + 1}/{epochs}] [{i + 1}/{num_batches}] Loss: {loss_value:.4f} LR: {current_lr:.6f}')

		train_loss /= num_batches

		# ============ Validation ============
		model.eval()
		val_loss = 0.0
		val_acc1 = 0.0
		val_acc5 = 0.0
		val_batches = len(val_loader)

		with torch.no_grad():
			for images, targets in val_loader:
				images = images.to(device, non_blocking=True)
				targets = targets.to(device, non_blocking=True)

				with torch.autocast('cuda', enabled=use_amp):
					outputs = model(images)
					loss = val_criterion(outputs, targets)

				val_loss += loss.item()

				# 计算准确率
				acc1, acc5 = accuracy(outputs, targets, topk=(1, min(5, num_classes)))
				val_acc1 += acc1
				val_acc5 += acc5

		val_loss /= val_batches
		val_acc1 /= val_batches
		val_acc5 /= val_batches

		print(
			f'Epoch {epoch + 1}/{epochs}, '
			f'Train Loss: {train_loss:.4f}, '
			f'Val Loss: {val_loss:.4f}, '
			f'Val Acc@1: {val_acc1:.2f}%, '
			f'Val Acc@5: {val_acc5:.2f}%'
		)

		# 写入日志
		if log_file is not None:
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(f'{epoch + 1},{current_lr:.6f},{train_loss:.4f},{val_loss:.4f},{val_acc1:.2f},{val_acc5:.2f}\n')

		# 保存 checkpoint
		if save_path is not None:
			# 保存最佳模型
			if val_acc1 > best_acc1:
				best_acc1 = val_acc1
				torch.save(
					{
						'model': model.state_dict(),
						'epoch': epoch,
						'acc1': val_acc1,
						'acc5': val_acc5,
					},
					save_path / 'best_model.pth',
				)
				print(f'Saved best model with Acc@1: {val_acc1:.2f}%')

			# 定期保存
			if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
				torch.save(
					{
						'model': model.state_dict(),
						'optimizer': optimizer.state_dict(),
						'scaler': scaler.state_dict(),
						'epoch': epoch,
						'acc1': val_acc1,
					},
					save_path / f'checkpoint_{epoch + 1:04d}.pth',
				)
				print(f'Saved checkpoint at epoch {epoch + 1}')

			# 保存最新
			torch.save(
				{
					'model': model.state_dict(),
					'optimizer': optimizer.state_dict(),
					'scaler': scaler.state_dict(),
					'epoch': epoch,
					'acc1': val_acc1,
				},
				save_path / 'checkpoint_latest.pth',
			)

	print(f'MAE Finetune completed! Best Acc@1: {best_acc1:.2f}%')
