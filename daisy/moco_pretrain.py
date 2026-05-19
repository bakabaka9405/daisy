"""MoCo 预训练器 (v2/v3 统一)

通过 engine 参数区分 v2 和 v3 的训练逻辑:
- v2: forward(im_q, im_k) -> (logits, labels), 外部 CrossEntropyLoss
- v3: forward(x1, x2, m) -> loss, momentum 从训练循环传入
"""

import math
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import Dataset


def moco_pretrain(
	device: torch.device,
	model: nn.Module,
	dataset: Dataset,
	engine: Literal['v2', 'v3'],
	epochs: int,
	batch_size: int = 256,
	optimizer_type: Literal['sgd', 'adamw', 'lars'] = 'sgd',
	lr: float = 0.03,
	momentum: float = 0.9,
	weight_decay: float = 1e-4,
	lr_schedule: Literal['step', 'cosine'] = 'cosine',
	lr_milestones: list[int] | None = None,
	warmup_epochs: int = 0,
	moco_m: float = 0.999,
	moco_m_cos: bool = False,
	use_amp: bool = True,
	num_workers: int = 4,
	pin_memory: bool = True,
	save_path: Path | str | None = None,
	save_freq: int = 20,
	log_dir: Path | str | None = None,
	resume: str | None = None,
):
	"""
	MoCo 预训练

	Args:
		device: 训练设备
		model: MoCoV2 或 MoCoV3 模型
		dataset: 训练数据集 (返回 [view1, view2] 或 (view1, dummy_label))
		engine: 'v2' 或 'v3'
		epochs: 训练轮数
		batch_size: 批大小
		optimizer_type: 优化器类型
		lr: 学习率
		momentum: SGD/LARS momentum
		weight_decay: 权重衰减
		lr_schedule: 学习率调度策略
		lr_milestones: step 调度的 milestones
		warmup_epochs: warmup 轮数
		moco_m: MoCo momentum (v2: 固定, v3: 可选 cosine schedule)
		moco_m_cos: 是否使用 cosine momentum schedule (仅 v3)
		use_amp: 是否使用混合精度
		num_workers: DataLoader workers 数量
		pin_memory: 是否 pin memory
		save_path: 模型保存路径
		save_freq: 保存频率（每 N 个 epoch）
		log_dir: 日志目录
		resume: 恢复训练的 checkpoint 路径
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
		log_file = log_dir / f'moco_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		if not log_file.exists():
			with open(log_file, 'w', encoding='utf-8') as f:
				f.write('epoch,lr,train_loss,moco_m\n')
	else:
		log_file = None

	# DataLoader
	print('Loading dataloader...')
	data_loader = MultiEpochsDataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=True,
	)

	model.to(device)
	num_batches = len(data_loader)

	# ========== 优化器 ==========
	if optimizer_type == 'lars':
		from daisy.model.moco.lars import LARS

		optimizer: torch.optim.Optimizer = LARS(
			model.parameters(),
			lr=lr,
			weight_decay=weight_decay,
			momentum=momentum,
		)
	elif optimizer_type == 'adamw':
		optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		optimizer = torch.optim.SGD(
			model.parameters(),
			lr=lr,
			momentum=momentum,
			weight_decay=weight_decay,
		)

	# ========== 学习率调度函数 ==========
	if lr_milestones is None:
		lr_milestones = [120, 160]

	def get_lr(epoch: int, step: int = 0) -> float:
		"""计算当前学习率"""
		if lr_schedule == 'cosine':
			if engine == 'v3':
				# v3: per-iteration cosine with warmup
				current = epoch + step / num_batches
			else:
				current = float(epoch)

			if current < warmup_epochs:
				return lr * current / max(warmup_epochs, 1e-8)
			else:
				progress = (current - warmup_epochs) / max(epochs - warmup_epochs, 1)
				return lr * 0.5 * (1.0 + math.cos(math.pi * progress))
		else:
			# step schedule
			factor = 1.0
			for milestone in lr_milestones:
				if epoch >= milestone:
					factor *= 0.1
			if epoch < warmup_epochs:
				return lr * epoch / max(warmup_epochs, 1e-8)
			return lr * factor

	# ========== Momentum schedule (v3) ==========
	def get_moco_momentum(epoch: int, step: int = 0) -> float:
		if not moco_m_cos:
			return moco_m
		# cosine schedule: m = 1 - (1 - m) * (1 + cos(pi * t / T)) / 2
		current = epoch + step / num_batches
		return 1 - (1 - moco_m) * (1 + math.cos(math.pi * current / epochs)) / 2

	# ========== Loss (v2) ==========
	criterion = nn.CrossEntropyLoss().to(device) if engine == 'v2' else None

	print(f'MoCo {engine} pretraining: {epochs} epochs, lr={lr}, batch_size={batch_size}')
	print(f'Optimizer: {optimizer_type}, LR schedule: {lr_schedule}, Warmup: {warmup_epochs}')
	scaler = torch.GradScaler(enabled=use_amp)

	start_epoch = 0

	# Resume from checkpoint
	if resume is not None:
		checkpoint = torch.load(resume, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		scaler.load_state_dict(checkpoint['scaler'])
		start_epoch = checkpoint['epoch'] + 1
		print(f'Resumed from epoch {start_epoch}')

	for epoch in range(start_epoch, epochs):
		model.train()
		train_loss = 0.0

		for i, batch in enumerate(data_loader):
			# 调整学习率
			current_lr = get_lr(epoch, i)
			for param_group in optimizer.param_groups:
				param_group['lr'] = current_lr

			# 解析 batch: TwoCropsTransform 返回 ([view1, view2], dummy_label)
			images, _ = batch
			# images 是一个 list: [view1_tensor, view2_tensor]
			im_1 = images[0].to(device, non_blocking=True)
			im_2 = images[1].to(device, non_blocking=True)

			with torch.autocast('cuda', enabled=use_amp):
				if engine == 'v2':
					assert criterion is not None
					logits, labels = model(im_1, im_2)
					loss = criterion(logits, labels)
				else:
					# v3: forward(x1, x2, m) -> loss
					m = get_moco_momentum(epoch, i)
					loss = model(im_1, im_2, m)

			loss_value = loss.item()

			if not math.isfinite(loss_value):
				print(f'Loss is {loss_value}, stopping training')
				raise RuntimeError(f'Loss is {loss_value}')

			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()

			train_loss += loss_value

			# 打印进度
			if (i + 1) % 20 == 0 or (i + 1) == num_batches:
				current_m = get_moco_momentum(epoch, i) if engine == 'v3' else moco_m
				print(f'Epoch [{epoch + 1}/{epochs}] [{i + 1}/{num_batches}] Loss: {loss_value:.4f} LR: {current_lr:.6f} M: {current_m:.4f}')

		train_loss /= num_batches
		current_m = get_moco_momentum(epoch) if engine == 'v3' else moco_m
		print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

		# 写入日志
		if log_file is not None:
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(f'{epoch + 1},{optimizer.param_groups[0]["lr"]:.6f},{train_loss:.4f},{current_m:.4f}\n')

		# 保存 checkpoint
		if save_path is not None:
			ckpt = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': scaler.state_dict(),
				'epoch': epoch,
				'engine': engine,
			}
			# 定期保存
			if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
				torch.save(ckpt, save_path / f'checkpoint_{epoch + 1:04d}.pth')
				print(f'Saved checkpoint at epoch {epoch + 1}')

			# 保存最新
			torch.save(ckpt, save_path / 'checkpoint_latest.pth')

	print('MoCo Pretraining completed!')
