"""MAE 预训练器"""

import math
import time
from pathlib import Path

import torch
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import Dataset


def mae_pretrain(
	device: torch.device,
	model: torch.nn.Module,
	dataset: Dataset,
	epochs: int,
	batch_size: int = 64,
	blr: float = 1.5e-4,
	weight_decay: float = 0.05,
	warmup_epochs: int = 40,
	mask_ratio: float = 0.75,
	accum_iter: int = 1,
	num_workers: int = 4,
	use_amp: bool = True,
	clip_grad: bool = False,
	max_norm: float = 1.0,
	pin_memory: bool = True,
	save_path: Path | str | None = None,
	save_freq: int = 20,
	log_dir: Path | str | None = None,
	resume: str | None = None,
):
	"""
	MAE 预训练

	Args:
		device: 训练设备
		model: MAE 模型
		dataset: 训练数据集
		epochs: 训练轮数
		batch_size: 批大小
		blr: 基础学习率 (实际 lr = blr * batch_size / 256)
		weight_decay: 权重衰减
		warmup_epochs: warmup 轮数
		mask_ratio: masking 比例
		accum_iter: 梯度累积迭代数
		num_workers: DataLoader workers 数量
		use_amp: 是否使用混合精度
		clip_grad: 是否裁剪梯度
		max_norm: 梯度裁剪的最大范数
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
		log_file = log_dir / f"mae_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
		if not log_file.exists():
			with open(log_file, 'w', encoding='utf-8') as f:
				f.write('epoch,lr,train_loss\n')
	else:
		log_file = None

	# 计算实际学习率: lr = blr * batch_size / 256
	eff_batch_size = batch_size * accum_iter
	lr = blr * eff_batch_size / 256
	print(f'Base LR: {blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

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

	# Optimizer
	# 使用 AdamW，参考 MAE 原论文
	param_groups = [
		{'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n]},
		{'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n], 'weight_decay': 0.0},
	]
	optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))

	# 学习率调度: warmup + cosine decay
	def lr_func(epoch: int, iter_in_epoch: float = 0.0) -> float:
		"""计算学习率倍率"""
		current = epoch + iter_in_epoch
		if current < warmup_epochs:
			return current / warmup_epochs
		else:
			# cosine decay
			return 0.5 * (1.0 + math.cos(math.pi * (current - warmup_epochs) / (epochs - warmup_epochs)))

	print('Ready to train...')
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
		num_batches = len(data_loader)

		for i, (images, _) in enumerate(data_loader):
			# 调整学习率 (per iteration)
			iter_ratio = i / num_batches
			lr_mult = lr_func(epoch, iter_ratio)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr * lr_mult

			images = images.to(device, non_blocking=True)

			with torch.autocast('cuda', enabled=use_amp):
				loss, _, _ = model(images, mask_ratio=mask_ratio)

			loss_value = loss.item()

			if not math.isfinite(loss_value):
				print(f'Loss is {loss_value}, stopping training')
				raise RuntimeError(f'Loss is {loss_value}')

			loss = loss / accum_iter
			scaler.scale(loss).backward()

			if (i + 1) % accum_iter == 0 or (i + 1) == num_batches:
				if clip_grad:
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad()

			train_loss += loss_value

			# 打印进度
			if (i + 1) % 20 == 0 or (i + 1) == num_batches:
				current_lr = optimizer.param_groups[0]['lr']
				print(
					f'Epoch [{epoch + 1}/{epochs}] [{i + 1}/{num_batches}] '
					f'Loss: {loss_value:.4f} LR: {current_lr:.6f}'
				)

		train_loss /= num_batches
		print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

		# 写入日志
		if log_file is not None:
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(f'{epoch + 1},{optimizer.param_groups[0]["lr"]:.6f},{train_loss:.4f}\n')

		# 保存 checkpoint
		if save_path is not None:
			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': scaler.state_dict(),
				'epoch': epoch,
			}
			# 定期保存
			if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
				torch.save(checkpoint, save_path / f'checkpoint_{epoch + 1:04d}.pth')
				print(f'Saved checkpoint at epoch {epoch + 1}')

			# 保存最新
			torch.save(checkpoint, save_path / 'checkpoint_latest.pth')

	print('MAE Pretraining completed!')
