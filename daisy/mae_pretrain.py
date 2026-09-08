"""MAE 预训练器"""

import math
import time
from pathlib import Path

import torch
from ignite.engine import Engine, Events
from ignite.metrics import Average
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import Dataset
from daisy.model.mae import MaskedAutoencoderViT


def mae_pretrain(
	device: torch.device,
	model: MaskedAutoencoderViT,
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
		log_file = log_dir / f'mae_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
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
	num_batches = len(data_loader)
	if num_batches == 0:
		raise ValueError('empty DataLoader')

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

	def process_function(engine: Engine, batch: torch.Tensor) -> float:
		images, _ = batch

		# 数据搬运
		images = images.to(device, non_blocking=True)

		# 前向
		with torch.autocast('cuda', enabled=use_amp):
			loss, _, _ = model(images, mask_ratio=mask_ratio)

		loss_value = loss.item()

		loss = loss / accum_iter
		scaler.scale(loss).backward()

		i = (engine.state.iteration - 1) % num_batches
		if (i + 1) % accum_iter == 0 or (i + 1) == num_batches:
			if clip_grad:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
			scaler.step(optimizer)
			scaler.update()
			optimizer.zero_grad()

		return loss_value

	engine = Engine(process_function)

	@engine.on(Events.EPOCH_STARTED)
	def on_epoch_started(_):
		model.train()
		optimizer.zero_grad()

	@engine.on(Events.ITERATION_STARTED)
	def on_iteration_started(engine: Engine):
		epoch = engine.state.epoch
		i = (engine.state.iteration - 1) % num_batches
		lr_mult = lr_func(epoch - 1, i / num_batches)
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr * lr_mult

	Average().attach(engine, 'train_loss')

	@engine.on(Events.EPOCH_COMPLETED)
	def on_epoch_completed(engine: Engine):
		train_loss: float = engine.state.metrics['train_loss']
		epoch = engine.state.epoch
		print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}')

		if log_file is not None:
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(f'{epoch},{optimizer.param_groups[0]["lr"]:.6f},{train_loss:.4f}\n')

		if save_path is not None:
			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': scaler.state_dict(),
				'epoch': epoch - 1,
			}
			if epoch % save_freq == 0 or epoch == epochs:
				torch.save(checkpoint, save_path / f'checkpoint_{epoch:04d}.pth')
				print(f'Saved checkpoint at epoch {epoch}')
			torch.save(checkpoint, save_path / 'checkpoint_latest.pth')

	if start_epoch >= epochs:
		return

	if start_epoch > 0:
		engine.load_state_dict(
			{
				'epoch': start_epoch,
				'max_epochs': epochs,
				'epoch_length': num_batches,
			}
		)

	engine.run(data_loader, max_epochs=epochs)
