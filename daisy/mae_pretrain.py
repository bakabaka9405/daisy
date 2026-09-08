"""MAE 预训练器"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from ignite.engine import Engine, Events
from ignite.metrics import Average
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import Dataset

from daisy.model.mae import MaskedAutoencoderViT
from daisy.typing import Replaceable


@dataclass(frozen=True, slots=True, kw_only=True)
class MAEPretrainParams(Replaceable):
	"""MAE 预训练超参数。"""

	epochs: int
	batch_size: int = 64
	blr: float = 1.5e-4
	weight_decay: float = 0.05
	warmup_epochs: int = 40
	mask_ratio: float = 0.75
	accum_iter: int = 1
	num_workers: int = 4
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	pin_memory: bool = True


def mae_pretrain(
	device: torch.device,
	model: MaskedAutoencoderViT,
	dataset: Dataset,
	params: MAEPretrainParams,
	*,
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
		params: 预训练超参数
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
	eff_batch_size = params.batch_size * params.accum_iter
	lr = params.blr * eff_batch_size / 256
	print(f'Base LR: {params.blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

	# DataLoader
	print('Loading dataloader...')
	data_loader = MultiEpochsDataLoader(
		dataset,
		batch_size=params.batch_size,
		shuffle=True,
		num_workers=params.num_workers,
		pin_memory=params.pin_memory,
		drop_last=True,
	)
	num_batches = len(data_loader)
	if num_batches == 0:
		raise ValueError('empty DataLoader')

	model.to(device)
	model.compile(backend='inductor')

	# Optimizer
	# 使用 AdamW，参考 MAE 原论文
	param_groups = [
		{'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n]},
		{'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n], 'weight_decay': 0.0},
	]
	optimizer = torch.optim.AdamW(
		param_groups,
		lr=lr,
		weight_decay=params.weight_decay,
		betas=(0.9, 0.95),
		fused=True,
	)

	# 学习率调度: warmup + cosine decay
	def lr_func(epoch: int, iter_in_epoch: float = 0.0) -> float:
		"""计算学习率倍率"""
		current = epoch + iter_in_epoch
		if current < params.warmup_epochs:
			return current / params.warmup_epochs
		else:
			# cosine decay
			return 0.5 * (1.0 + math.cos(math.pi * (current - params.warmup_epochs) / (params.epochs - params.warmup_epochs)))

	print('Ready to train...')
	scaler = torch.GradScaler(enabled=params.use_amp)

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
		with torch.autocast('cuda', enabled=params.use_amp):
			loss, _, _ = model(images, mask_ratio=params.mask_ratio)

		loss_value = loss.detach()

		loss = loss / params.accum_iter
		scaler.scale(loss).backward()

		i = (engine.state.iteration - 1) % num_batches
		if (i + 1) % params.accum_iter == 0 or (i + 1) == num_batches:
			if params.clip_grad:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.max_norm)
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

	Average(device=device).attach(engine, 'train_loss')

	@engine.on(Events.EPOCH_COMPLETED)
	def on_epoch_completed(engine: Engine):
		train_loss: float = engine.state.metrics['train_loss']
		epoch = engine.state.epoch
		print(f'Epoch {epoch}/{params.epochs}, Train Loss: {train_loss:.4f}')

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
			if epoch % save_freq == 0 or epoch == params.epochs:
				torch.save(checkpoint, save_path / f'checkpoint_{epoch:04d}.pth')
				print(f'Saved checkpoint at epoch {epoch}')
			torch.save(checkpoint, save_path / 'checkpoint_latest.pth')

	if start_epoch >= params.epochs:
		return

	if start_epoch > 0:
		engine.load_state_dict(
			{
				'epoch': start_epoch,
				'max_epochs': params.epochs,
				'epoch_length': num_batches,
			}
		)

	engine.run(data_loader, max_epochs=params.epochs)
