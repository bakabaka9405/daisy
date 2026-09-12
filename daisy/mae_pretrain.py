"""MAE 预训练器"""

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from daisy.dataset.index_dataset import IndexDataset
from daisy.method.mae import MAEPretrainMethod
from daisy.model.mae import MaskedAutoencoderViT
from daisy.training import CosineAnnealingLR, Timer, TrainState
from daisy.typing import Replaceable
from daisy.util import make_dataloader
from daisy.util.transform import LazyNormalizedTensor


@dataclass(frozen=True, slots=True, kw_only=True)
class MAEPretrainParams(Replaceable):
	"""MAE 预训练超参数。"""

	epochs: int
	batch_size: int = 64
	blr: float = 1.5e-4
	weight_decay: float = 0.05
	warmup_epochs: int = 40
	mask_ratio: float = 0.75
	norm_pix_loss: bool = False
	accum_iter: int = 1
	num_workers: int = 4
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	pin_memory: bool = True


def mae_pretrain[ImageT: torch.Tensor | LazyNormalizedTensor](
	device: torch.device,
	model: MaskedAutoencoderViT,
	dataset: IndexDataset[ImageT, None],
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

	eff_batch_size = params.batch_size * params.accum_iter
	lr = params.blr * eff_batch_size / 256
	print(f'Base LR: {params.blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

	print('Loading dataloader...')
	loader = make_dataloader(
		dataset,
		batch_size=params.batch_size,
		device=device,
		shuffle=True,
		drop_last=True,
		num_workers=params.num_workers,
		pin_memory=params.pin_memory,
	)

	model.to(device).compile(backend='inductor')

	method = MAEPretrainMethod(
		model,
		device=device,
		lr=lr,
		weight_decay=params.weight_decay,
		mask_ratio=params.mask_ratio,
		norm_pix_loss=params.norm_pix_loss,
		use_amp=params.use_amp,
		accum_iter=params.accum_iter,
		clip_grad=params.max_norm if params.clip_grad else None,
	)
	trainer = method.trainer
	optimizer = trainer.optimizer
	assert optimizer is not None

	timer = Timer()
	trainer.use(
		CosineAnnealingLR(lr=lr, min_lr=0, warmup_epochs=params.warmup_epochs, per_iteration=True),
		timer,
	)

	def on_epoch_end(state: TrainState) -> None:
		# Trainer 的 epoch 从零开始；对外展示与 checkpoint 文件名沿用从一开始的编号。
		epoch = state.epoch + 1
		train_loss = state.epoch_train_loss
		print(f'Epoch {epoch}/{params.epochs}, Train Loss: {train_loss:.4f}, Time: {timer.elapsed:.1f}s')

		if log_file is not None:
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(f'{epoch},{state.current_lr:.6f},{train_loss:.4f}\n')

		if save_path is not None:
			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': trainer.scaler.state_dict(),
				'epoch': state.epoch,
			}
			if epoch % save_freq == 0 or epoch == params.epochs:
				torch.save(checkpoint, save_path / f'checkpoint_{epoch:04d}.pth')
				print(f'Saved checkpoint at epoch {epoch}')
			torch.save(checkpoint, save_path / 'checkpoint_latest.pth')

	trainer.on_epoch_end(on_epoch_end)

	print('Ready to train...')

	start_epoch = 0
	if resume is not None:
		checkpoint = torch.load(resume, map_location='cpu')
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		trainer.scaler.load_state_dict(checkpoint['scaler'])
		start_epoch = checkpoint['epoch'] + 1
		print(f'Resumed from epoch {start_epoch}')

	if start_epoch >= params.epochs:
		return

	method.fit(loader, params.epochs, start_epoch=start_epoch)
