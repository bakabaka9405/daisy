"""MAE 预训练任务执行器"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import time
from typing import ClassVar

import icontract
import torch
from torchvision.transforms import v2 as transforms, InterpolationMode

from daisy.dataset.disk_dataset import DiskDataset
from daisy.method.mae import MAEPretrainMethod, mae_reconstruction_loss
from daisy.model.mae import MaskedAutoencoderViT, create_mae_model, load_timm_pretrained_encoder_weights
from daisy.training import CosineAnnealingLR, CSVLog, RichEpochPrint, Timer, Trainer, TrainState
from daisy.util import Prefetcher, make_dataloader, set_global_seed, to_2tuple
from daisy.util.transform import LazyNormalization, LazyNormalizedTensor, get_rectangle_val_transform
from ...feed import feed
from ...protocol import RunContext
from ...runtime import save_json
from .config import MAEPretrainConfig

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@icontract.require(lambda ratio_min, ratio_max: 0 < ratio_min <= ratio_max)
@icontract.require(lambda scale_min, scale_max: 0 < scale_min <= scale_max)
@icontract.require(lambda input_size: min(to_2tuple(input_size)) > 0)
def get_mae_transform(
	input_size: int | tuple[int, int] = 224,
	scale_min: float = 0.2,
	scale_max: float = 1.0,
	ratio_min: float = 0.75,
	ratio_max: float = 1.3333,
	hflip: bool = True,
) -> Callable[[torch.Tensor], LazyNormalizedTensor]:
	"""获取 MAE 预训练的 transform

	参考 MAE 原论文：RandomResizedCrop + HorizontalFlip + Normalize
	"""
	trans: list = [
		transforms.RandomResizedCrop(
			input_size,
			scale=(scale_min, scale_max),
			ratio=(ratio_min, ratio_max),
			interpolation=InterpolationMode.BICUBIC,
		),
	]

	if hflip:
		trans.append(transforms.RandomHorizontalFlip())

	trans.append(LazyNormalization(list(MEAN), list(STD)))
	return transforms.Compose(trans)


@dataclass
class MAEPretrainRun:
	"""已完成 MAE 预训练的分析视图：最新完整检查点与原始数据划分在首次访问时加载"""

	run_dir: Path
	config: MAEPretrainConfig
	device: torch.device

	@cached_property
	def model(self) -> MaskedAutoencoderViT:
		"""checkpoint_latest.pth 严格加载的 eager 评估模型"""
		model = create_mae_model(self.config.model.name, img_size=self.config.model.img_size)
		checkpoint = torch.load(self.run_dir / 'checkpoint_latest.pth', map_location='cpu', weights_only=False)
		model.load_state_dict(checkpoint['model'], strict=True)
		return model.to(self.device).eval()

	@cached_property
	def trainer(self) -> Trainer:
		return Trainer(model=self.model, device=self.device, use_amp=self.config.training.use_amp)

	@cached_property
	def datasets(self) -> dict[str, DiskDataset]:
		return feed(self.config.dataset, seed=self.config.training.seed)

	def loader(self, split: str = 'train', *, batch_size: int | None = None, num_workers: int | None = None) -> Prefetcher:
		"""按保存的划分顺序提供无标签、固定尺寸的归一化图像"""
		dataset = self.datasets[split]
		transform = get_rectangle_val_transform(input_size=to_2tuple(self.config.model.img_size), mean=MEAN, std=STD)
		return make_dataloader(
			DiskDataset(dataset.file_paths, None, transform, backend=dataset.backend),
			batch_size if batch_size is not None else self.config.training.batch_size, self.device,
			shuffle=False, drop_last=False,
			num_workers=num_workers if num_workers is not None else self.config.training.num_workers,
			pin_memory=self.device.type == 'cuda',
		)

	@torch.no_grad()
	@icontract.require(lambda images: images.ndim == 4)
	def loss(self, images: torch.Tensor) -> torch.Tensor:
		"""按运行配置计算 MAE 训练损失，使用调用方当前的掩码随机状态"""
		model = self.model
		pred, mask = model(images, mask_ratio=self.config.training.mask_ratio)
		return mae_reconstruction_loss(pred, model.patchify(images), mask, norm_pix_loss=self.config.training.norm_pix_loss)


class MAEPretrainTask:
	"""MAE 预训练任务"""

	config_type: ClassVar = MAEPretrainConfig

	def __init__(self, config: MAEPretrainConfig) -> None:
		self.config = config

	def run(self, ctx: RunContext) -> None:
		"""执行 MAE 预训练任务"""
		config = self.config
		device = ctx.device
		training_cfg = config.training
		output_path = ctx.output_dir
		if training_cfg.seed is not None:
			set_global_seed(training_cfg.seed)

		# 计算实际学习率与有效批大小
		eff_batch_size = training_cfg.batch_size * training_cfg.accum_iter
		lr = training_cfg.blr * eff_batch_size / 256.0
		print(f'Base LR: {training_cfg.blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

		# 加载自监督数据集
		print('\nLoading dataset...')
		datasets = feed(config.dataset, seed=training_cfg.seed)
		if 'train' not in datasets:
			raise ValueError("MAE pretrain requires 'train' dataset split")
		dataset = datasets['train']
		print(f'Total samples: {len(dataset)}')

		# 获取 transform 并应用至数据集
		transform_cfg = training_cfg.transform
		transform = get_mae_transform(
			input_size=config.model.img_size,
			scale_min=transform_cfg.scale_min,
			scale_max=transform_cfg.scale_max,
			ratio_min=transform_cfg.ratio_min,
			ratio_max=transform_cfg.ratio_max,
			hflip=transform_cfg.hflip,
		)
		save_json(output_path / 'runtime.json', {'resolved_lr': lr, 'train_transform': repr(transform)})
		transformed_dataset = DiskDataset(
			dataset.file_paths,
			None,
			transform,
			backend=dataset.backend,
		)

		# 构建 DataLoader
		print('Loading dataloader...')
		loader = make_dataloader(
			transformed_dataset,
			batch_size=training_cfg.batch_size,
			device=device,
			shuffle=True,
			drop_last=True,
			num_workers=training_cfg.num_workers,
			pin_memory=True,
		)

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_mae_model(
			model_cfg.name,
			img_size=model_cfg.img_size,
		)
		if training_cfg.resume is not None:
			if model_cfg.pretrained is not None:
				print('Skipping pretrained checkpoint because resume is configured')
			if model_cfg.timm_pretrained is not None:
				print('Skipping timm pretrained encoder because resume is configured')
		elif model_cfg.pretrained is not None:
			print(f'Loading pretrained checkpoint: {model_cfg.pretrained}')
			checkpoint_model = torch.load(model_cfg.pretrained, map_location='cpu', weights_only=True)['model']
			checkpoint_model['pos_embed'] = model.pos_embed
			checkpoint_model['decoder_pos_embed'] = model.decoder_pos_embed
			model.load_state_dict(checkpoint_model, strict=True)
		elif model_cfg.timm_pretrained is not None:
			print(f'Loading timm pretrained encoder: {model_cfg.timm_pretrained}')
			load_timm_pretrained_encoder_weights(model, model_cfg.timm_pretrained)

		print(f'Model: {model_cfg.name}')

		model = model.to(device)
		if training_cfg.compile_backend is not None:
			model.compile(backend=training_cfg.compile_backend)

		# 构建 MAE 方法并作为核心插件装配训练器
		method = MAEPretrainMethod(
			model,
			lr=lr,
			weight_decay=training_cfg.weight_decay,
			mask_ratio=training_cfg.mask_ratio,
			norm_pix_loss=training_cfg.norm_pix_loss,
			freeze_encoder_epochs=training_cfg.freeze_encoder_epochs,
		)
		trainer = Trainer(
			device=device,
			use_amp=training_cfg.use_amp,
			accum_iter=training_cfg.accum_iter,
			clip_grad=training_cfg.max_norm if training_cfg.clip_grad else None,
		)
		trainer.use(method)
		optimizer = trainer.optimizer
		assert optimizer is not None

		# 配置调度器与计时器插件
		trainer.use(
			CosineAnnealingLR(lr=lr, min_lr=0, warmup_epochs=training_cfg.warmup_epochs, per_iteration=True),
			Timer(),
		)

		# 准备日志输出与记录插件
		trainer.use(RichEpochPrint(watch=['optim/lr', 'train/loss', 'time/epoch']))

		if config.output.log:
			log_file = output_path / 'logs' / f'mae_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
			trainer.use(CSVLog(log_path=log_file, watch=['optim/lr', 'train/loss', 'time/epoch']))

		# 注册 epoch 结束回调：完整检查点保存
		save_freq = training_cfg.save_freq
		epochs = training_cfg.epochs

		def on_epoch_end(state: TrainState) -> None:
			epoch = state.epoch + 1

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': trainer.scaler.state_dict(),
				'epoch': state.epoch,
				'config': config.model_dump(mode='json'),
			}
			if (save_freq > 0 and epoch % save_freq == 0) or epoch == epochs:
				torch.save(checkpoint, output_path / f'checkpoint_{epoch:04d}.pth')
				print(f'Saved checkpoint at epoch {epoch}')
			torch.save(checkpoint, output_path / 'checkpoint_latest.pth')

		trainer.on_epoch_end(on_epoch_end)

		# 恢复训练状态
		start_epoch = 0
		if training_cfg.resume is not None:
			checkpoint = torch.load(training_cfg.resume, map_location='cpu')
			model.load_state_dict(checkpoint['model'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			trainer.scaler.load_state_dict(checkpoint['scaler'])
			start_epoch = checkpoint['epoch'] + 1
			print(f'Resumed from epoch {start_epoch}')

		if start_epoch >= epochs:
			return

		print('Ready to train...')
		print('\nStarting MAE pretraining...')
		trainer.fit(loader, epochs, start_epoch=start_epoch)

	@classmethod
	def open_run(cls, run_dir: Path, config: MAEPretrainConfig, device: torch.device) -> MAEPretrainRun:
		return MAEPretrainRun(run_dir, config, device)
