"""MAE 预训练任务执行器"""

from collections.abc import Callable
from pathlib import Path
import random
import time
from typing import override

import torch
from torchvision.transforms import v2 as transforms, InterpolationMode

from daisy.dataset import DiskDataset, load_files_from_folder
from daisy.method.mae import MAEPretrainMethod
from daisy.model.mae import create_mae_model, load_timm_pretrained_encoder_weights
from daisy.training import CosineAnnealingLR, Timer, TrainState
from daisy.util import make_dataloader
from daisy.util.transform import LazyNormalization, LazyNormalizedTensor
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from .config import MAEPretrainConfig


def get_mae_transform(
	input_size: int = 224,
	scale_min: float = 0.2,
	scale_max: float = 1.0,
	hflip: bool = True,
) -> Callable[[torch.Tensor], LazyNormalizedTensor]:
	"""获取 MAE 预训练的 transform

	参考 MAE 原论文：RandomResizedCrop + HorizontalFlip + Normalize
	"""
	trans: list = [
		transforms.RandomResizedCrop(
			input_size,
			scale=(scale_min, scale_max),
			interpolation=InterpolationMode.BICUBIC,
		),
	]

	if hflip:
		trans.append(transforms.RandomHorizontalFlip())

	trans.append(LazyNormalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
	return transforms.Compose(trans)


@TaskRegistry.register
class MAEPretrainRunner(TaskRunner[MAEPretrainConfig]):
	"""MAE 预训练任务执行器"""

	@classmethod
	@override
	def get_config_class(cls) -> type[MAEPretrainConfig]:
		return MAEPretrainConfig

	@override
	def run(self, config: MAEPretrainConfig, device: torch.device) -> Path:
		"""执行 MAE 预训练任务"""
		training_cfg = config.training
		run_context = prepare_task_run(config, device, seed=training_cfg.seed)
		output_path = run_context.output_path

		# 计算实际学习率与有效批大小
		eff_batch_size = training_cfg.batch_size * training_cfg.accum_iter
		lr = training_cfg.blr * eff_batch_size / 256.0
		print(f'Base LR: {training_cfg.blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {lr:.2e}')

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset

		# 加载所有图像文件（支持多个文件夹，用分号分割）
		files: list[Path] = []
		roots = [r.strip() for r in dataset_cfg.root.split(';') if r.strip()]
		for root in roots:
			root_files = load_files_from_folder(
				Path(root),
				extensions=tuple(dataset_cfg.extensions),
			)
			print(f'Loaded {len(root_files)} samples from {root}')
			files.extend(root_files)

		if dataset_cfg.sample_ratio < 1.0 and files:
			sample_seed = dataset_cfg.sample_seed if dataset_cfg.sample_seed is not None else training_cfg.seed
			k = max(1, round(len(files) * dataset_cfg.sample_ratio))
			files = random.Random(sample_seed).sample(files, k)
		print(f'Total samples: {len(files)}')

		# 保存统一的运行快照信息
		save_run_snapshot(
			output_path,
			config,
			run_context,
			extra={'resolved_lr': lr},
		)

		# 获取 transform 并构建数据集（无标签，LabelT=None）
		transform_cfg = training_cfg.transform
		transform = get_mae_transform(
			input_size=transform_cfg.input_size,
			scale_min=transform_cfg.scale_min,
			scale_max=transform_cfg.scale_max,
			hflip=transform_cfg.hflip,
		)
		dataset: DiskDataset[LazyNormalizedTensor, None] = DiskDataset(files, None, transform)

		# 构建 DataLoader
		print('Loading dataloader...')
		loader = make_dataloader(
			dataset,
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
		if model_cfg.timm_pretrained and training_cfg.resume is None:
			print(f'Loading timm pretrained encoder: {model_cfg.timm_pretrained}')
			load_timm_pretrained_encoder_weights(model, model_cfg.timm_pretrained)
		elif model_cfg.timm_pretrained:
			print('Skipping timm pretrained encoder because resume is configured')

		print(f'Model: {model_cfg.name}')

		model.to(device).compile(backend='inductor')

		# 构建 MAE 方法实例
		method = MAEPretrainMethod(
			model,
			device=device,
			lr=lr,
			weight_decay=training_cfg.weight_decay,
			mask_ratio=training_cfg.mask_ratio,
			norm_pix_loss=training_cfg.norm_pix_loss,
			use_amp=training_cfg.use_amp,
			accum_iter=training_cfg.accum_iter,
			clip_grad=training_cfg.max_norm if training_cfg.clip_grad else None,
		)
		trainer = method.trainer
		optimizer = trainer.optimizer
		assert optimizer is not None

		# 配置调度器与计时器插件
		timer = Timer()
		trainer.use(
			CosineAnnealingLR(lr=lr, min_lr=0, warmup_epochs=training_cfg.warmup_epochs, per_iteration=True),
			timer,
		)

		# 准备日志文件
		if config.output.log:
			log_dir = output_path / 'logs'
			log_dir.mkdir(parents=True, exist_ok=True)
			log_file = log_dir / f'mae_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
			if not log_file.exists():
				with open(log_file, 'w', encoding='utf-8') as f:
					f.write('epoch,lr,train_loss\n')
		else:
			log_file = None

		# 注册 epoch 结束回调：指标输出、CSV 记录与完整检查点保存
		save_freq = training_cfg.save_freq
		epochs = training_cfg.epochs

		def on_epoch_end(state: TrainState) -> None:
			epoch = state.epoch + 1
			train_loss = state.epoch_train_loss
			print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Time: {timer.elapsed:.1f}s')

			if log_file is not None:
				with open(log_file, 'a', encoding='utf-8') as f:
					f.write(f'{epoch},{state.current_lr:.6f},{train_loss:.4f}\n')

			checkpoint = {
				'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
				'scaler': trainer.scaler.state_dict(),
				'epoch': state.epoch,
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
			print_task_completed(output_path)
			return output_path

		print('Ready to train...')
		print('\nStarting MAE pretraining...')
		method.fit(loader, epochs, start_epoch=start_epoch)

		print_task_completed(output_path)
		return output_path
