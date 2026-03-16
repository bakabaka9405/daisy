"""MAE 预训练任务执行器"""

from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms, InterpolationMode

import daisy
from daisy.model.mae import create_mae_model
from daisy.dataset import UnlabeledDiskDataset, load_files_from_folder
from daisy.util.transform import ZeroOneNormalize
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...ui_config import UIFieldConfig
from .config import MAEPretrainConfig


def get_mae_transform(
	input_size: int = 224,
	scale_min: float = 0.2,
	scale_max: float = 1.0,
	hflip: bool = True,
):
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

	trans.extend(
		[
			ZeroOneNormalize(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)

	return transforms.Compose(trans)


def sample_files_by_ratio(
	files: list[Path],
	*,
	ratio: float,
	seed: int | None = None,
) -> list[Path]:
	"""按比例采样文件列表"""
	if not files:
		return files
	if ratio <= 0 or ratio > 1:
		raise ValueError(f'sample_ratio must be in (0, 1], got {ratio}')
	if ratio >= 1:
		return files

	rng = torch.Generator()
	if seed is not None:
		rng.manual_seed(seed)
	indices = torch.randperm(len(files), generator=rng).tolist()
	sample_size = max(1, int(round(len(files) * ratio)))
	selected_indices = sorted(indices[:sample_size])
	return [files[index] for index in selected_indices]


@TaskRegistry.register
class MAEPretrainRunner(TaskRunner):
	"""MAE 预训练任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'mae_pretrain'

	@classmethod
	def get_config_class(cls) -> type[MAEPretrainConfig]:
		return MAEPretrainConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'MAE 预训练'

	@classmethod
	def get_ui_field_overrides(cls) -> dict[str, UIFieldConfig]:
		return {
			'meta.title': UIFieldConfig(label='任务标题'),
			'meta.description': UIFieldConfig(label='描述', component='textarea'),
			'meta.created_at': UIFieldConfig(hidden=True),
			'meta.commit': UIFieldConfig(hidden=True),
			'dataset.root': UIFieldConfig(label='数据目录'),
			'model.name': UIFieldConfig(
				label='模型',
				component='dropdown',
				choices=('mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'),
				allow_custom=True,
			),
			'training.epochs': UIFieldConfig(label='训练轮数'),
			'training.batch_size': UIFieldConfig(label='Batch Size'),
			'training.blr': UIFieldConfig(label='基础学习率'),
			'training.warmup_epochs': UIFieldConfig(label='Warmup 轮数'),
			'training.mask_ratio': UIFieldConfig(label='Mask 比例', component='slider', min_value=0.5, max_value=0.9, step=0.05),
			'output.save_path': UIFieldConfig(hidden=True),
		}

	def run(self, config: MAEPretrainConfig, device: torch.device) -> Path:  # type: ignore[override]
		"""执行 MAE 预训练任务"""
		training_cfg = config.training
		run_context = prepare_task_run(config, device, seed=training_cfg.seed)
		output_path = run_context.output_path

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset

		# 加载所有图像文件（支持多个文件夹，用分号分割）
		files = []
		roots = [r.strip() for r in dataset_cfg.root.split(';') if r.strip()]
		if not roots:
			raise ValueError('dataset.root is required for mae_pretrain')
		for root in roots:
			root_files = load_files_from_folder(
				Path(root),
				extensions=tuple(dataset_cfg.extensions),
			)
			print(f'Loaded {len(root_files)} samples from {root}')
			files.extend(root_files)

		sample_seed = dataset_cfg.sample_seed if dataset_cfg.sample_seed is not None else training_cfg.seed
		files = sample_files_by_ratio(files, ratio=dataset_cfg.sample_ratio, seed=sample_seed)
		print(f'Total samples: {len(files)}')
		save_run_snapshot(
			output_path,
			config,
			run_context,
		)

		# 获取 transform
		transform_cfg = config.training.transform
		transform = get_mae_transform(
			input_size=transform_cfg.input_size,
			scale_min=transform_cfg.scale_min,
			scale_max=transform_cfg.scale_max,
			hflip=transform_cfg.hflip,
		)

		# 创建数据集
		dataset = UnlabeledDiskDataset(files, transform=transform)

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_mae_model(
			model_cfg.name,
			img_size=model_cfg.img_size,
			norm_pix_loss=model_cfg.norm_pix_loss,
		)

		print(f'Model: {model_cfg.name}')

		# 训练
		print('\nStarting MAE pretraining...')

		daisy.mae_pretrain.mae_pretrain(
			device=device,
			model=model,
			dataset=dataset,
			epochs=training_cfg.epochs,
			batch_size=training_cfg.batch_size,
			blr=training_cfg.blr,
			weight_decay=training_cfg.weight_decay,
			warmup_epochs=training_cfg.warmup_epochs,
			mask_ratio=training_cfg.mask_ratio,
			accum_iter=training_cfg.accum_iter,
			num_workers=training_cfg.num_workers,
			use_amp=training_cfg.use_amp,
			clip_grad=training_cfg.clip_grad,
			max_norm=training_cfg.max_norm,
			save_path=output_path,
			save_freq=training_cfg.save_freq,
			log_dir=output_path / 'logs' if config.output.log else None,
			resume=training_cfg.resume,
		)

		print_task_completed(output_path)

		return output_path
