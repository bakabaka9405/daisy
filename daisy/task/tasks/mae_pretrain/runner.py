"""MAE 预训练任务执行器"""

from datetime import datetime
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms, InterpolationMode

import daisy
from daisy.model.mae import create_mae_model
from daisy.dataset import UnlabeledDiskDataset, load_files_from_folder
from daisy.util.transform import ZeroOneNormalize
from ...base import TaskRunner
from ...registry import TaskRegistry
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
	def get_ui_field_overrides(cls) -> dict[str, dict]:
		return {
			'meta.title': {'label': '任务标题'},
			'meta.description': {'label': '描述', 'component': 'textarea'},
			'meta.created_at': {'hidden': True},
			'meta.commit': {'hidden': True},
			'dataset.root': {'label': '数据目录'},
			'model.name': {
				'label': '模型',
				'component': 'dropdown',
				'choices': ['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'],
				'allow_custom': True,
			},
			'training.epochs': {'label': '训练轮数'},
			'training.batch_size': {'label': 'Batch Size'},
			'training.blr': {'label': '基础学习率'},
			'training.warmup_epochs': {'label': 'Warmup 轮数'},
			'training.mask_ratio': {
				'label': 'Mask 比例',
				'component': 'slider',
				'min_value': 0.5,
				'max_value': 0.9,
				'step': 0.05,
			},
			'output.save_path': {'hidden': True},
		}

	def run(self, config: MAEPretrainConfig, device: torch.device) -> Path:  # type: ignore[override]
		"""执行 MAE 预训练任务"""
		print('=' * 60)
		print(f'Task: {config.meta.title or config.task_id}')
		print(f'Description: {config.meta.description}')
		print(f'Device: {device}')
		print('=' * 60)

		# 获取 git commit
		if config.meta.commit == 'auto':
			config.meta.commit = daisy.util.get_git_commit()
		print(f'Git commit: {config.meta.commit}')

		# 准备输出目录
		output_path = config.output.save_path.format(
			task_id=config.task_id,
			date=datetime.now().strftime('%Y%m%d'),
		)
		output_path = Path(output_path)
		output_path.mkdir(parents=True, exist_ok=True)
		print(f'Output path: {output_path}')

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset

		# 加载所有图像文件（支持多个文件夹，用分号分割）
		files = []
		roots = [r.strip() for r in dataset_cfg.root.split(';') if r.strip()]
		for root in roots:
			root_files = load_files_from_folder(
				Path(root),
				extensions=tuple(dataset_cfg.extensions),
			)
			print(f'Loaded {len(root_files)} samples from {root}')
			files.extend(root_files)
		print(f'Total samples: {len(files)}')

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
		training_cfg = config.training

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

		print('\n' + '=' * 60)
		print('Task completed!')
		print(f'Output saved to: {output_path}')
		print('=' * 60)

		return output_path
