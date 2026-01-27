"""MAE Finetune 任务执行器"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from torchvision.transforms import v2 as transforms, InterpolationMode
from timm.data.transforms_factory import create_transform

import daisy
from daisy.model.mae import create_vit_model, load_mae_pretrained_weights
from daisy.util.transform import ZeroOneNormalize
from ...base import TaskRunner
from ...registry import TaskRegistry
from .config import MAEFinetuneConfig

if TYPE_CHECKING:
	import torch


def get_finetune_train_transform(
	input_size: int = 224,
	aa: str = 'rand-m9-mstd0.5-inc1',
	reprob: float = 0.25,
	remode: str = 'pixel',
	recount: int = 1,
):
	"""获取 MAE Finetune 训练 transform

	使用 timm.data.create_transform 创建，包含:
	- RandomResizedCrop
	- RandAugment
	- Random Erasing
	- ImageNet Normalize
	"""
	transform = create_transform(
		input_size=input_size,
		is_training=True,
		color_jitter=0.0,  # 使用 RandAugment 代替
		auto_augment=aa,
		interpolation='bicubic',
		re_prob=reprob,
		re_mode=remode,
		re_count=recount,
		mean=(0.485, 0.456, 0.406),
		std=(0.229, 0.224, 0.225),
	)
	return transform


def get_finetune_val_transform(input_size: int = 224):
	"""获取 MAE Finetune 验证 transform

	- Resize (input_size / 0.875)
	- CenterCrop (input_size)
	- ImageNet Normalize
	"""
	resize_size = int(input_size / 0.875)
	return transforms.Compose(
		[
			transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
			transforms.CenterCrop(input_size),
			ZeroOneNormalize(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)


@TaskRegistry.register
class MAEFinetuneRunner(TaskRunner['MAEFinetuneConfig']):
	"""MAE Finetune 任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'mae_finetune'

	@classmethod
	def get_config_class(cls) -> type[MAEFinetuneConfig]:
		return MAEFinetuneConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'MAE 微调'

	@classmethod
	def get_ui_field_overrides(cls) -> dict[str, dict]:
		return {
			'meta.title': {'label': '任务标题'},
			'meta.description': {'label': '描述', 'component': 'textarea'},
			'meta.created_at': {'hidden': True},
			'meta.commit': {'hidden': True},
			'dataset.root': {'label': '数据根目录'},
			'dataset.sheet': {'label': '标注文件'},
			'model.name': {
				'label': '模型',
				'component': 'dropdown',
				'choices': ['vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14'],
				'allow_custom': True,
			},
			'model.num_classes': {'label': '类别数'},
			'model.checkpoint': {'label': 'MAE 预训练权重'},
			'training.epochs': {'label': '训练轮数'},
			'training.batch_size': {'label': 'Batch Size'},
			'training.blr': {'label': '基础学习率'},
			'training.layer_decay': {
				'label': 'Layer Decay',
				'component': 'slider',
				'min_value': 0.5,
				'max_value': 0.9,
				'step': 0.05,
			},
			'training.warmup_epochs': {'label': 'Warmup 轮数'},
			'output.save_path': {'hidden': True},
		}

	def run(self, config: MAEFinetuneConfig, device: 'torch.device') -> Path:
		"""执行 MAE Finetune 任务"""
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

		if dataset_cfg.type == 'sheet':
			feeder = daisy.feeder.load_feeder_from_sheet(
				dataset_root=Path(dataset_cfg.root),
				sheet=Path(dataset_cfg.sheet),  # type: ignore
				sheet_name=dataset_cfg.sheet_name,
				column=dataset_cfg.column,
				label_offset=dataset_cfg.label_offset,
				have_header=dataset_cfg.have_header,
			)
		elif dataset_cfg.type == 'folder':
			feeder = daisy.feeder.load_feeder_from_folder(Path(dataset_cfg.root))
		else:
			raise ValueError(f'Unknown dataset type: {dataset_cfg.type}')

		files, labels = feeder.fetch()
		print(f'Total samples: {len(files)}')

		# 创建数据集
		dataset = daisy.dataset.DiskDataset(files, labels)

		# 数据划分
		split_cfg = dataset_cfg.split
		if split_cfg.method == 'ratio':
			train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(
				dataset, val_ratio=split_cfg.val_ratio
			)
		elif split_cfg.method == 'sheet':
			val_feeder = daisy.feeder.load_feeder_from_sheet(
				dataset_root=Path(dataset_cfg.root),
				sheet=Path(split_cfg.val_sheet),  # type: ignore
				sheet_name=split_cfg.val_sheet_name,
				column=dataset_cfg.column,
				label_offset=dataset_cfg.label_offset,
				have_header=dataset_cfg.have_header,
			)
			val_files, val_labels = val_feeder.fetch()
			train_dataset = dataset
			val_dataset = daisy.dataset.DiskDataset(val_files, val_labels)
		elif split_cfg.method == 'preset':
			# 使用预先划分的 train/val 目录
			train_feeder = daisy.feeder.load_feeder_from_folder(
				Path(dataset_cfg.root) / 'train'
			)
			val_feeder = daisy.feeder.load_feeder_from_folder(Path(dataset_cfg.root) / 'val')
			train_files, train_labels = train_feeder.fetch()
			val_files, val_labels = val_feeder.fetch()
			train_dataset = daisy.dataset.DiskDataset(train_files, train_labels)
			val_dataset = daisy.dataset.DiskDataset(val_files, val_labels)
		else:
			raise ValueError(f'Unknown split method: {split_cfg.method}')

		print(f'Train samples: {len(train_dataset)}')
		print(f'Val samples: {len(val_dataset)}')

		# 获取 transforms
		aug_cfg = config.training.augment
		train_transform = get_finetune_train_transform(
			input_size=aug_cfg.input_size,
			aa=aug_cfg.aa,
			reprob=aug_cfg.reprob,
			remode=aug_cfg.remode,
			recount=aug_cfg.recount,
		)
		val_transform = get_finetune_val_transform(input_size=aug_cfg.input_size)

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_vit_model(
			model_cfg.name,
			num_classes=model_cfg.num_classes,
			global_pool=model_cfg.global_pool,
			drop_path_rate=model_cfg.drop_path,
			img_size=model_cfg.img_size,
		)

		# 加载 MAE 预训练权重
		if model_cfg.checkpoint:
			print(f'Loading MAE checkpoint: {model_cfg.checkpoint}')
			load_mae_pretrained_weights(model, model_cfg.checkpoint)

		print(f'Model: {model_cfg.name}')

		# 训练
		print('\nStarting MAE finetuning...')
		training_cfg = config.training

		daisy.mae_finetune.mae_finetune(
			device=device,
			model=model,
			train_dataset=train_dataset,
			val_dataset=val_dataset,
			num_classes=model_cfg.num_classes,
			epochs=training_cfg.epochs,
			batch_size=training_cfg.batch_size,
			blr=training_cfg.blr,
			layer_decay=training_cfg.layer_decay,
			weight_decay=training_cfg.weight_decay,
			warmup_epochs=training_cfg.warmup_epochs,
			min_lr=training_cfg.min_lr,
			mixup=aug_cfg.mixup,
			cutmix=aug_cfg.cutmix,
			smoothing=aug_cfg.smoothing,
			accum_iter=training_cfg.accum_iter,
			use_amp=training_cfg.use_amp,
			clip_grad=training_cfg.clip_grad,
			num_workers=training_cfg.num_workers,
			save_path=output_path,
			save_freq=training_cfg.save_freq,
			log_dir=output_path / 'logs' if config.output.log else None,
			train_transform=train_transform,
			val_transform=val_transform,
		)

		print('\n' + '=' * 60)
		print('Task completed!')
		print(f'Output saved to: {output_path}')
		print('=' * 60)

		return output_path
