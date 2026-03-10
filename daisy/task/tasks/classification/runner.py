"""分类任务执行器"""

from pathlib import Path
from typing import TYPE_CHECKING

from timm import create_model

import daisy
from ...base import TaskRunner
from ...data import build_train_val_selection
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ...shared import get_classification_transform
from ...ui_config import UIFieldConfig
from .config import ClassificationConfig

if TYPE_CHECKING:
	import torch


@TaskRegistry.register
class ClassificationRunner(TaskRunner['ClassificationConfig']):
	"""分类任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'classification'

	@classmethod
	def get_config_class(cls) -> type[ClassificationConfig]:
		return ClassificationConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return '图像分类'

	@classmethod
	def get_ui_field_overrides(cls) -> dict[str, UIFieldConfig]:
		return {
			'meta.title': UIFieldConfig(label='任务标题'),
			'meta.description': UIFieldConfig(label='描述', component='textarea'),
			'meta.creator': UIFieldConfig(label='创建者'),
			'meta.created_at': UIFieldConfig(hidden=True),
			'meta.commit': UIFieldConfig(hidden=True),
			'dataset.type': UIFieldConfig(label='数据集类型'),
			'dataset.root': UIFieldConfig(label='数据根目录'),
			'dataset.sheet': UIFieldConfig(label='标注文件'),
			'dataset.column': UIFieldConfig(label='标签列'),
			'dataset.split.val_ratio': UIFieldConfig(label='验证集比例', component='slider', min_value=0.05, max_value=0.3, step=0.01),
			'model.name': UIFieldConfig(
				label='模型',
				component='dropdown',
				choices=('resnet34', 'resnet50', 'resnet101', 'efficientnet_b0', 'convnext_tiny'),
				allow_custom=True,
			),
			'model.num_classes': UIFieldConfig(label='类别数'),
			'model.pretrained': UIFieldConfig(label='使用预训练权重'),
			'training.epochs': UIFieldConfig(label='训练轮数'),
			'training.batch_size': UIFieldConfig(label='Batch Size'),
			'training.lr': UIFieldConfig(label='学习率'),
			'training.warmup_epochs': UIFieldConfig(label='Warmup 轮数'),
			'training.weight_decay': UIFieldConfig(label='Weight Decay'),
			'training.cmp_obj': UIFieldConfig(label='优化目标'),
			'training.transform.train': UIFieldConfig(
				label='训练 Transform',
				component='dropdown',
				choices=('rectangle_train', 'rectangle_train_slight', 'stretch_train'),
			),
			'training.transform.val': UIFieldConfig(
				label='验证 Transform',
				component='dropdown',
				choices=('rectangle_val', 'stretch_val'),
			),
			'output.save_path': UIFieldConfig(hidden=True),
			'output.keep_count': UIFieldConfig(label='保留检查点数'),
			'output.save_best': UIFieldConfig(label='保存最佳模型'),
			'output.log': UIFieldConfig(label='记录日志'),
		}

	def run(self, config: ClassificationConfig, device: 'torch.device') -> Path:
		"""执行分类训练任务"""
		import torch

		training_cfg = config.training
		run_context = prepare_task_run(config, device, seed=training_cfg.seed)
		output_path = run_context.output_path

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset
		split_selection = build_train_val_selection(
			dataset_cfg,
			default_seed=training_cfg.seed,
			context='classification splits',
		)
		print(f'Total samples: {split_selection.source_count}')
		train_dataset, val_dataset = split_selection.to_datasets()
		save_json(output_path / 'split_protocol.json', split_selection.protocol.to_dict())
		save_run_snapshot(
			output_path,
			config,
			run_context,
		)

		print(f'Train samples: {len(train_dataset)}')
		print(f'Val samples: {len(val_dataset)}')

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_model(
			model_cfg.name,
			pretrained=model_cfg.pretrained,
			num_classes=model_cfg.num_classes,
		)

		# 加载预训练权重
		if model_cfg.checkpoint:
			print(f'Loading checkpoint: {model_cfg.checkpoint}')
			model.load_state_dict(torch.load(model_cfg.checkpoint, map_location='cpu'))

		print(f'Model: {model_cfg.name}')

		# 获取 transforms
		train_transform = get_classification_transform(config.training.transform.train)
		val_transform = get_classification_transform(config.training.transform.val)

		# 训练
		print('\nStarting training...')

		daisy.classfier_trainer.fast_train_smile(
			device=device,
			model=model,
			dataset=(train_dataset, val_dataset),
			num_classes=model_cfg.num_classes,
			epochs=training_cfg.epochs,
			batch_size=training_cfg.batch_size,
			lr=training_cfg.lr,
			weight_decay=training_cfg.weight_decay,
			warmup_epochs=training_cfg.warmup_epochs,
			smoothing=training_cfg.smoothing,
			accum_iter=training_cfg.accum_iter,
			use_scheduler=training_cfg.use_scheduler,
			use_amp=training_cfg.use_amp,
			clip_grad=training_cfg.clip_grad,
			max_norm=training_cfg.max_norm,
			num_workers=training_cfg.num_workers,
			early_stop=training_cfg.early_stop,
			early_stop_epoch=training_cfg.early_stop_epoch,
			cmp_obj=training_cfg.cmp_obj,
			train_transform=train_transform,
			val_transform=val_transform,
			save_path=output_path if config.output.save_best else None,
			keep_count=config.output.keep_count,
			save_best=config.output.save_best,
			log_dir=output_path / 'logs' if config.output.log else None,
		)

		print_task_completed(output_path)

		return output_path
