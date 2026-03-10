"""MAE Finetune 任务执行器"""

from pathlib import Path
from typing import TYPE_CHECKING

import daisy
from daisy.model.mae import create_vit_model, load_mae_pretrained_weights
from ...base import TaskRunner
from ...data import build_train_val_selection
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ...shared import get_mae_finetune_train_transform, get_mae_finetune_val_transform
from ...ui_config import UIFieldConfig
from .config import MAEFinetuneConfig

if TYPE_CHECKING:
	import torch


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
	def get_ui_field_overrides(cls) -> dict[str, UIFieldConfig]:
		return {
			'meta.title': UIFieldConfig(label='任务标题'),
			'meta.description': UIFieldConfig(label='描述', component='textarea'),
			'meta.created_at': UIFieldConfig(hidden=True),
			'meta.commit': UIFieldConfig(hidden=True),
			'dataset.root': UIFieldConfig(label='数据根目录'),
			'dataset.sheet': UIFieldConfig(label='标注文件'),
			'model.name': UIFieldConfig(
				label='模型',
				component='dropdown',
				choices=('vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14'),
				allow_custom=True,
			),
			'model.num_classes': UIFieldConfig(label='类别数'),
			'model.checkpoint': UIFieldConfig(label='MAE 预训练权重'),
			'training.epochs': UIFieldConfig(label='训练轮数'),
			'training.batch_size': UIFieldConfig(label='Batch Size'),
			'training.blr': UIFieldConfig(label='基础学习率'),
			'training.layer_decay': UIFieldConfig(label='Layer Decay', component='slider', min_value=0.5, max_value=0.9, step=0.05),
			'training.warmup_epochs': UIFieldConfig(label='Warmup 轮数'),
			'output.save_path': UIFieldConfig(hidden=True),
		}

	def run(self, config: MAEFinetuneConfig, device: 'torch.device') -> Path:
		"""执行 MAE Finetune 任务"""
		training_cfg = config.training
		run_context = prepare_task_run(config, device, seed=training_cfg.seed)
		output_path = run_context.output_path

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset
		split_selection = build_train_val_selection(
			dataset_cfg,
			default_seed=training_cfg.seed,
			context='mae finetune splits',
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

		# 获取 transforms
		aug_cfg = config.training.augment
		train_transform = get_mae_finetune_train_transform(
			input_size=aug_cfg.input_size,
			aa=aug_cfg.aa,
			reprob=aug_cfg.reprob,
			remode=aug_cfg.remode,
			recount=aug_cfg.recount,
		)
		val_transform = get_mae_finetune_val_transform(input_size=aug_cfg.input_size)

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

		print_task_completed(output_path)

		return output_path
