"""分类任务执行器"""

from pathlib import Path
from typing import TYPE_CHECKING

from timm import create_model

import daisy
from ...base import TaskRunner
from ...data import build_train_val_selection
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...shared import get_classification_transform
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

		daisy.classifier_trainer.fast_train_smile(
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
