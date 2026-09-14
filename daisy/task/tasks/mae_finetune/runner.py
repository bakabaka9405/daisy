"""MAE Finetune 任务执行器"""

from pathlib import Path
import time
from typing import override

import torch

from daisy.classification import prepare_classification_training
from daisy.method.mae import MAEFinetuneMethod
from daisy.model.mae import create_vit_model, load_mae_pretrained_weights
from daisy.training import (
	BestModel,
	Checkpoint,
	CosineAnnealingLR,
	CSVLog,
	EpochPrint,
)
from ...base import TaskRunner
from ...data import build_train_val_selection
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...shared import get_mae_finetune_train_transform, get_mae_finetune_val_transform
from .config import MAEFinetuneConfig


@TaskRegistry.register
class MAEFinetuneRunner(TaskRunner[MAEFinetuneConfig]):
	"""MAE Finetune 任务执行器"""

	@classmethod
	@override
	def get_config_class(cls) -> type[MAEFinetuneConfig]:
		return MAEFinetuneConfig

	@override
	def run(self, config: MAEFinetuneConfig, device: torch.device) -> Path:
		"""执行 MAE Finetune 任务"""
		# 阶段 1: 初始化运行上下文
		training_cfg = config.training
		run_context = prepare_task_run(config, device, seed=training_cfg.seed)
		output_path = run_context.output_path

		# 阶段 2: 计算实际学习率
		eff_batch_size = training_cfg.batch_size * training_cfg.accum_iter
		if training_cfg.blr is not None:
			actual_lr = training_cfg.blr * eff_batch_size / 256
			print(f'Base LR: {training_cfg.blr:.2e}, Effective batch size: {eff_batch_size}, Actual LR: {actual_lr:.2e}')
		else:
			actual_lr = training_cfg.lr

		# 阶段 3: 加载数据集并保存运行快照
		print('\nLoading dataset...')
		dataset_cfg = config.dataset
		split_selection = build_train_val_selection(
			dataset_cfg,
			default_seed=training_cfg.seed,
			context='mae finetune splits',
		)
		print(f'Total samples: {split_selection.source_count}')
		train_dataset, val_dataset = split_selection.to_datasets()
		save_run_snapshot(
			output_path,
			config,
			run_context,
			extra={'resolved_lr': actual_lr},
		)

		print(f'Train samples: {len(train_dataset)}')
		print(f'Val samples: {len(val_dataset)}')

		# 阶段 4: 获取数据变换
		aug_cfg = config.training.augment
		train_transform = get_mae_finetune_train_transform(
			input_size=aug_cfg.input_size,
			aa=aug_cfg.aa,
			reprob=aug_cfg.reprob,
			remode=aug_cfg.remode,
			recount=aug_cfg.recount,
		)
		val_transform = get_mae_finetune_val_transform(input_size=aug_cfg.input_size)

		# 阶段 5: 创建模型并加载 MAE 预训练权重
		print('\nCreating model...')
		model_cfg = config.model
		model = create_vit_model(
			model_cfg.name,
			num_classes=model_cfg.num_classes,
			global_pool=model_cfg.global_pool,
			drop_path_rate=model_cfg.drop_path,
			img_size=model_cfg.img_size,
		)

		if model_cfg.checkpoint:
			print(f'Loading MAE checkpoint: {model_cfg.checkpoint}')
			load_mae_pretrained_weights(model, model_cfg.checkpoint)

		print(f'Model: {model_cfg.name}')

		# 阶段 6: 组装算法方法与数据加载器
		print('\nStarting MAE finetuning...')

		method = MAEFinetuneMethod(
			model,
			num_classes=model_cfg.num_classes,
			device=device,
			lr=actual_lr,
			weight_decay=training_cfg.weight_decay,
			layer_decay=training_cfg.layer_decay,
			mixup=aug_cfg.mixup,
			cutmix=aug_cfg.cutmix,
			smoothing=aug_cfg.smoothing,
			use_amp=training_cfg.use_amp,
			accum_iter=training_cfg.accum_iter,
			clip_grad=training_cfg.clip_grad,
		)

		train_loader, evaluator = prepare_classification_training(
			method,
			(train_dataset, val_dataset),
			batch_size=training_cfg.batch_size,
			train_transform=train_transform,
			val_transform=val_transform,
			num_workers=training_cfg.num_workers,
			pin_memory=True,
			drop_last=True,
		)

		# 阶段 7: 配置并注册训练插件
		trainer = method.trainer

		trainer.use(
			CosineAnnealingLR(
				lr=actual_lr,
				min_lr=training_cfg.min_lr,
				warmup_epochs=training_cfg.warmup_epochs,
				per_iteration=True,
			)
		)

		trainer.use(
			BestModel(
				evaluator,
				watch_metric='f1',
				mode='max',
				save_path=output_path,
			)
		)

		if training_cfg.save_freq > 0:
			trainer.use(
				Checkpoint(
					save_dir=output_path,
					save_freq=training_cfg.save_freq,
				)
			)

		if config.output.log:
			log_path = output_path / 'logs' / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
			trainer.use(CSVLog(log_path=log_path, evaluator=evaluator))

		trainer.use(EpochPrint(evaluator=evaluator))

		# 阶段 8: 执行模型训练
		print('Ready to train...')
		method.fit(train_loader, epochs=training_cfg.epochs)

		print_task_completed(output_path)

		return output_path
