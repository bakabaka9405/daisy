"""分类任务执行器"""

from pathlib import Path
import time
from typing import override

from timm import create_model
import torch

from daisy.classification import prepare_classification_training
from daisy.method.classification import ClassificationMethod
from daisy.training import BestModel, Checkpoint, CosineAnnealingLR, CSVLog, EarlyStop, EpochPrint
from ...base import TaskRunner
from ...data import build_train_val_selection
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...shared import get_classification_transform
from .config import ClassificationConfig


@TaskRegistry.register
class ClassificationRunner(TaskRunner[ClassificationConfig]):
	"""分类任务执行器"""

	@classmethod
	@override
	def get_config_class(cls) -> type[ClassificationConfig]:
		return ClassificationConfig

	@override
	def run(self, config: ClassificationConfig, device: torch.device) -> Path:
		"""执行分类训练任务"""
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
		train_transform = get_classification_transform(training_cfg.transform.train)
		val_transform = get_classification_transform(training_cfg.transform.val)

		# 训练
		print('\nStarting training...')

		# 实例化分类方法
		method = ClassificationMethod(
			model,
			num_classes=model_cfg.num_classes,
			device=device,
			lr=training_cfg.lr,
			weight_decay=training_cfg.weight_decay,
			smoothing=training_cfg.smoothing,
			use_amp=training_cfg.use_amp,
			accum_iter=training_cfg.accum_iter,
			clip_grad=training_cfg.max_norm if training_cfg.clip_grad else None,
		)

		# 构建 DataLoader 并自动注册 Eval 插件
		train_loader, evaluator = prepare_classification_training(
			method,
			(train_dataset, val_dataset),
			batch_size=training_cfg.batch_size,
			train_transform=train_transform,
			val_transform=val_transform,
			num_workers=training_cfg.num_workers,
		)

		# 配置并挂载训练插件
		trainer = method.trainer

		if training_cfg.use_scheduler:
			trainer.use(
				CosineAnnealingLR(
					lr=training_cfg.lr,
					warmup_epochs=training_cfg.warmup_epochs,
				)
			)

		watch_metric = 'precision' if training_cfg.cmp_obj == 'prec' else training_cfg.cmp_obj

		if config.output.save_best:
			trainer.use(
				BestModel(
					evaluator,
					watch_metric=watch_metric,
					mode='max',
					save_path=output_path,
				)
			)

		if config.output.keep_count > 0:
			trainer.use(
				Checkpoint(
					save_dir=output_path,
					keep_recent=config.output.keep_count,
				)
			)

		if config.output.log:
			log_path = output_path / 'logs' / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
			trainer.use(CSVLog(log_path=log_path, evaluator=evaluator))

		trainer.use(EpochPrint(evaluator=evaluator))

		if training_cfg.early_stop:
			trainer.use(
				EarlyStop(
					evaluator,
					patience=training_cfg.early_stop_epoch,
					watch_metric=watch_metric,
					mode='max',
				)
			)

		method.fit(train_loader, epochs=training_cfg.epochs)

		print_task_completed(output_path)

		return output_path
