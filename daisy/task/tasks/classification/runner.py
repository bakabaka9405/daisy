"""分类任务执行器"""

from pathlib import Path
import time

from timm import create_model
import torch

from daisy.classification import prepare_classification_training
from daisy.method.classification import ClassificationMethod
from daisy.task.data import build_datasets
from daisy.training import BestModel, Checkpoint, CosineAnnealingLR, CSVLog, EarlyStop, EpochPrint, Trainer
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...shared import SingleLabelConfig, get_classification_transform
from .config import ClassificationConfig


def run(config: ClassificationConfig, device: torch.device) -> Path:
	"""执行分类训练任务"""
	# 初始化运行环境
	training_cfg = config.training
	run_context = prepare_task_run(config, device, seed=training_cfg.seed)
	output_path = run_context.output_path

	# 加载数据集并校验划分
	print('\nLoading dataset...')
	dataset_cfg = config.dataset
	datasets = build_datasets(dataset_cfg, seed=training_cfg.seed)
	if 'train' not in datasets or 'val' not in datasets:
		raise ValueError("Classification requires 'train' and 'val' dataset splits")
	train_dataset = datasets['train']
	val_dataset = datasets['val']
	save_run_snapshot(
		output_path,
		config,
		run_context,
	)

	print(f'Train samples: {len(train_dataset)}')
	print(f'Val samples: {len(val_dataset)}')

	# 从目标配置中提取分类类别数作为单一来源
	if not isinstance(dataset_cfg.labels, SingleLabelConfig):
		raise ValueError("Classification task requires dataset.labels with mode='single_label'")
	num_classes = dataset_cfg.labels.target.num_classes

	# 创建模型
	print('\nCreating model...')
	model_cfg = config.model
	model = create_model(
		model_cfg.name,
		pretrained=model_cfg.pretrained,
		num_classes=num_classes,
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

	# 实例化分类方法并作为核心插件装配训练器
	method = ClassificationMethod(
		model,
		num_classes=num_classes,
		lr=training_cfg.lr,
		weight_decay=training_cfg.weight_decay,
		smoothing=training_cfg.smoothing,
	)

	trainer = Trainer(
		device=device,
		use_amp=training_cfg.use_amp,
		accum_iter=training_cfg.accum_iter,
		clip_grad=training_cfg.max_norm if training_cfg.clip_grad else None,
	)
	trainer.use(method)

	# 构建 DataLoader 并自动注册 Eval 插件
	train_loader = prepare_classification_training(
		method,
		trainer,
		(train_dataset, val_dataset),
		batch_size=training_cfg.batch_size,
		train_transform=train_transform,
		val_transform=val_transform,
		num_workers=training_cfg.num_workers,
	)

	# 配置并挂载训练插件
	if training_cfg.use_scheduler:
		trainer.use(
			CosineAnnealingLR(
				lr=training_cfg.lr,
				warmup_epochs=training_cfg.warmup_epochs,
			)
		)

	cmp_name = 'precision' if training_cfg.cmp_obj == 'prec' else training_cfg.cmp_obj
	watch_metric = f'val/{cmp_name}'

	if config.output.save_best:
		trainer.use(
			BestModel(
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

	eval_keys = [
		'train/loss',
		'optim/lr',
		'val/loss',
		'val/acc',
		'val/f1',
		'val/precision',
		'val/recall',
		'val/auroc',
	]

	if config.output.log:
		log_path = output_path / 'logs' / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		trainer.use(CSVLog(log_path=log_path, watch=eval_keys))

	trainer.use(EpochPrint(watch=eval_keys))

	if training_cfg.early_stop:
		trainer.use(
			EarlyStop(
				watch_metric=watch_metric,
				patience=training_cfg.early_stop_epoch,
				mode='max',
			)
		)

	trainer.fit(train_loader, epochs=training_cfg.epochs)

	print_task_completed(output_path)

	return output_path
