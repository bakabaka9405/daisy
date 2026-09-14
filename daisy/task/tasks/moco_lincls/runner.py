"""MoCo 线性探测评估任务执行器"""

from pathlib import Path
import time

import timm
import torch

from daisy.classification import prepare_classification_training
from daisy.method.classification import ClassificationMethod
from daisy.model.moco import load_moco_pretrained_weights
from daisy.training import BestModel, Checkpoint, CosineAnnealingLR, CSVLog, EpochPrint
from daisy.training.state import TrainState
from daisy.util import change_model_classifier, get_model_classifier
from ...data import build_train_val_selection
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from .config import MoCoLinclsConfig


def run(config: MoCoLinclsConfig, device: torch.device) -> Path:
	"""执行 MoCo 线性探测评估任务"""
	# 阶段 1: 准备任务运行时上下文与输出路径
	training_cfg = config.training
	run_context = prepare_task_run(config, device, seed=training_cfg.seed)
	output_path = run_context.output_path

	# 阶段 2: 加载数据集与划分
	print('\nLoading dataset...')
	dataset_cfg = config.dataset
	split_selection = build_train_val_selection(
		dataset_cfg,
		default_seed=training_cfg.seed,
		context='moco lincls splits',
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

	# 阶段 3: 创建基础模型并在替换分类头前加载 MoCo 预训练权重
	print('\nCreating model...')
	model_cfg = config.model
	model = timm.create_model(model_cfg.arch, pretrained=False, num_classes=model_cfg.num_classes)

	if model_cfg.pretrained_checkpoint:
		print(f'Loading MoCo checkpoint: {model_cfg.pretrained_checkpoint}')
		load_moco_pretrained_weights(model, model_cfg.pretrained_checkpoint, engine=model_cfg.engine)

	# 阶段 4: 冻结骨干网络所有层，重新初始化线性分类头并设为可训练
	for param in model.parameters():
		param.requires_grad = False

	change_model_classifier(model, num_classes=model_cfg.num_classes)
	classifier = get_model_classifier(model)
	assert isinstance(classifier, torch.nn.Linear)
	classifier.weight.data.normal_(mean=0.0, std=0.01)
	if classifier.bias is not None:
		classifier.bias.data.zero_()
	for param in classifier.parameters():
		param.requires_grad = True

	print(f'Model: {model_cfg.arch} (linear probe, {model_cfg.num_classes} classes)')

	# 阶段 5: 组装算法方法与数据加载器
	print('\nStarting linear evaluation...')
	method = ClassificationMethod(
		model,
		num_classes=model_cfg.num_classes,
		device=device,
		lr=training_cfg.lr,
		weight_decay=training_cfg.weight_decay,
		smoothing=training_cfg.smoothing,
		use_amp=training_cfg.use_amp,
		accum_iter=training_cfg.accum_iter,
		clip_grad=training_cfg.clip_grad,
	)

	train_loader, evaluator = prepare_classification_training(
		method,
		(train_dataset, val_dataset),
		batch_size=training_cfg.batch_size,
		num_workers=training_cfg.num_workers,
		pin_memory=(device.type == 'cuda'),
	)

	# 阶段 6: 配置并挂载训练插件与冻结钩子
	trainer = method.trainer

	# Trainer 在每个 epoch 开始时会调用 model.train()，故在 batch 开始时强制调用 model.eval()，
	# 从而冻结骨干网络层与 BatchNorm 运行统计量，且分类头梯度正常反向传播更新
	def _hold_eval_mode(state: TrainState) -> None:
		state.model.eval()

	trainer.on_batch_start(_hold_eval_mode)

	trainer.use(
		CosineAnnealingLR(
			lr=training_cfg.lr,
			min_lr=training_cfg.min_lr,
			warmup_epochs=training_cfg.warmup_epochs,
		)
	)

	if config.output.save_best:
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

	# 阶段 7: 执行模型训练
	method.fit(train_loader, epochs=training_cfg.epochs)

	print_task_completed(output_path)

	return output_path
