"""有序分类任务执行器"""

from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score
from timm import create_model
from timm.data.loader import MultiEpochsDataLoader
import torch

from daisy.method.ordinal import METHODS
from daisy.metrics import mae_score, rmse_score
from daisy.training import BestModel, Checkpoint, CosineAnnealingLR, CSVLog, EpochPrint, Eval, Trainer
from ...data import build_train_val_selection
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from ...shared import get_classification_transform
from ..inference_common import load_checkpoint_state_dict
from .config import OrdinalConfig


def run(config: OrdinalConfig, device: torch.device) -> Path:
	"""执行有序分类训练任务"""
	training_cfg = config.training
	model_cfg = config.model
	run_context = prepare_task_run(config, device, seed=training_cfg.seed)
	output_path = run_context.output_path

	# 数据加载与划分
	print('\nLoading dataset...')
	dataset_cfg = config.dataset
	split_selection = build_train_val_selection(
		dataset_cfg,
		default_seed=training_cfg.seed,
		context='ordinal splits',
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

	_, val_labels = val_dataset.getRawData()
	if val_labels is None:
		raise ValueError('Ordinal task requires a labeled dataset')
	val_targets = np.asarray(val_labels, dtype=np.int64)

	# 数据变换与 DataLoader 构建
	train_transform = get_classification_transform(training_cfg.transform.train)
	val_transform = get_classification_transform(training_cfg.transform.val)
	train_dataset.setTransform(train_transform)
	val_dataset.setTransform(val_transform)

	if isinstance(training_cfg.num_workers, int):
		workers = (training_cfg.num_workers, training_cfg.num_workers)
	else:
		workers = training_cfg.num_workers

	train_loader = MultiEpochsDataLoader(
		train_dataset,
		batch_size=training_cfg.batch_size,
		shuffle=True,
		num_workers=workers[0],
		pin_memory=device.type == 'cuda',
		drop_last=False,
	)

	val_loader = MultiEpochsDataLoader(
		val_dataset,
		batch_size=training_cfg.batch_size,
		shuffle=False,
		num_workers=workers[1],
		pin_memory=device.type == 'cuda',
		drop_last=False,
	)

	# 骨干网络与方法构建
	print('\nCreating model...')
	backbone = create_model(
		model_cfg.name,
		pretrained=model_cfg.pretrained,
		num_classes=0,
	)

	method_cls = METHODS[model_cfg.method]
	method = method_cls(
		backbone,
		num_classes=model_cfg.num_classes,
		device=device,
		lr=training_cfg.lr,
		weight_decay=training_cfg.weight_decay,
		use_amp=training_cfg.use_amp,
		accum_iter=training_cfg.accum_iter,
		clip_grad=training_cfg.clip_grad,
	)

	# 恢复模型权重（在方法构建后执行）
	if model_cfg.checkpoint:
		print(f'Loading checkpoint: {model_cfg.checkpoint}')
		state_dict = load_checkpoint_state_dict(model_cfg.checkpoint)
		method.model.load_state_dict(state_dict)

	print(f'Model: {model_cfg.name} (method={model_cfg.method}, num_classes={model_cfg.num_classes})')

	# 评估与训练插件挂载
	def eval_fn(trainer: Trainer) -> dict[str, float]:
		pred_np = method.predict(val_loader).numpy()
		val_targets_f = val_targets.astype(np.float64)
		pred_f = pred_np.astype(np.float64)
		return {
			'mae': float(mae_score(val_targets_f, pred_f)),
			'rmse': float(rmse_score(val_targets_f, pred_f)),
			'acc': float(accuracy_score(val_targets, pred_np)),
			'kappa': float(
				cohen_kappa_score(  # type: ignore[call-overload]
					val_targets,
					pred_np,
					weights='quadratic',
					labels=list(range(model_cfg.num_classes)),
				)
			),
		}

	trainer = method.trainer
	evaluator = Eval(eval_fn=eval_fn)
	trainer.use(evaluator)

	if training_cfg.use_scheduler:
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
				watch_metric='mae',
				mode='min',
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

	trainer.use(EpochPrint(evaluator=evaluator, metric_names=['mae', 'rmse', 'acc', 'kappa']))

	# 模型训练
	print('\nStarting training...')
	method.fit(train_loader, epochs=training_cfg.epochs)

	print_task_completed(output_path)
	return output_path
