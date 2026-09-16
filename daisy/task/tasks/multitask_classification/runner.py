"""多任务分类任务执行器"""

from pathlib import Path
import time
from typing import Any

import numpy as np
from timm import create_model
import torch

from daisy.dataset import DiskDataset
from daisy.feeder import load_feeder_from_sheet
from daisy.method.multihead_classification import MultiHeadClassificationMethod
from daisy.metrics import auroc_score, fast_calc_metrics
from daisy.training import BestModel, CosineAnnealingLR, CSVLog, EpochPrint, Eval
from daisy.typing import GenericDataLoader
from daisy.util.data import make_dataloader
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ...shared import get_classification_transform
from .config import MultiTaskClassificationConfig, MultiTaskDatasetConfig


def _resolve_sheet_name(name: int | str | None) -> int | str:
	return 0 if name is None else name


def _load_annotation(dataset_cfg: MultiTaskDatasetConfig) -> tuple[list[Path], list[list[int]]]:
	"""读取同一标注表上各任务列的样本与标签，要求所有任务共享同一图像集合。"""
	if dataset_cfg.annotation_sheet is None:
		raise ValueError('dataset.annotation_sheet is required')
	if not dataset_cfg.tasks:
		raise ValueError('dataset.tasks must not be empty')

	files: list[Path] | None = None
	labels_per_task: list[list[int]] = []
	for task in dataset_cfg.tasks:
		feeder = load_feeder_from_sheet(
			dataset_root=dataset_cfg.root,
			sheet_path=dataset_cfg.annotation_sheet,
			sheet_name=_resolve_sheet_name(dataset_cfg.annotation_sheet_name),
			column=task.column,
			label_offset=task.label_offset,
			have_header=dataset_cfg.have_header,
		)
		task_files, task_labels = feeder.fetch()
		if files is None:
			files = task_files
		elif files != task_files:
			raise ValueError('All tasks must annotate the same image set')
		labels_per_task.append(task_labels)
	assert files is not None
	return files, labels_per_task


def _load_split_map(dataset_cfg: MultiTaskDatasetConfig) -> dict[str, int]:
	"""读取统一划分列，返回文件名到划分值的映射。"""
	if dataset_cfg.split_sheet is None:
		raise ValueError('dataset.split_sheet is required')
	feeder = load_feeder_from_sheet(
		dataset_root=dataset_cfg.root,
		sheet_path=dataset_cfg.split_sheet,
		sheet_name=_resolve_sheet_name(dataset_cfg.split_sheet_name),
		column=dataset_cfg.split_column,
		have_header=dataset_cfg.have_header,
	)
	split_files, split_labels = feeder.fetch()
	return {file.name: int(label) for file, label in zip(split_files, split_labels)}


def _select(files: list[Path], split_map: dict[str, int], value: int) -> list[int]:
	return [index for index, file in enumerate(files) if split_map[file.name] == value]


def run(config: MultiTaskClassificationConfig, device: torch.device) -> Path:
	"""执行多任务分类训练任务"""
	training_cfg = config.training
	run_context = prepare_task_run(config, device, seed=training_cfg.seed)
	output_path = run_context.output_path

	# 解析学习率：blr 按有效批大小换算，与 lr 二选一
	eff_batch_size = training_cfg.batch_size * training_cfg.accum_iter
	if training_cfg.lr is not None:
		resolved_lr = training_cfg.lr
	elif training_cfg.blr is not None:
		resolved_lr = training_cfg.blr * eff_batch_size / 256.0
	else:
		raise ValueError('Either training.lr or training.blr must be set')
	print(f'Effective batch size: {eff_batch_size}, Learning rate: {resolved_lr:.2e}')

	# 加载标注与统一划分
	print('\nLoading dataset...')
	dataset_cfg = config.dataset
	files, labels_per_task = _load_annotation(dataset_cfg)
	split_map = _load_split_map(dataset_cfg)
	num_classes_list = [task.num_classes for task in dataset_cfg.tasks]

	missing = [file.name for file in files if file.name not in split_map]
	if missing:
		raise ValueError(f'Missing split entries for {len(missing)} files, e.g. {missing[:5]}')

	# 每个样本携带全部任务的标签，按任务头顺序排列
	labels = [tuple(task_labels[index] for task_labels in labels_per_task) for index in range(len(files))]

	train_index = _select(files, split_map, dataset_cfg.train_value)
	val_index = _select(files, split_map, dataset_cfg.val_value)
	test_index = _select(files, split_map, dataset_cfg.test_value)
	if not train_index or not val_index:
		raise ValueError('Train and val splits must not be empty')
	print(f'Train samples: {len(train_index)}')
	print(f'Val samples: {len(val_index)}')
	print(f'Test samples: {len(test_index)}')

	# 构建数据集
	train_transform = get_classification_transform(training_cfg.transform.train)
	val_transform = get_classification_transform(training_cfg.transform.val)

	train_dataset = DiskDataset([files[i] for i in train_index], [labels[i] for i in train_index], train_transform)
	val_dataset = DiskDataset([files[i] for i in val_index], [labels[i] for i in val_index], val_transform)
	test_dataset = DiskDataset([files[i] for i in test_index], [labels[i] for i in test_index], val_transform)

	save_run_snapshot(output_path, config, run_context, extra={'resolved_lr': resolved_lr})

	# 创建共享 backbone 与多任务头
	print('\nCreating model...')
	model_cfg = config.model
	model_kwargs: dict[str, Any] = {'pretrained': model_cfg.pretrained, 'num_classes': 0}
	if model_cfg.img_size is not None:
		model_kwargs['img_size'] = model_cfg.img_size
	if model_cfg.drop_path_rate > 0:
		model_kwargs['drop_path_rate'] = model_cfg.drop_path_rate
	backbone = create_model(model_cfg.name, **model_kwargs)
	print(f'Model: {model_cfg.name}, tasks: {len(num_classes_list)}')

	# 构建多任务分类方法
	method = MultiHeadClassificationMethod(
		backbone,
		num_classes_list,
		device=device,
		lr=resolved_lr,
		weight_decay=training_cfg.weight_decay,
		smoothing=training_cfg.smoothing,
		use_amp=training_cfg.use_amp,
		accum_iter=training_cfg.accum_iter,
		clip_grad=training_cfg.max_norm if training_cfg.clip_grad else None,
	)

	train_loader = make_dataloader(
		train_dataset,
		training_cfg.batch_size,
		device,
		shuffle=True,
		drop_last=True,
		num_workers=training_cfg.num_workers,
	)
	val_loader = make_dataloader(
		val_dataset,
		training_cfg.batch_size,
		device,
		shuffle=False,
		drop_last=False,
		num_workers=training_cfg.num_workers,
	)
	test_loader = make_dataloader(
		test_dataset,
		training_cfg.batch_size,
		device,
		shuffle=False,
		drop_last=False,
		num_workers=training_cfg.num_workers,
	)

	# 按任务收集预测与概率
	def evaluate_loader(loader: GenericDataLoader) -> tuple[list[list[int]], list[list[int]], list[np.ndarray]]:
		model = method.model
		was_training = model.training
		model.eval()
		preds: list[list[int]] = [[] for _ in num_classes_list]
		trues: list[list[int]] = [[] for _ in num_classes_list]
		probs: list[list[list[float]]] = [[] for _ in num_classes_list]
		try:
			with torch.no_grad():
				for images, targets in loader:
					logits_list = model(images)
					for task_idx in range(len(num_classes_list)):
						logits = logits_list[task_idx]
						preds[task_idx].extend(logits.argmax(dim=1).cpu().tolist())
						trues[task_idx].extend(targets[task_idx].cpu().tolist())
						probs[task_idx].extend(torch.softmax(logits, dim=1).cpu().tolist())
		finally:
			model.train(was_training)
		return trues, preds, [np.asarray(task_probs, dtype=np.float64) for task_probs in probs]

	# 训练期验证：各任务 f1 与平均 f1
	def eval_fn(trainer: Any) -> dict[str, float]:
		trues, preds, _ = evaluate_loader(val_loader)
		metrics: dict[str, float] = {}
		f1_values: list[float] = []
		for task_idx, num_classes in enumerate(num_classes_list):
			task_metrics = fast_calc_metrics(trues[task_idx], preds[task_idx], num_classes=num_classes)
			metrics[f'task{task_idx + 1}_f1'] = task_metrics.f1
			metrics[f'task{task_idx + 1}_acc'] = task_metrics.acc
			f1_values.append(task_metrics.f1)
		metrics['avg_f1'] = sum(f1_values) / len(f1_values)
		return metrics

	# 配置并挂载训练插件（Eval 先注册，确保指标先于依赖它的插件生成）
	trainer = method.trainer
	if training_cfg.use_scheduler:
		trainer.use(CosineAnnealingLR(lr=resolved_lr, warmup_epochs=training_cfg.warmup_epochs))

	evaluator = Eval(eval_fn=eval_fn)
	trainer.use(evaluator)

	if config.output.save_best:
		trainer.use(BestModel(evaluator, watch_metric='avg_f1', mode='max', save_path=output_path))
	if config.output.log:
		log_path = output_path / 'logs' / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		trainer.use(CSVLog(log_path=log_path, evaluator=evaluator))
	trainer.use(EpochPrint(evaluator=evaluator))

	# 执行训练
	print('\nStarting training...')
	method.fit(train_loader, epochs=training_cfg.epochs)

	# 加载最佳模型并在测试集上评估
	if config.output.save_best:
		method.model.load_state_dict(torch.load(output_path / 'best_model.pth', map_location='cpu', weights_only=True))

	print('\n=== Test evaluation ===')
	test_trues, test_preds, test_probs = evaluate_loader(test_loader)
	test_metrics: dict[str, dict[str, float]] = {}
	for task_idx, num_classes in enumerate(num_classes_list):
		task_metrics = fast_calc_metrics(test_trues[task_idx], test_preds[task_idx], num_classes=num_classes)
		auroc = auroc_score(
			np.asarray(test_trues[task_idx], dtype=np.int64),
			test_probs[task_idx],
			mode='macro',
		)
		auroc_value = float(auroc['macro'] if isinstance(auroc, dict) else auroc)
		test_metrics[f'task{task_idx + 1}'] = {
			'acc': float(task_metrics.acc),
			'precision': float(task_metrics.precision),
			'recall': float(task_metrics.recall),
			'f1': float(task_metrics.f1),
			'auroc': auroc_value,
		}
		print(
			f'  Task {task_idx + 1}: Acc={task_metrics.acc:.4f}, Prec={task_metrics.precision:.4f}, '
			f'Recall={task_metrics.recall:.4f}, F1={task_metrics.f1:.4f}, AUROC={auroc_value:.4f}'
		)
	save_json(output_path / 'test_metrics.json', test_metrics)

	print_task_completed(output_path)
	return output_path
