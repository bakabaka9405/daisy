"""多任务分类任务执行器"""

from functools import partial
from pathlib import Path
import time

import torch

from daisy.dataset import DiskDataset
from daisy.feeder import load_feeder_from_sheet
from daisy.method.classification import create_criterion
from daisy.method.multihead_classification import MultiHeadClassificationMethod
from daisy.method.ordinal import sord_loss
from daisy.multitask_classification import calculate_multihead_metrics, collect_multihead_predictions
from daisy.training import BestModel, CosineAnnealingLR, CSVLog, EpochPrint, Eval, Trainer
from daisy.util.data import make_dataloader
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ...shared import get_classification_transform
from .components import build_lrd_optimizer, create_multihead_model
from .config import MultiTaskClassificationConfig, MultiTaskDatasetConfig


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
			sheet_name=dataset_cfg.annotation_sheet_name,
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
		sheet_name=dataset_cfg.split_sheet_name,
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
	eval_cfg = config.evaluation
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

	extra: dict[str, float | int] = {'resolved_lr': resolved_lr}
	if config.model.checkpoint_path is not None:
		ckpt_stat = Path(config.model.checkpoint_path).stat()
		extra['checkpoint_size'] = ckpt_stat.st_size
		extra['checkpoint_mtime'] = ckpt_stat.st_mtime
	save_run_snapshot(output_path, config, run_context, extra=extra)

	# 创建共享 backbone 与多任务头
	print('\nCreating model...')
	model = create_multihead_model(config.model, num_classes_list)
	print(f'Model: {config.model.name}, tasks: {len(num_classes_list)}')

	# 选择目标损失函数
	if training_cfg.objective == 'ce':
		criterion = create_criterion(False, training_cfg.smoothing)
	elif training_cfg.objective == 'sord':
		criterion = partial(sord_loss, temperature=training_cfg.sord_temperature)
	else:
		raise ValueError(f'Unsupported objective: {training_cfg.objective}')

	# 构建多任务分类方法并作为核心插件装配训练器
	method = MultiHeadClassificationMethod(
		model,
		criterion,
		lr=resolved_lr,
		weight_decay=training_cfg.weight_decay,
	)
	trainer = Trainer(
		device=device,
		use_amp=training_cfg.use_amp,
		accum_iter=training_cfg.accum_iter,
		clip_grad=training_cfg.max_norm if training_cfg.clip_grad else None,
	)
	trainer.use(method)

	# layer-wise lr decay 时以自定义优化器覆盖方法默认的 AdamW
	if training_cfg.layer_decay is not None:
		trainer.optimizer = build_lrd_optimizer(
			model,
			resolved_lr,
			training_cfg.weight_decay,
			layer_decay=training_cfg.layer_decay,
			is_cuda=device.type == 'cuda',
		)

	if isinstance(training_cfg.num_workers, int):
		train_workers, val_workers = training_cfg.num_workers, training_cfg.num_workers
	else:
		train_workers, val_workers = training_cfg.num_workers

	train_loader = make_dataloader(
		train_dataset,
		training_cfg.batch_size,
		device,
		shuffle=True,
		drop_last=True,
		num_workers=train_workers,
	)
	val_loader = make_dataloader(
		val_dataset,
		training_cfg.batch_size,
		device,
		shuffle=False,
		drop_last=False,
		num_workers=val_workers,
	)
	test_loader = make_dataloader(
		test_dataset,
		training_cfg.batch_size,
		device,
		shuffle=False,
		drop_last=False,
		num_workers=val_workers,
	)

	# 训练期验证：各任务指标与平均指标
	def eval_fn(_trainer: Trainer) -> dict[str, float]:
		need_probs = 'auroc' in eval_cfg.val_metrics
		results = collect_multihead_predictions(model, val_loader, probabilities=need_probs)
		task_metrics = calculate_multihead_metrics(results, num_classes_list, eval_cfg.val_metrics)

		flattened: dict[str, float] = {}
		for task_key, m_dict in task_metrics.items():
			for m_name, val in m_dict.items():
				flattened[f'{task_key}_{m_name}'] = val

		num_tasks = len(num_classes_list)
		for avg_m in eval_cfg.average_metrics:
			vals = [task_metrics[f'task{t_idx + 1}'][avg_m] for t_idx in range(num_tasks)]
			flattened[f'avg_{avg_m}'] = sum(vals) / len(vals)

		return flattened

	# 配置并挂载训练插件（Eval 先注册，确保指标先于依赖它的插件生成）
	if training_cfg.use_scheduler:
		trainer.use(
			CosineAnnealingLR(
				lr=resolved_lr,
				min_lr=training_cfg.min_lr,
				warmup_epochs=training_cfg.warmup_epochs,
				per_iteration=training_cfg.per_iteration,
			)
		)

	evaluator = Eval(eval_fn=eval_fn)
	trainer.use(evaluator)

	if config.output.save_best:
		trainer.use(
			BestModel(
				evaluator,
				watch_metric=eval_cfg.watch_metric,
				mode=eval_cfg.watch_mode,
				save_path=output_path,
			)
		)
	if config.output.log:
		log_path = output_path / 'logs' / f'log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		trainer.use(CSVLog(log_path=log_path, evaluator=evaluator))
	trainer.use(EpochPrint(evaluator=evaluator))

	# 执行训练
	print('\nStarting training...')
	trainer.fit(train_loader, epochs=training_cfg.epochs)

	# 加载最佳模型并在测试集上评估
	if config.output.save_best:
		model.load_state_dict(torch.load(output_path / 'best_model.pth', map_location='cpu', weights_only=True))

	print('\n=== Test evaluation ===')
	if len(test_index) == 0:
		print('Test dataset is empty, skipping final test evaluation.')
	else:
		need_test_probs = 'auroc' in eval_cfg.test_metrics
		test_results = collect_multihead_predictions(model, test_loader, probabilities=need_test_probs)
		test_metrics = calculate_multihead_metrics(test_results, num_classes_list, eval_cfg.test_metrics)

		for task_idx in range(len(num_classes_list)):
			task_key = f'task{task_idx + 1}'
			m_dict = test_metrics[task_key]
			formatted = ', '.join(f'{k.capitalize()}={v:.4f}' for k, v in m_dict.items())
			print(f'  Task {task_idx + 1}: {formatted}')

		save_json(output_path / 'test_metrics.json', test_metrics)

	print_task_completed(output_path)
	return output_path
