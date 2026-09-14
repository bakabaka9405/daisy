"""checkpoint 评估任务执行器"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

from daisy.metrics.classification import fast_calc_metrics
from ...data import select_dataset_split
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ..inference_common import (
	build_prediction_rows,
	create_inference_model,
	execute_inference,
	get_inference_transform,
)
from .config import EvalCheckpointConfig


def run(config: EvalCheckpointConfig, device: torch.device) -> Path:
	runtime_cfg = config.evaluation
	run_context = prepare_task_run(config, device, seed=runtime_cfg.seed)
	output_path = run_context.output_path

	selection = select_dataset_split(
		config.dataset,
		split_name=runtime_cfg.split_name,
	)
	eval_dataset = selection.to_dataset()
	save_run_snapshot(
		output_path,
		config,
		run_context,
	)

	eval_files, eval_labels = eval_dataset.getRawData()
	if eval_labels is None:
		raise ValueError('eval_checkpoint requires a labeled dataset')
	print(f'Eval samples: {len(eval_dataset)}')

	model = create_inference_model(config.model)
	transform = get_inference_transform(config.model, runtime_cfg)
	y_pred, _ = execute_inference(
		model,
		eval_dataset,
		config.model,
		device=device,
		batch_size=runtime_cfg.batch_size,
		num_workers=runtime_cfg.num_workers,
		transform=transform,
	)
	y_true = [int(x) for x in eval_labels]
	metrics = fast_calc_metrics(
		y_true,
		y_pred,
		num_classes=config.model.num_classes,
	)

	metrics_data = {
		'acc': metrics.acc,
		'precision': metrics.precision,
		'recall': metrics.recall,
		'f1': metrics.f1,
		'confusion_matrix': metrics.confusion_matrix,
		'split_name': runtime_cfg.split_name,
		'sample_count': len(eval_dataset),
	}
	save_json(output_path / 'metrics.json', metrics_data)

	rows = build_prediction_rows(
		eval_files,
		eval_labels,
		y_pred,
		root=config.dataset.root,
	)
	with open(output_path / 'predictions.csv', 'w', encoding='utf-8', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=['file', 'sample_id', 'true', 'pred'])
		writer.writeheader()
		writer.writerows(rows)

	print(f'Metrics - acc: {metrics.acc:.4f}, f1: {metrics.f1:.4f}')
	print_task_completed(output_path)
	return output_path
