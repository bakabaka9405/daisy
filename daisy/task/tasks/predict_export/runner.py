"""预测导出任务执行器"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from ...data import select_dataset_split
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ..inference_common import (
	build_prediction_rows,
	create_inference_model,
	execute_inference,
	get_inference_transform,
)
from .config import PredictExportConfig


def run(config: PredictExportConfig, device: torch.device) -> Path:
	runtime_cfg = config.prediction
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
		raise ValueError('predict_export requires a labeled dataset')
	transform = get_inference_transform(config.model, runtime_cfg)
	model = create_inference_model(config.model)

	all_preds, all_logits = execute_inference(
		model,
		eval_dataset,
		config.model,
		device=device,
		batch_size=runtime_cfg.batch_size,
		num_workers=runtime_cfg.num_workers,
		include_logits=runtime_cfg.include_logits,
		transform=transform,
	)

	rows = build_prediction_rows(
		eval_files,
		eval_labels,
		all_preds,
		root=config.dataset.root,
		logits=all_logits if runtime_cfg.include_logits else None,
	)
	with open(output_path / config.output.filename, 'w', encoding='utf-8', newline='') as f:
		fieldnames = ['file', 'sample_id', 'true', 'pred']
		if runtime_cfg.include_logits:
			fieldnames.append('logits')
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			if 'logits' in row:
				row = {**row, 'logits': json.dumps(row['logits'], ensure_ascii=False)}
			writer.writerow(row)

	save_json(
		output_path / 'prediction_summary.json',
		{
			'split_name': runtime_cfg.split_name,
			'sample_count': len(rows),
			'include_logits': runtime_cfg.include_logits,
			'output_file': str(output_path / config.output.filename),
		},
	)

	print(f'Exported {len(rows)} predictions to {output_path / config.output.filename}')
	print_task_completed(output_path)
	return output_path
