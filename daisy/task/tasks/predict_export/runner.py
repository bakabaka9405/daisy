"""预测导出任务执行器"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from daisy.training import Trainer
from ...base import TaskRunner
from ...data import select_dataset_split
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ..inference_common import build_prediction_rows, create_inference_model, get_inference_transform
from .config import PredictExportConfig


@TaskRegistry.register
class PredictExportRunner(TaskRunner['PredictExportConfig']):
	"""预测导出任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'predict_export'

	@classmethod
	def get_config_class(cls) -> type[PredictExportConfig]:
		return PredictExportConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return '预测导出'

	def run(self, config: PredictExportConfig, device: torch.device) -> Path:
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
		transform = get_inference_transform(config.model, runtime_cfg)
		model = create_inference_model(config.model)

		eval_dataset.setTransform(transform)
		loader = torch.utils.data.DataLoader(
			eval_dataset,
			batch_size=runtime_cfg.batch_size,
			shuffle=False,
			num_workers=runtime_cfg.num_workers,
			pin_memory=True,
		)

		scores = Trainer(model=model, device=device, use_amp=False).inference(loader)
		all_preds = torch.argmax(scores, dim=1).tolist() if scores.numel() else []
		all_logits = scores.tolist() if runtime_cfg.include_logits else []

		rows = build_prediction_rows(
			eval_files,
			eval_labels,
			all_preds,
			root=Path(config.dataset.root),
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
