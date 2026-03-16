"""checkpoint 评估任务执行器"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

import daisy
from ...base import TaskRunner
from ...data import select_dataset_split
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
from ..inference_common import build_prediction_rows, create_inference_model, get_inference_transform
from .config import EvalCheckpointConfig


@TaskRegistry.register
class EvalCheckpointRunner(TaskRunner['EvalCheckpointConfig']):
	"""checkpoint 评估任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'eval_checkpoint'

	@classmethod
	def get_config_class(cls) -> type[EvalCheckpointConfig]:
		return EvalCheckpointConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'Checkpoint 评估'

	def run(self, config: EvalCheckpointConfig, device: torch.device) -> Path:
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
		print(f'Eval samples: {len(eval_dataset)}')

		model = create_inference_model(config.model)
		transform = get_inference_transform(config.model, runtime_cfg)
		y_true, y_pred = daisy.classfier_trainer.fast_eval(
			device,
			model,
			eval_dataset,
			transform,
			batch_size=runtime_cfg.batch_size,
			num_workers=runtime_cfg.num_workers,
		)
		metrics = daisy.classfier_trainer.fast_calc_metrics(
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
			root=Path(config.dataset.root),
		)
		with open(output_path / 'predictions.csv', 'w', encoding='utf-8', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=['file', 'sample_id', 'true', 'pred'])
			writer.writeheader()
			writer.writerows(rows)

		print(f'Metrics - acc: {metrics.acc:.4f}, f1: {metrics.f1:.4f}')
		print_task_completed(output_path)
		return output_path
