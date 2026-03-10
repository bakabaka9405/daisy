"""checkpoint 评估任务执行器"""

from __future__ import annotations

import csv
from pathlib import Path

import torch

import daisy
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import resolve_output_path, save_json, save_task_snapshot
from ..inference_common import build_prediction_rows, create_inference_model, get_inference_transform, select_inference_dataset
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
		if runtime_cfg.seed is not None:
			daisy.util.set_global_seed(runtime_cfg.seed)

		print('=' * 60)
		print(f'Task: {config.meta.title or config.task_id}')
		print(f'Description: {config.meta.description}')
		print(f'Device: {device}')
		print('=' * 60)

		if config.meta.commit == 'auto':
			config.meta.commit = daisy.util.get_git_commit()
		print(f'Git commit: {config.meta.commit}')

		output_path = resolve_output_path(config.output.save_path, config.task_id)
		print(f'Output path: {output_path}')

		eval_dataset, protocol_snapshot = select_inference_dataset(
			config.dataset,
			split_name=runtime_cfg.split_name,
		)
		save_json(output_path / 'eval_protocol.json', protocol_snapshot)
		save_task_snapshot(
			output_path,
			config,
			extra={
				'device': str(device),
				'commit': config.meta.commit,
				'seed': runtime_cfg.seed,
			},
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
		print('Task completed!')
		return output_path
