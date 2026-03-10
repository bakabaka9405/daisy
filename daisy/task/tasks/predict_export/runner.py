"""预测导出任务执行器"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

import daisy
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import resolve_output_path, save_json, save_task_snapshot
from ..inference_common import build_prediction_rows, create_inference_model, get_inference_transform, select_inference_dataset
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
		save_json(output_path / 'predict_protocol.json', protocol_snapshot)
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

		model.to(device)
		model.eval()
		all_preds: list[int] = []
		all_logits: list[list[float]] = []

		with torch.no_grad():
			for images, _ in loader:
				images = images.to(device, non_blocking=True)
				outputs = model(images)
				preds = torch.argmax(outputs, dim=1)
				all_preds.extend(preds.cpu().tolist())
				if runtime_cfg.include_logits:
					all_logits.extend(outputs.cpu().tolist())

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
		print('Task completed!')
		return output_path
