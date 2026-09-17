"""多任务多头分类评估模块。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from daisy.metrics import auroc_score, fast_calc_metrics, mae_score
from daisy.model.multihead import MultiHeadModel

type ClassificationMetric = Literal['acc', 'precision', 'recall', 'f1', 'auroc', 'mae']


@dataclass(frozen=True, slots=True)
class TaskPredictions:
	"""单个任务头的预测结果，包含 targets、predictions 及可选 probabilities 张量。"""

	targets: torch.Tensor
	predictions: torch.Tensor
	probabilities: torch.Tensor | None = None


def collect_multihead_predictions(
	model: MultiHeadModel,
	loader: Iterable[tuple[torch.Tensor, Sequence[torch.Tensor]]],
	*,
	probabilities: bool = False,
) -> list[TaskPredictions]:
	"""在评估模式下遍历 loader，收集各任务头的 CPU 张量预测结果。

	调用约定：loader 已负责将图像送至模型计算设备（如 Prefetcher）。
	"""
	was_training = model.training
	model.eval()

	num_tasks = len(model.num_classes_list)
	target_chunks: list[list[torch.Tensor]] = [[] for _ in range(num_tasks)]
	pred_chunks: list[list[torch.Tensor]] = [[] for _ in range(num_tasks)]
	prob_chunks: list[list[torch.Tensor]] = [[] for _ in range(num_tasks)]

	try:
		with torch.no_grad():
			for images, targets in loader:
				logits_list = model(images)
				if len(logits_list) != num_tasks or len(targets) != num_tasks:
					raise ValueError(
						f'Mismatch between task counts: model has {num_tasks} heads, '
						f'model output has {len(logits_list)}, batch targets has {len(targets)}'
					)

				for task_idx, (logits, target) in enumerate(zip(logits_list, targets, strict=True)):
					target_chunks[task_idx].append(target.detach().to(device='cpu'))
					pred_chunks[task_idx].append(logits.argmax(dim=1).detach().to(device='cpu'))
					if probabilities:
						prob_chunks[task_idx].append(torch.softmax(logits, dim=1).detach().to(device='cpu'))

		if not target_chunks[0]:
			raise ValueError('Loader yielded no batches')
	finally:
		model.train(was_training)

	predictions: list[TaskPredictions] = []
	for task_idx in range(num_tasks):
		task_targets = torch.cat(target_chunks[task_idx], dim=0)
		task_preds = torch.cat(pred_chunks[task_idx], dim=0)
		task_probs = torch.cat(prob_chunks[task_idx], dim=0) if probabilities else None
		predictions.append(
			TaskPredictions(
				targets=task_targets,
				predictions=task_preds,
				probabilities=task_probs,
			)
		)

	return predictions


def calculate_multihead_metrics(
	results: Sequence[TaskPredictions],
	num_classes: Sequence[int],
	metrics: Sequence[ClassificationMetric],
) -> dict[str, dict[str, float]]:
	"""根据各任务预测结果计算分类评估指标，输出键为 task1、task2 等的指标字典。"""
	if len(results) != len(num_classes):
		raise ValueError(f'Mismatch between results count ({len(results)}) and num_classes count ({len(num_classes)})')

	if 'auroc' in metrics:
		for task_idx, res in enumerate(results):
			if res.probabilities is None:
				raise ValueError(f'Task {task_idx + 1} missing probabilities required for AUROC calculation')

	valid_metrics = {'acc', 'precision', 'recall', 'f1', 'auroc', 'mae'}
	for m in metrics:
		if m not in valid_metrics:
			raise ValueError(f"Unsupported metric '{m}', expected one of {sorted(valid_metrics)}")

	need_fast_calc = any(m in {'acc', 'precision', 'recall', 'f1'} for m in metrics)
	all_metrics: dict[str, dict[str, float]] = {}

	for task_idx, (task_pred, n_cls) in enumerate(zip(results, num_classes, strict=True)):
		task_key = f'task{task_idx + 1}'
		y_true_np = task_pred.targets.detach().cpu().numpy()
		y_pred_np = task_pred.predictions.detach().cpu().numpy()

		computed: dict[str, float] = {}
		if need_fast_calc:
			fast_metrics = fast_calc_metrics(y_true_np, y_pred_np, num_classes=n_cls)
			computed['acc'] = float(fast_metrics.acc)
			computed['precision'] = float(fast_metrics.precision)
			computed['recall'] = float(fast_metrics.recall)
			computed['f1'] = float(fast_metrics.f1)

		if 'mae' in metrics:
			y_true_f64 = y_true_np.astype(np.float64, copy=False)
			y_pred_f64 = y_pred_np.astype(np.float64, copy=False)
			computed['mae'] = float(mae_score(y_true_f64, y_pred_f64))

		if 'auroc' in metrics:
			assert task_pred.probabilities is not None
			probs_f64 = task_pred.probabilities.detach().cpu().numpy().astype(np.float64, copy=False)
			targets_i64 = y_true_np.astype(np.int64, copy=False)
			auroc_result = auroc_score(targets_i64, probs_f64, mode='macro')
			computed['auroc'] = float(auroc_result['macro'] if isinstance(auroc_result, dict) else auroc_result)

		all_metrics[task_key] = {m: computed[m] for m in metrics}

	return all_metrics


__all__ = [
	'ClassificationMetric',
	'TaskPredictions',
	'calculate_multihead_metrics',
	'collect_multihead_predictions',
]
