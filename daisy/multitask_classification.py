"""多任务多头分类评估模块。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import icontract
import numpy as np
import torch

from daisy.metrics import auroc_score, fast_calc_metrics, mae_score

type ClassificationMetric = Literal['acc', 'precision', 'recall', 'f1', 'auroc', 'mae']


@dataclass(frozen=True, slots=True)
class TaskPredictions:
	"""单个任务头的预测结果，包含 targets、predictions 及可选 probabilities 张量。"""

	targets: torch.Tensor
	predictions: torch.Tensor
	probabilities: torch.Tensor | None = None


def _has_auroc_probabilities(results: Mapping[str, TaskPredictions], metrics: Sequence[ClassificationMetric]) -> bool:
	"""请求 auroc 时每个任务头均已提供 probabilities"""
	return 'auroc' not in metrics or all(res.probabilities is not None for res in results.values())


@icontract.require(
	lambda results, metrics: _has_auroc_probabilities(results, metrics),
	'Task missing probabilities required for AUROC calculation',
)
@icontract.require(
	lambda results, num_classes: results.keys() == num_classes.keys(),
	'Mismatch between results tasks and num_classes tasks',
)
def calculate_multihead_metrics(
	results: Mapping[str, TaskPredictions],
	num_classes: Mapping[str, int],
	metrics: Sequence[ClassificationMetric],
) -> dict[str, dict[str, float]]:
	"""根据各任务预测结果计算分类评估指标，输出键为任务名称的指标字典。"""
	need_fast_calc = any(m in {'acc', 'precision', 'recall', 'f1'} for m in metrics)
	all_metrics: dict[str, dict[str, float]] = {}

	for task_name, task_pred in results.items():
		n_cls = num_classes[task_name]
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

		all_metrics[task_name] = {m: computed[m] for m in metrics}

	return all_metrics


__all__ = [
	'ClassificationMetric',
	'TaskPredictions',
	'calculate_multihead_metrics',
]
