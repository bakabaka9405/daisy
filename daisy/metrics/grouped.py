"""按组（如数据集、站点）计算的回归指标，用于排除组间均值差异带来的相关。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from daisy.typing import VectorF64, VectorI64

from .regression import evaluate_regression_scores, pc_score


def _check_grouped(y_true: VectorF64, y_pred: VectorF64, groups: VectorI64) -> None:
	if not (y_true.shape == y_pred.shape == groups.shape):
		raise ValueError(f'y_true, y_pred and groups must have the same length, got {y_true.shape}, {y_pred.shape}, {groups.shape}')


def within_group_pc(y_true: VectorF64, y_pred: VectorF64, groups: VectorI64) -> float:
	"""在各组内分别计算 Pearson 相关系数，再取各组的平均。groups 为整数组标签。"""
	_check_grouped(y_true, y_pred, groups)
	return float(np.mean([pc_score(y_true[groups == g], y_pred[groups == g]) for g in np.unique(groups)]))


def within_group_centered_pc(y_true: VectorF64, y_pred: VectorF64, groups: VectorI64) -> float:
	"""在各组内分别减去组均值后，合并全部样本计算 Pearson 相关系数。groups 为整数组标签。"""
	_check_grouped(y_true, y_pred, groups)
	true, pred = y_true.copy(), y_pred.copy()
	for g in np.unique(groups):
		mask = groups == g
		true[mask] -= true[mask].mean()
		pred[mask] -= pred[mask].mean()
	return float(pc_score(true, pred))


def grouped_regression_scores(
	y_true: VectorF64,
	y_pred: VectorF64,
	groups: VectorI64,
	group_names: Sequence[str],
) -> dict[str, float]:
	"""全体与各组的 pc / icc / mae / rmse，以及组内 pc、icc 的平均（pc_within / icc_within）。

	groups 为 0..len(group_names)-1 的组下标；只对实际出现的组计算并参与平均。
	"""
	_check_grouped(y_true, y_pred, groups)
	present = np.unique(groups)
	if present.min() < 0 or present.max() >= len(group_names):
		raise ValueError(f'groups must lie in [0, {len(group_names) - 1}]')

	overall = evaluate_regression_scores(y_pred, y_true)
	scores = {'pc': float(overall['pc']), 'icc': float(overall['icc_a1']), 'mae': float(overall['mae']), 'rmse': float(overall['rmse'])}
	for g in present:
		mask = groups == g
		sub = evaluate_regression_scores(y_pred[mask], y_true[mask])
		name = group_names[g]
		scores[f'pc_{name}'] = float(sub['pc'])
		scores[f'icc_{name}'] = float(sub['icc_a1'])
		scores[f'mae_{name}'] = float(sub['mae'])
	scores['pc_within'] = float(np.mean([scores[f'pc_{group_names[g]}'] for g in present]))
	scores['icc_within'] = float(np.mean([scores[f'icc_{group_names[g]}'] for g in present]))
	return scores


__all__ = [
	'grouped_regression_scores',
	'within_group_centered_pc',
	'within_group_pc',
]
