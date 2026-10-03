"""按组（如数据集、站点）计算的回归指标，用于排除组间均值差异带来的相关。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from daisy.typing import VectorF64, VectorI64

from .regression import evaluate_regression_scores, pc_score


def _as_grouped(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> tuple[VectorF64, VectorF64, VectorI64]:
	true = np.asarray(y_true, dtype=np.float64).ravel()
	pred = np.asarray(y_pred, dtype=np.float64).ravel()
	grp = np.asarray(groups, dtype=np.int64).ravel()
	if not (true.shape == pred.shape == grp.shape):
		raise ValueError(f'y_true, y_pred and groups must have the same length, got {true.shape}, {pred.shape}, {grp.shape}')
	return true, pred, grp


def within_group_pc(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
	"""在各组内分别计算 Pearson 相关系数，再取各组的平均。groups 为整数组标签。"""
	true, pred, grp = _as_grouped(y_true, y_pred, groups)
	return float(np.mean([pc_score(true[grp == g], pred[grp == g]) for g in np.unique(grp)]))


def within_group_centered_pc(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
	"""在各组内分别减去组均值后，合并全部样本计算 Pearson 相关系数。groups 为整数组标签。"""
	true, pred, grp = _as_grouped(y_true, y_pred, groups)
	true, pred = true.copy(), pred.copy()
	for g in np.unique(grp):
		mask = grp == g
		true[mask] -= true[mask].mean()
		pred[mask] -= pred[mask].mean()
	return float(pc_score(true, pred))


def grouped_regression_scores(
	y_true: ArrayLike,
	y_pred: ArrayLike,
	groups: ArrayLike,
	group_names: Sequence[str],
) -> dict[str, float]:
	"""全体与各组的 pc / icc / mae / rmse，以及组内 pc、icc 的平均（pc_within / icc_within）。

	groups 为 0..len(group_names)-1 的组下标；只对实际出现的组计算并参与平均。
	"""
	true, pred, grp = _as_grouped(y_true, y_pred, groups)
	present = np.unique(grp)
	if present.min() < 0 or present.max() >= len(group_names):
		raise ValueError(f'groups must lie in [0, {len(group_names) - 1}]')

	overall = evaluate_regression_scores(pred, true)
	scores = {'pc': float(overall['pc']), 'icc': float(overall['icc_a1']), 'mae': float(overall['mae']), 'rmse': float(overall['rmse'])}
	for g in present:
		mask = grp == g
		sub = evaluate_regression_scores(pred[mask], true[mask])
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
