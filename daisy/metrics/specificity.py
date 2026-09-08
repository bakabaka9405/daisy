from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.multiclass import type_of_target, unique_labels  # type: ignore

from daisy.typing import MatrixI64, VectorF64, VectorI64

AverageMode = Literal['binary', 'micro', 'macro', 'weighted', 'samples'] | None


def _nanaverage(values: VectorF64, weights: VectorF64 | None = None) -> np.float64:
	mask = np.isnan(values)
	if mask.all():
		return np.float64('nan')
	if weights is None:
		return np.nanmean(values)

	masked_values = values[~mask]
	masked_weights = weights[~mask]
	try:
		return np.average(masked_values, weights=masked_weights)
	except ZeroDivisionError:
		return np.average(masked_values)


def _safe_specificity_divide(
	tn_sum: VectorF64,
	negative_sum: VectorF64,
	*,
	zero_division: Literal['warn', 0, 1, 'nan'],
) -> VectorF64:
	zero_division_value = np.float64(0 if zero_division == 'warn' else zero_division)
	result = np.divide(
		tn_sum,
		negative_sum,
		out=np.full_like(tn_sum, zero_division_value, dtype=np.float64),
		where=negative_sum != 0,
	)
	if zero_division == 'warn' and np.any(negative_sum == 0):
		warnings.warn('Undefined specificity.', UndefinedMetricWarning, stacklevel=3)
	return result


def _resolve_labels(
	y_true: VectorI64 | MatrixI64,
	y_pred: VectorI64 | MatrixI64,
	*,
	average: AverageMode,
	labels: VectorI64 | None,
	pos_label: int,
) -> VectorI64 | None:
	if average not in {None, 'binary', 'micro', 'macro', 'weighted', 'samples'}:
		raise ValueError('average 无效')

	if average == 'binary':
		if type_of_target(y_true) == 'binary':
			present_labels = unique_labels(y_true, y_pred)
			if present_labels.size >= 2 and not np.any(present_labels == pos_label):
				raise ValueError('pos_label 不是有效标签')
			return np.array([pos_label], dtype=np.int64)
		raise ValueError("非二分类目标不能使用 average='binary'")

	if pos_label != 1:
		warnings.warn("average 不是 'binary' 时将忽略 pos_label。", UserWarning, stacklevel=3)
	return labels


def specificity_score(
	y_true: VectorI64 | MatrixI64,
	y_pred: VectorI64 | MatrixI64,
	*,
	labels: VectorI64 | None = None,
	pos_label: int = 1,
	average: AverageMode = 'binary',
	sample_weight: VectorF64 | None = None,
	zero_division: Literal['warn', 0, 1, 'nan'] = 'warn',
) -> np.float64 | VectorF64:
	"""计算特异度（真负例率）。"""

	resolved_labels = _resolve_labels(y_true, y_pred, average=average, labels=labels, pos_label=pos_label)
	mcm = multilabel_confusion_matrix(
		y_true,
		y_pred,
		labels=resolved_labels,
		sample_weight=sample_weight,
		samplewise=average == 'samples',
	)

	tn_sum = mcm[:, 0, 0].astype(np.float64, copy=False)
	fp_sum = mcm[:, 0, 1].astype(np.float64, copy=False)
	if average == 'micro':
		tn_sum = tn_sum.sum(keepdims=True)
		fp_sum = fp_sum.sum(keepdims=True)

	specificity = _safe_specificity_divide(tn_sum, tn_sum + fp_sum, zero_division=zero_division)
	if average is None:
		return specificity

	if average == 'weighted':
		weights = (mcm[:, 1, 1] + mcm[:, 1, 0]).astype(np.float64, copy=False)
	elif average == 'samples':
		weights = sample_weight
	else:
		weights = None
	return _nanaverage(specificity, weights=weights)
