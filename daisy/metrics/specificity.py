from __future__ import annotations

from typing import Literal
import warnings

import icontract
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


def _resolve_labels(*, average: AverageMode, labels: VectorI64 | None, pos_label: int) -> VectorI64 | None:
	if average == 'binary':
		return np.array([pos_label], dtype=np.int64)

	if pos_label != 1:
		warnings.warn("average 不是 'binary' 时将忽略 pos_label。", UserWarning, stacklevel=3)
	return labels


def _valid_pos_label(y_true: VectorI64 | MatrixI64, y_pred: VectorI64 | MatrixI64, pos_label: int) -> bool:
	present_labels = unique_labels(y_true, y_pred)
	return present_labels.size < 2 or bool(np.any(present_labels == pos_label))


@icontract.require(lambda y_true, sample_weight: sample_weight is None or sample_weight.shape == (y_true.shape[0],))
@icontract.require(
	lambda y_true, y_pred, pos_label, average: average != 'binary' or _valid_pos_label(y_true, y_pred, pos_label), 'pos_label 不是有效标签'
)
@icontract.require(lambda y_true, average: average != 'binary' or type_of_target(y_true) == 'binary', "非二分类目标不能使用 average='binary'")
@icontract.require(lambda y_true, y_pred: y_true.shape == y_pred.shape)
@icontract.ensure(lambda result: np.all(np.isnan(result) | ((np.asarray(result) >= 0) & (np.asarray(result) <= 1))))
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

	resolved_labels = _resolve_labels(average=average, labels=labels, pos_label=pos_label)
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
