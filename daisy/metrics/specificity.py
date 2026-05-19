from __future__ import annotations

from numbers import Real
from typing import Any, Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.multiclass import type_of_target, unique_labels

AverageMode = Literal['binary', 'micro', 'macro', 'weighted', 'samples'] | None
SpecificityResult = float | NDArray[np.float64]


def _resolve_zero_division(zero_division: Literal['warn'] | int | float) -> float:
	if zero_division == 'warn':
		return 0.0

	if isinstance(zero_division, Real):
		value = float(zero_division)
		if value in {0.0, 1.0} or np.isnan(value):
			return value

	raise ValueError("zero_division must be one of {'warn', 0, 1, np.nan}")


def _nanaverage(values: NDArray[np.float64], weights: NDArray[np.float64] | None = None) -> float:
	if values.shape[0] == 0:
		return float(np.nan)

	mask = np.isnan(values)
	if bool(np.all(mask)):
		return float(np.nan)

	if weights is None:
		return float(np.nanmean(values))

	masked_values = values[~mask]
	masked_weights = weights[~mask]

	try:
		return float(np.average(masked_values, weights=masked_weights))
	except ZeroDivisionError:
		return float(np.average(masked_values))


def _warn_undefined_specificity(undefined_count: int, average: AverageMode) -> None:
	if undefined_count <= 0:
		return

	if average == 'samples':
		message = 'Specificity is ill-defined and being set to 0.0 in samples with no negative labels. Use `zero_division` parameter to control this behavior.'
	elif average == 'binary' or undefined_count == 1:
		message = (
			'Specificity is ill-defined and being set to 0.0 due to no negative samples. Use `zero_division` parameter to control this behavior.'
		)
	else:
		message = 'Specificity is ill-defined and being set to 0.0 in labels with no negative samples. Use `zero_division` parameter to control this behavior.'

	warnings.warn(message, UndefinedMetricWarning, stacklevel=3)


def _safe_specificity_divide(
	tn_sum: NDArray[np.float64],
	negative_sum: NDArray[np.float64],
	*,
	average: AverageMode,
	zero_division: Literal['warn'] | int | float,
) -> NDArray[np.float64]:
	zero_division_value = _resolve_zero_division(zero_division)
	result = np.divide(
		tn_sum,
		negative_sum,
		out=np.full_like(tn_sum, zero_division_value, dtype=np.float64),
		where=negative_sum != 0,
	)

	undefined_mask = negative_sum == 0
	if zero_division == 'warn' and bool(np.any(undefined_mask)):
		_warn_undefined_specificity(int(np.count_nonzero(undefined_mask)), average)

	return result


def _resolve_labels(
	y_true: ArrayLike,
	y_pred: ArrayLike,
	*,
	average: AverageMode,
	labels: ArrayLike | None,
	pos_label: Any,
) -> ArrayLike | None:
	average_options = (None, 'micro', 'macro', 'weighted', 'samples')
	if average not in average_options and average != 'binary':
		raise ValueError('average has to be one of ' + str(average_options))

	present_labels = np.asarray(unique_labels(y_true, y_pred), dtype=object).tolist()
	y_type = type_of_target(y_true)

	if average == 'binary':
		if y_type == 'binary':
			if pos_label not in present_labels and len(present_labels) >= 2:
				raise ValueError(f'pos_label={pos_label} is not a valid label. It should be one of {present_labels}')
			return [pos_label]

		average_choices = list(average_options)
		if y_type == 'multiclass':
			average_choices.remove('samples')
		raise ValueError(f"Target is {y_type} but average='binary'. Please choose another average setting, one of {average_choices}.")

	if pos_label not in (None, 1):
		warnings.warn(
			f"Note that pos_label (set to {pos_label!r}) is ignored when average != 'binary' (got {average!r}). "
			'You may use labels=[pos_label] to specify a single positive class.',
			UserWarning,
			stacklevel=3,
		)

	return labels


def specificity_score(
	y_true: ArrayLike,
	y_pred: ArrayLike,
	*,
	labels: ArrayLike | None = None,
	pos_label: Any = 1,
	average: AverageMode = 'binary',
	sample_weight: ArrayLike | None = None,
	zero_division: Literal['warn'] | int | float = 'warn',
) -> SpecificityResult:
	"""计算特异度（true negative rate）。"""

	# 1) 对齐 sklearn 风格的参数校验与标签选择。
	target_type = type_of_target(y_true)
	if average == 'samples' and not target_type.startswith('multilabel'):
		raise ValueError('Samplewise metrics are not available outside of multilabel classification.')

	resolved_labels = _resolve_labels(y_true, y_pred, average=average, labels=labels, pos_label=pos_label)

	# 2) 基于 one-vs-rest confusion matrix 统计 tn / (tn + fp)。
	mcm = multilabel_confusion_matrix(
		y_true,
		y_pred,
		labels=resolved_labels,
		sample_weight=sample_weight,
		samplewise=average == 'samples',
	)

	tn_sum = np.asarray(mcm[:, 0, 0], dtype=np.float64)
	fp_sum = np.asarray(mcm[:, 0, 1], dtype=np.float64)
	true_sum = np.asarray(mcm[:, 1, 1] + mcm[:, 1, 0], dtype=np.float64)

	if average == 'micro':
		tn_sum = np.asarray([np.sum(tn_sum)], dtype=np.float64)
		fp_sum = np.asarray([np.sum(fp_sum)], dtype=np.float64)
		true_sum = np.asarray([np.sum(true_sum)], dtype=np.float64)

	specificity = _safe_specificity_divide(
		tn_sum,
		tn_sum + fp_sum,
		average=average,
		zero_division=zero_division,
	)

	# 3) 根据 average 返回单值或逐标签结果。
	if average == 'weighted':
		weights = true_sum
	elif average == 'samples' and sample_weight is not None:
		weights = np.asarray(sample_weight, dtype=np.float64)
	else:
		weights = None

	if average is not None:
		return _nanaverage(specificity, weights=weights)

	return specificity


__all__ = ['specificity_score']
