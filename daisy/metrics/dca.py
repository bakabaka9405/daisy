from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

DcaMode = Literal['ovr', 'micro', 'macro', 'all']


@dataclass(slots=True, frozen=True)
class DcaCurve:
	thresholds: NDArray[np.float64]
	net_benefit: NDArray[np.float64]
	prevalence: float


def _validate_inputs(
	y_true: ArrayLike,
	y_prob: ArrayLike,
	num_classes: int | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
	y_true_array = np.asarray(y_true, dtype=np.int64).reshape(-1)
	y_prob_array = np.asarray(y_prob, dtype=np.float64)

	if y_true_array.size == 0:
		raise ValueError('y_true cannot be empty')

	if y_prob_array.ndim != 2:
		raise ValueError('y_prob must be a 2D array with shape (n_samples, num_classes)')

	if y_prob_array.shape[0] != y_true_array.shape[0]:
		raise ValueError('y_true and y_prob must contain the same number of samples')

	resolved_num_classes = int(y_prob_array.shape[1]) if num_classes is None else int(num_classes)
	if resolved_num_classes < 2:
		raise ValueError('num_classes must be at least 2')

	if y_prob_array.shape[1] != resolved_num_classes:
		raise ValueError(f'y_prob provides {y_prob_array.shape[1]} class columns, but num_classes={resolved_num_classes}')

	invalid_mask = (y_true_array < 0) | (y_true_array >= resolved_num_classes)
	if bool(np.any(invalid_mask)):
		invalid_labels = np.unique(y_true_array[invalid_mask]).tolist()
		raise ValueError(f'y_true contains labels outside [0, {resolved_num_classes - 1}]: {invalid_labels}')

	if bool(np.any(~np.isfinite(y_prob_array))):
		raise ValueError('y_prob must contain only finite values')

	if bool(np.any((y_prob_array < 0.0) | (y_prob_array > 1.0))):
		raise ValueError('y_prob must contain probabilities inside [0, 1]')

	row_sums = np.sum(y_prob_array, axis=1)
	if not bool(np.allclose(row_sums, 1.0, atol=1e-6, rtol=1e-6)):
		raise ValueError('Each y_prob row must sum to 1')

	return y_true_array, y_prob_array, resolved_num_classes


def _validate_thresholds(thresholds: ArrayLike | None) -> NDArray[np.float64]:
	if thresholds is None:
		return np.linspace(0.01, 0.99, num=99, dtype=np.float64)

	threshold_array = np.asarray(thresholds, dtype=np.float64).reshape(-1)
	if threshold_array.size == 0:
		raise ValueError('thresholds cannot be empty')

	if bool(np.any(~np.isfinite(threshold_array))):
		raise ValueError('thresholds must contain only finite values')

	if bool(np.any((threshold_array <= 0.0) | (threshold_array >= 1.0))):
		raise ValueError('thresholds must be strictly inside (0, 1)')

	return np.unique(threshold_array)


def _compute_binary_dca_curve(
	binary_true: NDArray[np.int64],
	probabilities: NDArray[np.float64],
	thresholds: NDArray[np.float64],
) -> DcaCurve:
	positive_mask = binary_true.astype(np.bool_)
	predicted_positive = probabilities[:, np.newaxis] >= thresholds[np.newaxis, :]

	true_positive = np.count_nonzero(predicted_positive & positive_mask[:, np.newaxis], axis=0)
	false_positive = np.count_nonzero(predicted_positive & ~positive_mask[:, np.newaxis], axis=0)

	sample_count = float(binary_true.size)
	threshold_odds = thresholds / (1.0 - thresholds)
	net_benefit = true_positive / sample_count - false_positive / sample_count * threshold_odds

	return DcaCurve(
		thresholds=thresholds,
		net_benefit=np.asarray(net_benefit, dtype=np.float64),
		prevalence=float(np.mean(positive_mask)),
	)


def _compute_ovr_curves(
	y_true: NDArray[np.int64],
	y_prob: NDArray[np.float64],
	num_classes: int,
	thresholds: NDArray[np.float64],
) -> dict[int, DcaCurve]:
	curves: dict[int, DcaCurve] = {}

	for class_index in range(num_classes):
		binary_true = (y_true == class_index).astype(np.int64)
		curves[class_index] = _compute_binary_dca_curve(binary_true, y_prob[:, class_index], thresholds)

	return curves


def _compute_micro_curve(
	y_true: NDArray[np.int64],
	y_prob: NDArray[np.float64],
	num_classes: int,
	thresholds: NDArray[np.float64],
) -> DcaCurve:
	binary_true = np.zeros((y_true.shape[0], num_classes), dtype=np.int64)
	binary_true[np.arange(y_true.shape[0]), y_true] = 1
	return _compute_binary_dca_curve(binary_true.ravel(), y_prob.ravel(), thresholds)


def _compute_macro_curve(ovr_curves: dict[int, DcaCurve]) -> DcaCurve:
	curve_values = list(ovr_curves.values())
	thresholds = curve_values[0].thresholds
	mean_net_benefit = np.mean(np.stack([curve.net_benefit for curve in curve_values], axis=0), axis=0)
	mean_prevalence = float(np.mean([curve.prevalence for curve in curve_values]))

	return DcaCurve(
		thresholds=thresholds,
		net_benefit=np.asarray(mean_net_benefit, dtype=np.float64),
		prevalence=mean_prevalence,
	)


def compute_multiclass_dca_curves(
	y_true: ArrayLike,
	y_prob: ArrayLike,
	*,
	num_classes: int | None = None,
	mode: DcaMode = 'all',
	thresholds: ArrayLike | None = None,
) -> dict[str, DcaCurve]:
	"""计算多分类 DCA 曲线。"""

	allowed_modes = {'ovr', 'micro', 'macro', 'all'}
	if mode not in allowed_modes:
		raise ValueError(f'mode must be one of {sorted(allowed_modes)}')

	y_true_array, y_prob_array, resolved_num_classes = _validate_inputs(y_true, y_prob, num_classes)
	threshold_array = _validate_thresholds(thresholds)
	curves: dict[str, DcaCurve] = {}

	ovr_curves: dict[int, DcaCurve] = {}
	if mode in {'ovr', 'macro', 'all'}:
		ovr_curves = _compute_ovr_curves(y_true_array, y_prob_array, resolved_num_classes, threshold_array)

	if mode in {'ovr', 'all'}:
		for class_index in sorted(ovr_curves):
			curves[f'class_{class_index}'] = ovr_curves[class_index]

	if mode in {'micro', 'all'}:
		curves['micro'] = _compute_micro_curve(y_true_array, y_prob_array, resolved_num_classes, threshold_array)

	if mode in {'macro', 'all'}:
		curves['macro'] = _compute_macro_curve(ovr_curves)

	return curves


__all__ = ['DcaCurve', 'DcaMode', 'compute_multiclass_dca_curves']
