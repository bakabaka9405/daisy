from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import auc, roc_curve


@dataclass(slots=True, frozen=True)
class RocCurve:
	fpr: NDArray[np.float64]
	tpr: NDArray[np.float64]
	auc: float


def _validate_inputs(
	y_true: ArrayLike,
	y_score: ArrayLike,
	num_classes: int | None,
) -> tuple[NDArray[np.int64], NDArray[np.float64], int]:
	y_true_array = np.asarray(y_true, dtype=np.int64).reshape(-1)
	y_score_array = np.asarray(y_score, dtype=np.float64)

	if y_true_array.size == 0:
		raise ValueError('y_true cannot be empty')

	if y_score_array.ndim != 2:
		raise ValueError('y_score must be a 2D array with shape (n_samples, num_classes)')

	if y_score_array.shape[0] != y_true_array.shape[0]:
		raise ValueError('y_true and y_score must contain the same number of samples')

	resolved_num_classes = int(y_score_array.shape[1]) if num_classes is None else int(num_classes)
	if resolved_num_classes < 2:
		raise ValueError('num_classes must be at least 2')

	if y_score_array.shape[1] != resolved_num_classes:
		raise ValueError(f'y_score provides {y_score_array.shape[1]} class columns, but num_classes={resolved_num_classes}')

	invalid_mask = (y_true_array < 0) | (y_true_array >= resolved_num_classes)
	if bool(np.any(invalid_mask)):
		invalid_labels = np.unique(y_true_array[invalid_mask]).tolist()
		raise ValueError(f'y_true contains labels outside [0, {resolved_num_classes - 1}]: {invalid_labels}')

	return y_true_array, y_score_array, resolved_num_classes


def _compute_binary_roc_curve(
	binary_true: NDArray[np.int64],
	scores: NDArray[np.float64],
) -> RocCurve | None:
	positive_count = int(np.count_nonzero(binary_true))
	negative_count = int(binary_true.size - positive_count)
	if positive_count == 0 or negative_count == 0:
		return None

	fpr, tpr, _ = roc_curve(binary_true, scores)
	fpr_array = np.asarray(fpr, dtype=np.float64)
	tpr_array = np.asarray(tpr, dtype=np.float64)
	return RocCurve(fpr=fpr_array, tpr=tpr_array, auc=float(auc(fpr_array, tpr_array)))


def _compute_ovr_curves(
	y_true: NDArray[np.int64],
	y_score: NDArray[np.float64],
	num_classes: int,
) -> dict[int, RocCurve]:
	curves: dict[int, RocCurve] = {}

	for class_index in range(num_classes):
		binary_true = (y_true == class_index).astype(np.int64)
		curve = _compute_binary_roc_curve(binary_true, y_score[:, class_index])
		if curve is not None:
			curves[class_index] = curve

	return curves


def _compute_micro_curve(
	y_true: NDArray[np.int64],
	y_score: NDArray[np.float64],
	num_classes: int,
) -> RocCurve:
	binary_true = np.zeros((y_true.shape[0], num_classes), dtype=np.int64)
	binary_true[np.arange(y_true.shape[0]), y_true] = 1

	micro_curve = _compute_binary_roc_curve(binary_true.ravel(), y_score.ravel())
	if micro_curve is None:
		raise ValueError('micro ROC curve is undefined for the current labels')

	return micro_curve


def _compute_macro_curve(ovr_curves: dict[int, RocCurve]) -> RocCurve:
	all_fpr = np.unique(np.concatenate([curve.fpr for curve in ovr_curves.values()]))
	mean_tpr = np.zeros_like(all_fpr)

	for curve in ovr_curves.values():
		mean_tpr += np.interp(all_fpr, curve.fpr, curve.tpr)

	mean_tpr /= float(len(ovr_curves))
	return RocCurve(fpr=all_fpr, tpr=mean_tpr, auc=float(auc(all_fpr, mean_tpr)))


def _compute_ovo_curves(
	y_true: NDArray[np.int64],
	y_score: NDArray[np.float64],
	num_classes: int,
) -> dict[tuple[int, int], RocCurve]:
	curves: dict[tuple[int, int], RocCurve] = {}

	for first_class in range(num_classes):
		for second_class in range(first_class + 1, num_classes):
			pair_mask = (y_true == first_class) | (y_true == second_class)
			if int(np.count_nonzero(pair_mask)) < 2:
				continue

			pair_true = (y_true[pair_mask] == first_class).astype(np.int64)
			pair_scores = y_score[pair_mask, first_class] - y_score[pair_mask, second_class]

			curve = _compute_binary_roc_curve(pair_true, pair_scores)
			if curve is not None:
				curves[(first_class, second_class)] = curve

	return curves


def compute_multiclass_roc_curves(
	y_true: ArrayLike,
	y_score: ArrayLike,
	*,
	num_classes: int | None = None,
	mode: Literal['ovr', 'micro', 'macro', 'all', 'ovo'] = 'all',
) -> dict[str, RocCurve]:
	"""计算多分类 ROC 曲线及其 AUC。"""

	allowed_modes = {'ovr', 'micro', 'macro', 'all', 'ovo'}
	if mode not in allowed_modes:
		raise ValueError(f'mode must be one of {sorted(allowed_modes)}')

	y_true_array, y_score_array, resolved_num_classes = _validate_inputs(y_true, y_score, num_classes)
	curves: dict[str, RocCurve] = {}

	ovr_curves: dict[int, RocCurve] = {}
	if mode in {'ovr', 'macro', 'all'}:
		ovr_curves = _compute_ovr_curves(y_true_array, y_score_array, resolved_num_classes)
		if not ovr_curves:
			raise ValueError('No valid one-vs-rest ROC curve can be computed from current labels')

	if mode in {'ovr', 'all'}:
		for class_index in sorted(ovr_curves):
			curves[f'class_{class_index}'] = ovr_curves[class_index]

	if mode in {'micro', 'all'}:
		curves['micro'] = _compute_micro_curve(y_true_array, y_score_array, resolved_num_classes)

	if mode in {'macro', 'all'}:
		curves['macro'] = _compute_macro_curve(ovr_curves)

	if mode == 'ovo':
		ovo_curves = _compute_ovo_curves(y_true_array, y_score_array, resolved_num_classes)
		if not ovo_curves:
			raise ValueError('No valid one-vs-one ROC curve can be computed from current labels')

		for (first_class, second_class), curve in sorted(ovo_curves.items()):
			curves[f'ovo_{first_class}_vs_{second_class}'] = curve

	return curves


def auroc_score(
	y_true: ArrayLike,
	y_score: ArrayLike,
	*,
	num_classes: int | None = None,
	mode: Literal['ovr', 'micro', 'macro', 'all', 'ovo'] = 'macro',
) -> float | dict[str, float]:
	"""计算多分类 AUROC 指标。"""

	curves = compute_multiclass_roc_curves(y_true, y_score, num_classes=num_classes, mode=mode)
	auc_scores = {curve_key: curve.auc for curve_key, curve in curves.items()}

	if mode == 'ovo' and auc_scores:
		auc_scores['ovo_mean'] = float(np.mean(list(auc_scores.values())))

	if mode in {'micro', 'macro'}:
		return auc_scores[mode]

	return auc_scores


__all__ = ['RocCurve', 'auroc_score', 'compute_multiclass_roc_curves']
