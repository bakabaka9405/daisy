from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from sklearn.metrics import roc_curve

from daisy.typing import MatrixF64, VectorF64, VectorI64

RocMode = Literal['ovr', 'micro', 'macro', 'all', 'ovo']


@dataclass(slots=True, frozen=True)
class RocCurve:
	fpr: VectorF64
	tpr: VectorF64
	auc: np.float64


def _validate_inputs(y_true: VectorI64, y_score: MatrixF64) -> None:
	if y_true.size != y_score.shape[0]:
		raise ValueError('标签与分数的样本数必须一致')
	if y_score.shape[1] < 2:
		raise ValueError('类别数必须至少为 2')
	if not np.all((y_true >= 0) & (y_true < y_score.shape[1])):
		raise ValueError('标签必须位于类别编号范围内')


def _compute_binary_roc_curve(
	positive_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
	scores: VectorF64,
) -> RocCurve | None:
	if not np.any(positive_mask) or np.all(positive_mask):
		return None

	fpr, tpr, _ = roc_curve(positive_mask, scores)
	return RocCurve(fpr=fpr, tpr=tpr, auc=cast(np.float64, np.trapezoid(tpr, fpr)))


def _compute_ovr_curves(y_true: VectorI64, y_score: MatrixF64) -> dict[int, RocCurve]:
	curves: dict[int, RocCurve] = {}
	_, num_classes = y_score.shape
	for class_index in range(num_classes):
		curve = _compute_binary_roc_curve(y_true == class_index, y_score[:, class_index])
		if curve is not None:
			curves[class_index] = curve
	return curves


def _compute_micro_curve(y_true: VectorI64, y_score: MatrixF64) -> RocCurve:
	n_samples, num_classes = y_score.shape
	binary_true = np.zeros((n_samples, num_classes), dtype=np.bool_)
	binary_true[np.arange(n_samples), y_true] = True
	micro_curve = _compute_binary_roc_curve(binary_true.ravel(), y_score.ravel())
	if micro_curve is None:
		raise ValueError('当前标签无法定义 micro ROC')
	return micro_curve


def _compute_macro_curve(ovr_curves: dict[int, RocCurve]) -> RocCurve:
	all_fpr = np.unique(np.concatenate([curve.fpr for curve in ovr_curves.values()]))
	mean_tpr = np.zeros_like(all_fpr)
	for curve in ovr_curves.values():
		mean_tpr += np.interp(all_fpr, curve.fpr, curve.tpr)
	mean_tpr /= len(ovr_curves)
	return RocCurve(fpr=all_fpr, tpr=mean_tpr, auc=cast(np.float64, np.trapezoid(mean_tpr, all_fpr)))


def _compute_ovo_curves(y_true: VectorI64, y_score: MatrixF64) -> dict[tuple[int, int], RocCurve]:
	curves: dict[tuple[int, int], RocCurve] = {}
	_, num_classes = y_score.shape
	for first_class in range(num_classes):
		for second_class in range(first_class + 1, num_classes):
			pair_mask = (y_true == first_class) | (y_true == second_class)
			pair_true = y_true[pair_mask] == first_class
			pair_scores = y_score[pair_mask, first_class] - y_score[pair_mask, second_class]
			curve = _compute_binary_roc_curve(pair_true, pair_scores)
			if curve is not None:
				curves[(first_class, second_class)] = curve
	return curves


def compute_multiclass_roc_curves(
	y_true: VectorI64,
	y_score: MatrixF64,
	*,
	mode: RocMode = 'all',
) -> dict[str, RocCurve]:
	"""由 (N,) 整数标签与 (N, C) 分数计算多分类 ROC。"""

	if mode not in {'ovr', 'micro', 'macro', 'all', 'ovo'}:
		raise ValueError('mode 无效')
	_validate_inputs(y_true, y_score)

	curves: dict[str, RocCurve] = {}
	ovr_curves: dict[int, RocCurve] = {}
	if mode in {'ovr', 'macro', 'all'}:
		ovr_curves = _compute_ovr_curves(y_true, y_score)
		if not ovr_curves:
			raise ValueError('当前标签无法计算 one-vs-rest ROC')
		if mode in {'ovr', 'all'}:
			for class_index, curve in ovr_curves.items():
				curves[f'class_{class_index}'] = curve

	if mode in {'micro', 'all'}:
		curves['micro'] = _compute_micro_curve(y_true, y_score)

	if mode in {'macro', 'all'}:
		curves['macro'] = _compute_macro_curve(ovr_curves)

	if mode == 'ovo':
		ovo_curves = _compute_ovo_curves(y_true, y_score)
		if not ovo_curves:
			raise ValueError('当前标签无法计算 one-vs-one ROC')
		for (first_class, second_class), curve in ovo_curves.items():
			curves[f'ovo_{first_class}_vs_{second_class}'] = curve

	return curves


def auroc_score(
	y_true: VectorI64,
	y_score: MatrixF64,
	*,
	mode: RocMode = 'macro',
) -> np.float64 | dict[str, np.float64]:
	"""计算多分类 AUROC。"""

	curves = compute_multiclass_roc_curves(y_true, y_score, mode=mode)
	auc_scores = {curve_key: curve.auc for curve_key, curve in curves.items()}
	if mode == 'ovo':
		ovo_mean: np.float64 = np.mean(np.stack(list(auc_scores.values())))
		auc_scores['ovo_mean'] = ovo_mean
	if mode in {'micro', 'macro'}:
		return auc_scores[mode]
	return auc_scores


__all__ = ['RocCurve', 'RocMode', 'auroc_score', 'compute_multiclass_roc_curves']
