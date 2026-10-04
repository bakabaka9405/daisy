from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import icontract
import numpy as np
from sklearn.metrics import roc_curve

from daisy.typing import MatrixF64, VectorF64, VectorI64

RocMode = Literal['ovr', 'micro', 'macro', 'all', 'ovo']

type BoolVector = np.ndarray[tuple[int], np.dtype[np.bool_]]


@dataclass(slots=True, frozen=True)
class RocCurve:
	fpr: VectorF64
	tpr: VectorF64
	auc: np.float64


def _has_both_classes(positive_mask: BoolVector) -> bool:
	return bool(np.any(positive_mask) and not np.all(positive_mask))


@icontract.require(lambda positive_mask: _has_both_classes(positive_mask), '正负样本都必须存在')
@icontract.require(lambda positive_mask, scores: positive_mask.shape == scores.shape)
@icontract.ensure(lambda result: 0 <= result.auc <= 1)
def _compute_binary_roc_curve(positive_mask: BoolVector, scores: VectorF64) -> RocCurve:
	fpr, tpr, _ = roc_curve(positive_mask, scores)
	return RocCurve(fpr=fpr, tpr=tpr, auc=cast(np.float64, np.trapezoid(tpr, fpr)))


def _compute_ovr_curves(y_true: VectorI64, y_score: MatrixF64) -> dict[int, RocCurve]:
	_, num_classes = y_score.shape
	return {
		class_index: _compute_binary_roc_curve(y_true == class_index, y_score[:, class_index])
		for class_index in range(num_classes)
		if _has_both_classes(y_true == class_index)
	}


def _compute_micro_curve(y_true: VectorI64, y_score: MatrixF64) -> RocCurve:
	n_samples, num_classes = y_score.shape
	binary_true = np.zeros((n_samples, num_classes), dtype=np.bool_)
	binary_true[np.arange(n_samples), y_true] = True
	return _compute_binary_roc_curve(binary_true.ravel(), y_score.ravel())


@icontract.require(lambda ovr_curves: len(ovr_curves) > 0)
def _compute_macro_curve(ovr_curves: dict[int, RocCurve]) -> RocCurve:
	all_fpr = np.unique(np.concatenate([curve.fpr for curve in ovr_curves.values()]))
	mean_tpr = np.zeros_like(all_fpr)
	for curve in ovr_curves.values():
		mean_tpr += np.interp(all_fpr, curve.fpr, curve.tpr)
	mean_tpr /= len(ovr_curves)
	macro_auc = cast(np.float64, np.mean([curve.auc for curve in ovr_curves.values()]))
	return RocCurve(fpr=all_fpr, tpr=mean_tpr, auc=macro_auc)


def _compute_ovo_curves(y_true: VectorI64, y_score: MatrixF64) -> dict[tuple[int, int], RocCurve]:
	curves: dict[tuple[int, int], RocCurve] = {}
	_, num_classes = y_score.shape
	for first_class in range(num_classes):
		for second_class in range(first_class + 1, num_classes):
			pair_mask = (y_true == first_class) | (y_true == second_class)
			pair_true = y_true[pair_mask] == first_class
			if _has_both_classes(pair_true):
				pair_scores = y_score[pair_mask, first_class] - y_score[pair_mask, second_class]
				curves[(first_class, second_class)] = _compute_binary_roc_curve(pair_true, pair_scores)
	return curves


@icontract.require(lambda y_true, mode: mode == 'micro' or np.unique(y_true).size >= 2, '除 micro 外至少需要两个类别的样本')
@icontract.require(lambda y_true, y_score: np.all((y_true >= 0) & (y_true < y_score.shape[1])), '标签必须位于类别编号范围内')
@icontract.require(lambda y_score: y_score.shape[1] >= 2, '类别数必须至少为 2')
@icontract.require(lambda y_true, y_score: y_true.shape == (y_score.shape[0],), '标签与分数的样本数必须一致')
@icontract.ensure(lambda result: len(result) > 0)
def compute_multiclass_roc_curves(
	y_true: VectorI64,
	y_score: MatrixF64,
	*,
	mode: RocMode = 'all',
) -> dict[str, RocCurve]:
	"""由 (N,) 整数标签与 (N, C) 分数计算多分类 ROC。"""

	curves: dict[str, RocCurve] = {}
	ovr_curves: dict[int, RocCurve] = {}
	if mode in {'ovr', 'macro', 'all'}:
		ovr_curves = _compute_ovr_curves(y_true, y_score)
		if mode in {'ovr', 'all'}:
			for class_index, curve in ovr_curves.items():
				curves[f'class_{class_index}'] = curve

	if mode in {'micro', 'all'}:
		curves['micro'] = _compute_micro_curve(y_true, y_score)

	if mode in {'macro', 'all'}:
		curves['macro'] = _compute_macro_curve(ovr_curves)

	if mode == 'ovo':
		for (first_class, second_class), curve in _compute_ovo_curves(y_true, y_score).items():
			curves[f'ovo_{first_class}_vs_{second_class}'] = curve

	return curves


@icontract.ensure(lambda result: all(0 <= v <= 1 for v in (result.values() if isinstance(result, dict) else [result])))
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
