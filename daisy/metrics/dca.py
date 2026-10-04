from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import icontract
import numpy as np

from daisy.typing import MatrixF64, VectorF64, VectorI64

DcaMode = Literal['ovr', 'micro', 'macro', 'all']


@dataclass(slots=True, frozen=True)
class DcaCurve:
	thresholds: VectorF64
	net_benefit: VectorF64
	prevalence: np.float64


@icontract.require(lambda positive_mask, probabilities: positive_mask.shape == probabilities.shape)
@icontract.ensure(lambda result: 0 <= result.prevalence <= 1)
@icontract.ensure(lambda thresholds, result: result.net_benefit.shape == thresholds.shape)
def _compute_binary_dca_curve(
	positive_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
	probabilities: VectorF64,
	thresholds: VectorF64,
) -> DcaCurve:
	predicted_positive = probabilities[:, np.newaxis] >= thresholds[np.newaxis, :]

	true_positive = np.sum(predicted_positive & positive_mask[:, np.newaxis], axis=0)
	false_positive = np.sum(predicted_positive & ~positive_mask[:, np.newaxis], axis=0)

	sample_count = positive_mask.size
	threshold_odds = thresholds / (1.0 - thresholds)
	net_benefit = true_positive / sample_count - false_positive / sample_count * threshold_odds

	return DcaCurve(
		thresholds=thresholds,
		net_benefit=net_benefit,
		prevalence=positive_mask.mean(),
	)


@icontract.ensure(lambda y_prob, result: len(result) == y_prob.shape[1])
def _compute_ovr_curves(
	y_true: VectorI64,
	y_prob: MatrixF64,
	thresholds: VectorF64,
) -> dict[int, DcaCurve]:
	curves: dict[int, DcaCurve] = {}

	for class_index in range(y_prob.shape[1]):
		curves[class_index] = _compute_binary_dca_curve(y_true == class_index, y_prob[:, class_index], thresholds)

	return curves


@icontract.require(lambda ovr_curves: len(ovr_curves) > 0)
def _compute_macro_curve(ovr_curves: dict[int, DcaCurve]) -> DcaCurve:
	curve_values = list(ovr_curves.values())
	thresholds = curve_values[0].thresholds
	mean_net_benefit = np.mean(np.stack([curve.net_benefit for curve in curve_values], axis=0), axis=0)
	mean_prevalence = np.mean(np.stack([curve.prevalence for curve in curve_values]))

	return DcaCurve(
		thresholds=thresholds,
		net_benefit=mean_net_benefit,
		prevalence=mean_prevalence,
	)


@icontract.require(lambda thresholds: thresholds is None or np.all(np.diff(thresholds) > 0), '阈值必须严格递增')
@icontract.require(lambda thresholds: thresholds is None or np.all((thresholds > 0) & (thresholds < 1)), '阈值必须位于 (0, 1)')
@icontract.require(lambda thresholds: thresholds is None or thresholds.size > 0, '阈值不能为空')
@icontract.require(lambda y_prob: np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6), '每个样本的类别概率之和必须为 1')
@icontract.require(lambda y_prob: np.all((y_prob >= 0) & (y_prob <= 1)), '概率必须位于 [0, 1]')
@icontract.require(lambda y_true, y_prob: np.all((y_true >= 0) & (y_true < y_prob.shape[1])), '标签必须位于类别编号范围内')
@icontract.require(lambda y_prob: y_prob.shape[1] >= 2, '类别数必须至少为 2')
@icontract.require(lambda y_true, y_prob: y_true.shape == (y_prob.shape[0],), '标签与概率的样本数必须一致')
@icontract.require(lambda y_true: y_true.size > 0, '样本不能为空')
def compute_multiclass_dca_curves(
	y_true: VectorI64,
	y_prob: MatrixF64,
	*,
	mode: DcaMode = 'all',
	thresholds: VectorF64 | None = None,
) -> dict[str, DcaCurve]:
	"""由 (N,) 整数标签、(N, C) 概率及严格递增的阈值计算多分类 DCA。"""

	threshold_array = np.linspace(0.01, 0.99, num=99, dtype=np.float64) if thresholds is None else thresholds

	# 各类别使用相同的样本与阈值计算净获益。
	ovr_curves = _compute_ovr_curves(y_true, y_prob, threshold_array)
	curves: dict[str, DcaCurve] = {}
	if mode in {'ovr', 'all'}:
		for class_index, curve in ovr_curves.items():
			curves[f'class_{class_index}'] = curve

	# 相同样本数及阈值下，micro 与类别净获益的等权平均相等。
	if mode != 'ovr':
		average_curve = _compute_macro_curve(ovr_curves)
		if mode in {'micro', 'all'}:
			curves['micro'] = average_curve
		if mode in {'macro', 'all'}:
			curves['macro'] = average_curve

	return curves


__all__ = ['DcaCurve', 'DcaMode', 'compute_multiclass_dca_curves']
