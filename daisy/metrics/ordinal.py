"""有序类别（评分）预测的评价指标。"""

from __future__ import annotations

from typing import Literal, overload

import icontract
import numpy as np
from sklearn.metrics import accuracy_score

from daisy.typing import Array3F64, MatrixF64, MatrixI64, VectorF64, VectorI64
from .regression import mae_score, rmse_score


def evaluate_ordinal_scores(y_true: VectorI64, y_pred: VectorI64, num_classes: int) -> dict[str, float]:
	"""由解码后的类别预测计算有序分类指标。"""
	true_f, pred_f = y_true.astype(np.float64), y_pred.astype(np.float64)
	return {
		'mae': float(mae_score(true_f, pred_f)),
		'rmse': float(rmse_score(true_f, pred_f)),
		'acc': float(accuracy_score(y_true, y_pred)),
		'kappa': qwk_score(y_true, y_pred, num_classes),
	}


@icontract.require(lambda y_pred, num_classes: y_pred.min() >= 0 and y_pred.max() < num_classes)
@icontract.require(lambda y_true, num_classes: y_true.min() >= 0 and y_true.max() < num_classes)
@icontract.require(lambda num_classes: num_classes >= 2)
@icontract.require(lambda y_true: y_true.size > 0)
@icontract.require(lambda y_true, y_pred: y_true.shape == y_pred.shape)
@icontract.ensure(lambda result: np.isnan(result) or -1 - 1e-9 <= result <= 1 + 1e-9)
def qwk_score(y_true: VectorI64, y_pred: VectorI64, num_classes: int) -> float:
	"""二次加权 Cohen kappa。标签取值为 0..num_classes-1。

	与 sklearn.metrics.cohen_kappa_score(weights='quadratic', labels=range(num_classes)) 一致。
	"""
	grid = np.arange(num_classes)
	weights = np.subtract.outer(grid, grid) ** 2 / (num_classes - 1) ** 2
	observed = np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes).astype(np.float64)
	expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
	return float(1.0 - (weights * observed).sum() / (weights * expected).sum())


@overload
def rps_score(y_true: VectorI64, probs: MatrixF64, *, reduce: Literal[True] = True) -> float: ...
@overload
def rps_score(y_true: MatrixI64, probs: Array3F64, *, reduce: Literal[True] = True) -> float: ...
@overload
def rps_score(y_true: VectorI64, probs: MatrixF64, *, reduce: Literal[False]) -> VectorF64: ...
@overload
def rps_score(y_true: MatrixI64, probs: Array3F64, *, reduce: Literal[False]) -> MatrixF64: ...
@icontract.require(lambda y_true, probs: y_true.min() >= 0 and y_true.max() < probs.shape[-1])
@icontract.require(lambda y_true, probs: y_true.shape == probs.shape[:-1])
@icontract.require(lambda probs: np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6))
@icontract.require(lambda probs: probs.shape[-1] >= 2)
@icontract.ensure(lambda result: np.all((np.asarray(result) >= -1e-12) & (np.asarray(result) <= 1 + 1e-12)))
@icontract.ensure(lambda y_true, reduce, result: reduce or result.shape == y_true.shape)
def rps_score(y_true: VectorI64 | MatrixI64, probs: MatrixF64 | Array3F64, *, reduce: bool = True) -> float | VectorF64 | MatrixF64:
	"""排序概率评分（ranked probability score），越低越好。

	按累积概率与真实标签累积指示的平方差求和，再除以 K-1，取值在 [0, 1]。
	y_true 为 (N,) 或 (N, R) 的标签（0..K-1），probs 相应为 (N, K) 或 (N, R, K)。
	reduce=False 时返回与 y_true 同形状的逐样本值。
	"""
	num_classes = probs.shape[-1]
	cdf_true = (y_true[..., None] <= np.arange(num_classes - 1)).astype(np.float64)
	scores = ((probs.cumsum(axis=-1)[..., :-1] - cdf_true) ** 2).sum(axis=-1) / (num_classes - 1)
	return float(scores.mean()) if reduce else scores


@overload
def expected_score(probs: MatrixF64, values: VectorF64 | None = None) -> VectorF64: ...
@overload
def expected_score(probs: Array3F64, values: VectorF64 | None = None) -> MatrixF64: ...
@icontract.require(lambda probs, values: values is None or values.shape == (probs.shape[-1],))
@icontract.require(lambda probs: np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6))
@icontract.require(lambda probs: probs.shape[-1] >= 2)
@icontract.ensure(lambda probs, result: result.shape == probs.shape[:-1])
def expected_score(probs: MatrixF64 | Array3F64, values: VectorF64 | None = None) -> VectorF64 | MatrixF64:
	"""(N, K) 或 (N, R, K) 概率分布的期望分，去掉最后一维。values 为各类别对应的分值，默认 0..K-1。"""
	v = np.arange(probs.shape[-1], dtype=np.float64) if values is None else values
	return probs @ v


__all__ = [
	'evaluate_ordinal_scores',
	'expected_score',
	'qwk_score',
	'rps_score',
]
