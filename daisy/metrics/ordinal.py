"""有序类别（评分）预测的评价指标。"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from daisy.typing import Array3F64, MatrixF64, MatrixI64, VectorF64, VectorI64


def _check_labels(labels: VectorI64 | MatrixI64, num_classes: int, name: str) -> None:
	if num_classes < 2:
		raise ValueError(f'num_classes must be at least 2, got {num_classes}')
	if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
		raise ValueError(f'{name} must lie in [0, {num_classes - 1}], got range [{labels.min()}, {labels.max()}]')


def _check_probs(probs: MatrixF64 | Array3F64) -> None:
	if probs.shape[-1] < 2:
		raise ValueError(f'probs must have a class dimension of size >= 2, got shape {probs.shape}')
	if not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6):
		raise ValueError('probs must sum to 1 along the last axis')


def qwk_score(y_true: VectorI64, y_pred: VectorI64, num_classes: int) -> float:
	"""二次加权 Cohen kappa。标签取值为 0..num_classes-1。

	与 sklearn.metrics.cohen_kappa_score(weights='quadratic', labels=range(num_classes)) 一致。
	"""
	if y_true.shape != y_pred.shape:
		raise ValueError(f'y_true and y_pred must have the same length, got {y_true.shape} and {y_pred.shape}')
	if y_true.size == 0:
		raise ValueError('qwk_score requires at least one sample')
	_check_labels(y_true, num_classes, 'y_true')
	_check_labels(y_pred, num_classes, 'y_pred')

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
def rps_score(y_true: VectorI64 | MatrixI64, probs: MatrixF64 | Array3F64, *, reduce: bool = True) -> float | VectorF64 | MatrixF64:
	"""排序概率评分（ranked probability score），越低越好。

	按累积概率与真实标签累积指示的平方差求和，再除以 K-1，取值在 [0, 1]。
	y_true 为 (N,) 或 (N, R) 的标签（0..K-1），probs 相应为 (N, K) 或 (N, R, K)。
	reduce=False 时返回与 y_true 同形状的逐样本值。
	"""
	_check_probs(probs)
	num_classes = probs.shape[-1]
	if y_true.shape != probs.shape[:-1]:
		raise ValueError(f'y_true shape {y_true.shape} does not match probs shape {probs.shape}')
	_check_labels(y_true, num_classes, 'y_true')

	cdf_true = (y_true[..., None] <= np.arange(num_classes - 1)).astype(np.float64)
	scores = ((probs.cumsum(axis=-1)[..., :-1] - cdf_true) ** 2).sum(axis=-1) / (num_classes - 1)
	return float(scores.mean()) if reduce else scores


@overload
def expected_score(probs: MatrixF64, values: VectorF64 | None = None) -> VectorF64: ...
@overload
def expected_score(probs: Array3F64, values: VectorF64 | None = None) -> MatrixF64: ...
def expected_score(probs: MatrixF64 | Array3F64, values: VectorF64 | None = None) -> VectorF64 | MatrixF64:
	"""(N, K) 或 (N, R, K) 概率分布的期望分，去掉最后一维。values 为各类别对应的分值，默认 0..K-1。"""
	_check_probs(probs)
	v = np.arange(probs.shape[-1], dtype=np.float64) if values is None else values
	if v.shape != (probs.shape[-1],):
		raise ValueError(f'values must have shape ({probs.shape[-1]},), got {v.shape}')
	return probs @ v


__all__ = [
	'expected_score',
	'qwk_score',
	'rps_score',
]
