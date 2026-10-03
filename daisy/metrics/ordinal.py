"""有序类别（评分）预测的评价指标。"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from daisy.typing import NDArrayF64, NDArrayI64


def _check_labels(labels: NDArrayI64, num_classes: int, name: str) -> None:
	if num_classes < 2:
		raise ValueError(f'num_classes must be at least 2, got {num_classes}')
	if labels.size and (labels.min() < 0 or labels.max() >= num_classes):
		raise ValueError(f'{name} must lie in [0, {num_classes - 1}], got range [{labels.min()}, {labels.max()}]')


def _check_probs(probs: NDArrayF64) -> None:
	if probs.ndim < 1 or probs.shape[-1] < 2:
		raise ValueError(f'probs must have a class dimension of size >= 2, got shape {probs.shape}')
	if not np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6):
		raise ValueError('probs must sum to 1 along the last axis')


def qwk_score(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> float:
	"""二次加权 Cohen kappa。标签取值为 0..num_classes-1。

	与 sklearn.metrics.cohen_kappa_score(weights='quadratic', labels=range(num_classes)) 一致。
	"""
	true = np.asarray(y_true, dtype=np.int64).ravel()
	pred = np.asarray(y_pred, dtype=np.int64).ravel()
	if true.shape != pred.shape:
		raise ValueError(f'y_true and y_pred must have the same length, got {true.shape} and {pred.shape}')
	if true.size == 0:
		raise ValueError('qwk_score requires at least one sample')
	_check_labels(true, num_classes, 'y_true')
	_check_labels(pred, num_classes, 'y_pred')

	grid = np.arange(num_classes)
	weights = np.subtract.outer(grid, grid) ** 2 / (num_classes - 1) ** 2
	observed = np.bincount(true * num_classes + pred, minlength=num_classes**2).reshape(num_classes, num_classes).astype(np.float64)
	expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
	return float(1.0 - (weights * observed).sum() / (weights * expected).sum())


@overload
def rps_score(y_true: ArrayLike, probs: ArrayLike, *, reduce: Literal[True] = True) -> float: ...
@overload
def rps_score(y_true: ArrayLike, probs: ArrayLike, *, reduce: Literal[False]) -> NDArrayF64: ...
def rps_score(y_true: ArrayLike, probs: ArrayLike, *, reduce: bool = True) -> float | NDArrayF64:
	"""排序概率评分（ranked probability score），越低越好。

	按累积概率与真实标签累积指示的平方差求和，再除以 K-1，取值在 [0, 1]。
	y_true 形状为 probs 去掉最后一维（标签 0..K-1），probs 最后一维为类别。
	reduce=False 时返回与 y_true 同形状的逐样本值。
	"""
	p = np.asarray(probs, dtype=np.float64)
	_check_probs(p)
	num_classes = p.shape[-1]
	true = np.asarray(y_true, dtype=np.int64)
	if true.shape != p.shape[:-1]:
		raise ValueError(f'y_true shape {true.shape} does not match probs shape {p.shape}')
	_check_labels(true, num_classes, 'y_true')

	cdf_true = (true[..., None] <= np.arange(num_classes - 1)).astype(np.float64)
	scores = ((p.cumsum(axis=-1)[..., :-1] - cdf_true) ** 2).sum(axis=-1) / (num_classes - 1)
	return float(scores.mean()) if reduce else scores


def expected_score(probs: ArrayLike, values: ArrayLike | None = None) -> NDArrayF64:
	"""概率分布的期望分。values 为各类别对应的分值，默认 0..K-1。"""
	p = np.asarray(probs, dtype=np.float64)
	_check_probs(p)
	v = np.arange(p.shape[-1], dtype=np.float64) if values is None else np.asarray(values, dtype=np.float64)
	if v.shape != (p.shape[-1],):
		raise ValueError(f'values must have shape ({p.shape[-1]},), got {v.shape}')
	return p @ v


__all__ = [
	'expected_score',
	'qwk_score',
	'rps_score',
]
