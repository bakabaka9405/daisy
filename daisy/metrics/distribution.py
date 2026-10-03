"""有序类别上概率分布之间的距离。"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import jensenshannon

from daisy.typing import MatrixF64, NDArrayF64


def _as_distributions(p: ArrayLike, q: ArrayLike) -> tuple[NDArrayF64, NDArrayF64]:
	p_arr = np.asarray(p, dtype=np.float64)
	q_arr = np.asarray(q, dtype=np.float64)
	if p_arr.shape != q_arr.shape:
		raise ValueError(f'p and q must have the same shape, got {p_arr.shape} and {q_arr.shape}')
	if p_arr.ndim < 1 or p_arr.shape[-1] < 2:
		raise ValueError(f'distributions must have a class dimension of size >= 2, got shape {p_arr.shape}')
	for name, arr in (('p', p_arr), ('q', q_arr)):
		if not np.allclose(arr.sum(axis=-1), 1.0, atol=1e-6):
			raise ValueError(f'{name} must sum to 1 along the last axis')
	return p_arr, q_arr


def label_histogram(labels: ArrayLike, num_classes: int) -> MatrixF64:
	"""(N, R) 的多评分者标签（0..K-1）→ (N, K) 每个样本的经验分布。"""
	arr = np.asarray(labels, dtype=np.int64)
	if arr.ndim != 2:
		raise ValueError(f'labels must have shape (N, R), got {arr.shape}')
	if num_classes < 2:
		raise ValueError(f'num_classes must be at least 2, got {num_classes}')
	if arr.min() < 0 or arr.max() >= num_classes:
		raise ValueError(f'labels must lie in [0, {num_classes - 1}]')
	return np.stack([(arr == k).mean(axis=1) for k in range(num_classes)], axis=1)


@overload
def emd_score(p: ArrayLike, q: ArrayLike, *, reduce: Literal[True] = True) -> float: ...
@overload
def emd_score(p: ArrayLike, q: ArrayLike, *, reduce: Literal[False]) -> NDArrayF64: ...
def emd_score(p: ArrayLike, q: ArrayLike, *, reduce: bool = True) -> float | NDArrayF64:
	"""一维有序分布的 Earth Mover's Distance（相邻类别间距为 1），即累积分布差的绝对值之和。"""
	p_arr, q_arr = _as_distributions(p, q)
	scores = np.abs(p_arr.cumsum(axis=-1) - q_arr.cumsum(axis=-1)).sum(axis=-1)
	return float(scores.mean()) if reduce else scores


@overload
def js_divergence(p: ArrayLike, q: ArrayLike, *, reduce: Literal[True] = True) -> float: ...
@overload
def js_divergence(p: ArrayLike, q: ArrayLike, *, reduce: Literal[False]) -> NDArrayF64: ...
def js_divergence(p: ArrayLike, q: ArrayLike, *, reduce: bool = True) -> float | NDArrayF64:
	"""Jensen-Shannon 散度（以 2 为底，取值 [0, 1]）。"""
	p_arr, q_arr = _as_distributions(p, q)
	scores = jensenshannon(p_arr, q_arr, axis=-1, base=2) ** 2  # pyright: ignore[reportCallIssue]
	return float(scores.mean()) if reduce else scores


__all__ = [
	'emd_score',
	'js_divergence',
	'label_histogram',
]
