"""有序类别上概率分布之间的距离。

分布矩阵 p、q 形状为 (N, K)：N 个样本，每行为 K 个有序类别上的概率分布。
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from scipy.spatial.distance import jensenshannon

from daisy.typing import MatrixF64, MatrixI64, VectorF64


def _check_distributions(p: MatrixF64, q: MatrixF64) -> None:
	if p.shape != q.shape:
		raise ValueError(f'p and q must have the same shape, got {p.shape} and {q.shape}')
	if p.shape[1] < 2:
		raise ValueError(f'distributions must have at least 2 classes, got shape {p.shape}')
	for name, arr in (('p', p), ('q', q)):
		if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-6):
			raise ValueError(f'{name} must sum to 1 along the class axis')


def label_histogram(labels: MatrixI64, num_classes: int) -> MatrixF64:
	"""(N, R) 的多评分者标签（0..K-1）→ (N, K) 每个样本的经验分布。"""
	if num_classes < 2:
		raise ValueError(f'num_classes must be at least 2, got {num_classes}')
	if labels.min() < 0 or labels.max() >= num_classes:
		raise ValueError(f'labels must lie in [0, {num_classes - 1}]')
	return np.stack([(labels == k).mean(axis=1) for k in range(num_classes)], axis=1)


@overload
def emd_score(p: MatrixF64, q: MatrixF64, *, reduce: Literal[True] = True) -> float: ...
@overload
def emd_score(p: MatrixF64, q: MatrixF64, *, reduce: Literal[False]) -> VectorF64: ...
def emd_score(p: MatrixF64, q: MatrixF64, *, reduce: bool = True) -> float | VectorF64:
	"""一维有序分布的 Earth Mover's Distance（相邻类别间距为 1），即累积分布差的绝对值之和。"""
	_check_distributions(p, q)
	scores = np.abs(p.cumsum(axis=1) - q.cumsum(axis=1)).sum(axis=1)
	return float(scores.mean()) if reduce else scores


@overload
def js_divergence(p: MatrixF64, q: MatrixF64, *, reduce: Literal[True] = True) -> float: ...
@overload
def js_divergence(p: MatrixF64, q: MatrixF64, *, reduce: Literal[False]) -> VectorF64: ...
def js_divergence(p: MatrixF64, q: MatrixF64, *, reduce: bool = True) -> float | VectorF64:
	"""Jensen-Shannon 散度（以 2 为底，取值 [0, 1]）。"""
	_check_distributions(p, q)
	scores = jensenshannon(p, q, axis=1, base=2) ** 2  # pyright: ignore[reportCallIssue]
	return float(scores.mean()) if reduce else scores


__all__ = [
	'emd_score',
	'js_divergence',
	'label_histogram',
]
