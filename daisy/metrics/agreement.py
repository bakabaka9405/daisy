"""多评分者评分矩阵的一致性与方差结构分析。

评分矩阵 ratings 形状为 (N, R)：N 个样本，R 位评分者，每个位置为该评分者对该样本的评分。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from daisy.typing import MatrixF64, VectorF64

from .regression import pc_score


def _as_ratings(ratings: ArrayLike) -> MatrixF64:
	arr = np.asarray(ratings, dtype=np.float64)
	if arr.ndim != 2:
		raise ValueError(f'ratings must have shape (N, R), got {arr.shape}')
	if arr.shape[1] < 2:
		raise ValueError(f'at least 2 raters are required, got {arr.shape[1]}')
	if not np.isfinite(arr).all():
		raise ValueError('ratings must be finite (missing ratings are not supported)')
	return arr


def others_mean(ratings: ArrayLike) -> MatrixF64:
	"""(N, R) → (N, R)，每个位置为同一样本其余 R-1 位评分者的平均分。"""
	arr = _as_ratings(ratings)
	return (arr.sum(axis=1, keepdims=True) - arr) / (arr.shape[1] - 1)


def rater_deviation(ratings: ArrayLike) -> MatrixF64:
	"""每个评分减去同一样本其余评分者的平均分，即评分者相对他人的偏好。"""
	arr = _as_ratings(ratings)
	return arr - others_mean(arr)


def leave_one_rater_out_pc(ratings: ArrayLike) -> VectorF64:
	"""每位评分者的评分与其余评分者平均分的 Pearson 相关系数，形状 (R,)。"""
	arr = _as_ratings(ratings)
	others = others_mean(arr)
	return np.array([pc_score(arr[:, j], others[:, j]) for j in range(arr.shape[1])], dtype=np.float64)


def split_half_reliability(
	ratings: ArrayLike,
	n_iter: int = 500,
	seed: int | np.random.Generator = 0,
) -> tuple[float, float]:
	"""随机将评分者对半分组，计算两半平均分的相关系数（重复 n_iter 次取平均），
	并用 Spearman-Brown 公式换算为全部 R 位评分者平均分的信度。

	R 为奇数时每次随机舍弃一位评分者。返回 (对半相关系数, 全体平均分信度)。
	"""
	arr = _as_ratings(ratings)
	if n_iter < 1:
		raise ValueError(f'n_iter must be positive, got {n_iter}')
	rng = np.random.default_rng(seed)
	n_raters = arr.shape[1]
	half = n_raters // 2
	correlations = [
		np.corrcoef(arr[:, p[:half]].mean(axis=1), arr[:, p[half : 2 * half]].mean(axis=1))[0, 1]
		for p in (rng.permutation(n_raters) for _ in range(n_iter))
	]
	half_r = float(np.mean(correlations))
	factor = n_raters / half
	return half_r, float(factor * half_r / (1 + (factor - 1) * half_r))


def two_way_interaction(values: ArrayLike, groups: ArrayLike) -> MatrixF64:
	"""(N, R) 的数值矩阵按样本分组后，计算评分者 × 组的交互分量，形状 (R, G)。

	单元均值减去评分者主效应与组主效应，再加回总均值；各行、各列之和为 0。
	"""
	arr = _as_ratings(values)
	grp = np.asarray(groups).ravel()
	if grp.shape[0] != arr.shape[0]:
		raise ValueError(f'groups must have length {arr.shape[0]}, got {grp.shape[0]}')
	cell = np.stack([arr[grp == g].mean(axis=0) for g in np.unique(grp)], axis=1)
	return cell - cell.mean(axis=1, keepdims=True) - cell.mean(axis=0, keepdims=True) + cell.mean()


def variance_components(ratings: ArrayLike) -> dict[str, float]:
	"""无重复双因素分解：样本主效应、评分者主效应与残差（评分者 × 样本交互 + 噪声）各占总平方和的比例。"""
	arr = _as_ratings(ratings)
	n, r = arr.shape
	grand = arr.mean()
	total = ((arr - grand) ** 2).sum()
	if total == 0:
		raise ValueError('ratings have zero variance')
	sample = r * ((arr.mean(axis=1) - grand) ** 2).sum()
	rater = n * ((arr.mean(axis=0) - grand) ** 2).sum()
	return {'sample': float(sample / total), 'rater': float(rater / total), 'residual': float((total - sample - rater) / total)}


__all__ = [
	'leave_one_rater_out_pc',
	'others_mean',
	'rater_deviation',
	'split_half_reliability',
	'two_way_interaction',
	'variance_components',
]
