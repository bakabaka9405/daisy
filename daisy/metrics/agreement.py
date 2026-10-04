"""多评分者评分矩阵的一致性与方差结构分析。

评分矩阵 ratings 形状为 (N, R)：N 个样本，R 位评分者，每个位置为该评分者对该样本的评分。
"""

from __future__ import annotations

import icontract
import numpy as np

from daisy.typing import MatrixF64, VectorF64, VectorI64

from .regression import pc_score


@icontract.require(lambda ratings: np.isfinite(ratings).all(), 'missing ratings are not supported')
@icontract.require(lambda ratings: ratings.shape[1] >= 2)
@icontract.ensure(lambda ratings, result: result.shape == ratings.shape)
def others_mean(ratings: MatrixF64) -> MatrixF64:
	"""(N, R) → (N, R)，每个位置为同一样本其余 R-1 位评分者的平均分。"""
	return (ratings.sum(axis=1, keepdims=True) - ratings) / (ratings.shape[1] - 1)


@icontract.ensure(lambda ratings, result: result.shape == ratings.shape)
def rater_deviation(ratings: MatrixF64) -> MatrixF64:
	"""每个评分减去同一样本其余评分者的平均分，即评分者相对他人的偏好。"""
	return ratings - others_mean(ratings)


@icontract.ensure(lambda ratings, result: result.shape == (ratings.shape[1],))
def leave_one_rater_out_pc(ratings: MatrixF64) -> VectorF64:
	"""每位评分者的评分与其余评分者平均分的 Pearson 相关系数，形状 (R,)。"""
	others = others_mean(ratings)
	return np.array([pc_score(ratings[:, j], others[:, j]) for j in range(ratings.shape[1])], dtype=np.float64)


@icontract.require(lambda n_iter: n_iter >= 1)
@icontract.require(lambda ratings: np.isfinite(ratings).all(), 'missing ratings are not supported')
@icontract.require(lambda ratings: ratings.shape[1] >= 2)
def split_half_reliability(
	ratings: MatrixF64,
	n_iter: int = 500,
	seed: int | np.random.Generator = 0,
) -> tuple[float, float]:
	"""随机将评分者对半分组，计算两半平均分的相关系数（重复 n_iter 次取平均），
	并用 Spearman-Brown 公式换算为全部 R 位评分者平均分的信度。

	R 为奇数时每次随机舍弃一位评分者。返回 (对半相关系数, 全体平均分信度)。
	"""
	rng = np.random.default_rng(seed)
	n_raters = ratings.shape[1]
	half = n_raters // 2
	correlations = [
		np.corrcoef(ratings[:, p[:half]].mean(axis=1), ratings[:, p[half : 2 * half]].mean(axis=1))[0, 1]
		for p in (rng.permutation(n_raters) for _ in range(n_iter))
	]
	half_r = float(np.mean(correlations))
	factor = n_raters / half
	return half_r, float(factor * half_r / (1 + (factor - 1) * half_r))


@icontract.require(lambda values, groups: groups.shape == (values.shape[0],))
@icontract.require(lambda values: np.isfinite(values).all())
@icontract.require(lambda values: values.shape[1] >= 2)
@icontract.ensure(lambda values, groups, result: result.shape == (values.shape[1], np.unique(groups).size))
def two_way_interaction(values: MatrixF64, groups: VectorI64) -> MatrixF64:
	"""(N, R) 的数值矩阵按样本分组后，计算评分者 × 组的交互分量，形状 (R, G)。

	单元均值减去评分者主效应与组主效应，再加回总均值；各行、各列之和为 0。
	"""
	cell = np.stack([values[groups == g].mean(axis=0) for g in np.unique(groups)], axis=1)
	return cell - cell.mean(axis=1, keepdims=True) - cell.mean(axis=0, keepdims=True) + cell.mean()


@icontract.require(lambda ratings: ratings.std() > 0, 'ratings have zero variance')
@icontract.require(lambda ratings: np.isfinite(ratings).all(), 'missing ratings are not supported')
@icontract.require(lambda ratings: ratings.shape[1] >= 2)
@icontract.ensure(lambda result: abs(sum(result.values()) - 1) < 1e-9)
def variance_components(ratings: MatrixF64) -> dict[str, float]:
	"""无重复双因素分解：样本主效应、评分者主效应与残差（评分者 × 样本交互 + 噪声）各占总平方和的比例。"""
	n, r = ratings.shape
	grand = ratings.mean()
	total = ((ratings - grand) ** 2).sum()
	sample = r * ((ratings.mean(axis=1) - grand) ** 2).sum()
	rater = n * ((ratings.mean(axis=0) - grand) ** 2).sum()
	return {'sample': float(sample / total), 'rater': float(rater / total), 'residual': float((total - sample - rater) / total)}


__all__ = [
	'leave_one_rater_out_pc',
	'others_mean',
	'rater_deviation',
	'split_half_reliability',
	'two_way_interaction',
	'variance_components',
]
