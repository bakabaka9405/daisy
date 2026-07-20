from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def icc_a1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
	"""计算 two-way random-effects、absolute agreement、single measurement ICC。

	输入会被展平，且形状必须一致并至少包含两个样本。分母为零时返回 NaN。
	"""
	y_true = np.asarray(y_true, dtype=np.float64).ravel()
	y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

	if y_true.shape != y_pred.shape:
		raise ValueError(f'形状不一致：y_true {y_true.shape} vs y_pred {y_pred.shape}')
	n = y_true.shape[0]
	if n < 2:
		raise ValueError(f'至少需要 2 个样本，当前仅有 {n} 个')

	# 将真实值和预测值视为两个评分者。
	data = np.column_stack((y_true, y_pred))
	k = 2

	grand_mean = data.mean()

	col_means = data.mean(axis=0)
	ms_raters = n * np.sum((col_means - grand_mean) ** 2) / (k - 1.0)

	row_means = data.mean(axis=1)
	ms_targets = k * np.sum((row_means - grand_mean) ** 2) / (n - 1.0)

	sse = np.sum((data - grand_mean) ** 2) - (k - 1) * ms_raters - (n - 1) * ms_targets
	df_error = (n - 1) * (k - 1)
	ms_error = sse / df_error

	# 绝对一致性需将评分者间差异计入分母。
	numer = ms_targets - ms_error
	denom = ms_targets + (k - 1) * ms_error + k * (ms_raters - ms_error) / n

	if denom == 0:
		return float('nan')

	return float(numer / denom)
