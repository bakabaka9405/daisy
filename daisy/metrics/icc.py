from __future__ import annotations

import numpy as np
from daisy.typing import VectorF64


def icc_a1_score(y_true: VectorF64, y_pred: VectorF64) -> np.float64:
	"""
	计算 two-way random-effects、absolute agreement、single measurement ICC。

	至少包含两个样本。分母为零时返回 NaN。
	"""

	n = y_true.shape[0]
	if n < 2:
		raise ValueError(f'need at least 2 samples, got {n}')

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
		return np.float64('nan')

	return numer / denom
