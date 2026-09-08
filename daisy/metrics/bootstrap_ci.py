from __future__ import annotations

from collections.abc import Callable

import numpy as np

from daisy.typing import VectorF64


def bootstrap_ci(
	y_true: VectorF64,
	y_pred: VectorF64,
	metric_fn: Callable[[VectorF64, VectorF64], np.float64],
	n_boot: int = 5000,
	alpha: float = 0.05,
	seed: int = 42,
) -> tuple[np.float64, np.float64]:
	"""对成对样本做有放回重抽样，返回度量的百分位置信区间。"""

	rng = np.random.default_rng(seed)
	n = y_true.shape[0]
	scores: VectorF64 = np.empty(n_boot, dtype=np.float64)
	for i in range(n_boot):
		idx = rng.integers(0, n, n)
		scores[i] = metric_fn(y_true[idx], y_pred[idx])

	lower, upper = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
	return lower, upper
