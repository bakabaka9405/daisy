import numpy as np
from numpy.typing import ArrayLike
from typing import Callable, Any


def bootstrap_ci(
	y_true: ArrayLike,
	y_pred: ArrayLike,
	metric_fn: Callable[[ArrayLike, ArrayLike], Any],
	n_boot=5000,
	alpha=0.05,
	seed=42,
):
	rng = np.random.default_rng(seed)
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	n = len(y_true)

	scores = []
	for _ in range(n_boot):
		idx = rng.integers(0, n, n)
		score = metric_fn(y_true[idx], y_pred[idx])
		scores.append(score)

	lower = np.percentile(scores, 100 * alpha / 2)
	upper = np.percentile(scores, 100 * (1 - alpha / 2))

	return lower, upper
