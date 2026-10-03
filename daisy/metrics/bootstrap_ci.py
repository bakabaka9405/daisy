from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from daisy.typing import VectorF64, VectorI64


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


def stratified_resample(strata: ArrayLike, rng: np.random.Generator) -> VectorI64:
	"""在各层内有放回地重采样样本下标（各层样本数不变），按层的取值顺序拼接。"""
	labels = np.asarray(strata).ravel()
	return np.concatenate([rng.choice(ix, len(ix)) for ix in (np.flatnonzero(labels == s) for s in np.unique(labels))])


@dataclass(slots=True, frozen=True)
class PairedBootstrapResult:
	"""配对 bootstrap 结果：差值（a − b）的点估计、百分位置信区间与双侧 p 值。"""

	delta: float
	ci_low: float
	ci_high: float
	p: float


def paired_bootstrap(
	stat_a: Callable[[VectorI64], float],
	stat_b: Callable[[VectorI64], float],
	n: int,
	*,
	strata: ArrayLike | None = None,
	n_boot: int = 2000,
	alpha: float = 0.05,
	seed: int | np.random.Generator = 0,
) -> PairedBootstrapResult:
	"""比较两个统计量在同一批样本上的差异。

	stat_a、stat_b 接收样本下标数组并返回统计量（例如在这些样本上计算的相关系数），
	每次重采样对两者使用同一组下标。strata 非空时在各层内重采样（保持各层样本数），否则整体重采样。
	p 值为双侧：2 · min(P(Δ* ≤ 0), P(Δ* ≥ 0))，上限为 1。
	seed 可传入 Generator，以便多次比较共享同一随机数序列。
	"""
	if n < 1:
		raise ValueError(f'n must be positive, got {n}')
	if n_boot < 1:
		raise ValueError(f'n_boot must be positive, got {n_boot}')
	if not 0 < alpha < 1:
		raise ValueError(f'alpha must be in (0, 1), got {alpha}')
	labels = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata).ravel()
	if labels.shape[0] != n:
		raise ValueError(f'strata must have length {n}, got {labels.shape[0]}')

	def delta(idx: VectorI64) -> float:
		return float(stat_a(idx) - stat_b(idx))

	rng = np.random.default_rng(seed)
	deltas = np.array([delta(stratified_resample(labels, rng)) for _ in range(n_boot)])
	return PairedBootstrapResult(
		delta=delta(np.arange(n)),
		ci_low=float(np.quantile(deltas, alpha / 2)),
		ci_high=float(np.quantile(deltas, 1 - alpha / 2)),
		p=float(min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))),
	)
