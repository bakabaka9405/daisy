from __future__ import annotations

import numpy as np
from daisy.typing import VectorF64, MatrixF64
from .icc import icc_a1_score


def pc_score(y_true: VectorF64 | MatrixF64, y_pred: VectorF64 | MatrixF64) -> np.float64:
	"""Pearson Correlation coefficient. Returns 0.0 for constant predictions."""
	y_true = y_true.reshape(y_true.shape[0], 1 if y_true.ndim == 1 else y_true.shape[-1])
	y_pred = y_pred.reshape(y_pred.shape[0], 1 if y_pred.ndim == 1 else y_pred.shape[-1])

	mu_t = y_true.mean(axis=0, keepdims=True)
	mu_p = y_pred.mean(axis=0, keepdims=True)
	cov = np.mean((y_true - mu_t) * (y_pred - mu_p), axis=0)
	std_t = y_true.std(axis=0)
	std_p = y_pred.std(axis=0)

	denom = std_t * std_p
	result = np.zeros_like(cov)
	mask = denom > 1e-12
	result[mask] = cov[mask] / denom[mask]
	return result.mean()


def mae_score(y_true: VectorF64, y_pred: VectorF64) -> np.float64:
	return np.mean(np.abs(y_true - y_pred))


def rmse_score(y_true: VectorF64, y_pred: VectorF64) -> np.float64:
	return np.sqrt(np.mean((y_true - y_pred) ** 2))


def evaluate_regression_scores(y_pred: VectorF64, y_true: VectorF64) -> dict[str, float]:
	"""计算回归指标。"""
	return {
		'mae': mae_score(y_true, y_pred),
		'rmse': rmse_score(y_true, y_pred),
		'pc': pc_score(y_true, y_pred),
		'icc_a1': icc_a1_score(y_true, y_pred),
	}
