from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
import torch

from .icc import icc_a1_score

RegressionValues = torch.Tensor | np.ndarray | ArrayLike | Sequence[float]


def _as_array(values: RegressionValues) -> np.ndarray:
	if isinstance(values, torch.Tensor):
		return values.detach().cpu().numpy()
	return np.asarray(values)


def pc_score(y_true: RegressionValues, y_pred: RegressionValues) -> float:
	"""Pearson Correlation coefficient. Returns 0.0 for constant predictions."""
	y_true = _as_array(y_true)
	y_pred = _as_array(y_pred)
	if y_true.ndim == 1:
		y_true = y_true.reshape(-1, 1)
	if y_pred.ndim == 1:
		y_pred = y_pred.reshape(-1, 1)

	mu_t = y_true.mean(axis=0, keepdims=True)
	mu_p = y_pred.mean(axis=0, keepdims=True)
	cov = np.mean((y_true - mu_t) * (y_pred - mu_p), axis=0)
	std_t = y_true.std(axis=0)
	std_p = y_pred.std(axis=0)

	denom = std_t * std_p
	result = np.zeros_like(cov)
	mask = denom > 1e-12
	result[mask] = cov[mask] / denom[mask]
	return float(result.mean())


def mae_score(y_true: RegressionValues, y_pred: RegressionValues) -> float:
	return float(np.mean(np.abs(_as_array(y_true) - _as_array(y_pred))))


def rmse_score(y_true: RegressionValues, y_pred: RegressionValues) -> float:
	return float(np.sqrt(np.mean((_as_array(y_true) - _as_array(y_pred)) ** 2)))


def evaluate_regression_scores(
	scores: RegressionValues,
	targets: RegressionValues,
) -> dict[str, float]:
	"""Evaluate regression metrics (PC, MAE, RMSE, ICC(A,1))."""
	y_true = np.asarray(_as_array(targets), dtype=np.float32)
	y_pred = np.asarray(_as_array(scores), dtype=np.float32)

	return {
		'pc': pc_score(y_true, y_pred),
		'mae': mae_score(y_true, y_pred),
		'rmse': rmse_score(y_true, y_pred),
		'icc_a1': icc_a1_score(y_true, y_pred),
	}
