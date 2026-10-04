"""分类评估指标计算模块。"""

from __future__ import annotations

from dataclasses import dataclass

import icontract
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True, slots=True)
class EvalMetrics:
	"""评估指标容器"""

	loss: float = 0.0
	acc: float = 0.0  # 准确率
	precision: float = 0.0
	recall: float = 0.0
	f1: float = 0.0
	auroc: float = 0.0
	confusion_matrix: list[list[int]] | None = None

	def get_metric(self, name: str) -> float:
		"""根据名称获取指标值"""
		if name == 'acc' or name == 'acc1':
			return self.acc
		elif name == 'prec':
			return self.precision
		elif name == 'recall':
			return self.recall
		elif name == 'f1':
			return self.f1
		elif name == 'auroc':
			return self.auroc
		else:
			return 0.0


@icontract.require(lambda num_classes: num_classes >= 1)
@icontract.require(lambda y_true, y_pred: np.shape(y_true) == np.shape(y_pred))
@icontract.ensure(lambda num_classes, result: len(result) == num_classes and all(len(row) == num_classes for row in result))
def calc_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int) -> list[list[int]]:
	"""计算混淆矩阵"""
	labels = list(range(num_classes))
	matrix = confusion_matrix(y_true, y_pred, labels=labels)
	return matrix.tolist()  # type: ignore[return-value]


@icontract.require(lambda num_classes: num_classes >= 0)
@icontract.require(lambda y_true, y_pred: np.shape(y_true) == np.shape(y_pred))
@icontract.ensure(lambda result: all(0 <= v <= 1 for v in (result.acc, result.precision, result.recall, result.f1)))
def fast_calc_metrics(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int = 0) -> EvalMetrics:
	"""快速计算分类评估指标"""
	acc = accuracy_score(y_true, y_pred)
	prec = precision_score(y_true, y_pred, average='macro', zero_division=0)  # type: ignore[call-overload]
	rec = recall_score(y_true, y_pred, average='macro', zero_division=0)  # type: ignore[call-overload]
	f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)  # type: ignore[call-overload]
	if num_classes == 0:
		matrix = None
	else:
		matrix = calc_confusion_matrix(y_true, y_pred, num_classes)

	return EvalMetrics(
		acc=float(acc),
		precision=float(prec),
		recall=float(rec),
		f1=float(f1),
		confusion_matrix=matrix,
	)
