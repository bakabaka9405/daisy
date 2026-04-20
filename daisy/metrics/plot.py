from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import auc, roc_curve


@dataclass(slots=True, frozen=True)
class _RocCurve:
	fpr: NDArray[np.float64]
	tpr: NDArray[np.float64]
	auc: float


def _resolve_class_names(
	class_names: Sequence[str] | None,
	num_classes: int,
) -> list[str]:
	if class_names is None:
		return [f'class_{i}' for i in range(num_classes)]

	names = [str(name) for name in class_names]
	if len(names) < num_classes:
		names.extend(f'class_{i}' for i in range(len(names), num_classes))

	return names[:num_classes]


def _compute_binary_roc_curve(
	binary_true: NDArray[np.int64],
	scores: NDArray[np.float64],
) -> _RocCurve | None:
	positive_count = int(np.count_nonzero(binary_true))
	negative_count = int(binary_true.size - positive_count)
	if positive_count == 0 or negative_count == 0:
		return None

	fpr, tpr, _ = roc_curve(binary_true, scores)
	fpr_array = np.asarray(fpr, dtype=np.float64)
	tpr_array = np.asarray(tpr, dtype=np.float64)
	return _RocCurve(fpr=fpr_array, tpr=tpr_array, auc=float(auc(fpr_array, tpr_array)))


def _compute_ovr_curves(
	y_true: NDArray[np.int64],
	y_pred: NDArray[np.float64],
	num_classes: int,
) -> dict[int, _RocCurve]:
	curves: dict[int, _RocCurve] = {}

	for class_index in range(num_classes):
		binary_true = (y_true == class_index).astype(np.int64)
		curve = _compute_binary_roc_curve(binary_true, y_pred[:, class_index])
		if curve is not None:
			curves[class_index] = curve

	return curves


def _compute_micro_curve(
	y_true: NDArray[np.int64],
	y_pred: NDArray[np.float64],
	num_classes: int,
) -> _RocCurve:
	binary_true = np.zeros((y_true.shape[0], num_classes), dtype=np.int64)
	binary_true[np.arange(y_true.shape[0]), y_true] = 1

	micro_curve = _compute_binary_roc_curve(binary_true.ravel(), y_pred.ravel())
	if micro_curve is None:
		raise ValueError('micro ROC curve is undefined for the current labels')

	return micro_curve


def _compute_macro_curve(ovr_curves: dict[int, _RocCurve]) -> _RocCurve:
	all_fpr = np.unique(np.concatenate([curve.fpr for curve in ovr_curves.values()]))
	mean_tpr = np.zeros_like(all_fpr)

	for curve in ovr_curves.values():
		mean_tpr += np.interp(all_fpr, curve.fpr, curve.tpr)

	mean_tpr /= float(len(ovr_curves))
	return _RocCurve(fpr=all_fpr, tpr=mean_tpr, auc=float(auc(all_fpr, mean_tpr)))


def _compute_ovo_curves(
	y_true: NDArray[np.int64],
	y_pred: NDArray[np.float64],
	num_classes: int,
) -> dict[tuple[int, int], _RocCurve]:
	curves: dict[tuple[int, int], _RocCurve] = {}

	for first_class in range(num_classes):
		for second_class in range(first_class + 1, num_classes):
			pair_mask = (y_true == first_class) | (y_true == second_class)
			if int(np.count_nonzero(pair_mask)) < 2:
				continue

			pair_true = (y_true[pair_mask] == first_class).astype(np.int64)
			pair_scores = y_pred[pair_mask, first_class] - y_pred[pair_mask, second_class]

			curve = _compute_binary_roc_curve(pair_true, pair_scores)
			if curve is not None:
				curves[(first_class, second_class)] = curve

	return curves


def _build_curve_colors(total_curves: int, cmap: str) -> list[Any]:
	if total_curves <= 0:
		return []

	color_map = plt.get_cmap(cmap)
	positions = np.linspace(0.0, 1.0, num=total_curves, endpoint=False)
	return [color_map(float(position)) for position in positions]


def plot_roc_curve(
	y_pred: ArrayLike,  # (N, num_classes)
	y_true: ArrayLike,  # (N,)
	num_classes: int,
	mode: Literal['ovr', 'micro', 'macro', 'all', 'ovo'],
	ax: Axes,
	*,
	class_names: Sequence[str] | None = None,
	title: str | None = None,
	plot_chance: bool = True,
	chance_label: str = 'Chance',
	cmap: str = 'tab10',
	linewidth: float = 1.8,
	alpha: float = 0.9,
	grid: bool = True,
	legend: bool = True,
	legend_loc: str = 'lower right',
	include_auc_in_label: bool = True,
	plot_kwargs: dict[str, Any] | None = None,
	chance_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
	"""绘制多分类 ROC 曲线并返回 AUC。"""

	# 1) 规范化输入并进行必要校验。

	y_true_array = np.asarray(y_true, dtype=np.int64).reshape(-1)
	y_pred_array = np.asarray(y_pred, dtype=np.float64)

	name_list = _resolve_class_names(class_names, num_classes)

	# 2) 根据 mode 计算待绘制曲线。
	plot_items: list[tuple[str, str, _RocCurve]] = []
	auc_scores: dict[str, float] = {}

	ovr_curves = _compute_ovr_curves(y_true_array, y_pred_array, num_classes)
	if mode in {'ovr', 'macro', 'all'} and not ovr_curves:
		raise ValueError('No valid one-vs-rest ROC curve can be computed from current labels')

	if mode in {'ovr', 'all'}:
		for class_index in sorted(ovr_curves):
			plot_items.append((f'class_{class_index}', f'{name_list[class_index]} vs rest', ovr_curves[class_index]))

	if mode in {'micro', 'all'}:
		micro_curve = _compute_micro_curve(y_true_array, y_pred_array, num_classes)
		plot_items.append(('micro', 'micro-average', micro_curve))

	if mode in {'macro', 'all'}:
		macro_curve = _compute_macro_curve(ovr_curves)
		plot_items.append(('macro', 'macro-average', macro_curve))

	if mode == 'ovo':
		ovo_curves = _compute_ovo_curves(y_true_array, y_pred_array, num_classes)
		if not ovo_curves:
			raise ValueError('No valid one-vs-one ROC curve can be computed from current labels')

		for (first_class, second_class), curve in sorted(ovo_curves.items()):
			curve_name = f'{name_list[first_class]} vs {name_list[second_class]}'
			curve_key = f'ovo_{first_class}_vs_{second_class}'
			plot_items.append((curve_key, curve_name, curve))

	# 3) 执行绘图和外观设置。
	if plot_chance:
		chance_style: dict[str, Any] = {
			'label': chance_label,
			'linestyle': '--',
			'color': '0.5',
			'linewidth': 1.0,
			'alpha': 0.8,
		}
		if chance_kwargs is not None:
			chance_style.update(chance_kwargs)
		ax.plot([0.0, 1.0], [0.0, 1.0], **chance_style)

	line_style = plot_kwargs.copy() if plot_kwargs is not None else {}
	for color, (curve_key, curve_title, curve) in zip(_build_curve_colors(len(plot_items), cmap), plot_items):
		curve_style = line_style.copy()
		curve_style.setdefault('color', color)
		curve_style.setdefault('linewidth', linewidth)
		curve_style.setdefault('alpha', alpha)

		label = curve_title
		if include_auc_in_label:
			label = f'{curve_title} (AUC={curve.auc:.4f})'
		curve_style.setdefault('label', label)

		ax.plot(curve.fpr, curve.tpr, **curve_style)
		auc_scores[curve_key] = curve.auc

	if mode == 'ovo' and auc_scores:
		auc_scores['ovo_mean'] = float(np.mean(list(auc_scores.values())))

	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.05)
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title(title if title is not None else f'Multi-class ROC ({mode})')

	if grid:
		ax.grid(True, linestyle='--', alpha=0.3)

	if legend:
		ax.legend(loc=legend_loc)

	return auc_scores


__all__ = ['plot_roc']
