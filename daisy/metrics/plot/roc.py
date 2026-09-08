from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from daisy.typing import MatrixF64, VectorI64

from ..auroc import RocCurve, RocMode, compute_multiclass_roc_curves


def _resolve_class_names(
	class_names: Sequence[str] | None,
	num_classes: int,
) -> list[str]:
	if class_names is not None:
		return [class_names[i] if i < len(class_names) else f'class_{i}' for i in range(num_classes)]
	return [f'class_{i}' for i in range(num_classes)]


def _build_plot_items(
	curves: dict[str, RocCurve],
	mode: RocMode,
	class_names: Sequence[str],
) -> list[tuple[str, str, RocCurve]]:
	plot_items: list[tuple[str, str, RocCurve]] = []

	if mode in {'ovr', 'all'}:
		for curve_key, curve in curves.items():
			if curve_key.startswith('class_'):
				class_index = int(curve_key.removeprefix('class_'))
				plot_items.append((curve_key, f'{class_names[class_index]} vs rest', curve))

	if mode in {'micro', 'all'}:
		plot_items.append(('micro', 'micro-average', curves['micro']))

	if mode in {'macro', 'all'}:
		plot_items.append(('macro', 'macro-average', curves['macro']))

	if mode == 'ovo':
		for curve_key, curve in curves.items():
			first_class_str, second_class_str = curve_key.removeprefix('ovo_').split('_vs_')
			curve_title = f'{class_names[int(first_class_str)]} vs {class_names[int(second_class_str)]}'
			plot_items.append((curve_key, curve_title, curve))

	return plot_items


def _build_curve_colors(total_curves: int, cmap: str) -> list[Any]:
	color_map = plt.get_cmap(cmap)
	positions = np.linspace(0.0, 1.0, num=total_curves, endpoint=False)
	return [color_map(float(position)) for position in positions]


def plot_roc_curve(
	y_pred: MatrixF64,
	y_true: VectorI64,
	mode: RocMode,
	ax: Axes,
	*,
	class_names: Sequence[str] | None = None,
	title: str | None = None,
	plot_chance: bool = True,
	chance_label: str = 'Chance',
	cmap: str = 'tab10',
	linewidth: float = 1.8,
	alpha: float = 0.9,
	legend: bool = True,
	legend_loc: str = 'lower right',
	include_auc_in_label: bool = True,
	plot_kwargs: dict[str, Any] | None = None,
	chance_kwargs: dict[str, Any] | None = None,
) -> dict[str, np.float64]:
	"""绘制多分类 ROC 曲线并返回 AUC。"""

	curves = compute_multiclass_roc_curves(y_true, y_pred, mode=mode)
	_, num_classes = y_pred.shape
	name_list = _resolve_class_names(class_names, num_classes)
	plot_items = _build_plot_items(curves, mode, name_list)
	auc_scores: dict[str, np.float64] = {}

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

	if mode == 'ovo':
		ovo_mean: np.float64 = np.mean(np.stack(list(auc_scores.values())))
		auc_scores['ovo_mean'] = ovo_mean

	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.05)
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_title(title if title is not None else f'Multi-class ROC ({mode})')

	if legend:
		ax.legend(loc=legend_loc)

	return auc_scores


__all__ = ['plot_roc_curve']
