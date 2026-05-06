from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from ..auroc import RocCurve, compute_multiclass_roc_curves


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


def _build_plot_items(
	curves: dict[str, RocCurve],
	mode: Literal['ovr', 'micro', 'macro', 'all', 'ovo'],
	class_names: Sequence[str],
) -> list[tuple[str, str, RocCurve]]:
	plot_items: list[tuple[str, str, RocCurve]] = []

	if mode in {'ovr', 'all'}:
		class_curve_keys = sorted(
			(curve_key for curve_key in curves if curve_key.startswith('class_')),
			key=lambda curve_key: int(curve_key.removeprefix('class_')),
		)
		for curve_key in class_curve_keys:
			class_index = int(curve_key.removeprefix('class_'))
			plot_items.append((curve_key, f'{class_names[class_index]} vs rest', curves[curve_key]))

	if mode in {'micro', 'all'} and 'micro' in curves:
		plot_items.append(('micro', 'micro-average', curves['micro']))

	if mode in {'macro', 'all'} and 'macro' in curves:
		plot_items.append(('macro', 'macro-average', curves['macro']))

	if mode == 'ovo':
		ovo_curve_keys = sorted(
			(curve_key for curve_key in curves if curve_key.startswith('ovo_')),
			key=lambda curve_key: tuple(int(index) for index in curve_key.removeprefix('ovo_').split('_vs_')),
		)
		for curve_key in ovo_curve_keys:
			first_class_str, second_class_str = curve_key.removeprefix('ovo_').split('_vs_')
			first_class = int(first_class_str)
			second_class = int(second_class_str)
			curve_title = f'{class_names[first_class]} vs {class_names[second_class]}'
			plot_items.append((curve_key, curve_title, curves[curve_key]))

	return plot_items


def _build_curve_colors(total_curves: int, cmap: str) -> list[Any]:
	if total_curves <= 0:
		return []

	color_map = plt.get_cmap(cmap)
	positions = np.linspace(0.0, 1.0, num=total_curves, endpoint=False)
	return [color_map(float(position)) for position in positions]


def plot_roc_curve(
	y_pred: ArrayLike,
	y_true: ArrayLike,
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
	legend: bool = True,
	legend_loc: str = 'lower right',
	include_auc_in_label: bool = True,
	plot_kwargs: dict[str, Any] | None = None,
	chance_kwargs: dict[str, Any] | None = None,
) -> dict[str, float]:
	"""绘制多分类 ROC 曲线并返回 AUC。"""

	# 1) 解析类别名并准备待绘制曲线。
	name_list = _resolve_class_names(class_names, num_classes)
	curves = compute_multiclass_roc_curves(y_true, y_pred, num_classes=num_classes, mode=mode)
	plot_items = _build_plot_items(curves, mode, name_list)
	auc_scores: dict[str, float] = {}

	# 2) 执行绘图和外观设置。
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

	if legend:
		ax.legend(loc=legend_loc)

	return auc_scores


__all__ = ['plot_roc_curve']
