from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from ..dca import DcaCurve, DcaMode, compute_multiclass_dca_curves


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
	curves: dict[str, DcaCurve],
	mode: DcaMode,
	class_names: Sequence[str],
) -> list[tuple[str, str, DcaCurve]]:
	plot_items: list[tuple[str, str, DcaCurve]] = []

	if mode in {'ovr', 'all'}:
		class_curve_keys = sorted(
			(curve_key for curve_key in curves if curve_key.startswith('class_')),
			key=lambda curve_key: int(curve_key.removeprefix('class_')),
		)
		for curve_key in class_curve_keys:
			class_index = int(curve_key.removeprefix('class_'))
			plot_items.append((curve_key, f'{class_names[class_index]} vs rest', curves[curve_key]))

	# multi-class DCA 下 micro 与 macro 理论上重合，all 模式默认不重复绘制。
	if mode == 'micro' and 'micro' in curves:
		plot_items.append(('micro', 'micro-average', curves['micro']))

	if mode in {'macro', 'all'} and 'macro' in curves:
		plot_items.append(('macro', 'macro-average', curves['macro']))

	return plot_items


def _build_curve_colors(total_curves: int, cmap: str) -> list[Any]:
	if total_curves <= 0:
		return []

	color_map = plt.get_cmap(cmap)
	positions = np.linspace(0.0, 1.0, num=total_curves, endpoint=False)
	return [color_map(float(position)) for position in positions]


def _compute_treat_all_net_benefit(
	thresholds: np.ndarray,
	prevalence: float,
) -> np.ndarray:
	threshold_odds = thresholds / (1.0 - thresholds)
	return prevalence - (1.0 - prevalence) * threshold_odds


def _resolve_reference_prevalence(plot_items: list[tuple[str, str, DcaCurve]]) -> tuple[float, str]:
	class_curves = [curve for curve_key, _, curve in plot_items if curve_key.startswith('class_')]
	if class_curves:
		return float(np.mean([curve.prevalence for curve in class_curves])), 'Treat all (mean prevalence)'

	if len(plot_items) == 1:
		return plot_items[0][2].prevalence, 'Treat all'

	return float(np.mean([curve.prevalence for _, _, curve in plot_items])), 'Treat all (mean prevalence)'


def _resolve_y_limits(
	plot_items: list[tuple[str, str, DcaCurve]],
	plot_treat_none: bool,
) -> tuple[float, float]:
	y_values = [curve.net_benefit for _, _, curve in plot_items]

	# Treat-all 在阈值接近 1 时会出现很大的负值，这里不让它主导纵轴缩放。
	thresholds = plot_items[0][2].thresholds

	if plot_treat_none:
		y_values.append(np.zeros_like(thresholds))

	min_value = min(float(np.min(values)) for values in y_values)
	max_value = max(float(np.max(values)) for values in y_values)
	padding = max(0.02, (max_value - min_value) * 0.05)
	return min_value - padding, max_value + padding


def plot_dca_curve(
	y_prob: ArrayLike,
	y_true: ArrayLike,
	num_classes: int,
	mode: DcaMode,
	ax: Axes,
	*,
	thresholds: ArrayLike | None = None,
	class_names: Sequence[str] | None = None,
	title: str | None = None,
	plot_treat_all: bool = True,
	plot_treat_none: bool = True,
	treat_all_label: str | None = None,
	treat_none_label: str = 'Treat none',
	cmap: str = 'tab10',
	linewidth: float = 1.8,
	alpha: float = 0.9,
	grid: bool = True,
	legend: bool = True,
	legend_loc: str = 'best',
	plot_kwargs: dict[str, Any] | None = None,
	treat_all_kwargs: dict[str, Any] | None = None,
	treat_none_kwargs: dict[str, Any] | None = None,
) -> dict[str, DcaCurve]:
	"""绘制多分类 DCA 曲线。"""

	# 1) 计算待绘制曲线与公共参考线。
	name_list = _resolve_class_names(class_names, num_classes)
	curves = compute_multiclass_dca_curves(y_true, y_prob, num_classes=num_classes, mode=mode, thresholds=thresholds)
	plot_items = _build_plot_items(curves, mode, name_list)
	if not plot_items:
		raise ValueError('No DCA curve is available for plotting')

	threshold_axis = plot_items[0][2].thresholds
	default_treat_all_label: str | None = None
	shared_treat_all: np.ndarray | None = None
	if plot_treat_all:
		reference_prevalence, default_treat_all_label = _resolve_reference_prevalence(plot_items)
		shared_treat_all = _compute_treat_all_net_benefit(threshold_axis, reference_prevalence)

	# 2) 执行绘图和外观设置。
	if plot_treat_none:
		treat_none_style: dict[str, Any] = {
			'label': treat_none_label,
			'linestyle': '--',
			'color': '0.55',
			'linewidth': 1.0,
			'alpha': 0.8,
		}
		if treat_none_kwargs is not None:
			treat_none_style.update(treat_none_kwargs)
		ax.plot(threshold_axis, np.zeros_like(threshold_axis), **treat_none_style)

	if plot_treat_all and shared_treat_all is not None:
		treat_all_style: dict[str, Any] = {
			'label': treat_all_label if treat_all_label is not None else default_treat_all_label,
			'linestyle': ':',
			'color': '0.35',
			'linewidth': 1.2,
			'alpha': 0.9,
		}
		if treat_all_kwargs is not None:
			treat_all_style.update(treat_all_kwargs)
		ax.plot(threshold_axis, shared_treat_all, **treat_all_style)

	line_style = plot_kwargs.copy() if plot_kwargs is not None else {}
	for color, (curve_key, curve_title, curve) in zip(_build_curve_colors(len(plot_items), cmap), plot_items):
		curve_style = line_style.copy()
		curve_style.setdefault('color', color)
		curve_style.setdefault('linewidth', linewidth)
		curve_style.setdefault('alpha', alpha)
		curve_style.setdefault('label', curve_title)

		ax.plot(curve.thresholds, curve.net_benefit, **curve_style)

	ax.set_xlim(float(threshold_axis[0]), float(threshold_axis[-1]))
	ax.set_ylim(*_resolve_y_limits(plot_items, plot_treat_none))
	ax.set_xlabel('Threshold Probability')
	ax.set_ylabel('Net Benefit')
	ax.set_title(title if title is not None else f'Multi-class DCA ({mode})')

	if grid:
		ax.grid(True, linestyle='--', alpha=0.3)

	if legend:
		ax.legend(loc=legend_loc)

	return curves


__all__ = ['plot_dca_curve']
