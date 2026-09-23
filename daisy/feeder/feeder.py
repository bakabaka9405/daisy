"""数据供给与样本解析实现"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .primitives import (
	ColumnLabelSpec,
	get_column_series,
	read_single_label,
)

type Samples[S, L] = tuple[list[S], list[L] | None]


def parse_table_paths(
	df: pd.DataFrame,
	dataset_root: Path | str,
	image_column: int | str = 0,
) -> list[Path]:
	"""从数据表中提取图像路径列表"""
	root = Path(dataset_root) if isinstance(dataset_root, str) else dataset_root
	img_series = get_column_series(df, image_column)
	return [root / str(x) for x in img_series.tolist()]


def parse_single_label_table(
	df: pd.DataFrame,
	dataset_root: Path | str,
	spec: ColumnLabelSpec,
	*,
	image_column: int | str = 0,
) -> tuple[list[Path], list[int]]:
	"""从数据表中解析图像路径与单任务标签"""
	paths = parse_table_paths(df, dataset_root, image_column=image_column)
	labels = read_single_label(df, spec)
	return paths, labels


def read_multitask_labels(
	df: pd.DataFrame,
	specs: Mapping[str, ColumnLabelSpec],
) -> list[dict[str, int]]:
	"""从数据表中提取并解析多任务标签"""
	# 解析各子任务的目标标签列表
	parsed: dict[str, list[int]] = {name: read_single_label(df, spec, target_name=name) for name, spec in specs.items()}

	# 按样本索引组装多任务标签字典
	num_samples = len(df)
	return [{name: parsed[name][i] for name in specs} for i in range(num_samples)]


def parse_multitask_table(
	df: pd.DataFrame,
	dataset_root: Path | str,
	specs: Mapping[str, ColumnLabelSpec],
	*,
	image_column: int | str = 0,
) -> tuple[list[Path], list[dict[str, int]]]:
	"""从数据表中解析图像路径与多任务标签"""
	paths = parse_table_paths(df, dataset_root, image_column=image_column)
	labels = read_multitask_labels(df, specs)
	return paths, labels


def merge_samples[S, L](
	samples: Sequence[Samples[S, L]],
) -> Samples[S, L]:
	"""合并多组样本数据"""
	if not samples:
		return [], None

	# 校验有标签与无标签样本的一致性并执行合并
	merged_items: list[S] = []
	has_labels = samples[0][1] is not None
	merged_labels: list[L] = []
	for items, labels in samples:
		if (labels is not None) != has_labels:
			raise ValueError('Cannot merge labeled samples with unlabeled samples')
		merged_items.extend(items)
		if has_labels:
			assert labels is not None
			merged_labels.extend(labels)

	return merged_items, (merged_labels if has_labels else None)
