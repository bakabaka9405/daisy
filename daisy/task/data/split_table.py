"""按数据表解析列划分索引"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from daisy.feeder import get_column_series


def parse_column_split_indices(
	file_paths: Sequence[Path],
	table: pd.DataFrame,
	*,
	root: Path,
	image_column: int | str,
	assignment_column: int | str,
	partition_values: Sequence[str | int],
) -> list[list[int]]:
	"""解析表格中的划分列并返回各分区对应的样本索引列表。

	外层列表顺序与 partition_values 一一对应；各内层列表按输入顺序保存样本在数据集中的索引；
	未匹配样本被排除；未分配到任何样本的分区保留为空列表。
	"""
	# 第一阶段：解析图像与划分列构建规范化路径映射
	split_img_series = get_column_series(table, image_column)
	split_col_series = get_column_series(table, assignment_column)

	split_map: dict[str, int | None] = {}
	for raw_img, split_val in zip(split_img_series.tolist(), split_col_series.tolist()):
		img_key = (root / str(raw_img)).resolve().as_posix()
		if img_key in split_map:
			raise ValueError(f'Duplicate image path key in split table: {img_key}')
		partition_index: int | None = None
		for group_idx, target in enumerate(partition_values):
			if split_val == target or str(split_val) == str(target):
				partition_index = group_idx
				break
		split_map[img_key] = partition_index

	# 第二阶段：样本覆盖完整性比对
	main_keys = [p.resolve().as_posix() for p in file_paths]
	main_key_set = set(main_keys)
	split_key_set = set(split_map.keys())

	missing = main_key_set - split_key_set
	if missing:
		raise ValueError(
			f'Split table is missing {len(missing)} dataset samples, e.g. {sorted(missing)[:5]}'
		)
	extra = split_key_set - main_key_set
	if extra:
		raise ValueError(
			f'Split table has {len(extra)} extra rows not in dataset, e.g. {sorted(extra)[:5]}'
		)

	# 第三阶段：按输入顺序分配样本索引至对应分区
	groups: list[list[int]] = [[] for _ in partition_values]
	for idx, key in enumerate(main_keys):
		partition_index = split_map[key]
		if partition_index is not None:
			groups[partition_index].append(idx)

	return groups
