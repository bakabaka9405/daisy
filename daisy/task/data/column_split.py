"""按数据表列划分数据集"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never

import pandas as pd

from daisy.dataset.disk_dataset import DiskDataset
from daisy.feeder import get_column_series, read_sheet
from .model import ColumnSplit, ExternalTable, MainTable


def split_by_column[ImageT, LabelT](
	dataset: DiskDataset[ImageT, LabelT],
	split: ColumnSplit,
	*,
	source_df: pd.DataFrame | None = None,
) -> dict[str, DiskDataset[ImageT, LabelT]]:
	"""根据数据表划分列构建训练/验证/测试子集"""
	is_separate = isinstance(split.table, ExternalTable)

	split_df: pd.DataFrame
	split_root: Path
	split_image_column: int | str

	match split.table:
		case ExternalTable(path=path, sheet_name=sheet_name, have_header=have_header, root=root, image_column=image_column):
			split_df = read_sheet(path, sheet_name=sheet_name, have_header=have_header)
			split_root = root
			split_image_column = image_column
		case MainTable(table=sheet_reader):
			assert source_df is not None
			split_df = source_df
			split_root = sheet_reader.root
			split_image_column = sheet_reader.image_column
		case _ as unreachable:
			assert_never(unreachable)

	# 第一阶段：解析图像与划分列构建规范化路径映射
	split_img_series = get_column_series(split_df, split_image_column)
	split_col_series = get_column_series(split_df, split.column)

	split_map: dict[str, object] = {}
	for raw_img, split_val in zip(split_img_series.tolist(), split_col_series.tolist()):
		img_key = (split_root / str(raw_img)).resolve().as_posix()
		if img_key in split_map:
			raise ValueError(f'Duplicate image path key in split table: {img_key}')
		split_map[img_key] = split_val

	# 第二阶段：样本覆盖完整性双向比对
	main_keys = [p.resolve().as_posix() for p in dataset.file_paths]
	main_key_set = set(main_keys)
	split_key_set = set(split_map.keys())

	missing = main_key_set - split_key_set
	if missing:
		raise ValueError(
			f'Split table is missing {len(missing)} dataset samples, e.g. {sorted(missing)[:5]}'
		)
	if is_separate:
		extra = split_key_set - main_key_set
		if extra:
			raise ValueError(
				f'Separate split table has {len(extra)} extra rows not in dataset, e.g. {sorted(extra)[:5]}'
			)

	# 第三阶段：分配样本索引
	train_idx: list[int] = []
	val_idx: list[int] = []
	test_idx: list[int] = []

	for idx, key in enumerate(main_keys):
		s_val = split_map[key]
		if s_val == split.train or str(s_val) == str(split.train):
			train_idx.append(idx)
		elif split.val is not None and (
			s_val == split.val or str(s_val) == str(split.val)
		):
			val_idx.append(idx)
		elif split.test is not None and (
			s_val == split.test or str(s_val) == str(split.test)
		):
			test_idx.append(idx)

	# 第四阶段：空划分校验并构建子集
	if not train_idx:
		raise ValueError('Train split is empty')

	splits: dict[str, DiskDataset[ImageT, LabelT]] = {'train': dataset.subset(train_idx)}

	if split.val is not None:
		if not val_idx:
			raise ValueError(f'Validation split declared with val={split.val!r} is empty')
		splits['val'] = dataset.subset(val_idx)

	if split.test is not None:
		if not test_idx:
			raise ValueError(f'Test split declared with test={split.test!r} is empty')
		splits['test'] = dataset.subset(test_idx)

	return splits
