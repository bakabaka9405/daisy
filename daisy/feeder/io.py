"""供外部 feeder 使用的样本读取与标签解析工具"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from daisy.dataset.disk_dataset import load_files_from_folder

DEFAULT_IMAGE_EXTENSIONS: list[str] = [
	'.jpg',
	'.jpeg',
	'.png',
	'.bmp',
	'.webp',
	'.tiff',
	'.tif',
]


@dataclass(slots=True, frozen=True)
class ColumnLabelSpec:
	"""标签解析所需的描述约束"""

	column: int | str
	num_classes: int
	offset: int
	class_dict: dict[str, int] | None


def read_sheet(
	sheet_path: Path,
	*,
	sheet_name: int | str = 0,
	have_header: bool = False,
) -> pd.DataFrame:
	"""按文件后缀读取 CSV 或 Excel 表格文件为 DataFrame"""
	suffix = sheet_path.suffix.lower()

	if suffix == '.csv':
		return pd.read_csv(
			sheet_path,
			header=0 if have_header else None,
		)
	if suffix in ('.xlsx', '.xls'):
		return pd.read_excel(
			sheet_path,
			sheet_name=sheet_name,
			header=0 if have_header else None,
		)
	raise ValueError(f'Unknown file type {sheet_path.suffix}')


def get_column_series(df: pd.DataFrame, col: int | str) -> pd.Series:
	"""从 DataFrame 中根据名称或位置索引获取列 Series"""
	if col in df.columns:
		res = df[col]
		if isinstance(res, pd.DataFrame):
			return res.iloc[:, 0]  # type: ignore
		return res
	if isinstance(col, int):
		return df.iloc[:, col]
	if isinstance(col, str) and col.isdigit() and int(col) in df.columns:
		res = df[int(col)]
		if isinstance(res, pd.DataFrame):
			return res.iloc[:, 0]  # type: ignore
		return res
	raise KeyError(f'Column {col!r} not found in sheet columns: {list(df.columns)}')


def load_samples_from_sheet(
	dataset_root: Path,
	sheet_path: Path,
	sheet_name: int | str = 0,
	have_header: bool = False,
	column: int | str = 1,
	label_offset: int = 0,
	label_dtype: type = int,
	image_column: int | str = 0,
) -> tuple[list[Path], list[Any]]:
	"""从表格加载图像路径与标签列表"""
	df = read_sheet(
		sheet_path,
		sheet_name=sheet_name,
		have_header=have_header,
	)

	img_series = get_column_series(df, image_column)
	files = [dataset_root / str(x) for x in img_series.tolist()]
	label_series = get_column_series(df, column)
	raw_labels = label_series.astype(label_dtype).tolist()
	labels = [x + label_offset for x in raw_labels]

	return files, labels


def load_samples_from_folder(
	root: Path,
	class_dict: dict[str, int] | None = None,
	extensions: Sequence[str] | None = None,
	*,
	num_classes: int | None = None,
) -> tuple[list[Path], list[int], dict[str, int]]:
	"""从文件夹按类别子目录结构加载图像路径与标签"""
	subdirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)
	should_build_dict = class_dict is None

	if should_build_dict:
		resolved_class_dict: dict[str, int] = {folder.name: i for i, folder in enumerate(subdirs)}
	else:
		assert class_dict is not None
		resolved_class_dict = class_dict

	files: list[Path] = []
	labels: list[int] = []

	ext_tuple = tuple(extensions) if extensions is not None else None

	for folder in subdirs:
		if folder.name not in resolved_class_dict:
			raise ValueError(f'Class {folder.name} not in class_dict')
		label = resolved_class_dict[folder.name]

		folder_files = load_files_from_folder(folder, extensions=ext_tuple)
		for file in folder_files:
			files.append(file)
			labels.append(label)

	if num_classes is not None:
		if class_dict is None and len(resolved_class_dict) > num_classes:
			raise ValueError(f'Discovered {len(resolved_class_dict)} classes in folder, exceeding num_classes={num_classes}')
		for lbl in labels:
			if lbl < 0 or lbl >= num_classes:
				raise ValueError(f'Label {lbl} out of range [0, {num_classes - 1}]')

	return files, labels, resolved_class_dict


def parse_target_labels(
	series: pd.Series,
	spec: ColumnLabelSpec,
	name: str,
) -> list[int]:
	"""解析并校验标签列，利用 Pandas 类型系统保证整型约束及类别范围"""
	if series.isna().any():
		raise ValueError(f'Missing label for target {name!r}')

	if spec.class_dict is not None:
		mapped = series.astype(str).str.strip().map(spec.class_dict)
		if mapped.isna().any():
			unmapped = series.loc[mapped.isna()].iloc[0]
			raise ValueError(f'Value {unmapped!r} for target {name!r} not in declared class_dict')
		labels = mapped.astype('Int64')
	else:
		try:
			numeric = pd.Series(pd.to_numeric(series, errors='raise'))
		except (ValueError, TypeError) as err:
			raise ValueError(f'Cannot parse label for target {name!r} as integer: {err}') from err

		try:
			labels = numeric.astype('Int64')
		except TypeError as err:
			bad = series.loc[numeric != numeric.astype('int64')].iloc[0]
			raise ValueError(f'Label {bad!r} for target {name!r} is fractional; non-integral labels are not allowed') from err

		labels = labels + spec.offset

	if not labels.between(0, spec.num_classes - 1).all():
		bad = labels.loc[~labels.between(0, spec.num_classes - 1)].iloc[0]
		raise ValueError(f'Label {bad} for target {name!r} out of range [0, {spec.num_classes - 1}]')

	return labels.tolist()


def read_single_label(
	df: pd.DataFrame,
	spec: ColumnLabelSpec,
	target_name: str = 'label',
) -> list[int]:
	"""从数据表中提取并解析单任务标签"""
	lbl_series = get_column_series(df, spec.column)
	return parse_target_labels(lbl_series, spec, target_name)


def parse_table_paths(
	df: pd.DataFrame,
	dataset_root: Path,
	image_column: int | str = 0,
) -> list[Path]:
	"""从数据表中提取图像路径列表"""
	img_series = get_column_series(df, image_column)
	return [dataset_root / str(x) for x in img_series.tolist()]


def parse_single_label_table(
	df: pd.DataFrame,
	dataset_root: Path,
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
	dataset_root: Path,
	specs: Mapping[str, ColumnLabelSpec],
	*,
	image_column: int | str = 0,
) -> tuple[list[Path], list[dict[str, int]]]:
	"""从数据表中解析图像路径与多任务标签"""
	paths = parse_table_paths(df, dataset_root, image_column=image_column)
	labels = read_multitask_labels(df, specs)
	return paths, labels


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
