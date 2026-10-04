"""供外部 feeder 使用的样本读取与标签解析工具"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import icontract
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

	def __post_init__(self) -> None:
		if self.num_classes < 1:
			raise ValueError(f'num_classes must be >= 1, got {self.num_classes}')


@icontract.require(lambda sheet_path: sheet_path.suffix.lower() in {'.csv', '.xlsx', '.xls'}, 'Unknown file type')
def read_sheet(
	sheet_path: Path,
	*,
	sheet_name: int | str = 0,
	have_header: bool = False,
) -> pd.DataFrame:
	"""按文件后缀读取 CSV 或 Excel 表格文件为 DataFrame"""
	if sheet_path.suffix.lower() == '.csv':
		return pd.read_csv(
			sheet_path,
			header=0 if have_header else None,
		)
	return pd.read_excel(
		sheet_path,
		sheet_name=sheet_name,
		header=0 if have_header else None,
	)


def _has_column(df: pd.DataFrame, col: int | str) -> bool:
	if col in df.columns:
		return True
	if isinstance(col, int):
		return -df.shape[1] <= col < df.shape[1]
	return col.isdigit() and int(col) in df.columns


@icontract.require(lambda df, col: _has_column(df, col), 'Column not found in sheet columns')
@icontract.ensure(lambda df, result: len(result) == len(df))
def get_column_series(df: pd.DataFrame, col: int | str) -> pd.Series:
	"""从 DataFrame 中根据名称或位置索引获取列 Series"""
	if col in df.columns:
		res = df[col]
		if isinstance(res, pd.DataFrame):
			return res.iloc[:, 0]  # type: ignore
		return res
	if isinstance(col, int):
		return df.iloc[:, col]
	res = df[int(col)]
	if isinstance(res, pd.DataFrame):
		return res.iloc[:, 0]  # type: ignore
	return res


@icontract.ensure(lambda result: len(result[0]) == len(result[1]))
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


def _class_subdirs(root: Path) -> list[Path]:
	return sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)


def _folder_labels_in_range(root: Path, class_dict: dict[str, int], extensions: Sequence[str] | None, num_classes: int) -> bool:
	"""含有图像文件的类别子目录，其 class_dict 标签都位于 [0, num_classes)。"""
	ext_tuple = tuple(extensions) if extensions is not None else None
	return all(
		0 <= class_dict[d.name] < num_classes or not load_files_from_folder(d, extensions=ext_tuple) for d in _class_subdirs(root)
	)


@icontract.require(
	lambda root, class_dict, extensions, num_classes: (
		num_classes is None or class_dict is None or _folder_labels_in_range(root, class_dict, extensions, num_classes)
	),
	'Label out of range [0, num_classes - 1]',
)
@icontract.require(
	lambda root, class_dict, num_classes: num_classes is None or class_dict is not None or len(_class_subdirs(root)) <= num_classes,
	'Discovered classes in folder exceed num_classes',
)
@icontract.require(
	lambda root, class_dict: class_dict is None or all(d.name in class_dict for d in _class_subdirs(root)), 'Class folder not in class_dict'
)
@icontract.ensure(lambda result: len(result[0]) == len(result[1]))
def load_samples_from_folder(
	root: Path,
	class_dict: dict[str, int] | None = None,
	extensions: Sequence[str] | None = None,
	*,
	num_classes: int | None = None,
) -> tuple[list[Path], list[int], dict[str, int]]:
	"""从文件夹按类别子目录结构加载图像路径与标签"""
	subdirs = _class_subdirs(root)
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
		label = resolved_class_dict[folder.name]

		folder_files = load_files_from_folder(folder, extensions=ext_tuple)
		for file in folder_files:
			files.append(file)
			labels.append(label)

	return files, labels, resolved_class_dict


def _numeric_labels(series: pd.Series) -> pd.Series:
	return pd.Series(pd.to_numeric(series, errors='coerce'))


def _parse_labels(series: pd.Series, spec: ColumnLabelSpec) -> pd.Series:
	"""把标签列映射为整型类别编号（class_dict 映射，或数值化后加 offset）"""
	if spec.class_dict is not None:
		return series.astype(str).str.strip().map(spec.class_dict).astype('Int64')
	return pd.Series(pd.to_numeric(series)).astype('Int64') + spec.offset


@icontract.require(
	lambda series, spec: _parse_labels(series, spec).between(0, spec.num_classes - 1).all(), 'Label for target out of range [0, num_classes - 1]'
)
@icontract.require(
	lambda series, spec: spec.class_dict is not None or (_numeric_labels(series) % 1 == 0).all(),
	'Label for target is fractional; non-integral labels are not allowed',
)
@icontract.require(
	lambda series, spec: spec.class_dict is not None or _numeric_labels(series).notna().all(), 'Cannot parse label for target as integer'
)
@icontract.require(
	lambda series, spec: spec.class_dict is None or series.astype(str).str.strip().isin(list(spec.class_dict)).all(),
	'Value for target not in declared class_dict',
)
@icontract.require(lambda series: not series.isna().any(), 'Missing label for target')
@icontract.ensure(lambda series, result: len(result) == len(series))
def parse_target_labels(
	series: pd.Series,
	spec: ColumnLabelSpec,
	name: str,
) -> list[int]:
	"""解析并校验标签列，利用 Pandas 类型系统保证整型约束及类别范围"""
	return _parse_labels(series, spec).tolist()


@icontract.ensure(lambda df, result: len(result) == len(df))
def read_single_label(
	df: pd.DataFrame,
	spec: ColumnLabelSpec,
	target_name: str = 'label',
) -> list[int]:
	"""从数据表中提取并解析单任务标签"""
	lbl_series = get_column_series(df, spec.column)
	return parse_target_labels(lbl_series, spec, target_name)


@icontract.ensure(lambda df, result: len(result) == len(df))
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


@icontract.ensure(lambda df, result: len(result) == len(df))
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


def _split_table_keys(table: pd.DataFrame, root: Path, image_column: int | str) -> list[str]:
	"""划分表中各行图像的规范化路径键"""
	return [(root / str(raw_img)).resolve().as_posix() for raw_img in get_column_series(table, image_column).tolist()]


def _dataset_keys(file_paths: Sequence[Path]) -> list[str]:
	"""数据集样本的规范化路径键"""
	return [p.resolve().as_posix() for p in file_paths]


@icontract.require(
	lambda file_paths, table, root, image_column: set(_split_table_keys(table, root, image_column)) <= set(_dataset_keys(file_paths)),
	'Split table has extra rows not in dataset',
)
@icontract.require(
	lambda file_paths, table, root, image_column: set(_dataset_keys(file_paths)) <= set(_split_table_keys(table, root, image_column)),
	'Split table is missing dataset samples',
)
@icontract.require(
	lambda table, root, image_column: len(set(_split_table_keys(table, root, image_column))) == len(table),
	'Duplicate image path key in split table',
)
@icontract.ensure(lambda partition_values, result: len(result) == len(partition_values))
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
	split_col_series = get_column_series(table, assignment_column)

	split_map: dict[str, int | None] = {}
	for img_key, split_val in zip(_split_table_keys(table, root, image_column), split_col_series.tolist()):
		partition_index: int | None = None
		for group_idx, target in enumerate(partition_values):
			if split_val == target or str(split_val) == str(target):
				partition_index = group_idx
				break
		split_map[img_key] = partition_index

	# 第二阶段：按输入顺序分配样本索引至对应分区
	groups: list[list[int]] = [[] for _ in partition_values]
	for idx, key in enumerate(_dataset_keys(file_paths)):
		partition_index = split_map[key]
		if partition_index is not None:
			groups[partition_index].append(idx)

	return groups
