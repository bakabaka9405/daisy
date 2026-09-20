"""数据样本加载与标签解析底层工具"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import pandas as pd

from daisy.dataset.disk_dataset import load_files_from_folder


@dataclass(slots=True, frozen=True)
class LabelSpec:
	"""标签解析所需的描述约束"""

	column: int | str
	num_classes: int
	offset: int
	class_dict: dict[str, int] | None


def read_sheet(
	sheet_path: Path | str,
	*,
	sheet_name: int | str = 0,
	have_header: bool = False,
) -> pd.DataFrame:
	"""按文件后缀读取 CSV 或 Excel 表格文件为 DataFrame"""
	sheet_path = Path(sheet_path)
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
	dataset_root: Path | str,
	sheet_path: Path | str,
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

	root = Path(dataset_root) if isinstance(dataset_root, str) else dataset_root
	img_series = get_column_series(df, image_column)
	files = [root / str(x) for x in img_series.tolist()]
	label_series = get_column_series(df, column)
	raw_labels = label_series.astype(label_dtype).tolist()
	labels = [x + label_offset for x in raw_labels]

	return files, labels


def load_samples_from_folder(
	root: Path | str,
	class_dict: dict[str, int] | None = None,
	extensions: Sequence[str] | None = None,
	*,
	num_classes: int | None = None,
) -> tuple[list[Path], list[int], dict[str, int]]:
	"""从文件夹按类别子目录结构加载图像路径与标签"""
	if isinstance(root, str):
		root = Path(root)

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
			raise ValueError(
				f'Discovered {len(resolved_class_dict)} classes in folder, exceeding num_classes={num_classes}'
			)
		for lbl in labels:
			if lbl < 0 or lbl >= num_classes:
				raise ValueError(f'Label {lbl} out of range [0, {num_classes - 1}]')

	return files, labels, resolved_class_dict


def parse_target_labels(
	series: pd.Series,
	spec: LabelSpec,
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
			raise ValueError(
				f'Label {bad!r} for target {name!r} is fractional; non-integral labels are not allowed'
			) from err

		labels = labels + spec.offset

	if not labels.between(0, spec.num_classes - 1).all():
		bad = labels.loc[~labels.between(0, spec.num_classes - 1)].iloc[0]
		raise ValueError(f'Label {bad} for target {name!r} out of range [0, {spec.num_classes - 1}]')

	return labels.tolist()


def read_single_label(
	df: pd.DataFrame,
	spec: LabelSpec,
	target_name: str = 'label',
) -> list[int]:
	"""从数据表中提取并解析单任务标签"""
	lbl_series = get_column_series(df, spec.column)
	return parse_target_labels(lbl_series, spec, target_name)
