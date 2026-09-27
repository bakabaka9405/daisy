"""数据供给模块：feeder 注册表、内置 feeder 与样本读取工具"""

from . import builtin as builtin
from .builtin import ImageFolderParams
from .io import (
	DEFAULT_IMAGE_EXTENSIONS,
	ColumnLabelSpec,
	get_column_series,
	load_samples_from_folder,
	load_samples_from_sheet,
	parse_column_split_indices,
	parse_multitask_table,
	parse_single_label_table,
	parse_table_paths,
	parse_target_labels,
	read_multitask_labels,
	read_sheet,
	read_single_label,
)
from .registry import FEEDERS, FeederEntry, FeederFn, FeederSplits, get_feeder, register_feeder

__all__ = [
	'DEFAULT_IMAGE_EXTENSIONS',
	'FEEDERS',
	'ColumnLabelSpec',
	'FeederEntry',
	'FeederFn',
	'FeederSplits',
	'ImageFolderParams',
	'get_column_series',
	'get_feeder',
	'load_samples_from_folder',
	'load_samples_from_sheet',
	'parse_column_split_indices',
	'parse_multitask_table',
	'parse_single_label_table',
	'parse_table_paths',
	'parse_target_labels',
	'read_multitask_labels',
	'read_sheet',
	'read_single_label',
	'register_feeder',
]
