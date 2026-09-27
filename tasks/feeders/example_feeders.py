"""示例外部 feeder：表格单标签 + 比例划分

task 文件通过顶层 imports 字段导入本文件完成注册：

	imports = ['feeders/example_feeders.py']

	[dataset]
	feeder = 'example_sheet_ratio'
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from daisy.dataset.dataset_split import default_data_split, stratified_data_split
from daisy.dataset.disk_dataset import DiskDataset
from daisy.feeder import ColumnLabelSpec, FeederSplits, parse_single_label_table, read_sheet, register_feeder


class SheetRatioParams(BaseModel):
	"""表格单标签比例划分参数"""

	model_config = ConfigDict(extra='forbid')

	root: Path
	sheet: Path
	sheet_name: int | str = 0
	have_header: bool = True
	image_column: int | str = 0
	label_column: int | str = 1
	num_classes: int = Field(ge=2)
	label_offset: int = 0
	val_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
	split_seed: int | None = None
	stratify: bool = False


@register_feeder('example_sheet_ratio')
def sheet_ratio(params: SheetRatioParams, *, seed: int | None) -> FeederSplits:
	"""读取表格中的图像路径与单标签，再按比例划分 train / val"""
	df = read_sheet(params.sheet, sheet_name=params.sheet_name, have_header=params.have_header)
	spec = ColumnLabelSpec(column=params.label_column, num_classes=params.num_classes, offset=params.label_offset, class_dict=None)
	files, labels = parse_single_label_table(df, params.root, spec, image_column=params.image_column)
	dataset = DiskDataset(files, labels, nn.Identity())

	split_seed = params.split_seed if params.split_seed is not None else seed
	if params.stratify:
		train, val = stratified_data_split(dataset, val_ratio=params.val_ratio, seed=split_seed)
	else:
		train, val = default_data_split(dataset, val_ratio=params.val_ratio, seed=split_seed)
	return {'train': train, 'val': val}
