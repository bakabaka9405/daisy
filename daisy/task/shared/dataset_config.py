"""共享的带标签数据集配置"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DatasetSplitConfig(BaseModel):
	"""数据集划分配置"""

	method: Literal['none', 'ratio', 'sheet', 'preset'] = 'ratio'
	val_ratio: float = 0.1
	seed: int | None = None
	stratified: bool = False
	val_sheet: str | None = None
	val_sheet_name: str | None = None
	preset_train_dir: str = 'train'
	preset_val_dir: str = 'val'
	preset_test_dir: str = 'test'


class DatasetConfig(BaseModel):
	"""带标签数据集配置"""

	type: Literal['sheet', 'folder'] = 'sheet'
	root: str = ''
	sheet: str | None = None
	sheet_name: str | None = None
	column: int = 1
	label_offset: int = 0
	have_header: bool = True
	split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
