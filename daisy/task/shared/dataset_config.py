"""共享的带标签数据集配置"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DatasetSplitConfig(BaseModel):
	"""数据集划分配置"""

	method: Literal['none', 'ratio', 'sheet', 'preset', 'manifest'] = 'ratio'
	val_ratio: float = 0.1
	seed: int | None = None
	stratified: bool = False
	val_sheet: str | None = None
	val_sheet_name: str | None = None
	train_files: list[str] = Field(default_factory=list)
	val_files: list[str] = Field(default_factory=list)
	test_files: list[str] = Field(default_factory=list)
	preset_train_dir: str = 'train'
	preset_val_dir: str = 'val'
	preset_test_dir: str = 'test'
	manifest: str | None = None
	manifest_id_type: Literal['relative_path', 'name', 'path'] = 'relative_path'
	manifest_train_split: str = 'train'
	manifest_val_split: str = 'val'
	manifest_test_split: str = 'test'
	require_all_in_manifest: bool = False


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
