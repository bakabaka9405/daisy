"""共享的带标签数据集配置"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from ..base import BaseConfig


class DatasetSplitConfig(BaseConfig):
	"""数据集划分配置"""

	method: Literal['none', 'ratio', 'sheet', 'preset'] = 'ratio'
	val_ratio: float = 0.1
	seed: int | None = None
	stratified: bool = False
	val_sheet: Path | None = None
	val_sheet_name: str | None = None
	preset_train_dir: Path = Path('train')
	preset_val_dir: Path = Path('val')
	preset_test_dir: Path = Path('test')


class DatasetConfig(BaseConfig):
	"""带标签数据集配置"""

	type: Literal['sheet', 'folder'] = 'sheet'
	root: Path = Path()
	sheet: Path | None = None
	sheet_name: str | None = None
	column: int = 1
	label_offset: int = 0
	have_header: bool = True
	split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
