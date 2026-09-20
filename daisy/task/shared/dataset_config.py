"""共享的数据集配置"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
import tyro.conf

from ..base import BaseConfig

DEFAULT_IMAGE_EXTENSIONS: list[str] = [
	'.jpg',
	'.jpeg',
	'.png',
	'.bmp',
	'.webp',
	'.tiff',
	'.tif',
]


class FolderSourceConfig(BaseConfig):
	"""文件夹数据源配置"""

	type: Literal['folder'] = 'folder'
	root: Path | list[Path] = Field(default_factory=Path)
	extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_IMAGE_EXTENSIONS))


class SheetSourceConfig(BaseConfig):
	"""表格数据源配置"""

	type: Literal['sheet'] = 'sheet'
	root: Path = Field(default_factory=Path)
	path: Path
	sheet_name: int | str = 0
	have_header: bool = True
	image_column: int | str = 0


SourceConfig = Annotated[
	FolderSourceConfig | SheetSourceConfig,
	Field(discriminator='type'),
	tyro.conf.AvoidSubcommands,
]


class DatasetTargetConfig(BaseConfig):
	"""目标标签配置"""

	column: int | str = 1
	num_classes: int = Field(default=2, ge=2)
	offset: int = 0
	class_dict: dict[str, int] | None = None

	@model_validator(mode='after')
	def _validate_target(self) -> Self:
		if self.class_dict is not None:
			for name, label in self.class_dict.items():
				if label < 0 or label >= self.num_classes:
					raise ValueError(
						f"class_dict value {label} for class {name!r} is out of range [0, {self.num_classes - 1}]"
					)
		return self


class UnlabeledConfig(BaseConfig):
	"""自监督无标签配置"""

	mode: Literal['self_supervised'] = 'self_supervised'


class SingleLabelConfig(BaseConfig):
	"""单标签分类配置"""

	mode: Literal['single_label'] = 'single_label'
	target: DatasetTargetConfig = Field(default_factory=DatasetTargetConfig)


class MultiTaskConfig(BaseConfig):
	"""多任务标签配置"""

	mode: Literal['multi_task'] = 'multi_task'
	targets: dict[str, DatasetTargetConfig] = Field(default_factory=dict)

	@model_validator(mode='after')
	def _validate_targets(self) -> Self:
		if not self.targets:
			raise ValueError('multi_task mode requires at least 1 named target')
		return self


LabelConfig = Annotated[
	UnlabeledConfig | SingleLabelConfig | MultiTaskConfig,
	Field(discriminator='mode'),
	tyro.conf.AvoidSubcommands,
]


class NoSplitConfig(BaseConfig):
	"""不划分配置"""

	method: Literal['none'] = 'none'


class RatioSplitConfig(BaseConfig):
	"""比例划分配置"""

	method: Literal['ratio'] = 'ratio'
	val_ratio: float = Field(default=0.1, gt=0.0, lt=1.0)
	seed: int | None = None
	stratify: str | None = None


class ColumnSplitConfig(BaseConfig):
	"""按列划分配置"""

	method: Literal['column'] = 'column'
	column: int | str = 'split'
	train: str | int = 'train'
	val: str | int | None = 'val'
	test: str | int | None = None
	source: SheetSourceConfig | None = None


SplitConfig = Annotated[
	NoSplitConfig | RatioSplitConfig | ColumnSplitConfig,
	Field(discriminator='method'),
	tyro.conf.AvoidSubcommands,
]


class BaseDatasetConfig(BaseConfig):
	"""共享采样与标签字段的基类"""

	labels: LabelConfig = Field(default_factory=SingleLabelConfig)
	sample_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
	sample_seed: int | None = None


class SingleSourceDatasetConfig(BaseDatasetConfig):
	"""单一来源数据集配置"""

	type: Literal['single'] = 'single'
	source: SourceConfig = Field(default_factory=FolderSourceConfig)
	split: SplitConfig = Field(default_factory=RatioSplitConfig)


class MultiSourceDatasetConfig(BaseDatasetConfig):
	"""多来源划分数据集配置"""

	type: Literal['sources'] = 'sources'
	sources: dict[Literal['train', 'val', 'test'], SourceConfig] = Field(default_factory=dict)


DatasetConfig = Annotated[
	SingleSourceDatasetConfig | MultiSourceDatasetConfig,
	Field(discriminator='type'),
	tyro.conf.AvoidSubcommands,
]
