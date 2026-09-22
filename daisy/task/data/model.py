"""声明式数据集构建核心代数项定义"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from torch import Tensor

from daisy.dataset.disk_dataset import DiskDataset
from daisy.feeder import LabelSpec


@dataclass(slots=True, frozen=True)
class NoLabels:
	"""自监督无标签规范"""


@dataclass(slots=True, frozen=True)
class SingleLabels:
	"""单标签分类规范"""

	target: LabelSpec


@dataclass(slots=True, frozen=True)
class MultiLabels:
	"""多任务标签规范"""

	targets: tuple[tuple[str, LabelSpec], ...]


type LabelerTerm[L = Any] = NoLabels | SingleLabels | MultiLabels


@dataclass(slots=True, frozen=True)
class FolderReader:
	"""文件夹数据源读取规范"""

	roots: tuple[Path, ...]
	extensions: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SheetReader:
	"""表格数据源读取规范"""

	path: Path
	sheet_name: int | str
	have_header: bool
	root: Path
	image_column: int | str


type ReaderTerm = FolderReader | SheetReader


@dataclass(slots=True, frozen=True)
class NoSplit:
	"""不划分规范"""


@dataclass(slots=True, frozen=True)
class RatioSplit:
	"""按比例划分规范"""

	val_ratio: float
	seed: int | None
	stratify: str | None


@dataclass(slots=True, frozen=True)
class ColumnSplit:
	"""按数据表列划分规范"""

	column: int | str
	partitions: tuple[tuple[str, str | int], ...]
	source: SheetReader


type SplitTerm = NoSplit | RatioSplit | ColumnSplit


@dataclass(slots=True, frozen=True)
class Sample:
	"""数据子集采样规范"""

	ratio: float
	seed: int | None


@dataclass(slots=True, frozen=True)
class Source[L = Any]:
	"""单一数据源定义，绑定读取规范与标签规范"""

	reader: ReaderTerm
	labeler: LabelerTerm[L]


@dataclass(slots=True, frozen=True)
class MultiSource[L = Any]:
	"""多数据源定义，按划分槽位组织数据源"""

	train: Source[L]
	val: Source[L] | None = None
	test: Source[L] | None = None


@dataclass(slots=True, frozen=True)
class SingleSourcePlan[L = Any]:
	"""单数据源执行计划"""

	source: Source[L]
	split: SplitTerm
	sample: Sample | None


@dataclass(slots=True, frozen=True)
class MultiSourcePlan[L = Any]:
	"""多数据源执行计划"""

	sources: MultiSource[L]
	sample: Sample | None


type DatasetPlan[L = Any] = SingleSourcePlan[L] | MultiSourcePlan[L]

type DatasetSplits[L = Any] = dict[str, DiskDataset[Tensor, L]]
