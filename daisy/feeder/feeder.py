"""Feeder 数据供给抽象与多模态实现"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import pandas as pd

from daisy.dataset.disk_dataset import load_files_from_folder
from .primitives import (
	LabelSpec,
	get_column_series,
	load_samples_from_folder,
	read_single_label,
)


def _extract_table_files(
	df: pd.DataFrame,
	dataset_root: Path | str,
	image_column: int | str = 0,
) -> list[Path]:
	"""从数据表中提取图像路径列表"""
	root = Path(dataset_root) if isinstance(dataset_root, str) else dataset_root
	img_series = get_column_series(df, image_column)
	return [root / str(x) for x in img_series.tolist()]


class Feeder[L](Protocol):
	"""样本读取与供给协议，解耦数据来源介质与标签模式"""

	def from_folder(
		self,
		root: Path,
		extensions: Sequence[str] | None = None,
	) -> tuple[list[Path], list[L] | None]: ...

	def from_table(
		self,
		df: pd.DataFrame,
		dataset_root: Path | str,
		image_column: int | str = 0,
	) -> tuple[list[Path], list[L] | None]: ...

	def table_labels(self, df: pd.DataFrame) -> list[L] | None: ...

	def stratify(self, labels: list[L] | None, key: str) -> list[int]: ...


class UnlabeledFeeder:
	"""自监督无标签 Feeder 实现"""

	def from_folder(
		self,
		root: Path,
		extensions: Sequence[str] | None = None,
	) -> tuple[list[Path], list[None] | None]:
		files = load_files_from_folder(root, extensions=tuple(extensions) if extensions is not None else None)
		return files, None

	def from_table(
		self,
		df: pd.DataFrame,
		dataset_root: Path | str,
		image_column: int | str = 0,
	) -> tuple[list[Path], list[None] | None]:
		files = _extract_table_files(df, dataset_root, image_column)
		return files, None

	def table_labels(self, df: pd.DataFrame) -> list[None] | None:
		return None

	def stratify(self, labels: list[None] | None, key: str) -> list[int]:
		raise ValueError('Cannot stratify unlabeled dataset')


class SingleLabelFeeder:
	"""单标签分类 Feeder 实现，维持文件夹扫描推导的类别映射"""

	def __init__(self, spec: LabelSpec) -> None:
		self.spec = spec
		self.class_dict: dict[str, int] | None = dict(spec.class_dict) if spec.class_dict is not None else None

	def from_folder(
		self,
		root: Path,
		extensions: Sequence[str] | None = None,
	) -> tuple[list[Path], list[int]]:
		files, labels, resolved = load_samples_from_folder(
			root,
			class_dict=self.class_dict,
			extensions=extensions,
			num_classes=self.spec.num_classes,
		)
		if self.class_dict is None:
			self.class_dict = resolved
		return files, labels

	def from_table(
		self,
		df: pd.DataFrame,
		dataset_root: Path | str,
		image_column: int | str = 0,
	) -> tuple[list[Path], list[int]]:
		files = _extract_table_files(df, dataset_root, image_column)
		return files, self.table_labels(df)

	def table_labels(self, df: pd.DataFrame, target_name: str = 'label') -> list[int]:
		return read_single_label(df, self.spec, target_name=target_name)

	def stratify(self, labels: list[int] | None, key: str) -> list[int]:
		assert labels is not None
		return labels


class MultiTaskFeeder:
	"""多任务标签 Feeder 实现，聚合各任务的 SingleLabelFeeder"""

	def __init__(self, specs: Mapping[str, LabelSpec]) -> None:
		self.feeders: dict[str, SingleLabelFeeder] = {name: SingleLabelFeeder(spec) for name, spec in specs.items()}

	def from_folder(
		self,
		root: Path,
		extensions: Sequence[str] | None = None,
	) -> tuple[list[Path], list[dict[str, int]]]:
		(name, feeder), = self.feeders.items()
		files, single_labels = feeder.from_folder(root, extensions)
		return files, [{name: lbl} for lbl in single_labels]

	def from_table(
		self,
		df: pd.DataFrame,
		dataset_root: Path | str,
		image_column: int | str = 0,
	) -> tuple[list[Path], list[dict[str, int]]]:
		files = _extract_table_files(df, dataset_root, image_column)
		return files, self.table_labels(df)

	def table_labels(self, df: pd.DataFrame) -> list[dict[str, int]]:
		parsed: dict[str, list[int]] = {}
		for name, feeder in self.feeders.items():
			parsed[name] = feeder.table_labels(df, target_name=name)

		num_samples = len(df)
		return [{name: parsed[name][i] for name in self.feeders} for i in range(num_samples)]

	def stratify(self, labels: list[dict[str, int]] | None, key: str) -> list[int]:
		assert labels is not None
		return [row[key] for row in labels]


def read_multitask_labels(
	df: pd.DataFrame,
	specs: Mapping[str, LabelSpec],
) -> list[dict[str, int]]:
	"""从数据表中提取并解析多任务标签"""
	return MultiTaskFeeder(specs).table_labels(df)


def merge_samples[L](
	samples: Sequence[tuple[list[Path], list[L] | None]],
) -> tuple[list[Path], list[L] | None]:
	"""合并多组 (files, labels) 样本数据"""
	if not samples:
		return [], None
	merged_files: list[Path] = []
	has_labels = samples[0][1] is not None
	merged_labels: list[L] = []
	for files, labels in samples:
		if (labels is not None) != has_labels:
			raise ValueError('Cannot merge labeled samples with unlabeled samples')
		merged_files.extend(files)
		if has_labels:
			assert labels is not None
			merged_labels.extend(labels)
	return merged_files, (merged_labels if has_labels else None)
