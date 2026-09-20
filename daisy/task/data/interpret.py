"""DatasetPlan 代数解释器"""

from __future__ import annotations

from typing import assert_never, cast
import pandas as pd
from torch import Tensor

from daisy.dataset.dataset_sample import random_subsample
from daisy.dataset.dataset_split import (
	assert_disjoint_file_splits,
	default_data_split,
	stratified_data_split,
)
from daisy.dataset.disk_dataset import DiskDataset
from daisy.feeder import (
	Feeder,
	MultiTaskFeeder,
	SingleLabelFeeder,
	UnlabeledFeeder,
	read_sheet,
)
from .column_split import split_by_column
from .model import (
	ColumnSplit,
	DatasetPlan,
	DatasetSplits,
	FolderReader,
	LabelerTerm,
	MultiLabels,
	MultiSourcePlan,
	NoLabels,
	NoSplit,
	RatioSplit,
	ReaderTerm,
	SheetReader,
	SingleLabels,
	SingleSourcePlan,
	SplitTerm,
)


def _identity_transform[T](x: T) -> T:
	"""模块级恒等变换，支持 spawn 进程安全序列化"""
	return x


def _create_feeder[L](labeler: LabelerTerm[L]) -> Feeder[L]:
	"""根据标签代数项创建对应的 Feeder 实例"""
	match labeler:
		case NoLabels():
			return cast(Feeder[L], UnlabeledFeeder())
		case SingleLabels(target=spec):
			return cast(Feeder[L], SingleLabelFeeder(spec))
		case MultiLabels(targets=targets):
			return cast(Feeder[L], MultiTaskFeeder(dict(targets)))
		case _ as unreachable:
			assert_never(unreachable)


def _read_source[L](
	reader: ReaderTerm,
	feeder: Feeder[L],
) -> tuple[DiskDataset[Tensor, L], pd.DataFrame | None]:
	"""读取单一数据源并构建 DiskDataset"""
	match reader:
		case FolderReader(roots=roots, extensions=extensions):
			datasets: list[DiskDataset[Tensor, L]] = [
				DiskDataset(*feeder.from_folder(r, extensions), _identity_transform, backend='tensor')
				for r in roots
			]
			dataset = DiskDataset.concat(datasets)
			if not dataset.file_paths:
				raise ValueError('Dataset source is empty')
			return dataset, None

		case SheetReader(
			path=path,
			sheet_name=sheet_name,
			have_header=have_header,
			root=root,
			image_column=image_column,
		):
			df = read_sheet(
				path,
				sheet_name=sheet_name,
				have_header=have_header,
			)
			files, labels = feeder.from_table(df, root, image_column)
			if not files:
				raise ValueError('Dataset source is empty')
			dataset: DiskDataset[Tensor, L] = DiskDataset(files, labels, _identity_transform, backend='tensor')
			return dataset, df

		case _ as unreachable:
			assert_never(unreachable)


def _split_single_dataset[L](
	dataset: DiskDataset[Tensor, L],
	split: SplitTerm,
	feeder: Feeder[L],
	*,
	caller_seed: int | None,
	source_df: pd.DataFrame | None,
) -> dict[str, DiskDataset[Tensor, L]]:
	"""对单一来源数据集执行划分代数项解释"""
	match split:
		case NoSplit():
			return {'train': dataset}

		case RatioSplit(val_ratio=val_ratio, seed=split_seed, stratify=stratify):
			effective_seed = split_seed if split_seed is not None else caller_seed
			if stratify is not None:
				strat_labels = feeder.stratify(dataset.labels, stratify)
				train_ds, val_ds = stratified_data_split(
					dataset,
					val_ratio=val_ratio,
					seed=effective_seed,
					stratify_labels=strat_labels,
				)
			else:
				train_ds, val_ds = default_data_split(
					dataset,
					val_ratio=val_ratio,
					seed=effective_seed,
				)

			if len(train_ds) == 0 or len(val_ds) == 0:
				raise ValueError('Train or validation split is empty')
			return {'train': train_ds, 'val': val_ds}

		case ColumnSplit() as col_split:
			return split_by_column(
				dataset,
				col_split,
				source_df=source_df,
			)

		case _ as unreachable:
			assert_never(unreachable)


def interpret[L](
	plan: DatasetPlan[L],
	*,
	seed: int | None = None,
) -> DatasetSplits[L]:
	"""解释 DatasetPlan 代数计划并构建 DiskDataset 字典"""
	splits: dict[str, DiskDataset[Tensor, L]]

	match plan:
		case SingleSourcePlan(source=source, split=split):
			feeder = _create_feeder(source.labeler)
			dataset, source_df = _read_source(source.reader, feeder)
			splits = _split_single_dataset(
				dataset,
				split,
				feeder,
				caller_seed=seed,
				source_df=source_df,
			)

		case MultiSourcePlan(sources=sources):
			feeder = _create_feeder(sources.train.labeler)
			splits = {}
			for name in ('train', 'val', 'test'):
				source = getattr(sources, name)
				if source is not None:
					split_ds, _ = _read_source(source.reader, feeder)
					splits[name] = split_ds

		case _ as unreachable:
			assert_never(unreachable)

	assert_disjoint_file_splits(splits)

	if plan.sample is not None:
		subsample_seed = plan.sample.seed if plan.sample.seed is not None else seed
		splits['train'] = random_subsample(splits['train'], plan.sample.ratio, subsample_seed)

	return splits
