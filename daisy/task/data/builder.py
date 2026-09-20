"""声明式数据集构建器"""

from __future__ import annotations

from typing import Any
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
from ..shared.dataset_config import (
	DatasetConfig,
	FolderSourceConfig,
	LabelConfig,
	MultiSourceDatasetConfig,
	NoSplitConfig,
	RatioSplitConfig,
	SheetSourceConfig,
	SingleLabelConfig,
	SingleSourceDatasetConfig,
	SourceConfig,
	SplitConfig,
	UnlabeledConfig,
)
from .column_split import split_by_column

type DatasetSplits = dict[str, DiskDataset[Tensor, Any]]


def _identity_transform[T](x: T) -> T:
	"""模块级恒等变换，支持 spawn 进程安全序列化"""
	return x


def create_feeder(labels: LabelConfig) -> Feeder[Any]:
	"""根据标签配置创建对应的 Feeder 实例"""
	if isinstance(labels, UnlabeledConfig):
		return UnlabeledFeeder()
	if isinstance(labels, SingleLabelConfig):
		return SingleLabelFeeder(labels.target)
	return MultiTaskFeeder(labels.targets)


def _read_source[L](
	source: SourceConfig,
	feeder: Feeder[L],
) -> tuple[DiskDataset[Tensor, L], pd.DataFrame | None]:
	"""读取单一数据源，分支处理 folder 与 sheet 并构建 DiskDataset"""
	if isinstance(source, FolderSourceConfig):
		roots = source.root if isinstance(source.root, list) else [source.root]
		datasets = [
			DiskDataset(*feeder.from_folder(r, source.extensions), _identity_transform, backend='tensor')
			for r in roots
		]
		dataset = DiskDataset.concat(datasets)
		if not dataset.file_paths:
			raise ValueError('Dataset source is empty')
		return dataset, None

	df = read_sheet(
		source.path,
		sheet_name=source.sheet_name,
		have_header=source.have_header,
	)
	files, labels = feeder.from_table(df, source.root, source.image_column)
	if not files:
		raise ValueError('Dataset source is empty')

	dataset: DiskDataset[Tensor, L] = DiskDataset(files, labels, _identity_transform, backend='tensor')
	return dataset, df


def _split_single_dataset[L](
	dataset: DiskDataset[Tensor, L],
	split_cfg: SplitConfig,
	feeder: Feeder[L],
	*,
	caller_seed: int | None,
	source_cfg: SourceConfig,
	source_df: pd.DataFrame | None,
) -> dict[str, DiskDataset[Tensor, L]]:
	"""对单一来源数据集执行划分"""
	if isinstance(split_cfg, NoSplitConfig):
		return {'train': dataset}

	if isinstance(split_cfg, RatioSplitConfig):
		effective_seed = split_cfg.seed if split_cfg.seed is not None else caller_seed
		if split_cfg.stratify is not None:
			strat_labels = feeder.stratify(dataset.labels, split_cfg.stratify)
			train_ds, val_ds = stratified_data_split(
				dataset,
				val_ratio=split_cfg.val_ratio,
				seed=effective_seed,
				stratify_labels=strat_labels,
			)
		else:
			train_ds, val_ds = default_data_split(
				dataset,
				val_ratio=split_cfg.val_ratio,
				seed=effective_seed,
			)

		if len(train_ds) == 0 or len(val_ds) == 0:
			raise ValueError('Train or validation split is empty')
		return {'train': train_ds, 'val': val_ds}

	# ColumnSplitConfig：根据是否配置独立 split 表进行分支
	if split_cfg.source is not None:
		split_source = split_cfg.source
		split_df = read_sheet(
			split_source.path,
			sheet_name=split_source.sheet_name,
			have_header=split_source.have_header,
		)
		return split_by_column(
			dataset,
			split_cfg,
			split_df,
			split_source.root,
			split_source.image_column,
			is_separate=True,
		)

	assert source_df is not None and isinstance(source_cfg, SheetSourceConfig)
	return split_by_column(
		dataset,
		split_cfg,
		source_df,
		source_cfg.root,
		source_cfg.image_column,
		is_separate=False,
	)


def build_datasets(
	config: DatasetConfig,
	*,
	seed: int | None = None,
) -> DatasetSplits:
	"""根据 DatasetConfig 声明式构建 DiskDataset 字典

	Args:
		config: 数据集配置对象（单来源或多来源划分）
		seed: 调用方提供的随机种子

	Returns:
		划分名称到 DiskDataset 的字典
	"""
	feeder = create_feeder(config.labels)

	if isinstance(config, SingleSourceDatasetConfig):
		main_dataset, source_df = _read_source(config.source, feeder)
		splits = _split_single_dataset(
			main_dataset,
			config.split,
			feeder,
			caller_seed=seed,
			source_cfg=config.source,
			source_df=source_df,
		)
	else:
		# 保证 train 优先遍历以确保类别映射先在训练集解析
		splits: dict[str, DiskDataset[Tensor, Any]] = {}
		sorted_sources = sorted(
			config.sources.items(),
			key=lambda item: 0 if item[0] == 'train' else 1,
		)
		for s_name, s_cfg in sorted_sources:
			split_ds, _ = _read_source(s_cfg, feeder)
			splits[s_name] = split_ds

	assert_disjoint_file_splits(splits)

	if config.sample_ratio < 1.0:
		if config.sample_seed is not None:
			subsample_seed = config.sample_seed
		elif (
			isinstance(config, SingleSourceDatasetConfig)
			and isinstance(config.split, RatioSplitConfig)
			and config.split.seed is not None
		):
			subsample_seed = config.split.seed
		else:
			subsample_seed = seed

		splits['train'] = random_subsample(splits['train'], config.sample_ratio, subsample_seed)

	return splits
