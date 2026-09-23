"""DatasetConfig 声明式配置解析器"""

from __future__ import annotations

from typing import Any, assert_never, cast

from daisy.feeder import ColumnLabelSpec
from ..shared.dataset_config import (
	ColumnSplitConfig,
	DatasetConfig,
	DatasetTargetConfig,
	FolderSourceConfig,
	LabelConfig,
	MultiSourceDatasetConfig,
	MultiTaskConfig,
	NoSplitConfig,
	RatioSplitConfig,
	SheetSourceConfig,
	SingleLabelConfig,
	SingleSourceDatasetConfig,
	SourceConfig,
	UnlabeledConfig,
)
from .model import (
	ColumnSplit,
	DatasetPlan,
	FolderReader,
	LabelerTerm,
	MultiLabels,
	MultiSource,
	MultiSourcePlan,
	NoLabels,
	NoSplit,
	RatioSplit,
	ReaderTerm,
	Sample,
	SheetReader,
	SingleLabels,
	SingleSourcePlan,
	Source,
	SplitTerm,
)


def _parse_target_spec(target_cfg: DatasetTargetConfig) -> ColumnLabelSpec:
	"""解析目标标签参数并复制类别映射字典"""
	return ColumnLabelSpec(
		column=target_cfg.column,
		num_classes=target_cfg.num_classes,
		offset=target_cfg.offset,
		class_dict=dict(target_cfg.class_dict) if target_cfg.class_dict is not None else None,
	)


def _parse_source_reader(source: SourceConfig) -> ReaderTerm:
	"""将数据源配置转换为对应的代数项 Reader"""
	match source:
		case FolderSourceConfig():
			roots = tuple(source.root) if isinstance(source.root, list) else (source.root,)
			extensions = tuple(source.extensions)
			return FolderReader(roots=roots, extensions=extensions)
		case SheetSourceConfig():
			return SheetReader(
				path=source.path,
				sheet_name=source.sheet_name,
				have_header=source.have_header,
				root=source.root,
				image_column=source.image_column,
			)
		case _ as unreachable:
			assert_never(unreachable)


def _parse_labeler(labels: LabelConfig) -> LabelerTerm[Any]:
	"""将标签配置转换为对应的代数项 Labeler"""
	match labels:
		case UnlabeledConfig():
			return NoLabels()
		case SingleLabelConfig():
			return SingleLabels(target=_parse_target_spec(labels.target))
		case MultiTaskConfig():
			targets = tuple((name, _parse_target_spec(t)) for name, t in labels.targets.items())
			return MultiLabels(targets=targets)
		case _ as unreachable:
			assert_never(unreachable)


def parse_dataset_config(config: DatasetConfig) -> DatasetPlan[Any]:
	"""解析 DatasetConfig 并构建 DatasetPlan 抽象语法树"""
	match config:
		case SingleSourceDatasetConfig():
			reader = _parse_source_reader(config.source)

			if isinstance(reader, FolderReader) and not isinstance(config.labels, UnlabeledConfig):
				if isinstance(config.labels, MultiTaskConfig) and len(config.labels.targets) > 1:
					raise ValueError('Folder source only supports at most one target for labeled data')

			labeler = _parse_labeler(config.labels)
			source = Source(reader=reader, labeler=labeler)

			split: SplitTerm
			match config.split:
				case NoSplitConfig():
					split = NoSplit()
				case RatioSplitConfig():
					if config.split.stratify is not None:
						match labeler:
							case NoLabels():
								raise ValueError('Cannot stratify self_supervised dataset')
							case SingleLabels():
								if config.split.stratify != 'label':
									raise ValueError(
										f"Stratify target {config.split.stratify!r} not found for single_label dataset (expected 'label')"
									)
							case MultiLabels():
								valid_targets = [name for name, _ in labeler.targets]
								if config.split.stratify not in valid_targets:
									raise ValueError(
										f"split.stratify {config.split.stratify!r} not found in dataset targets: {valid_targets}"
									)
							case _ as unreachable:
								assert_never(unreachable)
					split = RatioSplit(
						val_ratio=config.split.val_ratio,
						seed=config.split.seed,
						stratify=config.split.stratify,
					)
				case ColumnSplitConfig():
					split_source = config.split.source
					split_reader = SheetReader(
						path=split_source.path,
						sheet_name=split_source.sheet_name,
						have_header=split_source.have_header,
						root=split_source.root,
						image_column=split_source.image_column,
					)

					partitions: list[tuple[str, str | int]] = [('train', config.split.train)]
					if config.split.val is not None:
						partitions.append(('val', config.split.val))
					if config.split.test is not None:
						partitions.append(('test', config.split.test))

					split = ColumnSplit(
						column=config.split.column,
						partitions=tuple(partitions),
						source=split_reader,
					)
				case _ as unreachable:
					assert_never(unreachable)

			sample: Sample | None = None
			if config.sample_ratio < 1.0:
				if config.sample_seed is not None:
					subsample_seed = config.sample_seed
				elif isinstance(split, RatioSplit) and split.seed is not None:
					subsample_seed = split.seed
				else:
					subsample_seed = None
				sample = Sample(ratio=config.sample_ratio, seed=subsample_seed)

			return SingleSourcePlan(source=source, split=split, sample=sample)

		case MultiSourceDatasetConfig():
			if 'train' not in config.sources:
				raise ValueError("MultiSourceDatasetConfig requires 'train' in sources")

			for s_name, s_cfg in config.sources.items():
				if isinstance(s_cfg, FolderSourceConfig) and not isinstance(config.labels, UnlabeledConfig):
					if isinstance(config.labels, MultiTaskConfig) and len(config.labels.targets) > 1:
						raise ValueError(f'Folder source in {s_name!r} only supports at most one target')

			labeler = _parse_labeler(config.labels)

			train_source = Source(reader=_parse_source_reader(config.sources['train']), labeler=labeler)
			val_source = (
				Source(reader=_parse_source_reader(config.sources['val']), labeler=labeler)
				if 'val' in config.sources
				else None
			)
			test_source = (
				Source(reader=_parse_source_reader(config.sources['test']), labeler=labeler)
				if 'test' in config.sources
				else None
			)
			multi_sources = MultiSource(train=train_source, val=val_source, test=test_source)

			sample = (
				Sample(ratio=config.sample_ratio, seed=config.sample_seed)
				if config.sample_ratio < 1.0
				else None
			)
			return MultiSourcePlan(sources=multi_sources, sample=sample)

		case _ as unreachable:
			assert_never(unreachable)


def parse_single_label_dataset(config: DatasetConfig) -> DatasetPlan[int]:
	"""解析并校验单标签分类数据集计划"""
	plan = parse_dataset_config(config)
	match plan:
		case SingleSourcePlan():
			if not isinstance(plan.source.labeler, SingleLabels):
				raise ValueError("Classification task requires dataset.labels with mode='single_label'")
		case MultiSourcePlan():
			sources = [plan.sources.train]
			if plan.sources.val is not None:
				sources.append(plan.sources.val)
			if plan.sources.test is not None:
				sources.append(plan.sources.test)
			for s in sources:
				if not isinstance(s.labeler, SingleLabels):
					raise ValueError("Classification task requires dataset.labels with mode='single_label'")
		case _ as unreachable:
			assert_never(unreachable)
	return cast(DatasetPlan[int], plan)


def parse_unlabeled_dataset(config: DatasetConfig) -> DatasetPlan[None]:
	"""解析并校验无标签自监督数据集计划"""
	plan = parse_dataset_config(config)
	match plan:
		case SingleSourcePlan():
			if not isinstance(plan.source.labeler, NoLabels):
				raise ValueError("MAE pretrain task requires dataset.labels with mode='self_supervised'")
		case MultiSourcePlan():
			sources = [plan.sources.train]
			if plan.sources.val is not None:
				sources.append(plan.sources.val)
			if plan.sources.test is not None:
				sources.append(plan.sources.test)
			for s in sources:
				if not isinstance(s.labeler, NoLabels):
					raise ValueError("MAE pretrain task requires dataset.labels with mode='self_supervised'")
		case _ as unreachable:
			assert_never(unreachable)
	return cast(DatasetPlan[None], plan)


def single_label_target(plan: DatasetPlan[int]) -> ColumnLabelSpec:
	"""从单标签数据计划中提取首个数据源的目标标签参数"""
	match plan:
		case SingleSourcePlan():
			assert isinstance(plan.source.labeler, SingleLabels)
			return plan.source.labeler.target
		case MultiSourcePlan():
			assert isinstance(plan.sources.train.labeler, SingleLabels)
			return plan.sources.train.labeler.target
		case _ as unreachable:
			assert_never(unreachable)
