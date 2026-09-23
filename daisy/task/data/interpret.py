"""DatasetPlan 代数解释器"""

from __future__ import annotations

from pathlib import Path
from typing import assert_never, cast
from torch import Tensor

from daisy.dataset.dataset_sample import random_subsample
from daisy.dataset.dataset_split import (
	assert_disjoint_file_splits,
	default_data_split,
	stratified_data_split,
)
from daisy.dataset.disk_dataset import DiskDataset, load_files_from_folder
from daisy.feeder import (
	load_samples_from_folder,
	parse_multitask_table,
	parse_single_label_table,
	parse_table_paths,
	read_sheet,
)
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
	Source,
	SplitTerm,
)
from .split_table import parse_column_split_indices


def _identity_transform(x: Tensor) -> Tensor:
	"""模块级恒等变换，支持 spawn 进程安全序列化"""
	return x


def _read_source[L](
	reader: ReaderTerm,
	labeler: LabelerTerm[L],
	*,
	class_dict: dict[str, int] | None = None,
) -> tuple[DiskDataset[Tensor, L], dict[str, int] | None]:
	"""读取单一数据源并构建 DiskDataset"""
	match reader:
		case FolderReader(roots=roots, extensions=extensions):
			all_files: list[Path] = []

			match labeler:
				case NoLabels():
					for r in roots:
						all_files.extend(load_files_from_folder(r, extensions=extensions))

					if not all_files:
						raise ValueError('Dataset source is empty')

					unlabeled_dataset = DiskDataset(all_files, None, _identity_transform, backend='tensor')
					return cast(DiskDataset[Tensor, L], unlabeled_dataset), class_dict

				case SingleLabels(target=spec):
					target_spec = spec
					task_name: str | None = None

				case MultiLabels(targets=targets):
					(task_name, target_spec), = targets

				case _ as unreachable:
					assert_never(unreachable)

			curr_class_dict = class_dict if class_dict is not None else target_spec.class_dict

			raw_labels: list[int] = []
			for r in roots:
				folder_files, folder_labels, curr_class_dict = load_samples_from_folder(
					r,
					class_dict=curr_class_dict,
					extensions=extensions,
					num_classes=target_spec.num_classes,
				)
				all_files.extend(folder_files)
				raw_labels.extend(folder_labels)

			if not all_files:
				raise ValueError('Dataset source is empty')

			dataset_labels: list[L]
			if task_name is not None:
				dataset_labels = cast(list[L], [{task_name: lbl} for lbl in raw_labels])
			else:
				dataset_labels = cast(list[L], raw_labels)

			labeled_dataset = DiskDataset[Tensor, L](all_files, dataset_labels, _identity_transform, backend='tensor')
			return labeled_dataset, curr_class_dict

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

			files: list[Path]
			labels: list[L] | None

			match labeler:
				case NoLabels():
					files = parse_table_paths(df, root, image_column=image_column)
					labels = None

				case SingleLabels(target=spec):
					files, parsed_labels = parse_single_label_table(
						df,
						root,
						spec,
						image_column=image_column,
					)
					labels = cast(list[L] | None, parsed_labels)

				case MultiLabels(targets=targets):
					files, parsed_labels = parse_multitask_table(
						df,
						root,
						dict(targets),
						image_column=image_column,
					)
					labels = cast(list[L] | None, parsed_labels)

				case _ as unreachable:
					assert_never(unreachable)

			if not files:
				raise ValueError('Dataset source is empty')

			table_dataset = DiskDataset[Tensor, L](files, labels, _identity_transform, backend='tensor')
			return table_dataset, class_dict

		case _ as unreachable:
			assert_never(unreachable)


def _split_single_dataset[L](
	dataset: DiskDataset[Tensor, L],
	split: SplitTerm,
	labeler: LabelerTerm[L],
	*,
	caller_seed: int | None,
) -> dict[str, DiskDataset[Tensor, L]]:
	"""对单一来源数据集执行划分代数项解释"""
	match split:
		case NoSplit():
			return {'train': dataset}

		case RatioSplit(val_ratio=val_ratio, seed=split_seed, stratify=stratify):
			effective_seed = split_seed if split_seed is not None else caller_seed

			if stratify is not None:
				strat_labels: list[int]
				match labeler:
					case SingleLabels():
						strat_labels = cast(list[int], dataset.labels)
					case MultiLabels():
						multi_labels = cast(list[dict[str, int]], dataset.labels)
						strat_labels = [row[stratify] for row in multi_labels]
					case NoLabels():
						raise ValueError('Cannot stratify unlabeled dataset')
					case _ as unreachable:
						assert_never(unreachable)

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

		case ColumnSplit(column=column, partitions=partitions, source=split_source):
			split_df = read_sheet(
				split_source.path,
				sheet_name=split_source.sheet_name,
				have_header=split_source.have_header,
			)
			index_groups = parse_column_split_indices(
				dataset.file_paths,
				split_df,
				root=split_source.root,
				image_column=split_source.image_column,
				assignment_column=column,
				partition_values=[val for _, val in partitions],
			)

			splits: dict[str, DiskDataset[Tensor, L]] = {}
			for (name, val), indices in zip(partitions, index_groups):
				if not indices:
					raise ValueError(f'Split {name!r} declared with value {val!r} is empty')
				splits[name] = dataset.subset(indices)

			return splits

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
			dataset, _ = _read_source(source.reader, source.labeler)
			splits = _split_single_dataset(
				dataset,
				split,
				source.labeler,
				caller_seed=seed,
			)

		case MultiSourcePlan(sources=sources):
			splits = {}
			curr_class_dict: dict[str, int] | None = None
			for name in ('train', 'val', 'test'):
				source: Source[L] | None = getattr(sources, name)
				if source is not None:
					split_ds, curr_class_dict = _read_source(
						source.reader,
						source.labeler,
						class_dict=curr_class_dict,
					)
					splits[name] = split_ds

		case _ as unreachable:
			assert_never(unreachable)

	# 校验各分区文件不相交
	assert_disjoint_file_splits(splits)

	# 训练集子集采样
	if plan.sample is not None:
		subsample_seed = plan.sample.seed if plan.sample.seed is not None else seed
		splits['train'] = random_subsample(splits['train'], plan.sample.ratio, subsample_seed)

	return splits
