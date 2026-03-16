"""带标签数据集共享逻辑"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import daisy


@dataclass(slots=True)
class LabeledSamples:
	"""带标签样本集合"""

	files: list[Path]
	labels: list[int]

	def __len__(self) -> int:
		return len(self.files)

	def to_dataset(self) -> daisy.dataset.DiskDataset:
		return daisy.dataset.DiskDataset(self.files, self.labels)

	@classmethod
	def from_dataset(cls, dataset: daisy.dataset.IndexDataset) -> 'LabeledSamples':
		files, labels = dataset.getRawData()
		return cls(files=cast(list[Path], list(files)), labels=list(labels))


@dataclass(slots=True)
class TrainValSelection:
	"""训练/验证数据选择结果"""

	train: LabeledSamples
	val: LabeledSamples
	source_count: int

	def to_datasets(self) -> tuple[daisy.dataset.DiskDataset, daisy.dataset.DiskDataset]:
		return self.train.to_dataset(), self.val.to_dataset()


@dataclass(slots=True)
class DatasetSelection:
	"""单个 split 的数据选择结果"""

	split_name: str
	samples: LabeledSamples
	source_count: int

	def to_dataset(self) -> daisy.dataset.DiskDataset:
		return self.samples.to_dataset()


def _assert_train_val_disjoint(train: LabeledSamples, val: LabeledSamples, *, context: str) -> None:
	train_paths = {file_path.resolve().as_posix() for file_path in train.files}
	val_paths = {file_path.resolve().as_posix() for file_path in val.files}
	overlaps = sorted(train_paths & val_paths)
	if overlaps:
		raise ValueError(f'Found overlaps in {context}: {overlaps[:10]}')


def _relative_sample_id(file_path: Path, *, root: Path) -> str:
	try:
		return file_path.resolve().relative_to(root.resolve()).as_posix()
	except ValueError:
		return file_path.as_posix()


def _load_sheet_samples(
	*,
	dataset_root: Path,
	sheet_path: str | Path,
	sheet_name: str | int | None,
	column: int,
	label_offset: int,
	have_header: bool,
) -> LabeledSamples:
	feeder = daisy.feeder.load_feeder_from_sheet(
		dataset_root=dataset_root,
		sheet_path=Path(sheet_path),
		sheet_name=sheet_name if sheet_name is not None else 0,
		column=column,
		label_offset=label_offset,
		have_header=have_header,
	)
	files, labels = feeder.fetch()
	return LabeledSamples(files=files, labels=labels)


def _load_folder_samples(root: Path) -> LabeledSamples:
	feeder = daisy.feeder.load_feeder_from_folder(root)
	files, labels = feeder.fetch()
	return LabeledSamples(files=files, labels=labels)


def load_labeled_samples(dataset_cfg: Any) -> LabeledSamples:
	"""加载带标签样本"""
	if dataset_cfg.type == 'sheet':
		if not dataset_cfg.sheet:
			raise ValueError('dataset.sheet is required when dataset.type = "sheet"')
		return _load_sheet_samples(
			dataset_root=Path(dataset_cfg.root),
			sheet_path=dataset_cfg.sheet,
			sheet_name=dataset_cfg.sheet_name,
			column=dataset_cfg.column,
			label_offset=dataset_cfg.label_offset,
			have_header=dataset_cfg.have_header,
		)
	if dataset_cfg.type == 'folder':
		return _load_folder_samples(Path(dataset_cfg.root))
	raise ValueError(f'Unknown dataset type: {dataset_cfg.type}')


def build_train_val_selection(
	dataset_cfg: Any,
	*,
	default_seed: int | None = None,
	context: str,
) -> TrainValSelection:
	"""构建训练/验证数据划分"""
	dataset_root = Path(dataset_cfg.root)
	source_samples = load_labeled_samples(dataset_cfg)
	dataset = source_samples.to_dataset()
	split_cfg = dataset_cfg.split
	split_seed = split_cfg.seed if split_cfg.seed is not None else default_seed

	if split_cfg.method == 'ratio':
		split_fn = daisy.dataset.dataset_split.stratified_data_split if split_cfg.stratified else daisy.dataset.dataset_split.default_data_split
		train_dataset, val_dataset = split_fn(
			dataset,
			val_ratio=split_cfg.val_ratio,
			seed=split_seed,
		)
		train = LabeledSamples.from_dataset(train_dataset)
		val = LabeledSamples.from_dataset(val_dataset)
		result = TrainValSelection(train=train, val=val, source_count=len(source_samples))
	elif split_cfg.method == 'sheet':
		if not split_cfg.val_sheet:
			raise ValueError('dataset.split.val_sheet is required when split.method = "sheet"')
		val = _load_sheet_samples(
			dataset_root=dataset_root,
			sheet_path=split_cfg.val_sheet,
			sheet_name=split_cfg.val_sheet_name,
			column=dataset_cfg.column,
			label_offset=dataset_cfg.label_offset,
			have_header=dataset_cfg.have_header,
		)
		val_file_ids = {file_path.resolve().as_posix() for file_path in val.files}
		train_files: list[Path] = []
		train_labels: list[int] = []
		for file_path, label in zip(source_samples.files, source_samples.labels):
			if file_path.resolve().as_posix() in val_file_ids:
				continue
			train_files.append(file_path)
			train_labels.append(label)
		train = LabeledSamples(files=train_files, labels=train_labels)
		result = TrainValSelection(train=train, val=val, source_count=len(source_samples))
	elif split_cfg.method == 'preset':
		train = _load_folder_samples(dataset_root / split_cfg.preset_train_dir)
		val = _load_folder_samples(dataset_root / split_cfg.preset_val_dir)
		result = TrainValSelection(train=train, val=val, source_count=len(source_samples))
	else:
		raise ValueError(f'Unknown split method: {split_cfg.method}')

	_assert_train_val_disjoint(result.train, result.val, context=context)
	return result


def select_dataset_split(
	dataset_cfg: Any,
	*,
	split_name: str,
	default_seed: int | None = None,
) -> DatasetSelection:
	"""为评估/导出选择目标 split"""
	dataset_root = Path(dataset_cfg.root)
	source_samples = load_labeled_samples(dataset_cfg)
	dataset = source_samples.to_dataset()
	split_cfg = dataset_cfg.split
	split_seed = split_cfg.seed if split_cfg.seed is not None else default_seed

	if split_cfg.method == 'none':
		return DatasetSelection(
			split_name=split_name,
			samples=source_samples,
			source_count=len(source_samples),
		)

	if split_cfg.method == 'ratio':
		split_fn = daisy.dataset.dataset_split.stratified_data_split if split_cfg.stratified else daisy.dataset.dataset_split.default_data_split
		train_dataset, val_dataset = split_fn(
			dataset,
			val_ratio=split_cfg.val_ratio,
			seed=split_seed,
		)
		selection_map = {
			'train': LabeledSamples.from_dataset(train_dataset),
			'val': LabeledSamples.from_dataset(val_dataset),
		}
		if split_name not in selection_map:
			raise ValueError(f'ratio split only supports train/val, got {split_name!r}')
		selected = selection_map[split_name]
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			source_count=len(source_samples),
		)

	if split_cfg.method == 'sheet':
		if split_name == 'train':
			raise ValueError('sheet split does not support selecting train directly')
		if not split_cfg.val_sheet:
			raise ValueError('dataset.split.val_sheet is required when split.method = "sheet"')
		selected = _load_sheet_samples(
			dataset_root=dataset_root,
			sheet_path=split_cfg.val_sheet,
			sheet_name=split_cfg.val_sheet_name,
			column=dataset_cfg.column,
			label_offset=dataset_cfg.label_offset,
			have_header=dataset_cfg.have_header,
		)
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			source_count=len(source_samples),
		)

	if split_cfg.method == 'preset':
		dir_map = {
			'train': split_cfg.preset_train_dir,
			'val': split_cfg.preset_val_dir,
			'test': split_cfg.preset_test_dir,
		}
		if split_name not in dir_map:
			raise ValueError(f'Unknown preset split name: {split_name!r}')
		selected = _load_folder_samples(dataset_root / dir_map[split_name])
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			source_count=len(source_samples),
		)

	raise ValueError(f'Unknown split method: {split_cfg.method}')
