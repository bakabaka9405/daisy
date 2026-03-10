"""带标签数据集共享逻辑"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import daisy
from daisy.protocol import apply_named_splits, apply_split_manifest, assert_collections_disjoint, collect_sample_ids, normalize_sample_id


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
class ProtocolSplit:
	"""协议中的单个 split"""

	name: str
	sample_ids: list[str]


@dataclass(slots=True)
class SplitProtocolSnapshot:
	"""训练/验证划分协议快照"""

	method: str
	id_type: str
	root: str
	splits: list[ProtocolSplit] = field(default_factory=list)
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		data = {
			'method': self.method,
			'id_type': self.id_type,
			'root': self.root,
			'splits': {split.name: split.sample_ids for split in self.splits},
			'counts': {split.name: len(split.sample_ids) for split in self.splits},
		}
		data.update(self.metadata)
		return data


@dataclass(slots=True)
class SelectionProtocolSnapshot:
	"""单个 split 选择协议快照"""

	method: str
	id_type: str
	root: str
	split_name: str
	sample_ids: list[str]
	metadata: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		data = {
			'method': self.method,
			'id_type': self.id_type,
			'root': self.root,
			'split_name': self.split_name,
			'count': len(self.sample_ids),
			'sample_ids': self.sample_ids,
		}
		data.update(self.metadata)
		return data


@dataclass(slots=True)
class TrainValSelection:
	"""训练/验证数据选择结果"""

	train: LabeledSamples
	val: LabeledSamples
	protocol: SplitProtocolSnapshot
	source_count: int

	def to_datasets(self) -> tuple[daisy.dataset.DiskDataset, daisy.dataset.DiskDataset]:
		return self.train.to_dataset(), self.val.to_dataset()


@dataclass(slots=True)
class DatasetSelection:
	"""单个 split 的数据选择结果"""

	split_name: str
	samples: LabeledSamples
	protocol: SelectionProtocolSnapshot
	source_count: int

	def to_dataset(self) -> daisy.dataset.DiskDataset:
		return self.samples.to_dataset()


def _build_protocol_split(name: str, files: list[Path], *, root: Path, id_type: str) -> ProtocolSplit:
	return ProtocolSplit(
		name=name,
		sample_ids=collect_sample_ids(files, id_type=id_type, root=root),
	)


def _build_split_protocol_snapshot(
	*,
	method: str,
	root: Path,
	train: LabeledSamples,
	val: LabeledSamples,
	id_type: str = 'relative_path',
	metadata: dict[str, Any] | None = None,
) -> SplitProtocolSnapshot:
	return SplitProtocolSnapshot(
		method=method,
		id_type=id_type,
		root=str(root),
		splits=[
			_build_protocol_split('train', train.files, root=root, id_type=id_type),
			_build_protocol_split('val', val.files, root=root, id_type=id_type),
		],
		metadata=metadata or {},
	)


def _build_selection_protocol_snapshot(
	*,
	method: str,
	root: Path,
	split_name: str,
	files: list[Path],
	id_type: str = 'relative_path',
	metadata: dict[str, Any] | None = None,
) -> SelectionProtocolSnapshot:
	return SelectionProtocolSnapshot(
		method=method,
		id_type=id_type,
		root=str(root),
		split_name=split_name,
		sample_ids=collect_sample_ids(files, id_type=id_type, root=root),
		metadata=metadata or {},
	)


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
		protocol = _build_split_protocol_snapshot(
			method='ratio',
			root=dataset_root,
			train=train,
			val=val,
			metadata={
				'seed': split_seed,
				'stratified': split_cfg.stratified,
				'val_ratio': split_cfg.val_ratio,
			},
		)
		result = TrainValSelection(train=train, val=val, protocol=protocol, source_count=len(source_samples))
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
		val_file_ids = {normalize_sample_id(file_path, id_type='path') for file_path in val.files}
		train_files: list[Path] = []
		train_labels: list[int] = []
		for file_path, label in zip(source_samples.files, source_samples.labels):
			if normalize_sample_id(file_path, id_type='path') in val_file_ids:
				continue
			train_files.append(file_path)
			train_labels.append(label)
		train = LabeledSamples(files=train_files, labels=train_labels)
		protocol = _build_split_protocol_snapshot(
			method='sheet',
			root=dataset_root,
			train=train,
			val=val,
			metadata={
				'val_sheet': split_cfg.val_sheet,
				'val_sheet_name': split_cfg.val_sheet_name,
			},
		)
		result = TrainValSelection(train=train, val=val, protocol=protocol, source_count=len(source_samples))
	elif split_cfg.method == 'preset':
		if split_cfg.train_files or split_cfg.val_files:
			split_mapping, report = apply_named_splits(
				source_samples.files,
				source_samples.labels,
				{
					'train': split_cfg.train_files,
					'val': split_cfg.val_files,
				},
				id_type=split_cfg.manifest_id_type,
				root=dataset_root,
				strict=split_cfg.require_all_in_manifest,
			)
			train = LabeledSamples(*split_mapping['train'])
			val = LabeledSamples(*split_mapping['val'])
			protocol = _build_split_protocol_snapshot(
				method='preset',
				root=dataset_root,
				train=train,
				val=val,
				id_type=split_cfg.manifest_id_type,
				metadata={
					'selection_report': report,
					'from_explicit_file_lists': True,
				},
			)
		else:
			train = _load_folder_samples(dataset_root / split_cfg.preset_train_dir)
			val = _load_folder_samples(dataset_root / split_cfg.preset_val_dir)
			protocol = _build_split_protocol_snapshot(
				method='preset',
				root=dataset_root,
				train=train,
				val=val,
				metadata={
					'preset_train_dir': split_cfg.preset_train_dir,
					'preset_val_dir': split_cfg.preset_val_dir,
					'from_explicit_file_lists': False,
				},
			)
		result = TrainValSelection(train=train, val=val, protocol=protocol, source_count=len(source_samples))
	elif split_cfg.method == 'manifest':
		if not split_cfg.manifest:
			raise ValueError('dataset.split.manifest is required when split.method = "manifest"')
		split_mapping, report = apply_split_manifest(
			source_samples.files,
			source_samples.labels,
			split_cfg.manifest,
			dataset_root=dataset_root,
			split_names=(split_cfg.manifest_train_split, split_cfg.manifest_val_split),
			strict=split_cfg.require_all_in_manifest,
		)
		train = LabeledSamples(*split_mapping[split_cfg.manifest_train_split])
		val = LabeledSamples(*split_mapping[split_cfg.manifest_val_split])
		protocol = _build_split_protocol_snapshot(
			method='manifest',
			root=dataset_root,
			train=train,
			val=val,
			id_type=report['id_type'],
			metadata={
				'manifest': split_cfg.manifest,
				'manifest_train_split': split_cfg.manifest_train_split,
				'manifest_val_split': split_cfg.manifest_val_split,
				'selection_report': report,
			},
		)
		result = TrainValSelection(train=train, val=val, protocol=protocol, source_count=len(source_samples))
	else:
		raise ValueError(f'Unknown split method: {split_cfg.method}')

	assert_collections_disjoint(
		{
			'train': result.train.files,
			'val': result.val.files,
		},
		id_type='path',
		context=context,
	)
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
		protocol = _build_selection_protocol_snapshot(
			method='none',
			root=dataset_root,
			split_name=split_name,
			files=source_samples.files,
		)
		return DatasetSelection(
			split_name=split_name,
			samples=source_samples,
			protocol=protocol,
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
		protocol = _build_selection_protocol_snapshot(
			method='ratio',
			root=dataset_root,
			split_name=split_name,
			files=selected.files,
			metadata={
				'seed': split_seed,
				'stratified': split_cfg.stratified,
				'val_ratio': split_cfg.val_ratio,
			},
		)
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			protocol=protocol,
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
		protocol = _build_selection_protocol_snapshot(
			method='sheet',
			root=dataset_root,
			split_name=split_name,
			files=selected.files,
			metadata={
				'source_sheet': split_cfg.val_sheet,
				'source_sheet_name': split_cfg.val_sheet_name,
			},
		)
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			protocol=protocol,
			source_count=len(source_samples),
		)

	if split_cfg.method == 'preset':
		if split_cfg.train_files or split_cfg.val_files or split_cfg.test_files:
			split_mapping, report = apply_named_splits(
				source_samples.files,
				source_samples.labels,
				{
					'train': split_cfg.train_files,
					'val': split_cfg.val_files,
					'test': split_cfg.test_files,
				},
				id_type=split_cfg.manifest_id_type,
				root=dataset_root,
				strict=split_cfg.require_all_in_manifest,
			)
			if split_name not in split_mapping:
				raise ValueError(f'Unknown preset split name: {split_name!r}')
			selected = LabeledSamples(*split_mapping[split_name])
			protocol = _build_selection_protocol_snapshot(
				method='preset',
				root=dataset_root,
				split_name=split_name,
				files=selected.files,
				id_type=split_cfg.manifest_id_type,
				metadata={
					'selection_report': report,
					'from_explicit_file_lists': True,
				},
			)
			return DatasetSelection(
				split_name=split_name,
				samples=selected,
				protocol=protocol,
				source_count=len(source_samples),
			)

		dir_map = {
			'train': split_cfg.preset_train_dir,
			'val': split_cfg.preset_val_dir,
			'test': split_cfg.preset_test_dir,
		}
		if split_name not in dir_map:
			raise ValueError(f'Unknown preset split name: {split_name!r}')
		selected = _load_folder_samples(dataset_root / dir_map[split_name])
		protocol = _build_selection_protocol_snapshot(
			method='preset',
			root=dataset_root,
			split_name=split_name,
			files=selected.files,
			metadata={
				'directory': dir_map[split_name],
				'from_explicit_file_lists': False,
			},
		)
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			protocol=protocol,
			source_count=len(source_samples),
		)

	if split_cfg.method == 'manifest':
		if not split_cfg.manifest:
			raise ValueError('dataset.split.manifest is required when split.method = "manifest"')
		manifest_split_map = {
			'train': split_cfg.manifest_train_split,
			'val': split_cfg.manifest_val_split,
			'test': split_cfg.manifest_test_split,
		}
		target_split = manifest_split_map.get(split_name, split_name)
		split_mapping, report = apply_split_manifest(
			source_samples.files,
			source_samples.labels,
			split_cfg.manifest,
			dataset_root=dataset_root,
			split_names=(target_split,),
			strict=split_cfg.require_all_in_manifest,
		)
		selected = LabeledSamples(*split_mapping[target_split])
		protocol = _build_selection_protocol_snapshot(
			method='manifest',
			root=dataset_root,
			split_name=split_name,
			files=selected.files,
			id_type=report['id_type'],
			metadata={
				'manifest': split_cfg.manifest,
				'target_split': target_split,
				'selection_report': report,
			},
		)
		return DatasetSelection(
			split_name=split_name,
			samples=selected,
			protocol=protocol,
			source_count=len(source_samples),
		)

	raise ValueError(f'Unknown split method: {split_cfg.method}')
