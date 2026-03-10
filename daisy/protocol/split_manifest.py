"""固定划分 manifest 解析与应用"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import tomli


VALID_ID_TYPES = {'relative_path', 'path', 'name'}


@dataclass(slots=True)
class SplitManifest:
	"""固定划分 manifest"""

	path: Path
	id_type: str = 'relative_path'
	root: str | None = None
	splits: dict[str, list[str]] = field(default_factory=dict)
	metadata: dict[str, Any] = field(default_factory=dict)

	def all_ids(self) -> list[str]:
		ids: list[str] = []
		for split_ids in self.splits.values():
			ids.extend(split_ids)
		return ids


def _coerce_roots(root: Path | str | Iterable[Path | str] | None) -> list[Path]:
	if root is None:
		return []
	if isinstance(root, (Path, str)):
		return [Path(root)]
	return [Path(item) for item in root]


def _root_alias(root: Path, index: int) -> str:
	alias = root.name or root.stem
	return alias if alias else f'root{index}'


def _normalize_declared_id(sample_id: str, id_type: str) -> str:
	value = str(sample_id).strip()
	if id_type == 'name':
		return Path(value).name
	if id_type in {'relative_path', 'path'}:
		value = value.replace('\\', '/')
		while value.startswith('./'):
			value = value[2:]
		if id_type == 'relative_path':
			value = value.lstrip('/')
	return value


def normalize_sample_id(
	sample: Path | str,
	*,
	id_type: str = 'relative_path',
	root: Path | str | Iterable[Path | str] | None = None,
) -> str:
	"""将样本路径规范化为 manifest 可用的标识符"""
	if id_type not in VALID_ID_TYPES:
		raise ValueError(f'Unsupported id_type: {id_type}')

	sample_path = Path(sample)
	if id_type == 'name':
		return sample_path.name
	if id_type == 'path':
		return sample_path.resolve().as_posix()

	roots = _coerce_roots(root)
	resolved_sample = sample_path.resolve()
	for index, root_path in enumerate(roots):
		try:
			rel_path = resolved_sample.relative_to(root_path.resolve())
		except ValueError:
			continue
		if len(roots) == 1:
			return rel_path.as_posix()
		return f'{_root_alias(root_path, index)}/{rel_path.as_posix()}'

	return sample_path.as_posix()


def collect_sample_ids(
	samples: Iterable[Path | str],
	*,
	id_type: str = 'relative_path',
	root: Path | str | Iterable[Path | str] | None = None,
) -> list[str]:
	"""收集样本标识符列表"""
	return [normalize_sample_id(sample, id_type=id_type, root=root) for sample in samples]


def _build_sample_id_map(
	files: list[Path],
	labels: list[int],
	*,
	id_type: str,
	root: Path | str | Iterable[Path | str] | None,
) -> tuple[dict[str, list[tuple[Path, int]]], list[str]]:
	mapping: dict[str, list[tuple[Path, int]]] = {}
	duplicates: set[str] = set()
	for file_path, label in zip(files, labels):
		sample_id = normalize_sample_id(file_path, id_type=id_type, root=root)
		bucket = mapping.setdefault(sample_id, [])
		bucket.append((file_path, label))
		if len(bucket) > 1:
			duplicates.add(sample_id)
	return mapping, sorted(duplicates)


def _find_split_id_overlaps(split_to_ids: Mapping[str, list[str]]) -> dict[str, list[str]]:
	seen: dict[str, str] = {}
	overlaps: dict[str, list[str]] = {}
	for split_name, sample_ids in split_to_ids.items():
		for sample_id in sample_ids:
			other_split = seen.get(sample_id)
			if other_split is None:
				seen[sample_id] = split_name
				continue
			key = '__'.join(sorted((other_split, split_name)))
			overlaps.setdefault(key, []).append(sample_id)
	return {key: sorted(set(value)) for key, value in overlaps.items()}


def apply_named_splits(
	files: list[Path],
	labels: list[int],
	split_to_ids: Mapping[str, list[str]],
	*,
	id_type: str = 'relative_path',
	root: Path | str | Iterable[Path | str] | None = None,
	strict: bool = True,
) -> tuple[dict[str, tuple[list[Path], list[int]]], dict[str, Any]]:
	"""按给定的 split-id 映射选择样本"""
	if len(files) != len(labels):
		raise ValueError('files and labels must have the same length')

	normalized_split_ids = {
		split_name: [_normalize_declared_id(sample_id, id_type) for sample_id in sample_ids] for split_name, sample_ids in split_to_ids.items()
	}

	sample_id_map, duplicate_dataset_ids = _build_sample_id_map(
		files,
		labels,
		id_type=id_type,
		root=root,
	)
	split_id_overlaps = _find_split_id_overlaps(normalized_split_ids)

	if strict and duplicate_dataset_ids:
		raise ValueError(f'Found duplicate dataset ids under current id_type {id_type}: {duplicate_dataset_ids[:10]}')
	if strict and split_id_overlaps:
		raise ValueError(f'Split ids overlap across manifest splits: {split_id_overlaps}')

	assigned_dataset_ids: set[str] = set()
	selected_splits: dict[str, tuple[list[Path], list[int]]] = {}
	report: dict[str, Any] = {
		'id_type': id_type,
		'root': [str(item) for item in _coerce_roots(root)] if root is not None else None,
		'dataset_sample_count': len(files),
		'duplicate_dataset_ids': duplicate_dataset_ids,
		'split_id_overlaps': split_id_overlaps,
		'splits': {},
	}

	for split_name, sample_ids in normalized_split_ids.items():
		selected_files: list[Path] = []
		selected_labels: list[int] = []
		missing_ids: list[str] = []
		duplicate_requested_ids: list[str] = []
		seen_ids: set[str] = set()

		for sample_id in sample_ids:
			if sample_id in seen_ids:
				duplicate_requested_ids.append(sample_id)
				continue
			seen_ids.add(sample_id)
			matches = sample_id_map.get(sample_id)
			if matches is None:
				missing_ids.append(sample_id)
				continue
			for file_path, label in matches:
				selected_files.append(file_path)
				selected_labels.append(label)
			assigned_dataset_ids.add(sample_id)

		if strict and missing_ids:
			raise ValueError(f'Manifest split {split_name!r} contains ids not found in dataset: {missing_ids[:10]}')

		selected_splits[split_name] = (selected_files, selected_labels)
		report['splits'][split_name] = {
			'requested_count': len(sample_ids),
			'matched_count': len(selected_files),
			'missing_ids': missing_ids,
			'duplicate_requested_ids': sorted(set(duplicate_requested_ids)),
			'sample_ids': sorted(seen_ids),
		}

	report['unassigned_count'] = len(sample_id_map) - len(assigned_dataset_ids)
	return selected_splits, report


def _load_toml_manifest(path: Path) -> dict[str, Any]:
	with open(path, 'rb') as f:
		return tomli.load(f)


def _load_json_manifest(path: Path) -> dict[str, Any]:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def _load_csv_manifest(path: Path) -> dict[str, Any]:
	with open(path, 'r', encoding='utf-8-sig', newline='') as f:
		reader = csv.DictReader(f)
		field_map = {field.lower(): field for field in (reader.fieldnames or [])}
		sample_col = next(
			(field_map[name] for name in ('sample_id', 'id', 'file', 'path', 'relative_path', 'name') if name in field_map),
			None,
		)
		split_col = next(
			(field_map[name] for name in ('split', 'set', 'subset') if name in field_map),
			None,
		)
		if sample_col is None or split_col is None:
			raise ValueError('CSV split manifest requires sample_id/id/file/path column and split/set/subset column')

		if sample_col.lower() == 'name':
			id_type = 'name'
		elif sample_col.lower() == 'path':
			id_type = 'path'
		else:
			id_type = 'relative_path'

		splits: dict[str, list[str]] = {}
		for row in reader:
			split_name = str(row[split_col]).strip()
			sample_id = str(row[sample_col]).strip()
			splits.setdefault(split_name, []).append(sample_id)

		return {
			'id_type': id_type,
			'splits': splits,
		}


def load_split_manifest(path: str | Path) -> SplitManifest:
	"""加载固定划分 manifest"""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f'Split manifest not found: {path}')

	suffix = path.suffix.lower()
	if suffix == '.toml':
		data = _load_toml_manifest(path)
	elif suffix == '.json':
		data = _load_json_manifest(path)
	elif suffix == '.csv':
		data = _load_csv_manifest(path)
	else:
		raise ValueError(f'Unsupported split manifest format: {path.suffix}')

	id_type = data.get('id_type', 'relative_path')
	if id_type not in VALID_ID_TYPES:
		raise ValueError(f'Unsupported id_type in manifest: {id_type}')

	splits = data.get('splits')
	if not isinstance(splits, dict):
		raise ValueError('Split manifest must contain a [splits] mapping')

	normalized_splits = {
		str(split_name): [_normalize_declared_id(sample_id, id_type) for sample_id in sample_ids] for split_name, sample_ids in splits.items()
	}
	metadata = {k: v for k, v in data.items() if k not in {'id_type', 'root', 'splits'}}

	return SplitManifest(
		path=path,
		id_type=id_type,
		root=data.get('root'),
		splits=normalized_splits,
		metadata=metadata,
	)


def apply_split_manifest(
	files: list[Path],
	labels: list[int],
	manifest: SplitManifest | str | Path,
	*,
	dataset_root: Path | str | Iterable[Path | str] | None = None,
	split_names: Iterable[str] | None = None,
	strict: bool = True,
) -> tuple[dict[str, tuple[list[Path], list[int]]], dict[str, Any]]:
	"""根据 manifest 选择固定 split"""
	if not isinstance(manifest, SplitManifest):
		manifest = load_split_manifest(manifest)

	if split_names is None:
		selected_split_to_ids = manifest.splits
	else:
		selected_split_to_ids = {}
		for split_name in split_names:
			if split_name not in manifest.splits:
				raise ValueError(f'Split {split_name!r} not found in manifest {manifest.path}')
			selected_split_to_ids[split_name] = manifest.splits[split_name]

	selected_splits, report = apply_named_splits(
		files,
		labels,
		selected_split_to_ids,
		id_type=manifest.id_type,
		root=dataset_root if dataset_root is not None else manifest.root,
		strict=strict,
	)
	report['manifest_path'] = str(manifest.path)
	report['manifest_root'] = manifest.root
	report['manifest_metadata'] = manifest.metadata
	return selected_splits, report
