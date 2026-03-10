"""数据泄露检查工具"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path

from .split_manifest import SplitManifest, collect_sample_ids, load_split_manifest


def find_collection_overlaps(
	collections: Mapping[str, Iterable[Path | str]],
	*,
	id_type: str = 'relative_path',
	roots: Mapping[str, Path | str | Iterable[Path | str]] | Path | str | Iterable[Path | str] | None = None,
) -> dict[str, list[str]]:
	"""检查多个样本集合之间的重叠"""
	normalized_collections: dict[str, set[str]] = {}
	for name, samples in collections.items():
		root = roots.get(name) if isinstance(roots, Mapping) else roots
		normalized_collections[name] = set(collect_sample_ids(samples, id_type=id_type, root=root))

	overlaps: dict[str, list[str]] = {}
	for left, right in combinations(sorted(normalized_collections), 2):
		common = sorted(normalized_collections[left] & normalized_collections[right])
		if common:
			overlaps[f'{left}__{right}'] = common
	return overlaps


def assert_collections_disjoint(
	collections: Mapping[str, Iterable[Path | str]],
	*,
	id_type: str = 'relative_path',
	roots: Mapping[str, Path | str | Iterable[Path | str]] | Path | str | Iterable[Path | str] | None = None,
	context: str = 'collections',
) -> dict[str, list[str]]:
	"""断言多个集合互不重叠"""
	overlaps = find_collection_overlaps(collections, id_type=id_type, roots=roots)
	if overlaps:
		raise ValueError(f'Found overlaps in {context}: {overlaps}')
	return overlaps


def assert_manifest_splits_disjoint(manifest: SplitManifest | str | Path) -> dict[str, list[str]]:
	"""断言 manifest 中各 split 互不重叠"""
	if not isinstance(manifest, SplitManifest):
		manifest = load_split_manifest(manifest)
	return assert_collections_disjoint(
		manifest.splits,
		id_type=manifest.id_type,
		context=f'manifest {manifest.path}',
	)
