"""实验协议层"""

from .leakage_check import (
	assert_collections_disjoint,
	assert_manifest_splits_disjoint,
	collect_sample_ids,
	find_collection_overlaps,
)
from .split_manifest import (
	SplitManifest,
	apply_named_splits,
	apply_split_manifest,
	load_split_manifest,
	normalize_sample_id,
)

__all__ = [
	'SplitManifest',
	'load_split_manifest',
	'apply_named_splits',
	'apply_split_manifest',
	'normalize_sample_id',
	'collect_sample_ids',
	'find_collection_overlaps',
	'assert_collections_disjoint',
	'assert_manifest_splits_disjoint',
]
