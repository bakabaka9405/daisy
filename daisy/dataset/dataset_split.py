from .disk_dataset import DiskDataset
from .index_dataset import IndexDataset
from collections import Counter
import icontract
import numpy
from ..util import (
	gather_list_by_indexes as gather,
	shuffle_correlated_lists,
)
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Any


def _get_rng(seed: int | None = None):
	return numpy.random.default_rng(seed)


def _labels_of(dataset: IndexDataset, stratify_labels: list[int] | None = None) -> Sequence:
	"""分层依据：优先使用显式给定的 stratify_labels，否则使用数据集自身标签。"""
	if stratify_labels is not None:
		return stratify_labels
	labels = dataset.getRawData()[1]
	assert labels is not None
	return labels


def _has_dict_labels(dataset: IndexDataset) -> bool:
	labels = dataset.getRawData()[1]
	return labels is not None and len(labels) > 0 and isinstance(labels[0], dict)


def _min_class_size(dataset: IndexDataset, stratify_labels: list[int] | None = None) -> int:
	return min(Counter(_labels_of(dataset, stratify_labels)).values())


def _has_path_overlap(splits: Mapping[str, DiskDataset[Any, Any]]) -> bool:
	"""任一卷签的文件路径是否与其他卷签相交"""
	seen: set[Path] = set()
	for split in splits.values():
		paths = {path.resolve() for path in split.file_paths}
		if not seen.isdisjoint(paths):
			return True
		seen.update(paths)
	return False


@icontract.require(lambda splits: not _has_path_overlap(splits), 'Split path overlap detected')
def assert_disjoint_file_splits[ImageT, LabelT](splits: Mapping[str, DiskDataset[ImageT, LabelT]]) -> None:
	"""集中检查各 split 间文件路径互斥性（校验全部由前置条件完成）"""


@icontract.require(lambda dataset: dataset.getRawData()[1] is not None, 'group_by_label requires a labeled dataset')
@icontract.ensure(lambda dataset, result: sum(len(v) for v in result.values()) == len(dataset))
def group_by_label[ImageT, LabelT](dataset: IndexDataset[ImageT, LabelT]) -> dict[LabelT, list]:
	"""将数据集按标签分割为多个子集"""
	label_to_indices: dict[LabelT, list] = {}
	data, labels = dataset.getRawData()
	assert labels is not None
	for i, j in zip(data, labels):
		if j not in label_to_indices:
			label_to_indices[j] = []
		label_to_indices[j].append(i)
	return label_to_indices


@icontract.require(lambda val_ratio: 0 <= val_ratio <= 1)
@icontract.ensure(lambda dataset, result: len(result[0]) + len(result[1]) == len(dataset))
def default_data_split[T: IndexDataset](
	dataset: T,
	val_ratio: float = 0.1,
	seed: int | None = None,
) -> tuple[T, T]:
	"""随机划分训练/验证集，与标签无关"""
	data, _ = dataset.getRawData()

	rng = _get_rng(seed)
	idx = rng.permutation(len(data)).tolist()

	train_size = int(len(data) * (1 - val_ratio))

	return (
		dataset.subset(idx[:train_size]),
		dataset.subset(idx[train_size:]),
	)


@icontract.require(
	lambda dataset, stratify_labels: len(dataset) == 0 or _min_class_size(dataset, stratify_labels) >= 2,
	'every class must have at least 2 samples to be stratified',
)
@icontract.require(
	lambda dataset, stratify_labels: stratify_labels is not None or not _has_dict_labels(dataset),
	'Multi-task dataset requires stratify_labels to be specified for stratified_data_split',
)
@icontract.require(
	lambda dataset, stratify_labels: stratify_labels is not None or dataset.getRawData()[1] is not None,
	'stratified_data_split requires a labeled dataset',
)
@icontract.require(
	lambda dataset, stratify_labels: stratify_labels is None or len(stratify_labels) == len(dataset),
	'stratify_labels length does not match dataset length',
)
@icontract.require(lambda val_ratio: 0 <= val_ratio <= 1)
@icontract.ensure(lambda dataset, result: len(result[0]) + len(result[1]) == len(dataset))
def stratified_data_split[T: IndexDataset](
	dataset: T,
	val_ratio: float = 0.1,
	seed: int | None = None,
	shuffle: bool = True,
	stratify_labels: list[int] | None = None,
) -> tuple[T, T]:
	"""按类别分层划分训练/验证集"""
	rng = _get_rng(seed)

	labels_seq = _labels_of(dataset, stratify_labels)

	label_to_indices: dict[object, list[int]] = {}
	for idx, label in enumerate(labels_seq):
		label_to_indices.setdefault(label, []).append(idx)

	train_indices: list[int] = []
	val_indices: list[int] = []

	for label, indices in label_to_indices.items():
		perm = rng.permutation(len(indices))
		val_size = max(1, int(round(len(indices) * val_ratio)))
		if val_size >= len(indices):
			val_size = len(indices) - 1
		train_size = len(indices) - val_size
		train_indices.extend(indices[i] for i in perm[:train_size])
		val_indices.extend(indices[i] for i in perm[train_size:])

	if shuffle:
		train_perm = rng.permutation(len(train_indices)) if train_indices else []
		val_perm = rng.permutation(len(val_indices)) if val_indices else []
		train_indices = [train_indices[i] for i in train_perm]
		val_indices = [val_indices[i] for i in val_perm]

	return (
		dataset.subset(train_indices),
		dataset.subset(val_indices),
	)


@icontract.require(lambda groups: len(set(groups)) >= 2, 'Not enough groups to create a validation split')
@icontract.require(lambda dataset, groups: len(groups) == len(dataset), 'groups length must match dataset length')
@icontract.require(lambda val_ratio: 0 <= val_ratio <= 1)
@icontract.ensure(lambda dataset, result: len(result[0]) + len(result[1]) == len(dataset))
def split_by_groups[T: IndexDataset](
	dataset: T,
	groups: list[str | int],
	val_ratio: float = 0.1,
	seed: int | None = None,
) -> tuple[T, T]:
	"""按 group 划分，保证同组样本不会跨 split 泄露"""
	group_to_indices: dict[str | int, list[int]] = {}
	for index, group in enumerate(groups):
		group_to_indices.setdefault(group, []).append(index)

	rng = _get_rng(seed)
	group_keys = list(group_to_indices.keys())
	shuffled_group_keys = [group_keys[i] for i in rng.permutation(len(group_keys))]
	val_group_count = max(1, int(round(len(shuffled_group_keys) * val_ratio)))
	if val_group_count >= len(shuffled_group_keys):
		val_group_count = len(shuffled_group_keys) - 1

	val_groups = set(shuffled_group_keys[:val_group_count])
	train_indices: list[int] = []
	val_indices: list[int] = []
	for group, indices in group_to_indices.items():
		if group in val_groups:
			val_indices.extend(indices)
		else:
			train_indices.extend(indices)

	return (
		dataset.subset(train_indices),
		dataset.subset(val_indices),
	)


@icontract.require(
	lambda dataset, val_ratio, val_minimum_size, val_maximum_ratio: (
		max(int(_min_class_size(dataset) * val_ratio), val_minimum_size) / _min_class_size(dataset) <= val_maximum_ratio
	),
	'类型不均衡太严重或设置的 val_maximum_ratio 阈值过低：有类别用于验证集的样本比例超过 val_maximum_ratio',
)
@icontract.require(
	lambda dataset, val_ratio, val_minimum_size, force_fetch_minimum_size: (
		force_fetch_minimum_size or int(_min_class_size(dataset) * val_ratio) >= val_minimum_size
	),
	'类型不均衡太严重或设置的 val_minimum_size 阈值过高：最少类别按 val_ratio 抽取的数量不足 val_minimum_size',
)
@icontract.require(lambda dataset: len(dataset) > 0)
@icontract.require(lambda dataset: dataset.getRawData()[1] is not None, 'minimum_class_proportional_split requires a labeled dataset')
@icontract.require(lambda val_ratio: 0 <= val_ratio <= 1)
def minimum_class_proportional_split[T: IndexDataset](
	dataset: T,
	val_ratio: float = 0.1,
	val_minimum_size: int = 100,
	force_fetch_minimum_size: bool = False,
	val_maximum_ratio: float = 0.5,
	shuffle: bool = True,
	seed: int | None = None,
) -> tuple[T, T]:
	dataset_type = type(dataset)
	rng = _get_rng(seed)

	label_dict = group_by_label(dataset)

	train_data = []
	train_labels = []
	val_data = []
	val_labels = []

	val_size: int = int(min(len(i) for _, i in label_dict.items()) * val_ratio)
	if force_fetch_minimum_size and val_size < val_minimum_size:
		val_size = val_minimum_size

	for i, j in label_dict.items():
		idx = rng.permutation(len(j))
		train_size = len(j) - val_size
		train_data += gather(j, idx[:train_size])
		train_labels += [i] * train_size
		val_data += gather(j, idx[train_size:])
		val_labels += [i] * val_size

	if shuffle:
		train_idx = rng.permutation(len(train_data)) if train_data else []
		val_idx = rng.permutation(len(val_data)) if val_data else []
		train_data = gather(train_data, train_idx) if len(train_data) > 0 else []
		train_labels = gather(train_labels, train_idx) if len(train_labels) > 0 else []
		val_data = gather(val_data, val_idx) if len(val_data) > 0 else []
		val_labels = gather(val_labels, val_idx) if len(val_labels) > 0 else []

	return (
		dataset_type(train_data, train_labels, dataset.transform),
		dataset_type(val_data, val_labels, dataset.transform),
	)


@icontract.require(lambda dataset, k: len(dataset) == 0 or _min_class_size(dataset) >= k, '每个类别的样本数量不能小于 K 折数')
@icontract.require(lambda dataset: dataset.getRawData()[1] is not None, 'balanced_k_fold requires a labeled dataset')
@icontract.require(lambda k: k >= 1)
def balanced_k_fold[T: IndexDataset](
	dataset: T,
	k: int,
	seed: int | None = None,
) -> Generator[tuple[T, T]]:
	"""K 折验证，保证同类别样本均匀分布在每一折中"""

	dataset_type = type(dataset)

	_ = _get_rng(seed)

	for fold in range(k):
		train_data = []
		train_labels = []
		val_data = []
		val_labels = []

		yield (
			dataset_type(train_data, train_labels, dataset.transform),
			dataset_type(val_data, val_labels, dataset.transform),
		)


@icontract.require(lambda k: k >= 1)
def k_fold[T: IndexDataset](
	dataset: T,
	k: int,
	seed: int | None = None,
) -> Generator[tuple[T, T]]:
	"""K 折验证，类别不均匀的情况下不保证同类别样本均匀分布在每一折中"""

	data, _ = dataset.getRawData()
	rng = _get_rng(seed)
	idx = rng.permutation(len(data))
	idxes = numpy.array_split(idx, k)
	for fold in range(k):
		train_idx = numpy.concatenate(idxes[:fold] + idxes[fold + 1 :]).tolist()
		val_idx = idxes[fold].tolist()
		yield (
			dataset.subset(train_idx),
			dataset.subset(val_idx),
		)


@icontract.require(lambda dataset: len(dataset) > 0)
@icontract.require(lambda dataset: dataset.getRawData()[1] is not None, 'trunc_max_class requires a labeled dataset')
@icontract.ensure(
	lambda dataset, result: len(result) == len(Counter(_labels_of(dataset))) * _min_class_size(dataset)
)
def trunc_max_class[T: IndexDataset](dataset: T) -> T:
	dataset_type = type(dataset)
	label_dict = group_by_label(dataset)
	minimum_size = min(len(i) for _, i in label_dict.items())
	rng = _get_rng(None)

	data = []
	labels = []
	for i, j in label_dict.items():
		idx = rng.permutation(len(j))
		data += gather(j, idx[:minimum_size])
		labels += [i] * minimum_size

	shuffle_correlated_lists(data, labels)
	return dataset_type(data, labels, dataset.transform)
