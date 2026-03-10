from .index_dataset import IndexDataset
import numpy
from ..util import (
	gather_list_by_indexes as gather,
	shuffle_correlated_lists,
)
from collections.abc import Generator


def _get_rng(seed: int | None = None):
	return numpy.random.default_rng(seed)


def split_by_label(dataset: IndexDataset) -> dict[int, list]:
	"""将数据集按标签分割为多个子集"""
	label_to_indices: dict[int, list] = {}
	data, labels = dataset.getRawData()
	for i, j in zip(data, labels):
		if j not in label_to_indices:
			label_to_indices[j] = []
		label_to_indices[j].append(i)
	return label_to_indices


def default_data_split(
	dataset: IndexDataset,
	val_ratio: float = 0.1,
	seed: int | None = None,
) -> tuple[IndexDataset, IndexDataset]:
	dataset_type = type(dataset)

	data, labels = dataset.getRawData()

	rng = _get_rng(seed)
	idx = rng.permutation(len(data))

	train_size = int(len(data) * (1 - val_ratio))

	return (
		dataset_type(gather(data, idx[:train_size]), gather(labels, idx[:train_size])),
		dataset_type(gather(data, idx[train_size:]), gather(labels, idx[train_size:])),
	)


def stratified_data_split(
	dataset: IndexDataset,
	val_ratio: float = 0.1,
	seed: int | None = None,
	shuffle: bool = True,
) -> tuple[IndexDataset, IndexDataset]:
	"""按类别分层划分训练/验证集"""
	dataset_type = type(dataset)
	data, labels = dataset.getRawData()
	label_dict = split_by_label(dataset)
	rng = _get_rng(seed)

	train_pairs: list[tuple[object, int]] = []
	val_pairs: list[tuple[object, int]] = []

	for label, items in label_dict.items():
		if len(items) < 2:
			raise ValueError(f'Class {label} has fewer than 2 samples and cannot be stratified')
		idx = rng.permutation(len(items))
		val_size = max(1, int(round(len(items) * val_ratio)))
		if val_size >= len(items):
			val_size = len(items) - 1
		train_size = len(items) - val_size
		train_pairs.extend((items[i], label) for i in idx[:train_size])
		val_pairs.extend((items[i], label) for i in idx[train_size:])

	if shuffle:
		train_idx = rng.permutation(len(train_pairs)) if train_pairs else []
		val_idx = rng.permutation(len(val_pairs)) if val_pairs else []
		train_pairs = [train_pairs[i] for i in train_idx]
		val_pairs = [val_pairs[i] for i in val_idx]

	train_data = [item for item, _ in train_pairs]
	train_labels = [label for _, label in train_pairs]
	val_data = [item for item, _ in val_pairs]
	val_labels = [label for _, label in val_pairs]

	return dataset_type(train_data, train_labels), dataset_type(val_data, val_labels)


def split_by_groups(
	dataset: IndexDataset,
	groups: list[str | int],
	val_ratio: float = 0.1,
	seed: int | None = None,
) -> tuple[IndexDataset, IndexDataset]:
	"""按 group 划分，保证同组样本不会跨 split 泄露"""
	if len(groups) != len(dataset):
		raise ValueError('groups length must match dataset length')

	dataset_type = type(dataset)
	data, labels = dataset.getRawData()
	group_to_indices: dict[str | int, list[int]] = {}
	for index, group in enumerate(groups):
		group_to_indices.setdefault(group, []).append(index)

	rng = _get_rng(seed)
	group_keys = list(group_to_indices.keys())
	shuffled_group_keys = [group_keys[i] for i in rng.permutation(len(group_keys))]
	val_group_count = max(1, int(round(len(shuffled_group_keys) * val_ratio)))
	if val_group_count >= len(shuffled_group_keys):
		val_group_count = len(shuffled_group_keys) - 1
	if val_group_count <= 0:
		raise ValueError('Not enough groups to create a validation split')

	val_groups = set(shuffled_group_keys[:val_group_count])
	train_indices: list[int] = []
	val_indices: list[int] = []
	for group, indices in group_to_indices.items():
		if group in val_groups:
			val_indices.extend(indices)
		else:
			train_indices.extend(indices)

	return (
		dataset_type(gather(data, train_indices), gather(labels, train_indices)),
		dataset_type(gather(data, val_indices), gather(labels, val_indices)),
	)


def minimum_class_proportional_split(
	dataset: IndexDataset,
	val_ratio: float = 0.1,
	val_minimum_size: int = 100,
	force_fetch_minimum_size: bool = False,
	val_maximum_ratio: float = 0.5,
	shuffle: bool = True,
	seed: int | None = None,
) -> tuple[IndexDataset, IndexDataset]:
	dataset_type = type(dataset)
	rng = _get_rng(seed)

	label_dict = split_by_label(dataset)

	train_data = []
	train_labels = []
	val_data = []
	val_labels = []

	val_size: int = int(min(len(i) for _, i in label_dict.items()) * val_ratio)
	if val_size < val_minimum_size:
		if force_fetch_minimum_size:
			val_size = val_minimum_size
		else:
			raise ValueError(
				f'类型不均衡太严重或设置的 val_minimum_size 阈值过高，设定 {val_minimum_size}，检查到最少的类别数量 {val_size} 不满足要求'
			)
	for _, i in label_dict.items():
		if val_size / len(i) > val_maximum_ratio:
			raise ValueError(
				f'类型不均衡太严重或设置的 val_maximum_ratio 阈值过低，设定 {val_maximum_ratio}，'
				f'检查到有类别需要 {val_size}/{len(i)} 个样本用于验证集，不满足要求'
			)

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

	return dataset_type(train_data, train_labels), dataset_type(val_data, val_labels)


def balanced_k_fold(
	dataset: IndexDataset,
	k: int,
	seed: int | None = None,
) -> Generator[tuple[IndexDataset, IndexDataset]]:
	"""K 折验证，保证同类别样本均匀分布在每一折中"""

	dataset_type = type(dataset)

	label_dict = split_by_label(dataset)

	for _, i in label_dict.items():
		if len(i) < k:
			raise ValueError(f'类别 {i} 的样本数量 {len(i)} 小于 K 折数 {k}')

	_ = _get_rng(seed)

	for fold in range(k):
		train_data = []
		train_labels = []
		val_data = []
		val_labels = []

		yield dataset_type(train_data, train_labels), dataset_type(val_data, val_labels)


def k_fold(
	dataset: IndexDataset,
	k: int,
	seed: int | None = None,
) -> Generator[tuple[IndexDataset, IndexDataset]]:
	"""K 折验证，类别不均匀的情况下不保证同类别样本均匀分布在每一折中"""

	dataset_type = type(dataset)

	data, labels = dataset.getRawData()
	rng = _get_rng(seed)
	idx = rng.permutation(len(data))
	idxes = numpy.array_split(idx, k)
	for fold in range(k):
		train_data = gather(data, numpy.concatenate(idxes[:fold] + idxes[fold + 1 :]))
		train_labels = gather(labels, numpy.concatenate(idxes[:fold] + idxes[fold + 1 :]))
		val_data = gather(data, idxes[fold])
		val_labels = gather(labels, idxes[fold])
		yield dataset_type(train_data, train_labels), dataset_type(val_data, val_labels)


def trunc_max_class(dataset: IndexDataset):
	dataset_type = type(dataset)
	label_dict = split_by_label(dataset)
	minimum_size = min(len(i) for _, i in label_dict.items())
	rng = _get_rng(None)

	data = []
	labels = []
	for i, j in label_dict.items():
		idx = rng.permutation(len(j))
		data += gather(j, idx[:minimum_size])
		labels += [i] * minimum_size

	shuffle_correlated_lists(data, labels)
	return dataset_type(data, labels)
