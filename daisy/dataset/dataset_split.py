from .index_dataset import IndexDataset
import numpy
from ..util import (
	gather_list_by_indexes as gather,
	shuffle_correlated_lists,
)
from collections.abc import Generator


def split_by_label(dataset: IndexDataset) -> dict[int, list]:
	"""将数据集按标签分割为多个子集"""
	label_to_indices: dict[int, list] = {}
	data, labels = dataset.getRawData()
	for i, j in zip(data, labels):
		if j not in label_to_indices:
			label_to_indices[j] = []
		label_to_indices[j].append(i)
	return label_to_indices


def default_data_split(dataset: IndexDataset, val_ratio: float = 0.1) -> tuple[IndexDataset, IndexDataset]:
	dataset_type = type(dataset)

	data, labels = dataset.getRawData()

	idx = numpy.random.permutation(len(data))

	train_size = int(len(data) * (1 - val_ratio))

	return (
		dataset_type(gather(data, idx[:train_size]), gather(labels, idx[:train_size])),
		dataset_type(gather(data, idx[train_size:]), gather(labels, idx[train_size:])),
	)


def minimum_class_proportional_split(
	dataset: IndexDataset,
	val_ratio: float = 0.1,
	val_minimum_size: int = 100,
	force_fetch_minimum_size: bool = False,
	val_maximum_ratio: float = 0.5,
	shuffle: bool = True,
) -> tuple[IndexDataset, IndexDataset]:
	dataset_type = type(dataset)

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
		idx = numpy.random.permutation(len(j))
		train_size = len(j) - val_size
		train_data += gather(j, idx[:train_size])
		train_labels += [i] * train_size
		val_data += gather(j, idx[train_size:])
		val_labels += [i] * val_size

	if shuffle:
		shuffle_correlated_lists(train_data, train_labels)
		shuffle_correlated_lists(val_data, val_labels)

	return dataset_type(train_data, train_labels), dataset_type(val_data, val_labels)


def balanced_k_fold(dataset: IndexDataset, k: int) -> Generator[tuple[IndexDataset, IndexDataset]]:
	"""K 折验证，保证同类别样本均匀分布在每一折中"""

	dataset_type = type(dataset)

	label_dict = split_by_label(dataset)

	for _, i in label_dict.items():
		if len(i) < k:
			raise ValueError(f'类别 {i} 的样本数量 {len(i)} 小于 K 折数 {k}')

	idx = numpy.random.permutation(len(dataset))

	for fold in range(k):
		train_data = []
		train_labels = []
		val_data = []
		val_labels = []

		yield dataset_type(train_data, train_labels), dataset_type(val_data, val_labels)


def k_fold(dataset: IndexDataset, k: int) -> Generator[tuple[IndexDataset, IndexDataset]]:
	"""K 折验证，类别不均匀的情况下不保证同类别样本均匀分布在每一折中"""

	dataset_type = type(dataset)

	data, labels = dataset.getRawData()
	idx = numpy.random.permutation(len(data))
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

	data = []
	labels = []
	for i, j in label_dict.items():
		idx = numpy.random.permutation(len(j))
		data += gather(j, idx[:minimum_size])
		labels += [i] * minimum_size

	shuffle_correlated_lists(data, labels)
	return dataset_type(data, labels)
