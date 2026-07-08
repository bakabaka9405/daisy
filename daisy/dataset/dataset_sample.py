from .index_dataset import IndexDataset
import numpy
from typing import TypeVar


def _get_rng(seed: int | None = None):
	return numpy.random.default_rng(seed)


T = TypeVar('T', bound='IndexDataset[int]')


def balanced_sample(
	dataset: T,
	expected_size: int,
	num_classes: int,
	seed: int | None = None,
	strict: bool = False,  # 严格模式，若某类别样本不足，则抛出异常；否则尽可能均衡地抽样
) -> tuple[T, list[int]]:
	"""
	从数据集中抽取一个子集，使得子集中每个类别的样本数量尽可能相同

	优化目标：在保证每个类别样本数量尽可能相同的前提下，使得最大类别数量尽可能少
	"""

	if expected_size > len(dataset):
		raise ValueError('expected_size must be less than or equal to the size of the dataset')

	labels = dataset.getRawData()[1]
	rng = _get_rng(seed)

	class_index = [[] for _ in range(num_classes)]
	for i, label in enumerate(labels):
		class_index[label].append(i)

	L = 1
	R = len(dataset)
	mid = 0
	ans = -1

	def check(k: int) -> bool:
		count = 0
		for i in range(num_classes):
			count += min(k, len(class_index[i]))
		return count >= expected_size

	while L <= R:
		mid = (L + R) // 2
		if check(mid):
			ans = mid
			R = mid - 1
		else:
			L = mid + 1

	assert ans != -1

	expected_avg = expected_size // num_classes

	selected_indices = []
	class_counts = [0] * num_classes
	for i in range(num_classes):
		if strict and len(class_index[i]) < expected_avg:
			raise ValueError(f'Class {i} has only {len(class_index[i])} samples, which is less than the expected average {expected_avg}')
		selected_indices.extend(class_index[i][:ans])

	selected_indices = selected_indices[:expected_size]
	rng.shuffle(selected_indices)

	for idx in selected_indices:
		class_counts[labels[idx]] += 1

	return dataset.subset(selected_indices), class_counts
