from collections import Counter

import icontract
import numpy

from .index_dataset import IndexDataset


def _get_rng(seed: int | None = None):
	return numpy.random.default_rng(seed)


def _label_counts(dataset: IndexDataset) -> Counter:
	labels = dataset.getRawData()[1]
	assert labels is not None
	return Counter(labels)


@icontract.require(lambda ratio: ratio > 0)
@icontract.ensure(lambda dataset, result: len(result) <= len(dataset))
def random_subsample[T: IndexDataset](dataset: T, ratio: float, seed: int | None = None) -> T:
	"""按比例随机抽取数据集子集"""
	if ratio >= 1.0 or len(dataset) == 0:
		return dataset
	k = max(1, round(len(dataset) * ratio))
	rng = _get_rng(seed)
	selected = sorted(rng.choice(len(dataset), size=k, replace=False).tolist())
	return dataset.subset(selected)


@icontract.require(
	lambda dataset, expected_size, num_classes, strict: (
		not strict or all(_label_counts(dataset)[i] >= expected_size // num_classes for i in range(num_classes))
	),
	'strict 模式下每个类别的样本数不能少于 expected_size // num_classes',
)
@icontract.require(lambda dataset, num_classes: all(0 <= label < num_classes for label in _label_counts(dataset)))
@icontract.require(lambda dataset: dataset.getRawData()[1] is not None, 'balanced_sample requires a labeled dataset')
@icontract.require(
	lambda dataset, expected_size: expected_size <= len(dataset), 'expected_size must be less than or equal to the size of the dataset'
)
@icontract.require(lambda expected_size: expected_size >= 0)
@icontract.require(lambda num_classes: num_classes >= 1)
@icontract.ensure(lambda expected_size, result: sum(result[1]) == expected_size)
@icontract.ensure(lambda num_classes, result: len(result[1]) == num_classes)
@icontract.ensure(lambda expected_size, result: len(result[0]) == expected_size)
def balanced_sample[T: IndexDataset](
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

	labels = dataset.getRawData()[1]
	assert labels is not None
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

	selected_indices = []
	class_counts = [0] * num_classes
	for i in range(num_classes):
		selected_indices.extend(class_index[i][:ans])

	selected_indices = selected_indices[:expected_size]
	rng.shuffle(selected_indices)

	for idx in selected_indices:
		class_counts[labels[idx]] += 1

	return dataset.subset(selected_indices), class_counts
