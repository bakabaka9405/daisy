"""feeder 调用与通用数据检查"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

import icontract

from daisy.dataset.dataset_split import assert_disjoint_file_splits
from daisy.dataset.disk_dataset import DiskDataset
from daisy.feeder import FeederSplits, get_feeder
from .shared.feeder_config import FeederConfig


def feed(config: FeederConfig, *, seed: int | None) -> FeederSplits:
	"""调用 feeder 构建数据划分，并校验划分非空且文件互斥

	Args:
		config: 数据集配置
		seed: 调用方随机种子，透传给 feeder

	Returns:
		划分名称到 DiskDataset 的字典
	"""
	splits = get_feeder(config.feeder).fn(config.params, seed=seed)

	if not isinstance(splits, dict) or not splits:
		raise TypeError(f'Feeder {config.feeder!r} must return a non-empty dict of splits, got {splits!r}')
	for name, dataset in splits.items():
		if not isinstance(dataset, DiskDataset):
			raise TypeError(f'Feeder {config.feeder!r} split {name!r} must be DiskDataset, got {type(dataset).__name__}')
		if len(dataset) == 0:
			raise ValueError(f'Feeder {config.feeder!r} produced empty split {name!r}')

	assert_disjoint_file_splits(splits)
	return splits


def _is_class_index(label: Any, num_classes: int) -> bool:
	"""label 是否为 [0, num_classes) 内的整数（不含 bool）"""
	return isinstance(label, Integral) and not isinstance(label, bool) and 0 <= int(label) < num_classes


def _unlabeled_splits(splits: Mapping[str, DiskDataset[Any, Any]]) -> list[str]:
	"""无标签的划分名称"""
	return [name for name, dataset in splits.items() if dataset.labels is None]


def _labels_in_range(splits: Mapping[str, DiskDataset[Any, Any]], num_classes: int) -> bool:
	"""各划分的单标签均为 [0, num_classes) 内的整数"""
	return all(_is_class_index(label, num_classes) for dataset in splits.values() for label in dataset.labels or ())


def _multitask_labels_valid(splits: Mapping[str, DiskDataset[Any, Any]], targets: Mapping[str, int]) -> bool:
	"""各划分的多任务标签键与 targets 一致，且各值在对应类别范围内"""
	return all(
		isinstance(label, Mapping)
		and set(label) == set(targets)
		and all(_is_class_index(label[target], num_classes) for target, num_classes in targets.items())
		for dataset in splits.values()
		for label in dataset.labels or ()
	)


@icontract.require(lambda splits, num_classes: _labels_in_range(splits, num_classes), 'Label must be an integer in [0, num_classes)')
@icontract.require(lambda splits: not _unlabeled_splits(splits), 'Split is unlabeled')
def assert_label_range(splits: Mapping[str, DiskDataset[Any, Any]], num_classes: int) -> None:
	"""校验各划分的单标签均为 [0, num_classes) 内的整数（校验全部由前置条件完成）"""


@icontract.require(
	lambda splits, targets: _multitask_labels_valid(splits, targets),
	'Label keys must match targets and each value must be an integer in [0, num_classes)',
)
@icontract.require(lambda splits: not _unlabeled_splits(splits), 'Split is unlabeled')
def assert_multitask_label_range(splits: Mapping[str, DiskDataset[Any, Any]], targets: Mapping[str, int]) -> None:
	"""校验各划分的多任务标签键与 targets 一致且各值在对应类别范围内（校验全部由前置条件完成）"""
