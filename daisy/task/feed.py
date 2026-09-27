"""feeder 调用与通用数据检查"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

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


def _check_label(label: Any, num_classes: int, where: str) -> None:
	if not isinstance(label, Integral) or isinstance(label, bool):
		raise TypeError(f'Label {label!r} in {where} must be an integer')
	if not 0 <= int(label) < num_classes:
		raise ValueError(f'Label {label} in {where} out of range [0, {num_classes - 1}]')


def assert_label_range(splits: Mapping[str, DiskDataset[Any, Any]], num_classes: int) -> None:
	"""校验各划分的单标签均为 [0, num_classes) 内的整数"""
	for name, dataset in splits.items():
		if dataset.labels is None:
			raise ValueError(f'Split {name!r} is unlabeled')
		for label in dataset.labels:
			_check_label(label, num_classes, f'split {name!r}')


def assert_multitask_label_range(splits: Mapping[str, DiskDataset[Any, Any]], targets: Mapping[str, int]) -> None:
	"""校验各划分的多任务标签键与 targets 一致且各值在对应类别范围内"""
	expected = set(targets)
	for name, dataset in splits.items():
		if dataset.labels is None:
			raise ValueError(f'Split {name!r} is unlabeled')
		for label in dataset.labels:
			if not isinstance(label, Mapping) or set(label) != expected:
				raise ValueError(f'Label {label!r} in split {name!r} does not match targets {sorted(expected)}')
			for target, num_classes in targets.items():
				_check_label(label[target], num_classes, f'split {name!r} target {target!r}')
