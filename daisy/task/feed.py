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


def _is_class_index(label: Any, num_classes: int) -> bool:
	"""label 是否为 [0, num_classes) 内的整数（不含 bool）"""
	return isinstance(label, Integral) and not isinstance(label, bool) and 0 <= int(label) < num_classes


def assert_label_range(splits: Mapping[str, DiskDataset[Any, Any]], num_classes: int) -> None:
	"""逐样本校验单标签的类型与类别范围"""
	for name, dataset in splits.items():
		if dataset.labels is None:
			raise ValueError(f'Split {name!r} is unlabeled')
		for index, label in enumerate(dataset.labels):
			if not _is_class_index(label, num_classes):
				message = f'Split {name!r} sample {index}: label must be an integer in [0, {num_classes}), got {label!r}'
				if not isinstance(label, Integral) or isinstance(label, bool):
					raise TypeError(message)
				raise ValueError(message)


def assert_multitask_label_range(splits: Mapping[str, DiskDataset[Any, Any]], targets: Mapping[str, int]) -> None:
	"""逐样本校验多任务标签键与各任务类别范围"""
	target_keys = set(targets)
	for name, dataset in splits.items():
		if dataset.labels is None:
			raise ValueError(f'Split {name!r} is unlabeled')
		for index, label in enumerate(dataset.labels):
			context = f'Split {name!r} sample {index}'
			if not isinstance(label, Mapping):
				raise TypeError(f'{context}: label must be a Mapping, got {label!r}')
			label_keys = set(label)
			if label_keys != target_keys:
				raise ValueError(f'{context}: label keys must match targets {target_keys!r}, got {label_keys!r}')
			for target, num_classes in targets.items():
				value = label[target]
				if not _is_class_index(value, num_classes):
					message = f'{context} target {target!r}: label must be an integer in [0, {num_classes}), got {value!r}'
					if not isinstance(value, Integral) or isinstance(value, bool):
						raise TypeError(message)
					raise ValueError(message)
