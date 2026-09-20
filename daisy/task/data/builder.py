"""声明式数据集构建器"""

from __future__ import annotations

from typing import Any

from ..shared.dataset_config import DatasetConfig
from .interpret import interpret
from .model import DatasetSplits
from .parse import parse_dataset_config


def build_datasets(
	config: DatasetConfig,
	*,
	seed: int | None = None,
) -> DatasetSplits[Any]:
	"""根据 DatasetConfig 声明式构建 DiskDataset 字典

	Args:
		config: 数据集配置对象（单来源或多来源划分）
		seed: 调用方提供的随机种子

	Returns:
		划分名称到 DiskDataset 的字典
	"""
	return interpret(parse_dataset_config(config), seed=seed)
