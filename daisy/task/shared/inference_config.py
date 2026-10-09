"""共享的推理任务配置"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from daisy.method.ordinal import METHODS
from ..base import BaseConfig


class InferenceModelConfig(BaseConfig):
	"""推理模型配置"""

	family: Literal['classification', 'mae_finetune', 'ordinal'] = 'classification'
	method: str = 'ce'
	name: str = 'resnet34'
	pretrained: bool = False
	checkpoint: Path | None = None
	num_classes: int = 2
	global_pool: str = 'avg'
	drop_path: float = 0.1
	img_size: int = 224

	@model_validator(mode='after')
	def validate_config(self) -> Self:
		if self.family == 'ordinal' and self.method not in METHODS:
			raise ValueError('Unknown ordinal method')
		if self.family == 'ordinal' and self.num_classes < 2:
			raise ValueError('num_classes must be >= 2 for ordinal models')
		if not 0 <= self.drop_path < 1:
			raise ValueError('drop_path must satisfy 0 <= drop_path < 1')
		if not self.img_size > 0:
			raise ValueError('img_size must be > 0')
		return self


class InferenceRuntimeConfig(BaseConfig):
	"""推理运行配置"""

	split_name: str = 'val'
	batch_size: int = 16
	num_workers: int = 4
	transform: str = 'rectangle_val'
	input_size: int = 224
	seed: int | None = None

	@model_validator(mode='after')
	def validate_config(self) -> Self:
		if self.batch_size < 1:
			raise ValueError('batch_size must be >= 1')
		if self.num_workers < 0:
			raise ValueError('num_workers must be >= 0')
		if self.input_size <= 0:
			raise ValueError('input_size must be > 0')
		return self
