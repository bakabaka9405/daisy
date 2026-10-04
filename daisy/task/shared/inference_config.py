"""共享的推理任务配置"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import icontract

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

	@icontract.require(lambda self: self.img_size > 0)
	@icontract.require(lambda self: 0 <= self.drop_path < 1)
	@icontract.require(lambda self: self.family != 'ordinal' or self.num_classes >= 2, 'num_classes must be >= 2 for ordinal models')
	@icontract.require(lambda self: self.family != 'ordinal' or self.method in METHODS, 'Unknown ordinal method')
	def model_post_init(self, context: Any, /) -> None:
		pass


class InferenceRuntimeConfig(BaseConfig):
	"""推理运行配置"""

	split_name: str = 'val'
	batch_size: int = 16
	num_workers: int = 4
	transform: str = 'rectangle_val'
	input_size: int = 224
	seed: int | None = None

	@icontract.require(lambda self: self.input_size > 0)
	@icontract.require(lambda self: self.num_workers >= 0)
	@icontract.require(lambda self: self.batch_size >= 1)
	def model_post_init(self, context: Any, /) -> None:
		pass
