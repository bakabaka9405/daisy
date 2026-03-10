"""共享的推理任务配置"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class InferenceModelConfig(BaseModel):
	"""推理模型配置"""

	family: Literal['classification', 'mae_finetune'] = 'classification'
	name: str = 'resnet34'
	pretrained: bool = False
	checkpoint: str = ''
	num_classes: int = 2
	global_pool: str = 'avg'
	drop_path: float = 0.1
	img_size: int = 224


class InferenceRuntimeConfig(BaseModel):
	"""推理运行配置"""

	split_name: str = 'val'
	batch_size: int = 16
	num_workers: int = 4
	transform: str = 'rectangle_val'
	input_size: int = 224
	seed: int | None = None
