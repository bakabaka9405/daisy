"""MoCo 预训练任务配置"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseTaskConfig
from ...shared import LogOutputConfig


class MoCoDatasetConfig(BaseConfig):
	"""MoCo 数据集配置"""

	root: str = ''
	extensions: list[str] = Field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])


class MoCoModelConfig(BaseConfig):
	"""MoCo 模型配置"""

	arch: str = 'resnet50'
	# MoCo v2 参数
	moco_dim: int = 128  # v2 默认 128, v3 默认 256
	moco_k: int = 65536  # queue size (仅 v2)
	moco_m: float = 0.999  # momentum
	moco_t: float = 0.07  # temperature, v2 默认 0.07, v3 默认 1.0
	# MoCo v3 参数
	moco_mlp_dim: int = 4096  # MLP hidden dim (仅 v3)
	# ViT 参数
	stop_grad_conv1: bool = False  # 是否冻结 conv1 (仅 ViT)


class MoCoTrainingConfig(BaseConfig):
	"""MoCo 训练配置"""

	epochs: int = 200
	batch_size: int = 256
	optimizer: Literal['sgd', 'adamw', 'lars'] = 'sgd'
	lr: float = 0.03
	momentum: float = 0.9
	weight_decay: float = 1e-4
	lr_schedule: Literal['step', 'cosine'] = 'cosine'
	lr_milestones: list[int] = Field(default_factory=lambda: [120, 160])
	warmup_epochs: int = 0
	moco_m_cos: bool = False  # cosine momentum schedule (仅 v3)
	use_amp: bool = True
	num_workers: int = 4
	save_freq: int = 20
	resume: Path | None = None
	# 数据增强
	input_size: int = 224


class MoCoPretrainConfig(BaseTaskConfig):
	"""MoCo 预训练任务配置"""

	task_type: Annotated[Literal['moco_pretrain'], tyro.conf.Suppress] = 'moco_pretrain'  # type: ignore[assignment]
	engine: Literal['v2', 'v3'] = 'v3'
	output: LogOutputConfig = Field(default_factory=LogOutputConfig)
	dataset: MoCoDatasetConfig = Field(default_factory=MoCoDatasetConfig)
	model: MoCoModelConfig = Field(default_factory=MoCoModelConfig)
	training: MoCoTrainingConfig = Field(default_factory=MoCoTrainingConfig)
