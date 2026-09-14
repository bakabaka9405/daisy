"""MoCo 线性探测评估任务配置"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseTaskConfig
from ...shared import DatasetConfig, LogOutputConfig


class MoCoLinclsModelConfig(BaseConfig):
	"""MoCo 线性探测模型配置"""

	arch: str = 'resnet50'
	num_classes: int = 1000
	pretrained_checkpoint: Path | None = None  # MoCo 预训练 checkpoint 路径
	engine: Literal['v2', 'v3'] = 'v3'  # 用于确定权重提取方式


class MoCoLinclsTrainingConfig(BaseConfig):
	"""MoCo 线性探测训练配置"""

	seed: int | None = None
	epochs: int = 100
	batch_size: int = 256
	lr: float = 30.0  # 线性探测通常用较大学习率
	min_lr: float = 1e-6
	weight_decay: float = 0.0
	warmup_epochs: int = 0
	smoothing: float = 0.1
	accum_iter: int = 1
	use_amp: bool = True
	clip_grad: float | None = None
	num_workers: int | tuple[int, int] = 4
	save_freq: int = 20


class MoCoLinclsOutputConfig(LogOutputConfig):
	"""MoCo 线性探测输出配置"""

	save_best: bool = True


class MoCoLinclsConfig(BaseTaskConfig):
	"""MoCo 线性探测评估任务配置"""

	task_type: Annotated[Literal['moco_lincls'], tyro.conf.Suppress] = 'moco_lincls'  # type: ignore[assignment]
	output: MoCoLinclsOutputConfig = Field(default_factory=MoCoLinclsOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: MoCoLinclsModelConfig = Field(default_factory=MoCoLinclsModelConfig)
	training: MoCoLinclsTrainingConfig = Field(default_factory=MoCoLinclsTrainingConfig)
