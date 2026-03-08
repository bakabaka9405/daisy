"""MoCo 线性探测评估任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig
from ..classification.config import DatasetConfig


class MoCoLinclsModelConfig(BaseModel):
	"""MoCo 线性探测模型配置"""

	arch: str = 'resnet50'
	num_classes: int = 1000
	pretrained_checkpoint: str = ''  # MoCo 预训练 checkpoint 路径
	engine: Literal['v2', 'v3'] = 'v3'  # 用于确定权重提取方式


class MoCoLinclsTrainingConfig(BaseModel):
	"""MoCo 线性探测训练配置"""

	epochs: int = 100
	batch_size: int = 256
	lr: float = 30.0  # 线性探测通常用较大学习率
	momentum: float = 0.9
	weight_decay: float = 0.0
	lr_schedule: Literal['step', 'cosine'] = 'step'
	lr_milestones: list[int] = Field(default_factory=lambda: [60, 80])
	warmup_epochs: int = 0
	use_amp: bool = True
	num_workers: int = 4
	save_freq: int = 20


class MoCoLinclsOutputConfig(BaseOutputConfig):
	"""MoCo 线性探测输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = True


class MoCoLinclsConfig(BaseTaskConfig):
	"""MoCo 线性探测评估任务配置"""

	task_type: Literal['moco_lincls'] = 'moco_lincls'  # type: ignore[assignment]
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: MoCoLinclsOutputConfig = Field(default_factory=MoCoLinclsOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: MoCoLinclsModelConfig = Field(default_factory=MoCoLinclsModelConfig)
	training: MoCoLinclsTrainingConfig = Field(default_factory=MoCoLinclsTrainingConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'moco_lincls'
