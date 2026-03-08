"""MoCo 预训练任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig


class MoCoDatasetConfig(BaseModel):
	"""MoCo 数据集配置"""

	type: Literal['folder'] = 'folder'
	root: str = ''
	extensions: list[str] = Field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])


class MoCoModelConfig(BaseModel):
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


class MoCoTrainingConfig(BaseModel):
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
	resume: str | None = None
	# 数据增强
	input_size: int = 224


class MoCoOutputConfig(BaseOutputConfig):
	"""MoCo 输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = True


class MoCoPretrainConfig(BaseTaskConfig):
	"""MoCo 预训练任务配置"""

	task_type: Literal['moco_pretrain'] = 'moco_pretrain'  # type: ignore[assignment]
	engine: Literal['v2', 'v3'] = 'v3'
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: MoCoOutputConfig = Field(default_factory=MoCoOutputConfig)
	dataset: MoCoDatasetConfig = Field(default_factory=MoCoDatasetConfig)
	model: MoCoModelConfig = Field(default_factory=MoCoModelConfig)
	training: MoCoTrainingConfig = Field(default_factory=MoCoTrainingConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'moco_pretrain'
