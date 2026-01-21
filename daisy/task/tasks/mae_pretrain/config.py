"""MAE 预训练任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig


class MAEDatasetConfig(BaseModel):
	"""MAE 数据集配置"""

	type: Literal['folder'] = 'folder'
	root: str = ''
	extensions: list[str] = Field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])


class MAEModelConfig(BaseModel):
	"""MAE 模型配置"""

	name: str = 'mae_vit_base_patch16'
	img_size: int = 224
	norm_pix_loss: bool = False
	# 可选：加载预训练权重
	checkpoint: str | None = None


class MAETransformConfig(BaseModel):
	"""MAE 数据增强配置"""

	input_size: int = 224
	# RandomResizedCrop 的 scale 参数
	scale_min: float = 0.2
	scale_max: float = 1.0
	# 是否水平翻转
	hflip: bool = True


class MAETrainingConfig(BaseModel):
	"""MAE 训练配置"""

	epochs: int = 400
	batch_size: int = 64
	blr: float = 1.5e-4  # 基础学习率
	weight_decay: float = 0.05
	warmup_epochs: int = 40
	mask_ratio: float = 0.75
	accum_iter: int = 1
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	num_workers: int = 4
	save_freq: int = 20  # 保存频率
	transform: MAETransformConfig = Field(default_factory=MAETransformConfig)


class MAEOutputConfig(BaseOutputConfig):
	"""MAE 输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = True


class MAEPretrainConfig(BaseTaskConfig):
	"""MAE 预训练任务配置"""

	task_type: Literal['mae_pretrain'] = 'mae_pretrain'  # type: ignore[assignment]
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: MAEOutputConfig = Field(default_factory=MAEOutputConfig)
	dataset: MAEDatasetConfig = Field(default_factory=MAEDatasetConfig)
	model: MAEModelConfig = Field(default_factory=MAEModelConfig)
	training: MAETrainingConfig = Field(default_factory=MAETrainingConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'mae_pretrain'
