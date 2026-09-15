"""MAE 预训练任务配置"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseTaskConfig
from ...shared import LogOutputConfig


class MAEDatasetConfig(BaseConfig):
	"""MAE 数据集配置"""

	root: str = ''
	extensions: list[str] = Field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
	sample_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
	sample_seed: int | None = None


class MAEModelConfig(BaseConfig):
	"""MAE 模型配置"""

	name: str = 'mae_vit_base_patch16'
	img_size: int | tuple[int, int] = 224  # 图像尺寸，支持整数或 (H, W) 元组
	timm_pretrained: str | None = None


class MAETransformConfig(BaseConfig):
	"""MAE 数据增强配置"""
	# RandomResizedCrop 的 scale 参数
	scale_min: float = 0.2
	scale_max: float = 1.0
	# RandomResizedCrop 的 ratio 参数
	ratio_min: float = 0.75
	ratio_max: float = 1.3333
	# 是否水平翻转
	hflip: bool = True


class MAETrainingConfig(BaseConfig):
	"""MAE 训练配置"""

	seed: int | None = None
	epochs: int = 400
	batch_size: int = 64
	blr: float = 1.5e-4  # 基础学习率
	weight_decay: float = 0.05
	warmup_epochs: int = 40
	mask_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
	norm_pix_loss: bool = False
	accum_iter: int = 1
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	num_workers: int = 4
	save_freq: int = 20  # 保存频率
	resume: Path | None = None  # 恢复训练的 checkpoint 路径
	transform: MAETransformConfig = Field(default_factory=MAETransformConfig)


class MAEPretrainConfig(BaseTaskConfig):
	"""MAE 预训练任务配置"""

	task_type: Annotated[Literal['mae_pretrain'], tyro.conf.Suppress] = 'mae_pretrain'  # type: ignore[assignment]
	output: LogOutputConfig = Field(default_factory=LogOutputConfig)
	dataset: MAEDatasetConfig = Field(default_factory=MAEDatasetConfig)
	model: MAEModelConfig = Field(default_factory=MAEModelConfig)
	training: MAETrainingConfig = Field(default_factory=MAETrainingConfig)
