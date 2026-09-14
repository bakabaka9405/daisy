"""MAE Finetune 任务配置"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseTaskConfig
from ...shared import DatasetConfig, LogOutputConfig


class MAEFinetuneModelConfig(BaseConfig):
	"""MAE Finetune 模型配置"""

	name: str = 'vit_base_patch16'
	num_classes: int = 1000
	global_pool: str = 'avg'  # 'avg', 'token', ''
	drop_path: float = 0.1
	img_size: int = 224
	# MAE 预训练权重路径
	checkpoint: Path | None = None


class MAEFinetuneAugConfig(BaseConfig):
	"""MAE Finetune 数据增强配置"""

	input_size: int = 224
	# RandAugment 策略
	aa: str = 'rand-m9-mstd0.5-inc1'
	# Random Erasing
	reprob: float = 0.25
	remode: str = 'pixel'
	recount: int = 1
	# Mixup/CutMix
	mixup: float = 0.8
	cutmix: float = 1.0
	smoothing: float = 0.1


class MAEFinetuneTrainingConfig(BaseConfig):
	"""MAE Finetune 训练配置"""

	seed: int | None = None
	epochs: int = 100
	batch_size: int = 64
	lr: float = 0.0
	blr: float | None = 5e-4  # 基础学习率
	layer_decay: float = 0.65  # ViT-Base 推荐 0.65, ViT-Large/Huge 推荐 0.75
	weight_decay: float = 0.05
	warmup_epochs: int = 5
	min_lr: float = 1e-6
	accum_iter: int = 1
	use_amp: bool = True
	clip_grad: float | None = None  # None 表示不裁剪
	num_workers: int = 4
	save_freq: int = 20  # 保存频率
	augment: MAEFinetuneAugConfig = Field(default_factory=MAEFinetuneAugConfig)


class MAEFinetuneConfig(BaseTaskConfig):
	"""MAE Finetune 任务配置"""

	task_type: Annotated[Literal['mae_finetune'], tyro.conf.Suppress] = 'mae_finetune'  # type: ignore[assignment]
	output: LogOutputConfig = Field(default_factory=LogOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: MAEFinetuneModelConfig = Field(default_factory=MAEFinetuneModelConfig)
	training: MAEFinetuneTrainingConfig = Field(default_factory=MAEFinetuneTrainingConfig)
