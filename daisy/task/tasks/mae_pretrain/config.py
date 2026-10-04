"""MAE 预训练任务配置"""

from pathlib import Path
from typing import Annotated, Any, Literal

import icontract
from pydantic import Field
import tyro.conf

from daisy.feeder import ImageFolderParams
from ...base import BaseConfig, BaseTaskConfig
from ...shared import DatasetConfig, FeederConfig, LogOutputConfig


class MAEModelConfig(BaseConfig):
	"""MAE 模型配置"""

	name: str = 'mae_vit_base_patch16'
	img_size: int | tuple[int, int] = 224  # 图像尺寸，支持整数或 (H, W) 元组
	pretrained: Path | None = None  # 本地完整预训练检查点路径
	timm_pretrained: str | None = None

	@icontract.require(lambda self: min(self.img_size if isinstance(self.img_size, tuple) else (self.img_size,)) > 0)
	def model_post_init(self, context: Any, /) -> None:
		pass


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

	@icontract.require(lambda self: 0 < self.ratio_min <= self.ratio_max)
	@icontract.require(lambda self: 0 < self.scale_min <= self.scale_max)
	def model_post_init(self, context: Any, /) -> None:
		pass


class MAETrainingConfig(BaseConfig):
	"""MAE 训练配置"""

	seed: int | None = None
	epochs: int = 400
	batch_size: int = 64
	blr: float = 1.5e-4  # 基础学习率
	weight_decay: float = 0.05
	warmup_epochs: int = 40
	freeze_encoder_epochs: int = Field(default=0, ge=0)
	mask_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
	norm_pix_loss: bool = False
	accum_iter: int = 1
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	num_workers: int = 4
	save_freq: int = 20  # 保存频率
	resume: Path | None = None  # 恢复训练的 checkpoint 路径
	compile_backend: str | None = 'inductor'  # 编译后端，可选 'inductor'、'eager' 或 None
	transform: MAETransformConfig = Field(default_factory=MAETransformConfig)

	@icontract.require(lambda self: self.save_freq >= 0)
	@icontract.require(lambda self: self.num_workers >= 0)
	@icontract.require(lambda self: self.max_norm > 0)
	@icontract.require(lambda self: self.accum_iter >= 1)
	@icontract.require(lambda self: self.warmup_epochs >= 0)
	@icontract.require(lambda self: self.weight_decay >= 0)
	@icontract.require(lambda self: self.blr > 0)
	@icontract.require(lambda self: self.batch_size >= 1)
	@icontract.require(lambda self: self.epochs >= 1)
	def model_post_init(self, context: Any, /) -> None:
		pass


def _default_mae_dataset() -> FeederConfig:
	return FeederConfig(feeder='image_folder', params=ImageFolderParams())


class MAEPretrainConfig(BaseTaskConfig):
	"""MAE 预训练任务配置"""

	task_type: Annotated[Literal['mae_pretrain'], tyro.conf.Suppress] = 'mae_pretrain'  # type: ignore[assignment]
	output: LogOutputConfig = Field(default_factory=LogOutputConfig)
	dataset: DatasetConfig = Field(default_factory=_default_mae_dataset)
	model: MAEModelConfig = Field(default_factory=MAEModelConfig)
	training: MAETrainingConfig = Field(default_factory=MAETrainingConfig)
