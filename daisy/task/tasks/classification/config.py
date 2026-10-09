"""分类任务配置"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
import tyro.conf

from daisy.util import to_2tuple

from ...base import BaseConfig, BaseOutputConfig, BaseTaskConfig
from ...shared import DatasetConfig


class ModelConfig(BaseConfig):
	"""模型配置"""

	name: str = 'resnet34'
	pretrained: bool = True
	# 分类类别数，feeder 产出的标签须位于 [0, num_classes)
	num_classes: int = Field(default=2, ge=2)
	# 可选：加载预训练权重
	checkpoint: Path | None = None


class TransformConfig(BaseConfig):
	"""数据增强配置"""

	train: str = 'rectangle_train'
	val: str = 'rectangle_val'


class TrainingConfig(BaseConfig):
	"""训练配置"""

	seed: int | None = None
	epochs: int = 30
	batch_size: int = 64
	lr: float = 1e-3
	weight_decay: float = 1e-4
	warmup_epochs: int = 0
	smoothing: float = 0.1
	accum_iter: int = 1
	use_scheduler: bool = True
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	num_workers: int | tuple[int, int] = 4
	early_stop: bool = False
	early_stop_epoch: int = 5
	cmp_obj: Literal['acc', 'prec', 'recall', 'f1'] = 'f1'
	transform: TransformConfig = Field(default_factory=TransformConfig)

	@model_validator(mode='after')
	def validate_config(self) -> Self:
		if self.epochs < 1:
			raise ValueError('epochs must be >= 1')
		if self.batch_size < 1:
			raise ValueError('batch_size must be >= 1')
		if not self.lr > 0:
			raise ValueError('lr must be > 0')
		if not self.weight_decay >= 0:
			raise ValueError('weight_decay must be >= 0')
		if self.warmup_epochs < 0:
			raise ValueError('warmup_epochs must be >= 0')
		if not 0 <= self.smoothing < 1:
			raise ValueError('smoothing must satisfy 0 <= smoothing < 1')
		if self.accum_iter < 1:
			raise ValueError('accum_iter must be >= 1')
		if not self.max_norm > 0:
			raise ValueError('max_norm must be > 0')
		if min(to_2tuple(self.num_workers)) < 0:
			raise ValueError('num_workers must be >= 0')
		if self.early_stop_epoch < 1:
			raise ValueError('early_stop_epoch must be >= 1')
		return self


class ClassificationConfig(BaseTaskConfig):
	"""分类任务配置"""

	task_type: Annotated[Literal['classification'], tyro.conf.Suppress] = 'classification'  # type: ignore[assignment]
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: DatasetConfig
	model: ModelConfig = Field(default_factory=ModelConfig)
	training: TrainingConfig = Field(default_factory=TrainingConfig)
