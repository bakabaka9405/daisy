"""分类任务配置"""

from pathlib import Path
from typing import Annotated, Any, Literal

import icontract
from pydantic import Field
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

	@icontract.require(lambda self: self.early_stop_epoch >= 1)
	@icontract.require(lambda self: min(to_2tuple(self.num_workers)) >= 0)
	@icontract.require(lambda self: self.max_norm > 0)
	@icontract.require(lambda self: self.accum_iter >= 1)
	@icontract.require(lambda self: 0 <= self.smoothing < 1)
	@icontract.require(lambda self: self.warmup_epochs >= 0)
	@icontract.require(lambda self: self.weight_decay >= 0)
	@icontract.require(lambda self: self.lr > 0)
	@icontract.require(lambda self: self.batch_size >= 1)
	@icontract.require(lambda self: self.epochs >= 1)
	def model_post_init(self, context: Any, /) -> None:
		pass


class ClassificationConfig(BaseTaskConfig):
	"""分类任务配置"""

	task_type: Annotated[Literal['classification'], tyro.conf.Suppress] = 'classification'  # type: ignore[assignment]
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: DatasetConfig
	model: ModelConfig = Field(default_factory=ModelConfig)
	training: TrainingConfig = Field(default_factory=TrainingConfig)
