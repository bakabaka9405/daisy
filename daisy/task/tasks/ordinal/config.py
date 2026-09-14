"""有序分类任务配置"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseOutputConfig, BaseTaskConfig
from ...shared import DatasetConfig
from ..classification.config import TransformConfig

OrdinalMethodType = Literal[
	'ce',
	'label_smoothing_ce',
	'clm',
	'coral',
	'corn',
	'sord',
	'rps',
	'wasserstein',
	'distance_ce',
	'unimodal',
	'mse_ranknet',
	'mse_rank_hinge',
	'ordinal_contrastive',
]


class ModelConfig(BaseConfig):
	"""有序分类模型配置"""

	name: str = 'resnet34'
	pretrained: bool = True
	num_classes: int = Field(default=5, ge=2)
	method: OrdinalMethodType = 'ce'
	checkpoint: Path | None = None


class TrainingConfig(BaseConfig):
	"""训练配置"""

	seed: int | None = None
	epochs: int = 30
	batch_size: int = 64
	lr: float = 1e-4
	weight_decay: float = 0.05
	warmup_epochs: int = 0
	min_lr: float = 1e-6
	accum_iter: int = 1
	use_scheduler: bool = True
	use_amp: bool = True
	clip_grad: float | None = None
	num_workers: int | tuple[int, int] = 4
	transform: TransformConfig = Field(default_factory=TransformConfig)


class OrdinalConfig(BaseTaskConfig):
	"""有序分类任务配置"""

	task_type: Annotated[Literal['ordinal'], tyro.conf.Suppress] = 'ordinal'  # type: ignore[assignment]
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: ModelConfig = Field(default_factory=ModelConfig)
	training: TrainingConfig = Field(default_factory=TrainingConfig)
