"""分类任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig
from ...shared import DatasetConfig as DatasetConfig, DatasetSplitConfig as DatasetSplitConfig


class ModelConfig(BaseModel):
	"""模型配置"""

	name: str = 'resnet34'
	pretrained: bool = True
	num_classes: int = 2
	# 可选：加载预训练权重
	checkpoint: str | None = None


class TransformConfig(BaseModel):
	"""数据增强配置"""

	train: str = 'rectangle_train'
	val: str = 'rectangle_val'


class TrainingConfig(BaseModel):
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


class ClassificationConfig(BaseTaskConfig):
	"""分类任务配置"""

	task_type: Literal['classification'] = 'classification'  # type: ignore[assignment]
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: ModelConfig = Field(default_factory=ModelConfig)
	training: TrainingConfig = Field(default_factory=TrainingConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'classification'
