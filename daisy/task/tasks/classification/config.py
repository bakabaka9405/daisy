"""分类任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig


class DatasetSplitConfig(BaseModel):
	"""数据集划分配置"""
	method: Literal['ratio', 'sheet', 'preset'] = 'ratio'
	val_ratio: float = 0.1
	# 用于 sheet 方法
	val_sheet: str | None = None
	val_sheet_name: str | None = None
	# 用于 preset 方法（预先划分好的数据）
	train_files: list[str] = Field(default_factory=list)
	val_files: list[str] = Field(default_factory=list)


class DatasetConfig(BaseModel):
	"""数据集配置"""
	type: Literal['sheet', 'folder'] = 'sheet'
	root: str = ''
	# sheet 类型的配置
	sheet: str | None = None
	sheet_name: str | None = None
	column: int = 1
	label_offset: int = 0
	have_header: bool = True
	# folder 类型的配置
	# (folder 类型直接用 root 作为数据根目录)
	# 数据划分配置
	split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)


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

	task_type: Literal['classification'] = 'classification'
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: ModelConfig = Field(default_factory=ModelConfig)
	training: TrainingConfig = Field(default_factory=TrainingConfig)

	class Config:
		extra = 'forbid'

	@classmethod
	def get_task_type(cls) -> str:
		return 'classification'
