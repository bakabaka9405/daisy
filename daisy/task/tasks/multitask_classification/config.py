"""多任务分类任务配置"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseOutputConfig, BaseTaskConfig


class TaskHeadConfig(BaseConfig):
	"""单个任务的标注列与类别配置"""

	column: int
	num_classes: int
	label_offset: int = 0


class MultiTaskDatasetConfig(BaseConfig):
	"""多任务数据集配置

	所有任务共用同一张标注表，各自读取不同的标注列；训练/验证/测试划分统一
	读取划分表的 ``split_column`` 列，按 ``train_value`` / ``val_value`` /
	``test_value`` 切分。
	"""

	root: Path = Path()
	annotation_sheet: Path | None = None
	annotation_sheet_name: int | str | None = None
	have_header: bool = True
	split_sheet: Path | None = None
	split_sheet_name: int | str | None = None
	split_column: int = 1
	train_value: int = 1
	val_value: int = 2
	test_value: int = 3
	tasks: list[TaskHeadConfig] = Field(default_factory=list)


class MultiTaskModelConfig(BaseConfig):
	"""多任务模型配置"""

	name: str = 'vit_base_patch16_224'
	pretrained: bool = True
	img_size: int | tuple[int, int] | None = None
	drop_path_rate: float = 0.0


class MultiTaskTransformConfig(BaseConfig):
	"""数据增强配置"""

	train: str = 'rectangle_train'
	val: str = 'rectangle_val'


class MultiTaskTrainingConfig(BaseConfig):
	"""多任务训练配置"""

	seed: int | None = None
	epochs: int = 50
	batch_size: int = 64
	lr: float = 1e-3
	weight_decay: float = 0.05
	warmup_epochs: int = 5
	smoothing: float = 0.1
	accum_iter: int = 1
	use_scheduler: bool = True
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	num_workers: int = 6
	transform: MultiTaskTransformConfig = Field(default_factory=MultiTaskTransformConfig)


class MultiTaskClassificationConfig(BaseTaskConfig):
	"""多任务分类任务配置"""

	task_type: Annotated[Literal['multitask_classification'], tyro.conf.Suppress] = 'multitask_classification'  # type: ignore[assignment]
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: MultiTaskDatasetConfig = Field(default_factory=MultiTaskDatasetConfig)
	model: MultiTaskModelConfig = Field(default_factory=MultiTaskModelConfig)
	training: MultiTaskTrainingConfig = Field(default_factory=MultiTaskTrainingConfig)
