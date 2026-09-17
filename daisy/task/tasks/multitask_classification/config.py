"""多任务分类任务配置"""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator
import tyro.conf

from daisy.multitask_classification import ClassificationMetric
from ...base import BaseConfig, BaseOutputConfig, BaseTaskConfig


class TaskHeadConfig(BaseConfig):
	"""单个任务的标注列与类别配置"""

	column: int
	num_classes: int = Field(ge=2)
	label_offset: int = 0


class MultiTaskDatasetConfig(BaseConfig):
	"""多任务数据集配置

	所有任务共用同一张标注表，各自读取不同的标注列；训练/验证/测试划分统一
	读取划分表的 ``split_column`` 列，按 ``train_value`` / ``val_value`` /
	``test_value`` 切分。
	"""

	root: Path = Path()
	annotation_sheet: Path | None = None
	annotation_sheet_name: int | str = 0
	have_header: bool = True
	split_sheet: Path | None = None
	split_sheet_name: int | str = 0
	split_column: int = 1
	train_value: int = 1
	val_value: int = 2
	test_value: int = 3
	tasks: list[TaskHeadConfig] = Field(default_factory=list, min_length=1)


class MultiTaskModelConfig(BaseConfig):
	"""多任务模型配置"""

	name: str = 'vit_base_patch16_224'
	pretrained: bool = True
	img_size: int | tuple[int, int] | None = None
	drop_path_rate: float = 0.0
	checkpoint_path: Path | None = None
	source_grid_size: tuple[int, int] | None = None
	global_pool: str | None = None
	compile_backend: str | None = None


class MultiTaskTransformConfig(BaseConfig):
	"""数据增强配置"""

	train: str = 'rectangle_train'
	val: str = 'rectangle_val'


class MultiTaskTrainingConfig(BaseConfig):
	"""多任务训练配置"""

	seed: int | None = None
	epochs: int = 50
	batch_size: int = 64
	blr: float | None = None  # 基础学习率，按 lr = blr * batch_size * accum_iter / 256 换算
	lr: float | None = None  # 实际学习率，与 blr 二选一
	weight_decay: float = 0.05
	layer_decay: float | None = None
	warmup_epochs: int = 5
	min_lr: float = 1e-6
	per_iteration: bool = False
	smoothing: float = 0.1
	accum_iter: int = 1
	use_scheduler: bool = True
	use_amp: bool = True
	clip_grad: bool = False
	max_norm: float = 1.0
	num_workers: int | tuple[int, int] = 6
	objective: Literal['ce', 'sord'] = 'ce'
	sord_temperature: float = Field(default=1.0, gt=0, allow_inf_nan=False)
	transform: MultiTaskTransformConfig = Field(default_factory=MultiTaskTransformConfig)

	@model_validator(mode='after')
	def _check_learning_rate(self) -> Self:
		"""blr 与 lr 互斥，二者至多设置一个。"""
		if self.blr is not None and self.lr is not None:
			raise ValueError('training.blr and training.lr are mutually exclusive')
		return self


class MultiTaskEvaluationConfig(BaseConfig):
	"""多任务评估配置"""

	val_metrics: list[ClassificationMetric] = Field(default_factory=lambda: ['f1', 'acc'])
	test_metrics: list[ClassificationMetric] = Field(default_factory=lambda: ['acc', 'precision', 'recall', 'f1', 'auroc'])
	average_metrics: list[ClassificationMetric] = Field(default_factory=lambda: ['f1'])
	watch_metric: str = 'avg_f1'
	watch_mode: Literal['min', 'max'] = 'max'

	@field_validator('val_metrics', 'test_metrics')
	@classmethod
	def _validate_nonempty_metrics(cls, v: list[ClassificationMetric]) -> list[ClassificationMetric]:
		if not v:
			raise ValueError('Metrics list must not be empty')
		if len(v) != len(set(v)):
			raise ValueError(f'Duplicate metrics found: {v}')
		return v

	@field_validator('average_metrics')
	@classmethod
	def _validate_average_metrics(cls, v: list[ClassificationMetric]) -> list[ClassificationMetric]:
		if len(v) != len(set(v)):
			raise ValueError(f'Duplicate metrics found: {v}')
		return v

	@model_validator(mode='after')
	def _validate_averages_subset(self) -> Self:
		val_set = set(self.val_metrics)
		avg_set = set(self.average_metrics)
		if not avg_set.issubset(val_set):
			raise ValueError(f'average_metrics ({self.average_metrics}) must be a subset of val_metrics ({self.val_metrics})')
		return self


class MultiTaskClassificationConfig(BaseTaskConfig):
	"""多任务分类任务配置"""

	task_type: Annotated[Literal['multitask_classification'], tyro.conf.Suppress] = 'multitask_classification'  # type: ignore[assignment]
	output: BaseOutputConfig = Field(default_factory=BaseOutputConfig)
	dataset: MultiTaskDatasetConfig = Field(default_factory=MultiTaskDatasetConfig)
	model: MultiTaskModelConfig = Field(default_factory=MultiTaskModelConfig)
	training: MultiTaskTrainingConfig = Field(default_factory=MultiTaskTrainingConfig)
	evaluation: MultiTaskEvaluationConfig = Field(default_factory=MultiTaskEvaluationConfig)

	@model_validator(mode='after')
	def _validate_watched_metric(self) -> Self:
		if not self.output.save_best:
			return self
		eval_cfg = self.evaluation
		valid_watch: set[str] = {f'avg_{m}' for m in eval_cfg.average_metrics}
		for task_idx in range(len(self.dataset.tasks)):
			for m in eval_cfg.val_metrics:
				valid_watch.add(f'task{task_idx + 1}_{m}')
		if eval_cfg.watch_metric not in valid_watch:
			raise ValueError(
				f"watch_metric '{eval_cfg.watch_metric}' is not available in validation metrics. Available metrics: {sorted(valid_watch)}"
			)
		return self
