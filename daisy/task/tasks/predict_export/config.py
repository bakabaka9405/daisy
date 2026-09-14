"""预测导出任务配置"""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
import tyro.conf

from ...base import BaseConfig, BaseTaskConfig
from ...shared import DatasetConfig, InferenceModelConfig, InferenceRuntimeConfig


class PredictExportRuntimeConfig(InferenceRuntimeConfig):
	"""预测导出运行配置"""

	include_logits: bool = True


class PredictExportOutputConfig(BaseConfig):
	"""预测导出输出配置"""

	save_path: str = 'outputs/{task_id}'
	filename: str = 'predictions.csv'


class PredictExportConfig(BaseTaskConfig):
	"""预测导出任务配置"""

	task_type: Annotated[Literal['predict_export'], tyro.conf.Suppress] = 'predict_export'  # type: ignore[assignment]
	output: PredictExportOutputConfig = Field(default_factory=PredictExportOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: InferenceModelConfig = Field(default_factory=InferenceModelConfig)
	prediction: PredictExportRuntimeConfig = Field(default_factory=PredictExportRuntimeConfig)

	@model_validator(mode='after')
	def validate_ordinal_logits(self) -> Self:
		if self.model.family == 'ordinal' and self.prediction.include_logits:
			raise ValueError('include_logits=True is unsupported for ordinal models; set include_logits=False')
		return self
