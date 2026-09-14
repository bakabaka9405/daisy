"""预测导出任务配置"""

from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseOutputConfig, BaseTaskConfig
from ...shared import DatasetConfig, InferenceModelConfig, InferenceRuntimeConfig


class PredictExportRuntimeConfig(InferenceRuntimeConfig):
	"""预测导出运行配置"""

	include_logits: bool = True


class PredictExportOutputConfig(BaseOutputConfig):
	"""预测导出输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = False
	filename: str = 'predictions.csv'


class PredictExportConfig(BaseTaskConfig):
	"""预测导出任务配置"""

	task_type: Annotated[Literal['predict_export'], tyro.conf.Suppress] = 'predict_export'  # type: ignore[assignment]
	output: PredictExportOutputConfig = Field(default_factory=PredictExportOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: InferenceModelConfig = Field(default_factory=InferenceModelConfig)
	prediction: PredictExportRuntimeConfig = Field(default_factory=PredictExportRuntimeConfig)
