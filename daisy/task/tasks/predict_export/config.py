"""预测导出任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig
from ..classification.config import DatasetConfig
from ..eval_checkpoint.config import EvalCheckpointModelConfig


class PredictExportRuntimeConfig(BaseModel):
	"""预测导出运行配置"""

	split_name: str = 'val'
	batch_size: int = 16
	num_workers: int = 4
	transform: str = 'rectangle_val'
	input_size: int = 224
	seed: int | None = None
	include_logits: bool = True


class PredictExportOutputConfig(BaseOutputConfig):
	"""预测导出输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = False
	filename: str = 'predictions.csv'


class PredictExportConfig(BaseTaskConfig):
	"""预测导出任务配置"""

	task_type: Literal['predict_export'] = 'predict_export'  # type: ignore[assignment]
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: PredictExportOutputConfig = Field(default_factory=PredictExportOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: EvalCheckpointModelConfig = Field(default_factory=EvalCheckpointModelConfig)
	prediction: PredictExportRuntimeConfig = Field(default_factory=PredictExportRuntimeConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'predict_export'
