"""checkpoint 评估任务配置"""

from typing import Literal

from pydantic import Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig
from ...shared import DatasetConfig, InferenceModelConfig, InferenceRuntimeConfig


class EvalCheckpointModelConfig(InferenceModelConfig):
	"""评估模型配置"""


class EvalCheckpointRuntimeConfig(InferenceRuntimeConfig):
	"""评估运行配置"""


class EvalCheckpointOutputConfig(BaseOutputConfig):
	"""评估输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = False


class EvalCheckpointConfig(BaseTaskConfig):
	"""checkpoint 评估任务配置"""

	task_type: Literal['eval_checkpoint'] = 'eval_checkpoint'  # type: ignore[assignment]
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: EvalCheckpointOutputConfig = Field(default_factory=EvalCheckpointOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: EvalCheckpointModelConfig = Field(default_factory=EvalCheckpointModelConfig)
	evaluation: EvalCheckpointRuntimeConfig = Field(default_factory=EvalCheckpointRuntimeConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'eval_checkpoint'
