"""checkpoint 评估任务配置"""

from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseOutputConfig, BaseTaskConfig
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

	task_type: Annotated[Literal['eval_checkpoint'], tyro.conf.Suppress] = 'eval_checkpoint'  # type: ignore[assignment]
	output: EvalCheckpointOutputConfig = Field(default_factory=EvalCheckpointOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: EvalCheckpointModelConfig = Field(default_factory=EvalCheckpointModelConfig)
	evaluation: EvalCheckpointRuntimeConfig = Field(default_factory=EvalCheckpointRuntimeConfig)
