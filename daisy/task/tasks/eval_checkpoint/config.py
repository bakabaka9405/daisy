"""checkpoint 评估任务配置"""

from typing import Annotated, Literal

from pydantic import Field
import tyro.conf

from ...base import BaseConfig, BaseTaskConfig
from ...shared import DatasetConfig, InferenceModelConfig, InferenceRuntimeConfig


class EvalCheckpointOutputConfig(BaseConfig):
	"""评估输出配置"""

	save_path: str = 'outputs/{task_id}'


class EvalCheckpointConfig(BaseTaskConfig):
	"""checkpoint 评估任务配置"""

	task_type: Annotated[Literal['eval_checkpoint'], tyro.conf.Suppress] = 'eval_checkpoint'  # type: ignore[assignment]
	output: EvalCheckpointOutputConfig = Field(default_factory=EvalCheckpointOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: InferenceModelConfig = Field(default_factory=InferenceModelConfig)
	evaluation: InferenceRuntimeConfig = Field(default_factory=InferenceRuntimeConfig)
