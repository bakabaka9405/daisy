"""可解释性任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig
from ...shared import DatasetConfig, InferenceModelConfig


class InterpretabilityRuntimeConfig(BaseModel):
	"""可解释性运行配置"""

	split_name: str = 'val'
	max_samples: int | None = None
	target_class_mode: Literal['pred', 'true'] = 'pred'
	start_layer: int = 0
	input_size: int = 224
	seed: int | None = None


class InterpretabilityOutputConfig(BaseOutputConfig):
	"""可解释性输出配置"""

	save_path: str = 'outputs/{task_id}'
	log: bool = False


class InterpretabilityConfig(BaseTaskConfig):
	"""可解释性任务配置"""

	task_type: Literal['interpretability'] = 'interpretability'  # type: ignore[assignment]
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	output: InterpretabilityOutputConfig = Field(default_factory=InterpretabilityOutputConfig)
	dataset: DatasetConfig = Field(default_factory=DatasetConfig)
	model: InferenceModelConfig = Field(default_factory=InferenceModelConfig)
	runtime: InterpretabilityRuntimeConfig = Field(default_factory=InterpretabilityRuntimeConfig)

	@classmethod
	def get_task_type(cls) -> str:
		return 'interpretability'
