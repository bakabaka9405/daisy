"""checkpoint 评估任务配置"""

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseMetaConfig, BaseOutputConfig, BaseTaskConfig
from ..classification.config import DatasetConfig


class EvalCheckpointModelConfig(BaseModel):
	"""评估模型配置"""

	family: Literal['classification', 'mae_finetune'] = 'classification'
	name: str = 'resnet34'
	pretrained: bool = False
	checkpoint: str = ''
	num_classes: int = 2
	global_pool: str = 'avg'
	drop_path: float = 0.1
	img_size: int = 224


class EvalCheckpointRuntimeConfig(BaseModel):
	"""评估运行配置"""

	split_name: str = 'val'
	batch_size: int = 16
	num_workers: int = 4
	transform: str = 'rectangle_val'
	input_size: int = 224
	seed: int | None = None


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
