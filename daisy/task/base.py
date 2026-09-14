"""Task 模块抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated
import torch

from pydantic import BaseModel, Field
import tyro.conf


class BaseMetaConfig(BaseModel):
	"""任务元信息配置"""

	title: str = ''
	description: str = ''
	creator: str = ''
	created_at: str = ''
	commit: str = 'auto'  # "auto" 表示运行时自动获取


class BaseOutputConfig(BaseModel):
	"""输出配置"""

	save_path: str = 'outputs/{task_id}'  # 支持 {task_id} 变量
	keep_count: int = 1
	save_best: bool = True
	log: bool = True


class BaseTaskConfig(BaseModel, ABC):
	"""所有任务配置的基类"""

	# task_type 由子类定义具体 Literal 类型
	task_type: str
	# task_type 和 output 由子类定义
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	# 任务文件路径（加载后自动设置）
	task_file: Annotated[Path | None, tyro.conf.Suppress] = None
	task_id: Annotated[str, tyro.conf.Suppress] = ''

	class Config:
		extra = 'forbid'

	@classmethod
	@abstractmethod
	def get_task_type(cls) -> str:
		"""返回任务类型标识符"""
		pass


class TaskRunner[T_Config: BaseTaskConfig](ABC):
	"""任务执行器基类"""

	@classmethod
	@abstractmethod
	def get_task_type(cls) -> str:
		"""返回支持的任务类型"""
		pass

	@classmethod
	@abstractmethod
	def get_config_class(cls) -> type[BaseTaskConfig]:
		"""返回配置类"""
		pass

	@abstractmethod
	def run(self, config: T_Config, device: torch.device) -> Path:
		"""执行任务，返回输出路径"""
		pass
