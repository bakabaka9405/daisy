"""Task 模块抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path
import torch

from typing import Any, Callable, Generic, TypeVar

from pydantic import BaseModel, Field


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
	task_file: Path | None = None
	task_id: str = ''

	class Config:
		extra = 'forbid'

	@classmethod
	@abstractmethod
	def get_task_type(cls) -> str:
		"""返回任务类型标识符"""
		pass


# 泛型类型变量
T_Config = TypeVar('T_Config', bound=BaseTaskConfig)


class TaskRunner(ABC, Generic[T_Config]):
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

	# ========== UI 钩子方法 ==========

	@classmethod
	def get_ui_display_name(cls) -> str:
		"""UI 显示名称，默认返回 task_type"""
		return cls.get_task_type()

	@classmethod
	def get_ui_field_overrides(cls) -> dict[str, dict]:
		"""字段 UI 配置覆盖

		返回字典，键格式: 'section.field' 或 'section.subsection.field'

		示例:
			return {
				'training.lr': {'label': '学习率', 'component': 'number'},
				'dataset.split.val_ratio': {'component': 'slider', 'min_value': 0.05, 'max_value': 0.3},
			}
		"""
		return {}

	@classmethod
	def build_custom_ui(cls, gr_module: Any) -> tuple[list[Any], Callable[..., Any]] | None:
		"""完全自定义 UI

		如果返回 None，则使用自动生成的 UI。
		否则返回 (components, submit_callback):
			- components: Gradio 组件列表
			- submit_callback: 提交回调函数，接收所有组件值
		"""
		return None
