"""Task 模块基础配置"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
import tyro.conf


class BaseConfig(BaseModel):
	"""Daisy 任务配置基类"""

	model_config = ConfigDict(extra='forbid')


class BaseMetaConfig(BaseConfig):
	"""任务元信息配置"""

	title: str = ''
	description: str = ''
	creator: str = ''
	created_at: str = ''
	commit: str = 'auto'  # "auto" 表示运行时自动获取


class BaseOutputConfig(BaseConfig):
	"""输出配置"""

	save_path: str = 'outputs/{task_id}'  # 支持 {task_id} 变量
	keep_count: int = 1
	save_best: bool = True
	log: bool = True


class BaseTaskConfig(BaseConfig):
	"""所有任务配置的基类"""

	# task_type 由子类定义具体 Literal 类型
	task_type: Annotated[str, tyro.conf.Suppress] = ''
	# meta 由基类统一提供默认工厂
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	# 任务文件路径（加载后自动设置）
	task_file: Annotated[Path | None, tyro.conf.Suppress] = None
	task_id: Annotated[str, tyro.conf.Suppress] = ''
