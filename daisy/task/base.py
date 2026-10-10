"""Task 模块基础配置"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
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


class BaseOutputConfig(BaseConfig):
	"""输出配置"""

	save_path: str = 'outputs/{task_id}'  # 支持 {task_id}、{run_id}、{date} 及任意配置字段（如 {training.seed}）
	keep_count: int = 1
	save_best: bool = True
	log: bool = True

	@model_validator(mode='after')
	def validate_config(self) -> Self:
		if self.keep_count < 0:
			raise ValueError('keep_count must be >= 0')
		return self


class BaseTaskConfig(BaseConfig):
	"""所有任务配置的基类"""

	# task_type 由子类定义具体 Literal 类型
	task_type: Annotated[str, tyro.conf.Suppress] = ''
	# meta 由基类统一提供默认工厂
	meta: BaseMetaConfig = Field(default_factory=BaseMetaConfig)
	# 任务文件路径（加载后自动设置）
	task_file: Annotated[Path | None, tyro.conf.Suppress] = None
	task_id: Annotated[str, tyro.conf.Suppress] = ''
	# task 文件 imports 字段解析出的外部 Python 文件绝对路径（加载后自动设置）
	imports: Annotated[tuple[Path, ...], tyro.conf.Suppress] = ()
	# task 文件 run 数组中当前条目的 id、重载内容与是否默认运行（加载后自动设置；无 run 时分别为空字符串、None 与 True）
	run_id: Annotated[str, tyro.conf.Suppress] = ''
	run_overrides: Annotated[dict[str, Any] | None, tyro.conf.Suppress] = None
	# 只影响选择哪些 run，不进入快照；条目原文（含 default）已记录在 run_overrides 中
	run_default: Annotated[bool, tyro.conf.Suppress] = Field(default=True, exclude=True)
