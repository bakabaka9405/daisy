"""feeder 数据集配置"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, SerializeAsAny, model_validator
import tyro.conf

from daisy.feeder import get_feeder
from ..base import BaseConfig


class FeederConfig(BaseConfig):
	"""数据集配置：指定已注册 feeder 及其参数"""

	feeder: str
	params: SerializeAsAny[BaseModel]

	@model_validator(mode='before')
	@classmethod
	def _validate_params(cls, data: Any) -> Any:
		"""按 feeder 标识符查找参数模型并校验 params"""
		if not isinstance(data, dict) or not isinstance(data.get('feeder'), str):
			return data
		params_cls = get_feeder(data['feeder']).params_cls
		params = data.get('params', {})
		if not isinstance(params, params_cls):
			params = params_cls.model_validate(params)
		return {**data, 'params': params}


# 数据协议只由 task 文件决定，不参与 CLI 覆盖
DatasetConfig = Annotated[FeederConfig, tyro.conf.Suppress]
