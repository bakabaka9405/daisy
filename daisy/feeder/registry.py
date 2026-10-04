"""feeder 注册表

feeder 是签名为 ``(params: P, *, seed: int | None) -> dict[str, DiskDataset]`` 的函数，
负责数据定位、组装与划分；P 为 ``extra='forbid'`` 的 pydantic 模型，从首参注解推导。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Any, get_type_hints

import icontract
from pydantic import BaseModel

from daisy.dataset.disk_dataset import DiskDataset

type FeederSplits = dict[str, DiskDataset[Any, Any]]
type FeederFn = Callable[..., FeederSplits]


@dataclass(slots=True, frozen=True)
class FeederEntry:
	"""已注册的 feeder"""

	name: str
	params_cls: type[BaseModel]
	fn: FeederFn


FEEDERS: dict[str, FeederEntry] = {}


def _first_param_positional(fn: FeederFn) -> bool:
	parameters = list(inspect.signature(fn).parameters.values())
	return len(parameters) > 0 and parameters[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _seed_keyword_only(fn: FeederFn) -> bool:
	seed_param = inspect.signature(fn).parameters.get('seed')
	return seed_param is not None and seed_param.kind is inspect.Parameter.KEYWORD_ONLY


def _first_param_name(fn: FeederFn) -> str:
	return next(iter(inspect.signature(fn).parameters))


def _params_annotation(fn: FeederFn) -> Any:
	return get_type_hints(fn)[_first_param_name(fn)]


@icontract.require(
	lambda fn: _params_annotation(fn).model_config.get('extra') == 'forbid', "Feeder params model must set model_config extra='forbid'"
)
@icontract.require(
	lambda fn: isinstance(_params_annotation(fn), type) and issubclass(_params_annotation(fn), BaseModel),
	'Feeder params annotation must be a pydantic BaseModel subclass',
)
@icontract.require(lambda fn: _first_param_name(fn) in get_type_hints(fn), 'Feeder must annotate its params parameter with a pydantic model')
@icontract.require(lambda fn: _seed_keyword_only(fn), "Feeder must declare keyword-only parameter 'seed'")
@icontract.require(lambda fn: _first_param_positional(fn), 'Feeder must take the params model as its first positional parameter')
def _resolve_params_cls(fn: FeederFn) -> type[BaseModel]:
	"""从 feeder 首参注解推导参数模型"""
	return _params_annotation(fn)


def register_feeder[F: FeederFn](name: str) -> Callable[[F], F]:
	"""注册 feeder 的装饰器

	Args:
		name: feeder 唯一标识符，对应 TOML 中的 dataset.feeder
	"""

	@icontract.require(lambda: name not in FEEDERS, 'Feeder is already registered')
	def decorator(fn: F) -> F:
		FEEDERS[name] = FeederEntry(name=name, params_cls=_resolve_params_cls(fn), fn=fn)
		return fn

	return decorator


@icontract.require(lambda name: name in FEEDERS, 'Unknown feeder')
def get_feeder(name: str) -> FeederEntry:
	"""根据标识符获取已注册的 feeder"""
	return FEEDERS[name]
