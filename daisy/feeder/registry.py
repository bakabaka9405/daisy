"""feeder 注册表

feeder 是签名为 ``(params: P, *, seed: int | None) -> dict[str, DiskDataset]`` 的函数，
负责数据定位、组装与划分；P 为 ``extra='forbid'`` 的 pydantic 模型，从首参注解推导。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
from typing import Any, get_type_hints

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


def _resolve_params_cls(fn: FeederFn) -> type[BaseModel]:
	"""校验 feeder 签名并从首参注解推导参数模型"""
	qualname = getattr(fn, '__qualname__', repr(fn))
	parameters = list(inspect.signature(fn).parameters.values())

	if not parameters or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
		raise TypeError(f'Feeder {qualname} must take the params model as its first positional parameter')
	seed_param = inspect.signature(fn).parameters.get('seed')
	if seed_param is None or seed_param.kind is not inspect.Parameter.KEYWORD_ONLY:
		raise TypeError(f"Feeder {qualname} must declare keyword-only parameter 'seed'")

	hints = get_type_hints(fn)
	params_name = parameters[0].name
	if params_name not in hints:
		raise TypeError(f'Feeder {qualname} must annotate parameter {params_name!r} with a pydantic model')
	params_cls = hints[params_name]
	if not (isinstance(params_cls, type) and issubclass(params_cls, BaseModel)):
		raise TypeError(f'Feeder {qualname} params annotation must be a pydantic BaseModel subclass, got {params_cls!r}')
	if params_cls.model_config.get('extra') != 'forbid':
		raise TypeError(f"Feeder {qualname} params model {params_cls.__name__} must set model_config extra='forbid'")
	return params_cls


def register_feeder[F: FeederFn](name: str) -> Callable[[F], F]:
	"""注册 feeder 的装饰器

	Args:
		name: feeder 唯一标识符，对应 TOML 中的 dataset.feeder
	"""

	def decorator(fn: F) -> F:
		if name in FEEDERS:
			raise ValueError(f"Feeder '{name}' is already registered.")
		FEEDERS[name] = FeederEntry(name=name, params_cls=_resolve_params_cls(fn), fn=fn)
		return fn

	return decorator


def get_feeder(name: str) -> FeederEntry:
	"""根据标识符获取已注册的 feeder"""
	entry = FEEDERS.get(name)
	if entry is None:
		raise ValueError(f"Unknown feeder: '{name}'. Available feeders: {list(FEEDERS.keys())}")
	return entry
