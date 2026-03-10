"""任务类型模块

每个任务类型作为一个子模块实现，包含:
- config.py: 任务配置类
- runner.py: 任务执行器类

导入此模块会自动注册所有任务类型。
"""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules


_DISCOVERED_MODULES: set[str] = set()


def discover_tasks() -> list[str]:
	"""自动发现并导入所有任务模块"""
	discovered: list[str] = []
	for module_info in sorted(iter_modules(__path__), key=lambda item: item.name):
		module_name = module_info.name
		if not module_info.ispkg or module_name.startswith('_'):
			continue
		qualified_name = f'{__name__}.{module_name}'
		if qualified_name in _DISCOVERED_MODULES:
			discovered.append(module_name)
			continue
		import_module(qualified_name)
		_DISCOVERED_MODULES.add(qualified_name)
		discovered.append(module_name)
	return discovered


discover_tasks()


__all__ = ['discover_tasks']
