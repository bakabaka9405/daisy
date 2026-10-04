"""task 文件 imports 字段对应的外部 Python 文件导入"""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from pathlib import Path
import sys


def import_files(paths: Iterable[Path]) -> None:
	"""以文件名为顶层模块名导入外部 Python 文件，触发其中的 task / feeder 注册

	文件所在目录会加入 sys.path，使同目录模块互相导入可用，
	spawn 子进程继承 sys.path 后也能按模块名反序列化外部对象。
	"""
	for raw_path in paths:
		path = raw_path.resolve()
		if path.suffix != '.py':
			raise ValueError(f'Import file must be a .py file: {path}')
		name = path.stem
		if not name.isidentifier():
			raise ValueError(f'Import file name {path.name!r} is not a valid module name')

		# 同一文件重复导入直接跳过，同名不同文件视为冲突
		loaded = sys.modules.get(name)
		if loaded is not None:
			loaded_file = getattr(loaded, '__file__', None)
			if loaded_file is None or Path(loaded_file).resolve() != path:
				raise ImportError(f"Cannot import {path}: module name '{name}' is already bound to {loaded_file}")
			continue

		parent = str(path.parent)
		if parent not in sys.path:
			sys.path.insert(0, parent)
		module = importlib.import_module(name)
		module_file = getattr(module, '__file__', None)
		if module_file is None or Path(module_file).resolve() != path:
			raise ImportError(f"Cannot import {path}: module name '{name}' resolved to {module_file}")
