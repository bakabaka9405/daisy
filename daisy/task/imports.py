"""task 文件 imports 字段对应的外部 Python 文件导入"""

from __future__ import annotations

from collections.abc import Iterable
import importlib
from pathlib import Path
import sys


def import_files(paths: Iterable[Path]) -> None:
	"""按常规包名或独立文件名导入外部 Python 文件，触发 task / feeder 注册

	包的导入根目录或独立文件所在目录会加入 sys.path，
	spawn 子进程继承 sys.path 后也能按模块名反序列化外部对象。
	"""
	for raw_path in paths:
		path = raw_path.resolve()
		if path.suffix != '.py':
			raise ValueError(f'Import file must be a .py file: {path}')
		# 连续的常规包祖先决定模块全名及导入根目录
		packages: list[Path] = []
		root = path.parent
		while (root / '__init__.py').is_file():
			packages.insert(0, root)
			root = root.parent
		components = [package.name for package in packages]
		if path.name != '__init__.py' or not packages:
			components.append(path.stem)
		if not all(component.isidentifier() for component in components):
			raise ValueError(f'Import path {path} contains an invalid module name')
		name = '.'.join(components)

		# 先核对已加载的包，避免错误的缓存祖先执行其他来源的叶模块
		for index, package in enumerate(packages, start=1):
			package_name = '.'.join(components[:index])
			loaded_package = sys.modules.get(package_name)
			if loaded_package is not None:
				package_file = getattr(loaded_package, '__file__', None)
				if package_file is None or Path(package_file).resolve() != (package / '__init__.py').resolve():
					raise ImportError(
						f"Cannot import {path}: package name '{package_name}' is already bound to {package_file}"
					)

		# 同一文件重复导入直接跳过，同名不同文件视为冲突
		loaded = sys.modules.get(name)
		if loaded is not None:
			loaded_file = getattr(loaded, '__file__', None)
			if loaded_file is None or Path(loaded_file).resolve() != path:
				raise ImportError(f"Cannot import {path}: module name '{name}' is already bound to {loaded_file}")
			continue

		import_root = str(root)
		if import_root not in sys.path:
			sys.path.insert(0, import_root)
		module = importlib.import_module(name)
		module_file = getattr(module, '__file__', None)
		if module_file is None or Path(module_file).resolve() != path:
			raise ImportError(f"Cannot import {path}: module name '{name}' resolved to {module_file}")
