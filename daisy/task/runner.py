"""任务执行引擎"""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Any

from deepmerge import Merger
import torch

from .base import BaseTaskConfig
from .tasks import get_task

_merger = Merger([(dict, ['merge'])], ['override'], ['override'])


def _load_raw_config(path: Path, visited: tuple[Path, ...] = ()) -> dict[str, Any]:
	"""递归加载 TOML 文件及其继承链，返回合并后的原始字典。

	Args:
		path: 当前 TOML 文件路径
		visited: 继承链中已访问祖先路径元组，用于检测继承循环

	Returns:
		合并后的原始配置字典
	"""
	# 第一阶段：循环继承检测
	resolved_path = path.resolve()
	if resolved_path in visited:
		chain = ' -> '.join(str(p) for p in (*visited, resolved_path))
		raise ValueError(f'Cyclic inheritance detected: {chain}')

	# 第二阶段：读取当前文件数据
	with open(path, 'rb') as f:
		data = tomllib.load(f)

	# 第三阶段：解析 extends 字段并递归加载父级配置
	extends = data.pop('extends', None)
	if extends is not None:
		if not isinstance(extends, str) or not extends:
			raise ValueError(f"Invalid 'extends' in task file: {path}. Expected non-empty string.")

		parent_path = (path.parent / extends).resolve()
		parent_data = _load_raw_config(parent_path, (*visited, resolved_path))
		return _merger.merge(parent_data, data)

	return data


def load_config(path: str | Path) -> BaseTaskConfig:
	"""从 TOML 文件加载任务配置

	根据 TOML 中的 task_type 字段自动选择正确的配置类。
	支持通过 top-level extends 字段继承父级 TOML 配置。

	Args:
		path: TOML 文件路径

	Returns:
		对应任务类型的配置对象
	"""
	path = Path(path)

	# 第一阶段：递归加载并合并配置
	data = _load_raw_config(path)

	# 第二阶段：确定任务类型并获取配置类
	task_type = data.get('task_type')
	if task_type is None:
		raise ValueError(f"Missing 'task_type' in task file: {path}. Please specify a task type.")
	if not isinstance(task_type, str) or not task_type:
		raise ValueError(f"Invalid 'task_type' in task file: {path}. Expected non-empty string.")

	# 第三阶段：校验模型并设置运行时元信息
	config_cls, _ = get_task(task_type)
	config = config_cls.model_validate(data)

	config.task_file = path
	config.task_id = path.stem

	return config


def run_task(
	config: BaseTaskConfig | str | Path,
	device: torch.device | str | None = None,
) -> Path:
	"""运行任务

	Args:
		config: BaseTaskConfig 对象或 TOML 文件路径
		device: 设备，默认自动选择 CUDA/CPU

	Returns:
		输出路径
	"""
	if isinstance(config, (str, Path)):
		config = load_config(config)

	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	_, run_fn = get_task(config.task_type)
	return run_fn(config, device)
