"""任务执行引擎"""

from pathlib import Path

import tomli
import torch

from .base import BaseTaskConfig
from .registry import TaskRegistry

# 导入任务模块以触发注册
from . import tasks  # noqa: F401


def load_config(path: str | Path) -> BaseTaskConfig:
	"""从 TOML 文件加载任务配置

	根据 TOML 中的 task_type 字段自动选择正确的配置类。
	若无 task_type 字段，默认为 "classification" 以保持向后兼容。

	Args:
		path: TOML 文件路径

	Returns:
		对应任务类型的配置对象
	"""
	path = Path(path)

	with open(path, 'rb') as f:
		data = tomli.load(f)

	# 获取任务类型，默认为 classification 以保持向后兼容
	task_type = data.get('task_type', 'classification')

	# 获取对应的配置类
	config_cls = TaskRegistry.get_config_class(task_type)

	# 解析配置
	config = config_cls.model_validate(data)
	config.task_file = path
	config.task_id = path.stem  # 使用文件名（不含扩展名）作为 task_id

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
	# 加载配置
	if isinstance(config, (str, Path)):
		config = load_config(config)

	# 设置设备
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	elif isinstance(device, str):
		device = torch.device(device)

	# 获取并实例化执行器
	runner_cls = TaskRegistry.get_runner(config.task_type)
	runner = runner_cls()

	# 运行任务
	return runner.run(config, device)
