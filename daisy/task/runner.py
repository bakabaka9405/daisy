"""任务执行引擎"""

from pathlib import Path

from pydantic import BaseModel, ValidationError
import tomli
import torch

from .base import BaseTaskConfig
from .registry import TaskRegistry

from . import tasks


LEGACY_DEFAULT_TASK_TYPE = 'classification'


def _get_nested_model_class(annotation: object) -> type[BaseModel] | None:
	if isinstance(annotation, type) and issubclass(annotation, BaseModel):
		return annotation
	return None


def _collect_unexpected_fields(data: dict, model_cls: type[BaseModel], *, prefix: str = '') -> list[str]:
	unexpected_fields: list[str] = []
	for key, value in data.items():
		field_info = model_cls.model_fields.get(key)
		if field_info is None:
			unexpected_fields.append(f'{prefix}{key}')
			continue
		nested_model_class = _get_nested_model_class(field_info.annotation)
		if nested_model_class is None or not isinstance(value, dict):
			continue
		unexpected_fields.extend(
			_collect_unexpected_fields(
				value,
				nested_model_class,
				prefix=f'{prefix}{key}.',
			)
		)
	return unexpected_fields


def _resolve_task_type(data: dict, path: Path) -> tuple[str, dict]:
	"""解析任务类型，并兼容缺省的旧分类配置"""
	task_type = data.get('task_type')
	if task_type is not None:
		if not isinstance(task_type, str) or not task_type:
			raise ValueError(f'Invalid task_type in task file: {path}')
		return task_type, data

	classification_config = TaskRegistry.get_config_class(LEGACY_DEFAULT_TASK_TYPE)
	legacy_data = {**data, 'task_type': LEGACY_DEFAULT_TASK_TYPE}
	unexpected_fields = _collect_unexpected_fields(legacy_data, classification_config)
	if unexpected_fields:
		unexpected_preview = ', '.join(unexpected_fields[:3])
		raise ValueError(
			f'Missing task_type in task file: {path}. ' f'Found non-classification fields: {unexpected_preview}. ' 'Please add an explicit task_type.'
		)
	try:
		classification_config.model_validate(legacy_data)
	except ValidationError as exc:
		raise ValueError(f'Missing task_type in task file: {path}. ' 'Please add an explicit task_type for non-legacy task files.') from exc
	return LEGACY_DEFAULT_TASK_TYPE, legacy_data


def load_config(path: str | Path) -> BaseTaskConfig:
	"""从 TOML 文件加载任务配置

	根据 TOML 中的 task_type 字段自动选择正确的配置类。
	若无 task_type 字段，默认为 "classification" 以保持向后兼容。

	Args:
		path: TOML 文件路径

	Returns:
		对应任务类型的配置对象
	"""
	tasks.discover_tasks()
	path = Path(path)

	with open(path, 'rb') as f:
		data = tomli.load(f)

	# 解析任务类型，仅兼容缺省的旧 classification 配置
	task_type, resolved_data = _resolve_task_type(data, path)

	# 获取对应的配置类
	config_cls = TaskRegistry.get_config_class(task_type)

	# 解析配置
	config = config_cls.model_validate(resolved_data)
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
	tasks.discover_tasks()
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
