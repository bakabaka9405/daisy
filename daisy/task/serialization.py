"""Task 配置序列化"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import types
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
import tomli_w

from .base import BaseTaskConfig
from .registry import TaskRegistry


def _default_example_string(field_name: str) -> str:
	placeholders = {
		'root': '/path/to/data',
		'sheet': '/path/to/labels.xlsx',
		'sheet_name': 'Sheet1',
		'val_sheet': '/path/to/val_labels.xlsx',
		'val_sheet_name': 'Sheet1',
		'checkpoint': '/path/to/checkpoint.pth',
		'pretrained_checkpoint': '/path/to/checkpoint.pth',
		'creator': 'your_name',
		'created_at': date.today().isoformat(),
		'commit': 'auto',
	}
	return placeholders.get(field_name, '')


def _build_example_value(field_name: str, annotation: Any) -> Any:
	origin = get_origin(annotation)
	args = get_args(annotation)

	if origin is Literal:
		return args[0] if args else _default_example_string(field_name)

	if origin is types.UnionType or (origin is not None and getattr(origin, '__name__', None) == 'Union'):
		non_none_args = [arg for arg in args if arg is not type(None)]
		if non_none_args:
			return _build_example_value(field_name, non_none_args[0])
		return None

	if origin is list:
		return []

	if origin is dict:
		return {}

	if origin is tuple:
		return []

	if annotation is str:
		return _default_example_string(field_name)

	if annotation is bool:
		return False

	if annotation is int:
		return 0

	if annotation is float:
		return 0.0

	if isinstance(annotation, type) and issubclass(annotation, BaseModel):
		return _build_example_data(annotation)

	return _default_example_string(field_name)


def _get_field_example(field_name: str, field_info: Any) -> Any:
	if field_info.default_factory is not None:
		return field_info.default_factory()
	if not field_info.is_required():
		return deepcopy(field_info.default)
	return _build_example_value(field_name, field_info.annotation)


def _build_example_data(model_cls: type[BaseModel]) -> dict[str, Any]:
	data: dict[str, Any] = {}
	for field_name, field_info in model_cls.model_fields.items():
		if field_name in {'task_file', 'task_id'}:
			continue
		data[field_name] = _get_field_example(field_name, field_info)
	return data


def build_example_config(task_type: str) -> BaseTaskConfig:
	"""根据注册的配置类构建示例任务配置"""
	from . import tasks

	tasks.discover_tasks()
	config_cls = TaskRegistry.get_config_class(task_type)
	runner_cls = TaskRegistry.get_runner(task_type)
	config = config_cls.model_validate(_build_example_data(config_cls))

	display_name = runner_cls.get_ui_display_name()
	if not config.meta.title:
		config.meta.title = f'{display_name} 示例任务'
	if not config.meta.description:
		config.meta.description = f'根据 {task_type} 的注册配置自动生成'
	if not config.meta.creator:
		config.meta.creator = 'your_name'
	if not config.meta.created_at:
		config.meta.created_at = date.today().isoformat()
	if not config.meta.commit:
		config.meta.commit = 'auto'

	return config


def _get_default_example_path(task_type: str) -> Path:
	return Path('tasks') / f'example_{task_type}.toml'


def _get_available_output_path(path: Path) -> Path:
	if not path.exists():
		return path

	index = 1
	while True:
		candidate = path.with_name(f'{path.stem}_{index}{path.suffix}')
		if not candidate.exists():
			return candidate
		index += 1


def export_example_config(task_type: str, path: str | Path | None = None, *, force: bool = False) -> Path:
	"""导出某个任务类型的示例配置文件"""
	config = build_example_config(task_type)

	if path is None:
		default_path = _get_default_example_path(task_type)
		output_path = default_path if force else _get_available_output_path(default_path)
	else:
		output_path = Path(path)

	if output_path.exists() and not force:
		raise FileExistsError(f'Output file already exists: {output_path}')

	output_path.parent.mkdir(parents=True, exist_ok=True)
	save_config(config, output_path)
	return output_path


def save_config(config: BaseTaskConfig, path: str | Path) -> None:
	"""保存任务配置到 TOML 文件"""
	path = Path(path)
	data = config.model_dump(exclude={'task_file', 'task_id'}, exclude_none=True)

	with open(path, 'wb') as f:
		tomli_w.dump(data, f)
