"""Task 配置序列化"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import icontract
from pydantic import BaseModel
import tomli_w

from .base import BaseTaskConfig
from .shared import FeederConfig
from .tasks import get_task


class _PlaceholderParams(BaseModel):
	"""示例配置中的占位 feeder 参数"""


@icontract.ensure(lambda task_type, result: result.task_type == task_type)
def build_example_config(task_type: str) -> BaseTaskConfig:
	"""根据任务类型构建示例任务配置"""
	config_cls = get_task(task_type).config_type

	# 必填 dataset 无通用默认值，以跳过校验的占位 feeder 代替
	dataset_field = config_cls.model_fields.get('dataset')
	if dataset_field is not None and dataset_field.is_required():
		placeholder = FeederConfig.model_construct(feeder='<feeder_id>', params=_PlaceholderParams())
		config = config_cls.model_construct(dataset=placeholder)
	else:
		config = config_cls()

	if not config.meta.title:
		config.meta.title = f'{task_type} 示例任务'
	if not config.meta.description:
		config.meta.description = f'根据 {task_type} 的配置自动生成'
	if not config.meta.creator:
		config.meta.creator = 'your_name'
	if not config.meta.created_at:
		config.meta.created_at = date.today().isoformat()

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


def export_example_config(task_type: str, path: Path | None = None, *, force: bool = False) -> Path:
	"""导出某个任务类型的示例配置文件"""
	config = build_example_config(task_type)

	if path is None:
		default_path = _get_default_example_path(task_type)
		output_path = default_path if force else _get_available_output_path(default_path)
	else:
		output_path = path
		if not force and output_path.exists():
			raise FileExistsError(f'Output file already exists: {output_path}')

	output_path.parent.mkdir(parents=True, exist_ok=True)
	save_config(config, output_path)
	return output_path


def save_config(config: BaseTaskConfig, path: Path) -> None:
	"""保存任务配置到 TOML 文件"""
	data = config.model_dump(mode='json', exclude={'task_file', 'task_id', 'imports', 'run_id', 'run_overrides', 'run_default'}, exclude_none=True)

	with open(path, 'wb') as f:
		tomli_w.dump(data, f)
