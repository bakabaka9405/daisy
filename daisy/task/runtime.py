"""Task 运行期辅助函数"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from daisy.util import save_json

from .base import BaseTaskConfig

if TYPE_CHECKING:
	from .protocol import Task


def format_output_path(save_path_template: str, config: BaseTaskConfig) -> Path:
	"""按配置字段渲染输出路径模板（不创建目录）"""
	values: dict[str, Any] = dict(config)
	values['date'] = datetime.now().strftime('%Y%m%d')
	return Path(save_path_template.format(**values))


def print_task_header(config: BaseTaskConfig, device: torch.device, output_path: Path) -> None:
	"""打印统一的任务头部信息"""
	print('=' * 60)
	print(f'Task: {config.meta.title or config.task_id}')
	if config.run_id:
		print(f'Run: {config.run_id}')
	print(f'Description: {config.meta.description}')
	print(f'Device: {device}')
	print('=' * 60)
	print(f'Output path: {output_path}')


def print_task_completed(output_path: Path) -> None:
	"""打印统一的任务完成信息"""
	print('\n' + '=' * 60)
	print('Task completed!')
	print(f'Output saved to: {output_path}')
	print('=' * 60)


def save_task_snapshot(output_dir: Path, task: Task[Any, Any], device: torch.device) -> None:
	"""保存任务快照 task_snapshot.json：task 为任务类的完整名称，runtime 记录运行设备，其余为任务配置"""
	snapshot = {
		'task': f'{type(task).__module__}.{type(task).__qualname__}',
		'runtime': {'device': str(device)},
		**task.config.model_dump(mode='json', exclude_none=True),
	}
	save_json(output_dir / 'task_snapshot.json', snapshot)
