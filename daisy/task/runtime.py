"""Task 运行期辅助函数"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from daisy.util import save_json

from .base import BaseTaskConfig


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


def save_task_snapshot(
	output_path: Path,
	config: BaseTaskConfig,
	*,
	extra: dict[str, Any] | None = None,
	filename: str = 'task_snapshot.json',
) -> None:
	"""保存 task 配置快照"""
	# run_default 只影响选择哪些 run，条目原文（含 default）已记录在 run_overrides 中
	snapshot = config.model_dump(mode='json', exclude={'task_file', 'run_default'}, exclude_none=True)
	snapshot['task_id'] = config.task_id
	if not config.run_id:
		snapshot.pop('run_id')
	if config.task_file is not None:
		snapshot['task_file'] = str(config.task_file)
	if extra:
		snapshot['runtime'] = extra
	save_json(output_path / filename, snapshot)
