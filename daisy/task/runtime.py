"""Task 运行期辅助函数"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from .base import BaseTaskConfig


def resolve_output_path(save_path_template: str, task_id: str) -> Path:
	"""解析并创建输出目录"""
	output_path = Path(
		save_path_template.format(
			task_id=task_id,
			date=datetime.now().strftime('%Y%m%d'),
		)
	)
	output_path.mkdir(parents=True, exist_ok=True)
	return output_path


def _make_json_safe(value: Any) -> Any:
	if isinstance(value, Path):
		return str(value)
	if isinstance(value, dict):
		return {str(k): _make_json_safe(v) for k, v in value.items()}
	if isinstance(value, (list, tuple, set)):
		return [_make_json_safe(v) for v in value]
	return value


def save_json(path: str | Path, data: Any) -> None:
	"""保存 JSON 文件"""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(_make_json_safe(data), f, ensure_ascii=False, indent=2)


def save_task_snapshot(
	output_path: str | Path,
	config: BaseTaskConfig,
	*,
	extra: dict[str, Any] | None = None,
	filename: str = 'task_snapshot.json',
) -> None:
	"""保存 task 配置快照"""
	output_path = Path(output_path)
	snapshot = config.model_dump(exclude={'task_file'}, exclude_none=True)
	snapshot['task_id'] = config.task_id
	if config.task_file is not None:
		snapshot['task_file'] = str(config.task_file)
	if extra:
		snapshot['runtime'] = extra
	save_json(output_path / filename, snapshot)
