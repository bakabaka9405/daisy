"""Task 配置序列化"""

from __future__ import annotations

from pathlib import Path

import tomli_w

from .base import BaseTaskConfig


def save_config(config: BaseTaskConfig, path: str | Path) -> None:
	"""保存任务配置到 TOML 文件"""
	path = Path(path)
	data = config.model_dump(exclude={'task_file', 'task_id'}, exclude_none=True)

	with open(path, 'wb') as f:
		tomli_w.dump(data, f)
