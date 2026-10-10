"""已完成运行的读取：快照配置与任务提供的分析视图"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import icontract
import torch

from .base import BaseTaskConfig
from .protocol import TaskRun
from .tasks import get_task


def load_run_config(run_dir: Path) -> BaseTaskConfig:
	"""按快照中的 task_type 选择已注册任务的配置类，移除运行环境信息后校验 task_snapshot.json

	不导入快照记录的 imports（历史路径可能已失效），外部任务须由调用方预先导入注册。
	"""
	snapshot = json.loads((run_dir / 'task_snapshot.json').read_text(encoding='utf-8'))
	snapshot.pop('runtime', None)
	# 历史快照把源码版本写在 meta 中，当前由 provenance.json 单独记录
	snapshot.get('meta', {}).pop('commit', None)
	return get_task(snapshot['task_type']).config_type.model_validate(snapshot)


@icontract.ensure(lambda run_type, result: isinstance(result, run_type), 'Run directory belongs to a different task type')
def open_run[R: TaskRun](run_dir: Path, run_type: type[R], *, device: torch.device | None = None) -> R:
	"""打开已完成运行的分析视图，供离线统计分析复用配置、产物与模型

	Args:
		run_dir: 运行输出目录（含 task_snapshot.json）
		run_type: 期望的分析视图类型，由该任务的 open_run 决定
		device: 模型与数据加载设备，默认自动选择 CUDA/CPU
	"""
	if device is None:
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	config = load_run_config(run_dir)
	return cast(R, get_task(config.task_type).open_run(run_dir, config, device))
