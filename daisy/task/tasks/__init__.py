"""任务类型模块与任务注册表。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from ..base import BaseTaskConfig
from .classification.config import ClassificationConfig
from .classification.runner import run as run_classification
from .eval_checkpoint.config import EvalCheckpointConfig
from .eval_checkpoint.runner import run as run_eval_checkpoint
from .mae_finetune.config import MAEFinetuneConfig
from .mae_finetune.runner import run as run_mae_finetune
from .mae_pretrain.config import MAEPretrainConfig
from .mae_pretrain.runner import run as run_mae_pretrain
from .moco_lincls.config import MoCoLinclsConfig
from .moco_lincls.runner import run as run_moco_lincls
from .moco_pretrain.config import MoCoPretrainConfig
from .moco_pretrain.runner import run as run_moco_pretrain
from .ordinal.config import OrdinalConfig
from .ordinal.runner import run as run_ordinal
from .predict_export.config import PredictExportConfig
from .predict_export.runner import run as run_predict_export

type TaskEntry = tuple[type[BaseTaskConfig], Callable[[Any, torch.device], Path]]

TASKS: dict[str, TaskEntry] = {
	'classification': (ClassificationConfig, run_classification),
	'ordinal': (OrdinalConfig, run_ordinal),
	'mae_pretrain': (MAEPretrainConfig, run_mae_pretrain),
	'mae_finetune': (MAEFinetuneConfig, run_mae_finetune),
	'moco_pretrain': (MoCoPretrainConfig, run_moco_pretrain),
	'moco_lincls': (MoCoLinclsConfig, run_moco_lincls),
	'eval_checkpoint': (EvalCheckpointConfig, run_eval_checkpoint),
	'predict_export': (PredictExportConfig, run_predict_export),
}


def get_task(task_type: str) -> TaskEntry:
	"""根据任务类型获取 (配置类, 执行函数) 元组。"""
	task = TASKS.get(task_type)
	if task is None:
		raise ValueError(f"Unknown task type: '{task_type}'. Available types: {list(TASKS.keys())}")
	return task


__all__ = [
	'TASKS',
	'get_task',
	'TaskEntry',
]
