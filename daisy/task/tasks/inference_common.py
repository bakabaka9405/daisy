"""评估与预测导出的共享辅助函数"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from timm import create_model

from daisy.protocol import normalize_sample_id
from daisy.model.mae import create_vit_model
from ..shared import get_classification_transform, get_mae_finetune_val_transform


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, Any]:
	"""加载 checkpoint 并提取 state_dict"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	if isinstance(checkpoint, dict):
		if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
			return cast(dict[str, Any], checkpoint['model'])
		if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
			return cast(dict[str, Any], checkpoint['state_dict'])
	return cast(dict[str, Any], checkpoint)


def create_inference_model(model_cfg):
	"""构建并加载推理模型"""
	if model_cfg.family == 'classification':
		model = create_model(
			model_cfg.name,
			pretrained=model_cfg.pretrained,
			num_classes=model_cfg.num_classes,
		)
	elif model_cfg.family == 'mae_finetune':
		model = create_vit_model(
			model_cfg.name,
			num_classes=model_cfg.num_classes,
			global_pool=model_cfg.global_pool,
			drop_path_rate=model_cfg.drop_path,
			img_size=model_cfg.img_size,
		)
	else:
		raise ValueError(f'Unsupported model family: {model_cfg.family}')

	if not model_cfg.checkpoint:
		raise ValueError('model.checkpoint is required for inference tasks')
	state_dict = load_checkpoint_state_dict(model_cfg.checkpoint)
	model.load_state_dict(state_dict)
	return model


def get_inference_transform(model_cfg, inference_cfg):
	"""获取推理时使用的数据增强"""
	if model_cfg.family == 'mae_finetune':
		return get_mae_finetune_val_transform(input_size=inference_cfg.input_size)
	return get_classification_transform(inference_cfg.transform)


def build_prediction_rows(
	files: list[Path],
	labels: list[int],
	preds: list[int],
	*,
	root: Path,
	logits: list[list[float]] | None = None,
) -> list[dict[str, Any]]:
	"""构建逐样本预测导出行"""
	rows: list[dict[str, Any]] = []
	for index, (file_path, true_label, pred_label) in enumerate(zip(files, labels, preds)):
		row: dict[str, Any] = {
			'file': str(file_path),
			'sample_id': normalize_sample_id(file_path, id_type='relative_path', root=root),
			'true': int(true_label),
			'pred': int(pred_label),
		}
		if logits is not None:
			row['logits'] = logits[index]
		rows.append(row)
	return rows
