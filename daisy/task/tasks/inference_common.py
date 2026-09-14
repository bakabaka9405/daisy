"""评估与预测导出的共享辅助函数"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from timm import create_model

from daisy.dataset import IndexDataset
from daisy.method.ordinal import METHODS
from daisy.model.mae import create_vit_model
from daisy.model.ordinal import BackboneModel
from daisy.training import Trainer
from ..shared import (
	InferenceModelConfig,
	InferenceRuntimeConfig,
	get_classification_transform,
	get_mae_finetune_val_transform,
)


def _relative_sample_id(file_path: Path, *, root: Path) -> str:
	try:
		return file_path.resolve().relative_to(root.resolve()).as_posix()
	except ValueError:
		return file_path.as_posix()


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict[str, Any]:
	"""加载 checkpoint 并提取 state_dict"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	if isinstance(checkpoint, dict):
		if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
			return cast(dict[str, Any], checkpoint['model'])
		if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
			return cast(dict[str, Any], checkpoint['state_dict'])
	return cast(dict[str, Any], checkpoint)


def create_inference_model(model_cfg: InferenceModelConfig) -> nn.Module:
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
	elif model_cfg.family == 'ordinal':
		backbone = create_model(
			model_cfg.name,
			pretrained=model_cfg.pretrained,
			num_classes=0,
		)
		model = METHODS[model_cfg.method].create_model(backbone, model_cfg.num_classes)
	else:
		raise ValueError(f'Unsupported model family: {model_cfg.family}')

	if not model_cfg.checkpoint:
		raise ValueError('model.checkpoint is required for inference tasks')
	state_dict = load_checkpoint_state_dict(model_cfg.checkpoint)
	model.load_state_dict(state_dict)
	return model


def get_inference_transform(model_cfg: InferenceModelConfig, inference_cfg: InferenceRuntimeConfig) -> Any:
	"""获取推理时使用的数据增强"""
	if model_cfg.family == 'mae_finetune':
		return get_mae_finetune_val_transform(input_size=inference_cfg.input_size)
	return get_classification_transform(inference_cfg.transform)


def execute_inference(
	model: nn.Module,
	dataset: IndexDataset[Any, Any],
	model_cfg: InferenceModelConfig,
	*,
	device: torch.device,
	batch_size: int = 16,
	num_workers: int = 4,
	include_logits: bool = False,
	transform: Any = None,
) -> tuple[list[int], list[list[float]] | None]:
	"""执行数据集推理，返回预测类别列表与可选的类别 logits 列表。"""
	if transform is not None:
		dataset.setTransform(transform)

	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	trainer = Trainer(model=model, device=device, use_amp=False)
	if model_cfg.family == 'ordinal':
		method_cls = METHODS[model_cfg.method]
		ordinal_model = cast(BackboneModel, model)
		outputs = trainer.inference(
			loader,
			lambda batch: method_cls.decode_features(
				ordinal_model, ordinal_model.forward_features(batch[0]).float(), num_classes=model_cfg.num_classes
			),
		)
		preds = outputs.long().tolist() if outputs.numel() else []
		return preds, None

	scores = trainer.inference(loader, lambda batch: model(batch[0]))
	preds = torch.argmax(scores, dim=1).tolist() if scores.numel() else []
	logits = scores.tolist() if include_logits else None
	return preds, logits


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
			'sample_id': _relative_sample_id(file_path, root=root),
			'true': int(true_label),
			'pred': int(pred_label),
		}
		if logits is not None:
			row['logits'] = logits[index]
		rows.append(row)
	return rows
