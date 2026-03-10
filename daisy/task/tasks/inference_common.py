"""评估与预测导出的共享辅助函数"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from timm import create_model

import daisy
from daisy.model.mae import create_vit_model
from daisy.protocol import apply_named_splits, apply_split_manifest, collect_sample_ids, normalize_sample_id
from .classification.runner import get_transform as get_classification_transform
from .mae_finetune.runner import get_finetune_val_transform


def load_labeled_samples(dataset_cfg) -> tuple[list[Path], list[int]]:
	"""加载带标签样本"""
	if dataset_cfg.type == 'sheet':
		feeder = daisy.feeder.load_feeder_from_sheet(
			dataset_root=Path(dataset_cfg.root),
			sheet_path=Path(dataset_cfg.sheet),  # type: ignore[arg-type]
			sheet_name=dataset_cfg.sheet_name,
			column=dataset_cfg.column,
			label_offset=dataset_cfg.label_offset,
			have_header=dataset_cfg.have_header,
		)
		return feeder.fetch()
	if dataset_cfg.type == 'folder':
		feeder = daisy.feeder.load_feeder_from_folder(Path(dataset_cfg.root))
		return feeder.fetch()
	raise ValueError(f'Unknown dataset type: {dataset_cfg.type}')


def build_selection_snapshot(
	*,
	method: str,
	root: Path,
	split_name: str,
	files: list[Path],
	id_type: str = 'relative_path',
	extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
	snapshot: dict[str, Any] = {
		'method': method,
		'id_type': id_type,
		'root': str(root),
		'split_name': split_name,
		'count': len(files),
		'sample_ids': collect_sample_ids(files, id_type=id_type, root=root),
	}
	if extra:
		snapshot.update(extra)
	return snapshot


def select_inference_dataset(dataset_cfg, *, split_name: str) -> tuple[daisy.dataset.DiskDataset, dict[str, Any]]:
	"""为评估/预测选择目标数据集 split"""
	dataset_root = Path(dataset_cfg.root)
	files, labels = load_labeled_samples(dataset_cfg)
	dataset = daisy.dataset.DiskDataset(files, labels)
	split_cfg = dataset_cfg.split

	if split_cfg.method == 'none':
		return dataset, build_selection_snapshot(
			method='none',
			root=dataset_root,
			split_name=split_name,
			files=files,
		)

	if split_cfg.method == 'ratio':
		split_fn = daisy.dataset.dataset_split.stratified_data_split if split_cfg.stratified else daisy.dataset.dataset_split.default_data_split
		train_dataset, val_dataset = split_fn(
			dataset,
			val_ratio=split_cfg.val_ratio,
			seed=split_cfg.seed,
		)
		selection_map = {
			'train': train_dataset,
			'val': val_dataset,
		}
		if split_name not in selection_map:
			raise ValueError(f'ratio split only supports train/val, got {split_name!r}')
		selected_dataset = selection_map[split_name]
		selected_files, _ = selected_dataset.getRawData()
		return selected_dataset, build_selection_snapshot(
			method='ratio',
			root=dataset_root,
			split_name=split_name,
			files=selected_files,
			extra={
				'seed': split_cfg.seed,
				'stratified': split_cfg.stratified,
				'val_ratio': split_cfg.val_ratio,
			},
		)

	if split_cfg.method == 'sheet':
		if split_name == 'train':
			raise ValueError('sheet split does not support selecting train directly')
		val_feeder = daisy.feeder.load_feeder_from_sheet(
			dataset_root=dataset_root,
			sheet_path=Path(split_cfg.val_sheet),  # type: ignore[arg-type]
			sheet_name=split_cfg.val_sheet_name,
			column=dataset_cfg.column,
			label_offset=dataset_cfg.label_offset,
			have_header=dataset_cfg.have_header,
		)
		selected_files, selected_labels = val_feeder.fetch()
		return daisy.dataset.DiskDataset(selected_files, selected_labels), build_selection_snapshot(
			method='sheet',
			root=dataset_root,
			split_name=split_name,
			files=selected_files,
			extra={
				'source_sheet': split_cfg.val_sheet,
				'source_sheet_name': split_cfg.val_sheet_name,
			},
		)

	if split_cfg.method == 'preset':
		if split_cfg.train_files or split_cfg.val_files or split_cfg.test_files:
			split_mapping, report = apply_named_splits(
				files,
				labels,
				{
					'train': split_cfg.train_files,
					'val': split_cfg.val_files,
					'test': split_cfg.test_files,
				},
				id_type=split_cfg.manifest_id_type,
				root=dataset_root,
				strict=split_cfg.require_all_in_manifest,
			)
			if split_name not in split_mapping:
				raise ValueError(f'Unknown preset split name: {split_name!r}')
			selected_files, selected_labels = split_mapping[split_name]
			return daisy.dataset.DiskDataset(selected_files, selected_labels), build_selection_snapshot(
				method='preset',
				root=dataset_root,
				split_name=split_name,
				files=selected_files,
				id_type=split_cfg.manifest_id_type,
				extra={
					'selection_report': report,
					'from_explicit_file_lists': True,
				},
			)

		dir_map = {
			'train': split_cfg.preset_train_dir,
			'val': split_cfg.preset_val_dir,
			'test': split_cfg.preset_test_dir,
		}
		if split_name not in dir_map:
			raise ValueError(f'Unknown preset split name: {split_name!r}')
		target_dir = dir_map[split_name]
		feeder = daisy.feeder.load_feeder_from_folder(dataset_root / target_dir)
		selected_files, selected_labels = feeder.fetch()
		return daisy.dataset.DiskDataset(selected_files, selected_labels), build_selection_snapshot(
			method='preset',
			root=dataset_root,
			split_name=split_name,
			files=selected_files,
			extra={
				'directory': target_dir,
				'from_explicit_file_lists': False,
			},
		)

	if split_cfg.method == 'manifest':
		if not split_cfg.manifest:
			raise ValueError('dataset.split.manifest is required when split.method = "manifest"')
		manifest_split_map = {
			'train': split_cfg.manifest_train_split,
			'val': split_cfg.manifest_val_split,
			'test': split_cfg.manifest_test_split,
		}
		target_split = manifest_split_map.get(split_name, split_name)
		split_mapping, report = apply_split_manifest(
			files,
			labels,
			split_cfg.manifest,
			dataset_root=dataset_root,
			split_names=(target_split,),
			strict=split_cfg.require_all_in_manifest,
		)
		selected_files, selected_labels = split_mapping[target_split]
		return daisy.dataset.DiskDataset(selected_files, selected_labels), build_selection_snapshot(
			method='manifest',
			root=dataset_root,
			split_name=split_name,
			files=selected_files,
			id_type=report['id_type'],
			extra={
				'manifest': split_cfg.manifest,
				'target_split': target_split,
				'selection_report': report,
			},
		)

	raise ValueError(f'Unknown split method: {split_cfg.method}')


def load_checkpoint_state_dict(checkpoint_path: str | Path) -> dict:
	"""加载 checkpoint 并提取 state_dict"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	if isinstance(checkpoint, dict):
		if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
			return checkpoint['model']
		if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
			return checkpoint['state_dict']
	return checkpoint


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
		return get_finetune_val_transform(input_size=inference_cfg.input_size)
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
