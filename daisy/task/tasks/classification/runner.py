"""分类任务执行器"""

from pathlib import Path
from typing import TYPE_CHECKING

from timm import create_model

import daisy
from daisy.protocol import apply_named_splits, apply_split_manifest, assert_collections_disjoint, collect_sample_ids, normalize_sample_id
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import resolve_output_path, save_json, save_task_snapshot
from .config import ClassificationConfig

if TYPE_CHECKING:
	import torch


def get_transform(name: str):
	"""根据名称获取 transform"""
	transform_map = {
		'rectangle_train': daisy.util.transform.get_rectangle_train_transform,
		'rectangle_val': daisy.util.transform.get_rectangle_val_transform,
		'rectangle_train_slight': daisy.util.transform.get_rectangle_train_transform_slight,
		'stretch_train': daisy.util.transform.get_stretch_train_transform,
		'stretch_val': daisy.util.transform.get_stretch_val_transform,
	}

	if name not in transform_map:
		raise ValueError(f'Unknown transform: {name}. Available: {list(transform_map.keys())}')

	return transform_map[name]()


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


def build_split_snapshot(
	*,
	method: str,
	root: Path,
	train_files: list[Path],
	val_files: list[Path],
	id_type: str = 'relative_path',
	extra: dict | None = None,
) -> dict:
	snapshot = {
		'method': method,
		'id_type': id_type,
		'root': str(root),
		'splits': {
			'train': collect_sample_ids(train_files, id_type=id_type, root=root),
			'val': collect_sample_ids(val_files, id_type=id_type, root=root),
		},
		'counts': {
			'train': len(train_files),
			'val': len(val_files),
		},
	}
	if extra:
		snapshot.update(extra)
	return snapshot


@TaskRegistry.register
class ClassificationRunner(TaskRunner['ClassificationConfig']):
	"""分类任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'classification'

	@classmethod
	def get_config_class(cls) -> type[ClassificationConfig]:
		return ClassificationConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return '图像分类'

	@classmethod
	def get_ui_field_overrides(cls) -> dict[str, dict]:
		return {
			'meta.title': {'label': '任务标题'},
			'meta.description': {'label': '描述', 'component': 'textarea'},
			'meta.creator': {'label': '创建者'},
			'meta.created_at': {'hidden': True},
			'meta.commit': {'hidden': True},
			'dataset.type': {'label': '数据集类型'},
			'dataset.root': {'label': '数据根目录'},
			'dataset.sheet': {'label': '标注文件'},
			'dataset.column': {'label': '标签列'},
			'dataset.split.val_ratio': {
				'label': '验证集比例',
				'component': 'slider',
				'min_value': 0.05,
				'max_value': 0.3,
				'step': 0.01,
			},
			'model.name': {
				'label': '模型',
				'component': 'dropdown',
				'choices': ['resnet34', 'resnet50', 'resnet101', 'efficientnet_b0', 'convnext_tiny'],
				'allow_custom': True,
			},
			'model.num_classes': {'label': '类别数'},
			'model.pretrained': {'label': '使用预训练权重'},
			'training.epochs': {'label': '训练轮数'},
			'training.batch_size': {'label': 'Batch Size'},
			'training.lr': {'label': '学习率'},
			'training.warmup_epochs': {'label': 'Warmup 轮数'},
			'training.weight_decay': {'label': 'Weight Decay'},
			'training.cmp_obj': {'label': '优化目标'},
			'training.transform.train': {
				'label': '训练 Transform',
				'component': 'dropdown',
				'choices': ['rectangle_train', 'rectangle_train_slight', 'stretch_train'],
			},
			'training.transform.val': {
				'label': '验证 Transform',
				'component': 'dropdown',
				'choices': ['rectangle_val', 'stretch_val'],
			},
			'output.save_path': {'hidden': True},
			'output.keep_count': {'label': '保留检查点数'},
			'output.save_best': {'label': '保存最佳模型'},
			'output.log': {'label': '记录日志'},
		}

	def run(self, config: ClassificationConfig, device: 'torch.device') -> Path:
		"""执行分类训练任务"""
		import torch

		training_cfg = config.training
		if training_cfg.seed is not None:
			daisy.util.set_global_seed(training_cfg.seed)

		print('=' * 60)
		print(f'Task: {config.meta.title or config.task_id}')
		print(f'Description: {config.meta.description}')
		print(f'Device: {device}')
		if training_cfg.seed is not None:
			print(f'Seed: {training_cfg.seed}')
		print('=' * 60)

		# 获取 git commit
		if config.meta.commit == 'auto':
			config.meta.commit = daisy.util.get_git_commit()
		print(f'Git commit: {config.meta.commit}')

		# 准备输出目录
		output_path = resolve_output_path(config.output.save_path, config.task_id)
		print(f'Output path: {output_path}')

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset
		dataset_root = Path(dataset_cfg.root)
		files, labels = load_labeled_samples(dataset_cfg)
		print(f'Total samples: {len(files)}')

		# 创建数据集
		dataset = daisy.dataset.DiskDataset(files, labels)

		# 数据划分
		split_cfg = dataset_cfg.split
		split_seed = split_cfg.seed if split_cfg.seed is not None else training_cfg.seed
		protocol_snapshot: dict
		if split_cfg.method == 'ratio':
			split_fn = daisy.dataset.dataset_split.stratified_data_split if split_cfg.stratified else daisy.dataset.dataset_split.default_data_split
			train_dataset, val_dataset = split_fn(
				dataset,
				val_ratio=split_cfg.val_ratio,
				seed=split_seed,
			)
			train_files, _ = train_dataset.getRawData()
			val_files, _ = val_dataset.getRawData()
			protocol_snapshot = build_split_snapshot(
				method='ratio',
				root=dataset_root,
				train_files=train_files,
				val_files=val_files,
				extra={
					'seed': split_seed,
					'stratified': split_cfg.stratified,
					'val_ratio': split_cfg.val_ratio,
				},
			)
		elif split_cfg.method == 'sheet':
			# 从另一个 sheet 加载验证集
			val_feeder = daisy.feeder.load_feeder_from_sheet(
				dataset_root=Path(dataset_cfg.root),
				sheet_path=Path(split_cfg.val_sheet),  # type: ignore[arg-type]
				sheet_name=split_cfg.val_sheet_name,
				column=dataset_cfg.column,
				label_offset=dataset_cfg.label_offset,
				have_header=dataset_cfg.have_header,
			)
			val_files, val_labels = val_feeder.fetch()
			val_file_ids = {normalize_sample_id(file_path, id_type='path') for file_path in val_files}
			train_files = []
			train_labels = []
			for file_path, label in zip(files, labels):
				if normalize_sample_id(file_path, id_type='path') in val_file_ids:
					continue
				train_files.append(file_path)
				train_labels.append(label)
			train_dataset = daisy.dataset.DiskDataset(train_files, train_labels)
			val_dataset = daisy.dataset.DiskDataset(val_files, val_labels)
			protocol_snapshot = build_split_snapshot(
				method='sheet',
				root=dataset_root,
				train_files=train_files,
				val_files=val_files,
				extra={
					'val_sheet': split_cfg.val_sheet,
					'val_sheet_name': split_cfg.val_sheet_name,
				},
			)
		elif split_cfg.method == 'preset':
			if split_cfg.train_files or split_cfg.val_files:
				split_mapping, report = apply_named_splits(
					files,
					labels,
					{
						'train': split_cfg.train_files,
						'val': split_cfg.val_files,
					},
					id_type=split_cfg.manifest_id_type,
					root=dataset_root,
					strict=split_cfg.require_all_in_manifest,
				)
				train_dataset = daisy.dataset.DiskDataset(*split_mapping['train'])
				val_dataset = daisy.dataset.DiskDataset(*split_mapping['val'])
				protocol_snapshot = build_split_snapshot(
					method='preset',
					root=dataset_root,
					train_files=split_mapping['train'][0],
					val_files=split_mapping['val'][0],
					id_type=split_cfg.manifest_id_type,
					extra={
						'selection_report': report,
						'from_explicit_file_lists': True,
					},
				)
			else:
				train_feeder = daisy.feeder.load_feeder_from_folder(dataset_root / split_cfg.preset_train_dir)
				val_feeder = daisy.feeder.load_feeder_from_folder(dataset_root / split_cfg.preset_val_dir)
				train_files, train_labels = train_feeder.fetch()
				val_files, val_labels = val_feeder.fetch()
				train_dataset = daisy.dataset.DiskDataset(train_files, train_labels)
				val_dataset = daisy.dataset.DiskDataset(val_files, val_labels)
				protocol_snapshot = build_split_snapshot(
					method='preset',
					root=dataset_root,
					train_files=train_files,
					val_files=val_files,
					extra={
						'preset_train_dir': split_cfg.preset_train_dir,
						'preset_val_dir': split_cfg.preset_val_dir,
						'from_explicit_file_lists': False,
					},
				)
		elif split_cfg.method == 'manifest':
			if not split_cfg.manifest:
				raise ValueError('dataset.split.manifest is required when split.method = "manifest"')
			split_mapping, report = apply_split_manifest(
				files,
				labels,
				split_cfg.manifest,
				dataset_root=dataset_root,
				split_names=(split_cfg.manifest_train_split, split_cfg.manifest_val_split),
				strict=split_cfg.require_all_in_manifest,
			)
			train_dataset = daisy.dataset.DiskDataset(*split_mapping[split_cfg.manifest_train_split])
			val_dataset = daisy.dataset.DiskDataset(*split_mapping[split_cfg.manifest_val_split])
			protocol_snapshot = build_split_snapshot(
				method='manifest',
				root=dataset_root,
				train_files=split_mapping[split_cfg.manifest_train_split][0],
				val_files=split_mapping[split_cfg.manifest_val_split][0],
				id_type=report['id_type'],
				extra={
					'manifest': split_cfg.manifest,
					'manifest_train_split': split_cfg.manifest_train_split,
					'manifest_val_split': split_cfg.manifest_val_split,
					'selection_report': report,
				},
			)
		else:
			raise ValueError(f'Unknown split method: {split_cfg.method}')

		assert_collections_disjoint(
			{
				'train': train_dataset.getRawData()[0],
				'val': val_dataset.getRawData()[0],
			},
			id_type='path',
			context='classification splits',
		)
		save_json(output_path / 'split_protocol.json', protocol_snapshot)
		save_task_snapshot(
			output_path,
			config,
			extra={
				'device': str(device),
				'commit': config.meta.commit,
				'seed': training_cfg.seed,
			},
		)

		print(f'Train samples: {len(train_dataset)}')
		print(f'Val samples: {len(val_dataset)}')

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_model(
			model_cfg.name,
			pretrained=model_cfg.pretrained,
			num_classes=model_cfg.num_classes,
		)

		# 加载预训练权重
		if model_cfg.checkpoint:
			print(f'Loading checkpoint: {model_cfg.checkpoint}')
			model.load_state_dict(torch.load(model_cfg.checkpoint, map_location='cpu'))

		print(f'Model: {model_cfg.name}')

		# 获取 transforms
		train_transform = get_transform(config.training.transform.train)
		val_transform = get_transform(config.training.transform.val)

		# 训练
		print('\nStarting training...')

		daisy.classfier_trainer.fast_train_smile(
			device=device,
			model=model,
			dataset=(train_dataset, val_dataset),
			num_classes=model_cfg.num_classes,
			epochs=training_cfg.epochs,
			batch_size=training_cfg.batch_size,
			lr=training_cfg.lr,
			weight_decay=training_cfg.weight_decay,
			warmup_epochs=training_cfg.warmup_epochs,
			smoothing=training_cfg.smoothing,
			accum_iter=training_cfg.accum_iter,
			use_scheduler=training_cfg.use_scheduler,
			use_amp=training_cfg.use_amp,
			clip_grad=training_cfg.clip_grad,
			max_norm=training_cfg.max_norm,
			num_workers=training_cfg.num_workers,
			early_stop=training_cfg.early_stop,
			early_stop_epoch=training_cfg.early_stop_epoch,
			cmp_obj=training_cfg.cmp_obj,
			train_transform=train_transform,
			val_transform=val_transform,
			save_path=output_path if config.output.save_best else None,
			keep_count=config.output.keep_count,
			save_best=config.output.save_best,
			log_dir=output_path / 'logs' if config.output.log else None,
		)

		print('\n' + '=' * 60)
		print('Task completed!')
		print(f'Output saved to: {output_path}')
		print('=' * 60)

		return output_path
