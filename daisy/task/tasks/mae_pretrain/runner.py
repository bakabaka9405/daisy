"""MAE 预训练任务执行器"""

from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms, InterpolationMode

import daisy
from daisy.model.mae import create_mae_model
from daisy.dataset import UnlabeledDiskDataset, load_files_from_folder
from daisy.protocol import apply_split_manifest, assert_collections_disjoint, collect_sample_ids, load_split_manifest
from daisy.util.transform import ZeroOneNormalize
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import resolve_output_path, save_json, save_task_snapshot
from .config import MAEPretrainConfig


def get_mae_transform(
	input_size: int = 224,
	scale_min: float = 0.2,
	scale_max: float = 1.0,
	hflip: bool = True,
):
	"""获取 MAE 预训练的 transform

	参考 MAE 原论文：RandomResizedCrop + HorizontalFlip + Normalize
	"""
	trans: list = [
		transforms.RandomResizedCrop(
			input_size,
			scale=(scale_min, scale_max),
			interpolation=InterpolationMode.BICUBIC,
		),
	]

	if hflip:
		trans.append(transforms.RandomHorizontalFlip())

	trans.extend(
		[
			ZeroOneNormalize(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)

	return transforms.Compose(trans)


def sample_files_by_ratio(
	files: list[Path],
	*,
	ratio: float,
	seed: int | None = None,
) -> list[Path]:
	"""按比例采样文件列表"""
	if not files:
		return files
	if ratio <= 0 or ratio > 1:
		raise ValueError(f'sample_ratio must be in (0, 1], got {ratio}')
	if ratio >= 1:
		return files

	rng = torch.Generator()
	if seed is not None:
		rng.manual_seed(seed)
	indices = torch.randperm(len(files), generator=rng).tolist()
	sample_size = max(1, int(round(len(files) * ratio)))
	selected_indices = sorted(indices[:sample_size])
	return [files[index] for index in selected_indices]


@TaskRegistry.register
class MAEPretrainRunner(TaskRunner):
	"""MAE 预训练任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'mae_pretrain'

	@classmethod
	def get_config_class(cls) -> type[MAEPretrainConfig]:
		return MAEPretrainConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'MAE 预训练'

	@classmethod
	def get_ui_field_overrides(cls) -> dict[str, dict]:
		return {
			'meta.title': {'label': '任务标题'},
			'meta.description': {'label': '描述', 'component': 'textarea'},
			'meta.created_at': {'hidden': True},
			'meta.commit': {'hidden': True},
			'dataset.root': {'label': '数据目录'},
			'model.name': {
				'label': '模型',
				'component': 'dropdown',
				'choices': ['mae_vit_base_patch16', 'mae_vit_large_patch16', 'mae_vit_huge_patch14'],
				'allow_custom': True,
			},
			'training.epochs': {'label': '训练轮数'},
			'training.batch_size': {'label': 'Batch Size'},
			'training.blr': {'label': '基础学习率'},
			'training.warmup_epochs': {'label': 'Warmup 轮数'},
			'training.mask_ratio': {
				'label': 'Mask 比例',
				'component': 'slider',
				'min_value': 0.5,
				'max_value': 0.9,
				'step': 0.05,
			},
			'output.save_path': {'hidden': True},
		}

	def run(self, config: MAEPretrainConfig, device: torch.device) -> Path:  # type: ignore[override]
		"""执行 MAE 预训练任务"""
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

		# 加载所有图像文件（支持多个文件夹，用分号分割）
		files = []
		roots = [r.strip() for r in dataset_cfg.root.split(';') if r.strip()]
		if not roots:
			raise ValueError('dataset.root is required for mae_pretrain')
		root_paths = [Path(root) for root in roots]
		for root in roots:
			root_files = load_files_from_folder(
				Path(root),
				extensions=tuple(dataset_cfg.extensions),
			)
			print(f'Loaded {len(root_files)} samples from {root}')
			files.extend(root_files)

		protocol_snapshot = {
			'method': 'folder_scan',
			'roots': roots,
			'source_tag': dataset_cfg.source_tag,
			'id_type': dataset_cfg.sample_id_type,
		}

		if dataset_cfg.sample_manifest:
			manifest_splits, selection_report = apply_split_manifest(
				files,
				[0] * len(files),
				dataset_cfg.sample_manifest,
				dataset_root=root_paths,
				split_names=[dataset_cfg.sample_manifest_split] if dataset_cfg.sample_manifest_split else None,
				strict=True,
			)
			selected_files: list[Path] = []
			for split_files, _ in manifest_splits.values():
				selected_files.extend(split_files)
			files = selected_files
			protocol_snapshot['method'] = 'manifest'
			protocol_snapshot['sample_manifest'] = dataset_cfg.sample_manifest
			protocol_snapshot['sample_manifest_split'] = dataset_cfg.sample_manifest_split
			protocol_snapshot['selection_report'] = selection_report

		sample_seed = dataset_cfg.sample_seed if dataset_cfg.sample_seed is not None else training_cfg.seed
		files = sample_files_by_ratio(files, ratio=dataset_cfg.sample_ratio, seed=sample_seed)
		protocol_snapshot['sample_ratio'] = dataset_cfg.sample_ratio
		protocol_snapshot['sample_seed'] = sample_seed
		selected_ids = collect_sample_ids(
			files,
			id_type=dataset_cfg.sample_id_type,
			root=root_paths,
		)
		protocol_snapshot['selected_ids'] = selected_ids
		protocol_snapshot['selected_count'] = len(files)

		if dataset_cfg.reference_manifests:
			reference_collections = {
				'unlabeled_pool': selected_ids,
			}
			for manifest_path in dataset_cfg.reference_manifests:
				manifest = load_split_manifest(manifest_path)
				if manifest.id_type != dataset_cfg.sample_id_type:
					raise ValueError(f'Leakage reference manifest id_type does not match sample_id_type: {manifest_path}')
				reference_collections[Path(manifest_path).stem] = manifest.all_ids()
			assert_collections_disjoint(
				reference_collections,
				id_type=dataset_cfg.sample_id_type,
				context='mae pretrain leakage references',
			)
			protocol_snapshot['reference_manifests'] = dataset_cfg.reference_manifests
		print(f'Total samples: {len(files)}')

		save_json(output_path / 'data_protocol.json', protocol_snapshot)
		save_task_snapshot(
			output_path,
			config,
			extra={
				'device': str(device),
				'commit': config.meta.commit,
				'seed': training_cfg.seed,
			},
		)

		# 获取 transform
		transform_cfg = config.training.transform
		transform = get_mae_transform(
			input_size=transform_cfg.input_size,
			scale_min=transform_cfg.scale_min,
			scale_max=transform_cfg.scale_max,
			hflip=transform_cfg.hflip,
		)

		# 创建数据集
		dataset = UnlabeledDiskDataset(files, transform=transform)

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_mae_model(
			model_cfg.name,
			img_size=model_cfg.img_size,
			norm_pix_loss=model_cfg.norm_pix_loss,
		)

		print(f'Model: {model_cfg.name}')

		# 训练
		print('\nStarting MAE pretraining...')

		daisy.mae_pretrain.mae_pretrain(
			device=device,
			model=model,
			dataset=dataset,
			epochs=training_cfg.epochs,
			batch_size=training_cfg.batch_size,
			blr=training_cfg.blr,
			weight_decay=training_cfg.weight_decay,
			warmup_epochs=training_cfg.warmup_epochs,
			mask_ratio=training_cfg.mask_ratio,
			accum_iter=training_cfg.accum_iter,
			num_workers=training_cfg.num_workers,
			use_amp=training_cfg.use_amp,
			clip_grad=training_cfg.clip_grad,
			max_norm=training_cfg.max_norm,
			save_path=output_path,
			save_freq=training_cfg.save_freq,
			log_dir=output_path / 'logs' if config.output.log else None,
			resume=training_cfg.resume,
		)

		print('\n' + '=' * 60)
		print('Task completed!')
		print(f'Output saved to: {output_path}')
		print('=' * 60)

		return output_path
