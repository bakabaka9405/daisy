"""分类任务执行器"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from timm import create_model

import daisy
from ...base import TaskRunner
from ...registry import TaskRegistry
from .config import ClassificationConfig

if TYPE_CHECKING:
	import torch


def get_git_commit() -> str:
	"""获取当前 git commit hash"""
	try:
		result = subprocess.run(
			['git', 'rev-parse', 'HEAD'],
			capture_output=True,
			text=True,
			check=True,
		)
		return result.stdout.strip()[:8]
	except Exception:
		return 'unknown'


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


@TaskRegistry.register
class ClassificationRunner(TaskRunner):
	"""分类任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'classification'

	@classmethod
	def get_config_class(cls) -> type[ClassificationConfig]:
		return ClassificationConfig

	def run(self, config: ClassificationConfig, device: 'torch.device') -> Path:
		"""执行分类训练任务"""
		import torch

		print('=' * 60)
		print(f'Task: {config.meta.title or config.task_id}')
		print(f'Description: {config.meta.description}')
		print(f'Device: {device}')
		print('=' * 60)

		# 获取 git commit
		if config.meta.commit == 'auto':
			config.meta.commit = get_git_commit()
		print(f'Git commit: {config.meta.commit}')

		# 准备输出目录
		output_path = config.output.save_path.format(
			task_id=config.task_id,
			date=datetime.now().strftime('%Y%m%d'),
		)
		output_path = Path(output_path)
		output_path.mkdir(parents=True, exist_ok=True)
		print(f'Output path: {output_path}')

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset

		if dataset_cfg.type == 'sheet':
			feeder = daisy.feeder.load_feeder_from_sheet(
				dataset_root=Path(dataset_cfg.root),
				sheet=Path(dataset_cfg.sheet),  # type: ignore
				sheet_name=dataset_cfg.sheet_name,
				column=dataset_cfg.column,
				label_offset=dataset_cfg.label_offset,
				have_header=dataset_cfg.have_header,
			)
		elif dataset_cfg.type == 'folder':
			feeder = daisy.feeder.load_feeder_from_folder(Path(dataset_cfg.root))
		else:
			raise ValueError(f'Unknown dataset type: {dataset_cfg.type}')

		files, labels = feeder.fetch()
		print(f'Total samples: {len(files)}')

		# 创建数据集
		dataset = daisy.dataset.DiskDataset(files, labels)

		# 数据划分
		split_cfg = dataset_cfg.split
		if split_cfg.method == 'ratio':
			train_dataset, val_dataset = daisy.dataset.dataset_split.default_data_split(
				dataset, val_ratio=split_cfg.val_ratio
			)
		elif split_cfg.method == 'sheet':
			# 从另一个 sheet 加载验证集
			val_feeder = daisy.feeder.load_feeder_from_sheet(
				dataset_root=Path(dataset_cfg.root),
				sheet=Path(split_cfg.val_sheet),  # type: ignore
				sheet_name=split_cfg.val_sheet_name,
				column=dataset_cfg.column,
				label_offset=dataset_cfg.label_offset,
				have_header=dataset_cfg.have_header,
			)
			val_files, val_labels = val_feeder.fetch()
			train_dataset = dataset
			val_dataset = daisy.dataset.DiskDataset(val_files, val_labels)
		else:
			raise ValueError(f'Unknown split method: {split_cfg.method}')

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
		training_cfg = config.training

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
