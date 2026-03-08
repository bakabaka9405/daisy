"""MoCo 线性探测评估任务执行器"""

from datetime import datetime
from pathlib import Path

import timm
import torch

import daisy
from daisy.model.moco import load_moco_pretrained_weights
from daisy.util import get_model_classifier, change_model_classifier
from ...base import TaskRunner
from ...registry import TaskRegistry
from .config import MoCoLinclsConfig


@TaskRegistry.register
class MoCoLinclsRunner(TaskRunner['MoCoLinclsConfig']):
	"""MoCo 线性探测评估任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'moco_lincls'

	@classmethod
	def get_config_class(cls) -> type[MoCoLinclsConfig]:
		return MoCoLinclsConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'MoCo 线性探测'

	def run(self, config: MoCoLinclsConfig, device: torch.device) -> Path:  # type: ignore[override]
		"""执行 MoCo 线性探测评估任务"""
		print('=' * 60)
		print(f'Task: {config.meta.title or config.task_id}')
		print(f'Description: {config.meta.description}')
		print(f'Device: {device}')
		print('=' * 60)

		# 获取 git commit
		if config.meta.commit == 'auto':
			config.meta.commit = daisy.util.get_git_commit()
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
		elif split_cfg.method == 'preset':
			train_feeder = daisy.feeder.load_feeder_from_folder(Path(dataset_cfg.root) / 'train')
			val_feeder = daisy.feeder.load_feeder_from_folder(Path(dataset_cfg.root) / 'val')
			train_files, train_labels = train_feeder.fetch()
			val_files, val_labels = val_feeder.fetch()
			train_dataset = daisy.dataset.DiskDataset(train_files, train_labels)
			val_dataset = daisy.dataset.DiskDataset(val_files, val_labels)
		else:
			raise ValueError(f'Unknown split method: {split_cfg.method}')

		print(f'Train samples: {len(train_dataset)}')
		print(f'Val samples: {len(val_dataset)}')

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = timm.create_model(model_cfg.arch, pretrained=False, num_classes=model_cfg.num_classes)

		# 加载 MoCo 预训练权重
		if model_cfg.pretrained_checkpoint:
			print(f'Loading MoCo checkpoint: {model_cfg.pretrained_checkpoint}')
			load_moco_pretrained_weights(model, model_cfg.pretrained_checkpoint, engine=model_cfg.engine)

		# 冻结所有层
		for param in model.parameters():
			param.requires_grad = False

		# 重新初始化分类头并设为可训练
		change_model_classifier(model, num_classes=model_cfg.num_classes)
		classifier = get_model_classifier(model)
		assert isinstance(classifier, torch.nn.Linear)
		classifier.weight.data.normal_(mean=0.0, std=0.01)
		classifier.bias.data.zero_()  # type: ignore[union-attr]
		for param in classifier.parameters():
			param.requires_grad = True

		print(f'Model: {model_cfg.arch} (linear probe, {model_cfg.num_classes} classes)')

		# 训练 (复用 classifier trainer)
		print('\nStarting linear evaluation...')
		training_cfg = config.training

		daisy.classfier_trainer.train_classifier(
			device=device,
			model=model,
			num_classes=model_cfg.num_classes,
			epochs=training_cfg.epochs,
			dataset=(train_dataset, val_dataset),
			lr=training_cfg.lr,
			weight_decay=training_cfg.weight_decay,
			warmup_epochs=training_cfg.warmup_epochs,
			batch_size=training_cfg.batch_size,
			use_amp=training_cfg.use_amp,
			num_workers=training_cfg.num_workers,
			save_path=output_path,
			save_freq=training_cfg.save_freq,
			save_best=True,
			log_dir=output_path / 'logs' if config.output.log else None,
		)

		print('\n' + '=' * 60)
		print('Task completed!')
		print(f'Output saved to: {output_path}')
		print('=' * 60)

		return output_path
