"""MoCo 线性探测评估任务执行器"""

from pathlib import Path

import timm
import torch

import daisy
from daisy.model.moco import load_moco_pretrained_weights
from daisy.util import get_model_classifier, change_model_classifier
from ...base import TaskRunner
from ...data import build_train_val_selection
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_json, save_run_snapshot
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
		run_context = prepare_task_run(config, device)
		output_path = run_context.output_path

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset
		split_selection = build_train_val_selection(
			dataset_cfg,
			context='moco lincls splits',
		)
		print(f'Total samples: {split_selection.source_count}')
		train_dataset, val_dataset = split_selection.to_datasets()
		save_json(output_path / 'split_protocol.json', split_selection.protocol.to_dict())
		save_run_snapshot(
			output_path,
			config,
			run_context,
		)

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

		print_task_completed(output_path)

		return output_path
