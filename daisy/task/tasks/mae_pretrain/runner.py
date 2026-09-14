"""MAE 预训练任务执行器"""

from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
import random
from typing import override

import torch
from torchvision.transforms import v2 as transforms, InterpolationMode

import daisy
from daisy.mae_pretrain import MAEPretrainParams
from daisy.model.mae import create_mae_model, load_timm_pretrained_encoder_weights
from daisy.dataset import DiskDataset, load_files_from_folder
from daisy.util.transform import LazyNormalization, LazyNormalizedTensor
from ...base import TaskRunner
from ...registry import TaskRegistry
from ...runtime import prepare_task_run, print_task_completed, save_run_snapshot
from .config import MAEPretrainConfig


def get_mae_transform(
	input_size: int = 224,
	scale_min: float = 0.2,
	scale_max: float = 1.0,
	hflip: bool = True,
) -> Callable[[torch.Tensor], LazyNormalizedTensor]:
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

	trans.append(LazyNormalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
	return transforms.Compose(trans)


@TaskRegistry.register
class MAEPretrainRunner(TaskRunner):
	"""MAE 预训练任务执行器"""

	@classmethod
	@override
	def get_task_type(cls) -> str:
		return 'mae_pretrain'

	@classmethod
	@override
	def get_config_class(cls) -> type[MAEPretrainConfig]:
		return MAEPretrainConfig

	@override
	def run(self, config: MAEPretrainConfig, device: torch.device) -> Path:  # type: ignore[override]
		"""执行 MAE 预训练任务"""
		training_cfg = config.training
		run_context = prepare_task_run(config, device, seed=training_cfg.seed)
		output_path = run_context.output_path

		# 加载数据集
		print('\nLoading dataset...')
		dataset_cfg = config.dataset

		# 加载所有图像文件（支持多个文件夹，用分号分割）
		files = []
		roots = [r.strip() for r in dataset_cfg.root.split(';') if r.strip()]
		for root in roots:
			root_files = load_files_from_folder(
				Path(root),
				extensions=tuple(dataset_cfg.extensions),
			)
			print(f'Loaded {len(root_files)} samples from {root}')
			files.extend(root_files)

		if dataset_cfg.sample_ratio < 1.0 and files:
			sample_seed = dataset_cfg.sample_seed if dataset_cfg.sample_seed is not None else training_cfg.seed
			k = max(1, round(len(files) * dataset_cfg.sample_ratio))
			files = random.Random(sample_seed).sample(files, k)
		print(f'Total samples: {len(files)}')
		save_run_snapshot(
			output_path,
			config,
			run_context,
		)

		# 获取 transform
		transform_cfg = config.training.transform
		transform = get_mae_transform(
			input_size=transform_cfg.input_size,
			scale_min=transform_cfg.scale_min,
			scale_max=transform_cfg.scale_max,
			hflip=transform_cfg.hflip,
		)

		# 创建数据集（无标签，LabelT=None）
		dataset: DiskDataset[LazyNormalizedTensor, None] = DiskDataset(files, None, transform)

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		model = create_mae_model(
			model_cfg.name,
			img_size=model_cfg.img_size,
		)
		if model_cfg.timm_pretrained and training_cfg.resume is None:
			print(f'Loading timm pretrained encoder: {model_cfg.timm_pretrained}')
			load_timm_pretrained_encoder_weights(model, model_cfg.timm_pretrained)
		elif model_cfg.timm_pretrained:
			print('Skipping timm pretrained encoder because resume is configured')

		print(f'Model: {model_cfg.name}')

		# 训练
		print('\nStarting MAE pretraining...')

		params = MAEPretrainParams(**training_cfg.model_dump(include={field.name for field in fields(MAEPretrainParams)}))
		daisy.mae_pretrain.mae_pretrain(
			device=device,
			model=model,
			dataset=dataset,
			params=params,
			save_path=output_path,
			save_freq=training_cfg.save_freq,
			log_dir=output_path / 'logs' if config.output.log else None,
			resume=training_cfg.resume,
		)

		print_task_completed(output_path)

		return output_path
