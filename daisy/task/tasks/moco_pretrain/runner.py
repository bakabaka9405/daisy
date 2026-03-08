"""MoCo 预训练任务执行器"""

from datetime import datetime
from functools import partial
from pathlib import Path

import timm
import torch
from torchvision.transforms import v2 as transforms

import daisy
from daisy.model.moco import (
	MoCoV2,
	MoCoV3_ResNet,
	MoCoV3_ViT,
	MOCO_VIT_MODELS,
	TwoCropsTransform,
)
from daisy.dataset import UnlabeledDiskDataset, load_files_from_folder
from daisy.util.transform import ZeroOneNormalize
from ...base import TaskRunner
from ...registry import TaskRegistry
from .config import MoCoPretrainConfig


def _build_v2_transform(input_size: int = 224) -> TwoCropsTransform:
	"""MoCo v2: 对称增强"""
	base_transform = transforms.Compose([
		transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
		transforms.RandomApply([
			transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
		], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.RandomApply([
			transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
		], p=0.5),
		transforms.RandomHorizontalFlip(),
		ZeroOneNormalize(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return TwoCropsTransform(base_transform)


def _build_v3_transform(input_size: int = 224) -> TwoCropsTransform:
	"""MoCo v3: 非对称增强"""
	# aug1: stronger blur
	aug1 = transforms.Compose([
		transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
		transforms.RandomApply([
			transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
		], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
		transforms.RandomHorizontalFlip(),
		ZeroOneNormalize(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	# aug2: weaker blur + solarize
	# Solarize 需要 uint8 输入 (threshold=128)，所以放在 ZeroOneNormalize 之前
	aug2 = transforms.Compose([
		transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
		transforms.RandomApply([
			transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
		], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.RandomApply([
			transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
		], p=0.1),
		transforms.RandomSolarize(threshold=128, p=0.2),
		transforms.RandomHorizontalFlip(),
		ZeroOneNormalize(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	return TwoCropsTransform(aug1, aug2)


@TaskRegistry.register
class MoCoPretrainRunner(TaskRunner['MoCoPretrainConfig']):
	"""MoCo 预训练任务执行器"""

	@classmethod
	def get_task_type(cls) -> str:
		return 'moco_pretrain'

	@classmethod
	def get_config_class(cls) -> type[MoCoPretrainConfig]:
		return MoCoPretrainConfig

	@classmethod
	def get_ui_display_name(cls) -> str:
		return 'MoCo 预训练'

	def run(self, config: MoCoPretrainConfig, device: torch.device) -> Path:  # type: ignore[override]
		"""执行 MoCo 预训练任务"""
		print('=' * 60)
		print(f'Task: {config.meta.title or config.task_id}')
		print(f'Description: {config.meta.description}')
		print(f'Engine: MoCo {config.engine}')
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
		files: list[Path] = []
		roots = [r.strip() for r in dataset_cfg.root.split(';') if r.strip()]
		for root in roots:
			root_files = load_files_from_folder(
				Path(root),
				extensions=tuple(dataset_cfg.extensions),
			)
			print(f'Loaded {len(root_files)} samples from {root}')
			files.extend(root_files)
		print(f'Total samples: {len(files)}')

		# 构建 transform
		input_size = config.training.input_size
		if config.engine == 'v2':
			transform = _build_v2_transform(input_size)
		else:
			transform = _build_v3_transform(input_size)

		# 创建数据集
		dataset = UnlabeledDiskDataset(files, transform=transform)

		# 创建模型
		print('\nCreating model...')
		model_cfg = config.model
		engine = config.engine

		if engine == 'v2':
			encoder_factory = partial(timm.create_model, model_cfg.arch, pretrained=False)
			model = MoCoV2(
				base_encoder=encoder_factory,
				dim=model_cfg.moco_dim,
				K=model_cfg.moco_k,
				m=model_cfg.moco_m,
				T=model_cfg.moco_t,
			)
		else:
			# v3
			arch = model_cfg.arch
			if arch in MOCO_VIT_MODELS:
				# MoCo ViT
				vit_factory = partial(
					MOCO_VIT_MODELS[arch],
					stop_grad_conv1=model_cfg.stop_grad_conv1,
					img_size=input_size,
				)
				model = MoCoV3_ViT(
					base_encoder=vit_factory,
					dim=model_cfg.moco_dim,
					mlp_dim=model_cfg.moco_mlp_dim,
					T=model_cfg.moco_t,
				)
			else:
				# timm ResNet 等
				encoder_factory = partial(timm.create_model, arch, pretrained=False)
				model = MoCoV3_ResNet(
					base_encoder=encoder_factory,
					dim=model_cfg.moco_dim,
					mlp_dim=model_cfg.moco_mlp_dim,
					T=model_cfg.moco_t,
				)

		print(f'Model: {model_cfg.arch} (MoCo {engine})')

		# 训练
		print('\nStarting MoCo pretraining...')
		training_cfg = config.training

		daisy.moco_pretrain.moco_pretrain(
			device=device,
			model=model,
			dataset=dataset,
			engine=engine,
			epochs=training_cfg.epochs,
			batch_size=training_cfg.batch_size,
			optimizer_type=training_cfg.optimizer,
			lr=training_cfg.lr,
			momentum=training_cfg.momentum,
			weight_decay=training_cfg.weight_decay,
			lr_schedule=training_cfg.lr_schedule,
			lr_milestones=training_cfg.lr_milestones,
			warmup_epochs=training_cfg.warmup_epochs,
			moco_m=model_cfg.moco_m,
			moco_m_cos=training_cfg.moco_m_cos,
			use_amp=training_cfg.use_amp,
			num_workers=training_cfg.num_workers,
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
