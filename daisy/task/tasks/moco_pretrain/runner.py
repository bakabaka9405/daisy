"""MoCo 预训练任务执行器"""

from functools import partial
from pathlib import Path
import time

import timm
import torch
from torchvision.transforms import v2 as transforms

from daisy.dataset import DiskDataset, load_files_from_folder
from daisy.method.moco import MoCoPretrainMethod
from daisy.model.moco import (
	MoCoV2,
	MoCoV3_ResNet,
	MoCoV3_ViT,
	MOCO_VIT_MODELS,
	TwoCropsTransform,
)
from daisy.training import CosineAnnealingLR, TrainState
from daisy.util import make_dataloader
from daisy.util.transform import ZeroOneNormalize
from ...runtime import TaskHeaderLine, prepare_task_run, print_task_completed, save_run_snapshot
from .config import MoCoPretrainConfig


def _build_v2_transform(input_size: int = 224) -> TwoCropsTransform:
	"""构建 MoCo v2 对称数据增强管线。"""
	base_transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
			transforms.RandomApply(
				[
					transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
				],
				p=0.8,
			),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply(
				[
					transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
				],
				p=0.5,
			),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return TwoCropsTransform(base_transform)


def _build_v3_transform(input_size: int = 224) -> TwoCropsTransform:
	"""构建 MoCo v3 非对称双视图增强管线。"""
	# 第一分支增强：高斯模糊概率为 1.0
	aug1 = transforms.Compose(
		[
			transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
			transforms.RandomApply(
				[
					transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
				],
				p=0.8,
			),
			transforms.RandomGrayscale(p=0.2),
			transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	# 第二分支增强：降低模糊概率并引入日光化变换
	aug2 = transforms.Compose(
		[
			transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
			transforms.RandomApply(
				[
					transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
				],
				p=0.8,
			),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply(
				[
					transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
				],
				p=0.1,
			),
			transforms.RandomSolarize(threshold=128, p=0.2),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	return TwoCropsTransform(aug1, aug2)


def run(config: MoCoPretrainConfig, device: torch.device) -> Path:
	"""执行完整预训练工作流，覆盖数据装载、模型构建、方法调度与检查点持久化。"""
	run_context = prepare_task_run(
		config,
		device,
		header_lines=[TaskHeaderLine(label='Engine', value=f'MoCo {config.engine}')],
	)
	output_path = run_context.output_path

	# 递归装载文件夹数据源
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
	if len(files) == 0:
		raise RuntimeError('Dataset contains zero samples; check dataset directory and extensions.')
	save_run_snapshot(output_path, config, run_context)

	# 依据引擎版本选择对称或非对称双视图变换
	input_size = config.training.input_size
	if config.engine == 'v2':
		transform = _build_v2_transform(input_size)
	else:
		transform = _build_v3_transform(input_size)

	# 实例化无标签自监督数据集
	dataset: DiskDataset[list[torch.Tensor], None] = DiskDataset(files, None, transform)

	# 构建与引擎匹配的模型架构
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
		arch = model_cfg.arch
		if arch in MOCO_VIT_MODELS:
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
			encoder_factory = partial(timm.create_model, arch, pretrained=False)
			model = MoCoV3_ResNet(
				base_encoder=encoder_factory,
				dim=model_cfg.moco_dim,
				mlp_dim=model_cfg.moco_mlp_dim,
				T=model_cfg.moco_t,
			)

	print(f'Model: {model_cfg.arch} (MoCo {engine})')

	training_cfg = config.training

	# drop_last 会丢弃不足一批的数据，须在重复采样器启动前检查。
	if len(dataset) < training_cfg.batch_size:
		raise RuntimeError('DataLoader contains zero batches; check dataset directory and batch size.')

	print('Loading dataloader...')
	loader = make_dataloader(
		dataset,
		batch_size=training_cfg.batch_size,
		device=device,
		shuffle=True,
		drop_last=True,
		num_workers=training_cfg.num_workers,
		pin_memory=True,
	)

	# 初始化算法方法与核心训练循环
	method = MoCoPretrainMethod(
		model,
		device=device,
		engine=engine,
		lr=training_cfg.lr,
		weight_decay=training_cfg.weight_decay,
		momentum=training_cfg.momentum,
		optimizer_type=training_cfg.optimizer,
		moco_m=model_cfg.moco_m,
		moco_m_cos=training_cfg.moco_m_cos,
		use_amp=training_cfg.use_amp,
	)
	trainer = method.trainer
	optimizer = trainer.optimizer
	assert optimizer is not None

	# 注册学习率衰减策略
	if training_cfg.lr_schedule == 'cosine':
		trainer.use(
			CosineAnnealingLR(
				lr=training_cfg.lr,
				min_lr=0.0,
				warmup_epochs=training_cfg.warmup_epochs,
				per_iteration=(engine == 'v3'),
			)
		)
	elif training_cfg.lr_schedule == 'step':
		milestones = list(training_cfg.lr_milestones)
		warmup_epochs = training_cfg.warmup_epochs
		base_lr = training_cfg.lr

		def step_lr(state: TrainState) -> None:
			epoch = state.epoch
			if epoch < warmup_epochs:
				curr_lr = base_lr * epoch / max(warmup_epochs, 1e-8)
			else:
				factor = 1.0
				for milestone in milestones:
					if epoch >= milestone:
						factor *= 0.1
				curr_lr = base_lr * factor

			assert state.optimizer is not None
			for pg in state.optimizer.param_groups:
				pg['lr'] = curr_lr

		trainer.on_epoch_start(step_lr)

	# 批次监控与首批动量快照采集
	def on_batch_end(state: TrainState) -> None:
		if state.batch_idx == 0:
			state.extras['epoch_start_moco_m'] = method.current_m

		idx = state.batch_idx + 1
		if idx % 20 == 0 or idx == state.total_batches:
			cur_m = state.extras.get('moco_m', model_cfg.moco_m)
			loss_val = state.batch_loss.item()
			print(
				f'Epoch [{state.epoch + 1}/{training_cfg.epochs}] '
				f'[{idx}/{state.total_batches}] '
				f'Loss: {loss_val:.4f} LR: {state.current_lr:.6f} M: {cur_m:.4f}'
			)

	trainer.on_batch_end(on_batch_end)

	# 初始化指标记录日志文件
	if config.output.log:
		log_dir = output_path / 'logs'
		log_dir.mkdir(parents=True, exist_ok=True)
		log_file = log_dir / f'moco_log_{time.strftime("%Y%m%d_%H%M%S")}.csv'
		if not log_file.exists():
			with open(log_file, 'w', encoding='utf-8') as f:
				f.write('epoch,lr,train_loss,moco_m\n')
	else:
		log_file = None

	epochs = training_cfg.epochs
	save_freq = training_cfg.save_freq

	# 轮次结束指标汇总与检查点序列化
	def on_epoch_end(state: TrainState) -> None:
		epoch = state.epoch + 1
		train_loss = state.epoch_train_loss
		m_start = float(state.extras.get('epoch_start_moco_m', method.current_m))
		print(f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}')

		if log_file is not None:
			with open(log_file, 'a', encoding='utf-8') as f:
				f.write(f'{epoch},{state.current_lr:.6f},{train_loss:.4f},{m_start:.4f}\n')

		checkpoint = {
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'scaler': trainer.scaler.state_dict(),
			'epoch': state.epoch,
			'engine': engine,
		}
		if (save_freq > 0 and epoch % save_freq == 0) or epoch == epochs:
			torch.save(checkpoint, output_path / f'checkpoint_{epoch:04d}.pth')
			print(f'Saved checkpoint at epoch {epoch}')
		torch.save(checkpoint, output_path / 'checkpoint_latest.pth')

	trainer.on_epoch_end(on_epoch_end)

	# 检查点状态恢复与引擎版本校验
	start_epoch = 0
	if training_cfg.resume is not None:
		checkpoint = torch.load(training_cfg.resume, map_location='cpu')
		ckpt_engine = checkpoint.get('engine')
		if ckpt_engine is not None and ckpt_engine != engine:
			raise ValueError(
				f'Checkpoint engine {ckpt_engine} does not match task engine {engine}'
			)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		trainer.scaler.load_state_dict(checkpoint['scaler'])
		start_epoch = checkpoint['epoch'] + 1
		print(f'Resumed from epoch {start_epoch}')

	if start_epoch >= epochs:
		print_task_completed(output_path)
		return output_path

	print('Ready to train...')
	print('\nStarting MoCo pretraining...')
	method.fit(loader, epochs, start_epoch=start_epoch)

	print_task_completed(output_path)
	return output_path
