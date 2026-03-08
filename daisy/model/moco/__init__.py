"""MoCo (Momentum Contrast) 模型模块

提供 MoCo v2 (queue-based) 和 MoCo v3 (queue-free, symmetrized) 模型。
"""

from .builder_v2 import MoCoV2
from .builder_v3 import MoCoV3, MoCoV3_ResNet, MoCoV3_ViT
from .vit import VisionTransformerMoCo, ConvStem, MOCO_VIT_MODELS
from .lars import LARS
from .transforms import TwoCropsTransform

import torch
import torch.nn as nn


def load_moco_pretrained_weights(model: nn.Module, checkpoint_path: str, engine: str = 'v3'):
	"""从 MoCo checkpoint 中提取 encoder 权重到标准模型

	Args:
		model: 目标模型 (标准 timm 模型)
		checkpoint_path: MoCo checkpoint 路径
		engine: 'v2' 或 'v3'
	"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu')
	state_dict = checkpoint.get('model', checkpoint)

	new_state_dict: dict[str, torch.Tensor] = {}

	if engine == 'v2':
		# v2: 提取 encoder_q.* (去掉 MLP head 部分)
		prefix = 'encoder_q.'
		for k, v in state_dict.items():
			if k.startswith(prefix):
				new_key = k[len(prefix):]
				# 跳过 MLP head (fc.0.*, fc.1.*, fc.2.* 等 Sequential 层)
				if new_key.startswith('fc.') and new_key.split('.')[1].isdigit():
					continue
				new_state_dict[new_key] = v
	elif engine == 'v3':
		# v3: 提取 base_encoder.* (去掉 projector/predictor)
		prefix = 'base_encoder.'
		for k, v in state_dict.items():
			if k.startswith(prefix):
				new_key = k[len(prefix):]
				# 跳过 projector head (fc.*, head.* 中的 Sequential 层)
				parts = new_key.split('.')
				if len(parts) >= 2 and parts[1].isdigit():
					# 这是 Sequential MLP 层，跳过
					if parts[0] in ('fc', 'head'):
						continue
				new_state_dict[new_key] = v
	else:
		raise ValueError(f'Unknown engine: {engine}')

	msg = model.load_state_dict(new_state_dict, strict=False)
	print(f'Loaded MoCo {engine} pretrained weights from {checkpoint_path}')
	if msg.missing_keys:
		print(f'  Missing keys: {msg.missing_keys}')
	if msg.unexpected_keys:
		print(f'  Unexpected keys: {msg.unexpected_keys}')


__all__ = [
	'MoCoV2',
	'MoCoV3',
	'MoCoV3_ResNet',
	'MoCoV3_ViT',
	'VisionTransformerMoCo',
	'ConvStem',
	'MOCO_VIT_MODELS',
	'LARS',
	'TwoCropsTransform',
	'load_moco_pretrained_weights',
]
