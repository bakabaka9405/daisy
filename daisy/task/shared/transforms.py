"""Task 共享 transform 构建器"""

from __future__ import annotations

from timm.data.transforms_factory import create_transform
from torchvision.transforms import InterpolationMode, v2 as transforms

import daisy
from daisy.util.transform import ZeroOneNormalize
from typing import Literal


def get_classification_transform(name: str):
	"""根据名称获取分类任务 transform"""
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


def get_mae_finetune_train_transform(
	input_size: int = 224,
	aa: str = 'rand-m9-mstd0.5-inc1',
	reprob: float = 0.25,
	remode: str = 'pixel',
	recount: int = 1,
):
	"""获取 MAE Finetune 训练 transform"""
	return create_transform(
		input_size=input_size,
		is_training=True,
		color_jitter=0.0,
		auto_augment=aa,
		interpolation='bicubic',
		re_prob=reprob,
		re_mode=remode,
		re_count=recount,
		mean=(0.485, 0.456, 0.406),
		std=(0.229, 0.224, 0.225),
	)


def get_mae_finetune_val_transform(
	input_size: int | tuple[int, int] = 224,
	backend: Literal['pil', 'tensor'] = 'tensor',
):
	"""获取 MAE Finetune 验证 transform"""
	return transforms.Compose(
		[
			transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
			ZeroOneNormalize() if backend == 'tensor' else transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)
