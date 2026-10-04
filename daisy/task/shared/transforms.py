"""Task 共享 transform 构建器"""

from __future__ import annotations

import icontract
from timm.data.transforms_factory import create_transform
from torchvision.transforms import InterpolationMode, v2 as transforms

import daisy
from daisy.util.transform import ZeroOneNormalize
from typing import Any, Callable, Literal, cast


def _is_positive_size(size: int | tuple[int, int]) -> bool:
	"""图像尺寸（整数或 (H, W)）各维为正"""
	return all(s > 0 for s in (size if isinstance(size, tuple) else (size,)))


@icontract.require(
	lambda name: name in {'rectangle_train', 'rectangle_val', 'rectangle_train_slight', 'stretch_train', 'stretch_val'}, 'Unknown transform'
)
def get_classification_transform(name: str):
	"""根据名称获取分类任务 transform"""
	transform_map = {
		'rectangle_train': daisy.util.transform.get_rectangle_train_transform,
		'rectangle_val': daisy.util.transform.get_rectangle_val_transform,
		'rectangle_train_slight': daisy.util.transform.get_rectangle_train_transform_slight,
		'stretch_train': daisy.util.transform.get_stretch_train_transform,
		'stretch_val': daisy.util.transform.get_stretch_val_transform,
	}
	return transform_map[name]()


@icontract.require(lambda remode: remode.lower() in {'', 'const', 'rand', 'pixel'})
@icontract.require(lambda reprob: 0 <= reprob <= 1)
@icontract.require(lambda input_size: input_size > 0)
def get_mae_finetune_train_transform(
	input_size: int = 224,
	aa: str = 'rand-m9-mstd0.5-inc1',
	reprob: float = 0.25,
	remode: str = 'pixel',
	recount: int = 1,
):
	"""获取 MAE Finetune 训练 transform（输入 Tensor，输出归一化后的训练 Tensor）"""
	timm_transform = create_transform(
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
	return transforms.Compose([transforms.ToPILImage(), cast(Callable[..., Any], timm_transform)])


@icontract.require(lambda input_size: _is_positive_size(input_size))
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
