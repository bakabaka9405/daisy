from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision.transforms import v2 as transforms, InterpolationMode
import torch.nn.functional as F


@dataclass(slots=True, frozen=True)
class LazyNormalization:
	"""把张量包装为 LazyNormalizedTensor，归一化由 Prefetcher 在设备上执行。"""

	mean: tuple[float, ...]
	std: tuple[float, ...]

	def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
		object.__setattr__(self, 'mean', tuple(mean))
		object.__setattr__(self, 'std', tuple(std))

	def __call__(self, tensor: Tensor) -> 'LazyNormalizedTensor':
		return LazyNormalizedTensor(tensor, self)


@dataclass(slots=True, frozen=True)
class LazyNormalizedTensor:
	"""携带归一化参数的张量；同一 batch 内合并要求参数一致。"""

	tensor: Tensor
	policy: LazyNormalization

	def pin_memory(self) -> 'LazyNormalizedTensor':
		return LazyNormalizedTensor(self.tensor.pin_memory(), self.policy)


def _collate_lazy_normalized(
	batch: list[LazyNormalizedTensor],
	*,
	collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None,
) -> LazyNormalizedTensor:
	policy = batch[0].policy
	if any(item.policy != policy for item in batch):
		raise ValueError('同一 batch 内的 LazyNormalization 策略必须一致')
	collated = collate([item.tensor for item in batch], collate_fn_map=collate_fn_map)
	return LazyNormalizedTensor(collated, policy)


default_collate_fn_map[LazyNormalizedTensor] = _collate_lazy_normalized


class ZeroOneNormalize:
	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


class ResizeAndPad:
	size: int

	def __init__(self, size: int):
		self.size = size

	def __call__(self, tensor: torch.Tensor):
		# 确保张量是浮点类型，避免F.interpolate出错
		if tensor.dtype != torch.float32:
			tensor = tensor.float()

		# 获取图像尺寸 (C, H, W)
		c, h, w = tensor.shape

		# 确定长边并计算新尺寸
		if h > w:
			new_h = self.size
			new_w = int(w * (self.size / h))
		else:
			new_w = self.size
			new_h = int(h * (self.size / w))

		# 调整图像大小，保持长宽比
		resized = F.interpolate(
			tensor.unsqueeze(0),
			size=(new_h, new_w),
			mode='bilinear',
			align_corners=False,
		).squeeze(0)

		return torch.nn.functional.pad(
			resized,
			(
				(self.size - new_w + 1) // 2,
				(self.size - new_w) // 2,
				(self.size - new_h + 1) // 2,
				(self.size - new_h) // 2,
			),
		)


def get_default_train_transform():
	return transforms.Compose(
		[
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			ResizeAndPad(224),
			# transforms.CenterCrop(224),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_default_val_transform():
	return transforms.Compose(
		[
			transforms.Resize(224),
			transforms.CenterCrop(224),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_center_crop_train_transform():
	return transforms.Compose(
		[
			transforms.Resize(224),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			transforms.RandomHorizontalFlip(),
			transforms.CenterCrop(224),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_center_crop_val_transform():
	return transforms.Compose(
		[
			transforms.Resize(224),
			transforms.CenterCrop(224),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_pad_train_transform():
	return transforms.Compose(
		[
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			ResizeAndPad(224),
			# transforms.CenterCrop(224),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_pad_val_transform():
	return transforms.Compose(
		[
			ResizeAndPad(224),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_stretch_train_transform():
	return transforms.Compose(
		[
			transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(1.9, 2.1)),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			transforms.RandomHorizontalFlip(),
			LazyNormalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_stretch_train_transform_slight():
	return transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)


def get_stretch_linprobe_transform():
	return transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)


def get_stretch_val_transform():
	return transforms.Compose(
		[
			transforms.Resize((224, 224)),
			LazyNormalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# ZeroOneNormalize(),
			# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)


def get_rectangle_train_transform_slight():
	return transforms.Compose(
		[
			transforms.RandomResizedCrop((112, 224), scale=(0.9, 1.0), ratio=(1.9, 2.1)),
			transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)


def get_rectangle_train_transform():
	return transforms.Compose(
		[
			transforms.RandomZoomOut(),
			transforms.RandomRotation(degrees=(-10, 10), interpolation=InterpolationMode.BILINEAR),
			transforms.RandomResizedCrop((112, 224), scale=(0.9, 1.0), ratio=(1.9, 2.1)),
			transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_rectangle_val_transform():
	return transforms.Compose(
		[
			transforms.Resize((112, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_randaug_train_transform(
	input_size: int | tuple[int, int] = (224, 224),
	pad_size: int | Sequence[int] | None = None,
):
	if isinstance(input_size, int):
		resize_size = input_size * 256 // 224
	else:
		resize_size = input_size[0] * 256 // 224, input_size[1] * 256 // 224

	trans = []

	if pad_size is not None:
		trans += [transforms.Pad(pad_size)]

	trans += [
		transforms.RandAugment(2, interpolation=InterpolationMode.BILINEAR),
		# transforms.Resize(resize_size),
		# transforms.CenterCrop(input_size),
		transforms.Resize(input_size),
		ZeroOneNormalize(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	]

	return transforms.Compose(trans)


def get_autoaug_train_transform(
	input_size: int | tuple[int, int] = (224, 224),
	pad_size: int | Sequence[int] | None = None,
):
	if isinstance(input_size, int):
		resize_size = input_size * 256 // 224
	else:
		resize_size = input_size[0] * 256 // 224, input_size[1] * 256 // 224

	trans = []

	if pad_size is not None:
		trans += [transforms.Pad(pad_size)]

	trans += [
		transforms.AutoAugment(interpolation=InterpolationMode.BILINEAR),
		transforms.Resize(resize_size),
		transforms.CenterCrop(input_size),
		ZeroOneNormalize(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	]

	return transforms.Compose(trans)


def get_one_by_one_transform():
	return transforms.Compose(
		[
			transforms.Resize((224, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)
