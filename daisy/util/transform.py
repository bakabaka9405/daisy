import torch
from torchvision.transforms import v2 as transforms
import torch.nn.functional as F


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
		resized = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

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


def get_rectangle_train_transform():
	return transforms.Compose(
		[
			transforms.Resize((112, 224)),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			transforms.RandomHorizontalFlip(),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)


def get_rectangle_val_transform():
	return transforms.Compose(
		[
			transforms.Resize((112, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			# transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
		]
	)
