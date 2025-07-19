import random
import torch
import numpy
from numpy.typing import NDArray
from torch import Tensor
from torchvision.io.image import decode_image, ImageReadMode
from torchvision import transforms
from typing import TypeVar
from typing import Any


def set_global_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	numpy.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def enable_cudnn_benchmark():
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False


def DecodeImageFromFile(path: str, device: torch.device | None = None) -> Tensor:
	res = decode_image(path, ImageReadMode.RGB)
	if device is not None:
		res = res.to(device)
	return res


class ZeroOneNormalize:
	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


def get_default_train_transform():
	return transforms.Compose(
		[
			transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
			transforms.Resize((224, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def get_default_val_transform():
	return transforms.Compose(
		[
			transforms.Resize((224, 224)),
			ZeroOneNormalize(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		]
	)


def GetModelClassifier(model):
	if hasattr(model, 'fc'):
		return model.fc
	elif hasattr(model, 'head'):
		if not hasattr(model.head, 'fc'):
			return model.head
		else:
			return model.head.fc
	elif hasattr(model, 'classifier'):
		return model.classifier
	else:
		raise ValueError('Invalid model type')


def change_model_classifier(model, *, identity: bool = False, num_classes: int = 0):
	# Change the output layer to match the number of classes
	# The output layer is either model.fc or model.head.fc or model.classifier
	# Differ between models created by torchvision or timm

	if not identity and num_classes == 0:
		raise ValueError('num_classes must be specified if identity is False')
	if identity and num_classes != 0:
		raise ValueError('num_classes must be 0 if identity is True')

	if identity:
		if hasattr(model, 'fc'):
			assert isinstance(model.fc, torch.nn.Linear)
			model.fc = torch.nn.Identity().to(model.fc.weight.device)
		elif hasattr(model, 'head'):
			if not hasattr(model.head, 'fc'):
				assert isinstance(model.head, torch.nn.Linear)
				model.head = torch.nn.Identity().to(model.head.weight.device)
			else:
				assert isinstance(model.head.fc, torch.nn.Linear)
				model.head.fc = torch.nn.Identity().to(model.head.fc.weight.device)
		elif hasattr(model, 'classifier'):
			assert isinstance(model.classifier, torch.nn.Linear)
			model.classifier = torch.nn.Identity().to(model.classifier.weight.device)
		else:
			raise ValueError('Invalid model type')

	else:
		if hasattr(model, 'fc'):
			assert isinstance(model.fc, torch.nn.Linear)
			model.fc = torch.nn.Linear(model.fc.in_features, num_classes).to(model.fc.weight.device)
		elif hasattr(model, 'head'):
			if not hasattr(model.head, 'fc'):
				assert isinstance(model.head, torch.nn.Linear)
				model.head = torch.nn.Linear(model.head.in_features, num_classes).to(model.head.weight.device)
			else:
				assert isinstance(model.head.fc, torch.nn.Linear)
				model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes).to(model.head.fc.weight.device)
		elif hasattr(model, 'classifier'):
			assert isinstance(model.classifier, torch.nn.Linear)
			model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes).to(model.classifier.weight.device)
		else:
			raise ValueError('Invalid model type')


def FreezeModel(model):
	for param in model.parameters():
		param.requires_grad = False


def UnfreezeModel(model):
	for param in model.parameters():
		param.requires_grad = True


T = TypeVar('T')


def gather_list_by_indexes(lst: list[T], index: list[int] | NDArray) -> list[T]:
	return [lst[i] for i in index]


def shuffle_correlated_lists(*args: list):
	if len(args) == 0:
		raise ValueError('打乱至少需要一个参数')

	if max(len(i) for i in args) != min(len(i) for i in args):
		raise ValueError('数组长度不相等')

	length = len(args[0])

	idx = numpy.random.permutation(length)

	for i in args:
		tmp = gather_list_by_indexes(i, idx)
		for j in range(length):
			i[j] = tmp[j]


def copy_by_label(files: list, labels: list[int], label: int, val: int) -> None:
	mask = [1 if i == label else 0 for i in labels]
	copied_files = [i for i, j in zip(files, mask) if j == 1]

	files += copied_files * val
	labels += [label] * (len(copied_files) * val)


if __name__ == '__main__':
	files = [1, 2, 3, 4]
	labels = [1, 1, 2, 2]

	copy_by_label(files, labels, 1, 2)

	print(files)
