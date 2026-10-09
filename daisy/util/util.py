from pathlib import Path
import random

import icontract
import torch
import numpy
from torch import Tensor
from torchvision.io.image import decode_image, ImageReadMode
from typing import Any, SupportsIndex
from collections.abc import Iterable


def to_2tuple[T](value: T | tuple[T, T]) -> tuple[T, T]:
	if isinstance(value, tuple):
		return value
	return (value, value)


def _has_classifier(model: Any) -> bool:
	return hasattr(model, 'fc') or hasattr(model, 'head') or hasattr(model, 'classifier')


@icontract.require(lambda seed: 0 <= seed < 2**32)
def set_global_seed(seed: int):
	random.seed(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def enable_cudnn_benchmark():
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = False


@icontract.require(lambda blr: blr is None or blr > 0)
@icontract.require(lambda lr: lr is None or lr > 0)
@icontract.require(lambda lr, blr: lr is not None or blr is not None, 'Either lr or blr must be set')
@icontract.require(lambda accum_iter: accum_iter >= 1)
@icontract.require(lambda batch_size: batch_size >= 1)
def resolve_lr(batch_size: int, accum_iter: int, *, lr: float | None = None, blr: float | None = None) -> float:
	"""解析实际学习率：优先返回显式 lr，否则按 blr * 有效批大小 / 256 换算。"""
	if lr is not None:
		return lr
	assert blr is not None
	return blr * batch_size * accum_iter / 256.0


def DecodeImageFromFile(path: str, device: torch.device | None = None) -> Tensor:
	res = decode_image(path, ImageReadMode.RGB)
	if device is not None:
		res = res.to(device)
	return res


class ZeroOneNormalize:
	def __call__(self, tensor: torch.Tensor):
		return tensor.float().div(255)


@icontract.require(lambda model: _has_classifier(model), 'Invalid model type')
def get_model_classifier(model):
	if hasattr(model, 'fc'):
		return model.fc
	elif hasattr(model, 'head'):
		if not hasattr(model.head, 'fc'):
			return model.head
		else:
			return model.head.fc
	else:
		return model.classifier


@icontract.require(lambda identity, num_classes: not identity or num_classes == 0, 'num_classes must be 0 if identity is True')
@icontract.require(lambda identity, num_classes: identity or num_classes != 0, 'num_classes must be specified if identity is False')
@icontract.require(lambda num_classes: num_classes >= 0)
@icontract.require(lambda model: _has_classifier(model), 'Invalid model type')
def change_model_classifier(model, *, identity: bool = False, num_classes: int = 0):
	# Change the output layer to match the number of classes
	# The output layer is either model.fc or model.head.fc or model.classifier
	# Differ between models created by torchvision or timm

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
		else:
			assert isinstance(model.classifier, torch.nn.Linear)
			model.classifier = torch.nn.Identity().to(model.classifier.weight.device)

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
		else:
			assert isinstance(model.classifier, torch.nn.Linear)
			model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes).to(model.classifier.weight.device)


def freeze_model(model):
	for param in model.parameters():
		param.requires_grad = False


def unfreeze_model(model):
	for param in model.parameters():
		param.requires_grad = True


def gather_list_by_indexes[T](lst: list[T], index: Iterable[SupportsIndex]) -> list[T]:
	return [lst[i] for i in index]


# 可变位置参数在契约中通过 icontract 提供的 _ARGS（全部位置参数元组）访问
@icontract.require(lambda _ARGS: all(len(i) == len(_ARGS[0]) for i in _ARGS), '数组长度不相等')
@icontract.require(lambda _ARGS: len(_ARGS) > 0, '打乱至少需要一个参数')
def shuffle_correlated_lists(*args: list):
	length = len(args[0])

	idx = numpy.random.permutation(length)

	for i in args:
		tmp = gather_list_by_indexes(i, idx)
		for j in range(length):
			i[j] = tmp[j]


@icontract.require(lambda val: val >= 0)
@icontract.require(lambda files, labels: len(files) == len(labels))
def copy_by_label(files: list, labels: list[int], label: int, val: int) -> None:
	mask = [1 if i == label else 0 for i in labels]
	copied_files = [i for i, j in zip(files, mask) if j == 1]

	files += copied_files * val
	labels += [label] * (len(copied_files) * val)


@icontract.require(lambda labels, _ARGS: all(len(i) == len(labels) for i in _ARGS[2:]))
def extract_by_label(labels: list[int], target: int, *args: list):
	return ([j for j, k in zip(i, labels) if k == target] for i in args)


@icontract.require(lambda labels, _ARGS: all(len(i) == len(labels) for i in _ARGS[2:]))
def exclude_by_label(labels: list[int], target: int, *args: list):
	return ([j for j, k in zip(i, labels) if k != target] for i in args)


@icontract.require(lambda labels, _ARGS: all(len(i) == len(labels) for i in _ARGS[2:]))
def filter_by_label(labels: list[int], cond: Any, *args: list):
	return ([j for j, k in zip(i, labels) if cond(k)] for i in args)


if __name__ == '__main__':
	files = [1, 2, 3, 4]
	labels = [1, 1, 2, 2]

	a, b = extract_by_label(labels, 1, files, ['asdf', 666, 0, 1.2])
	print(a)
	print(b)
