from collections.abc import Callable

from torch import Tensor
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from typing import Any, cast, Literal

from .index_dataset import IndexDataset


class MemoryDataset[LabelT](IndexDataset[tuple[Tensor, LabelT]]):
	tensors: list[Tensor]
	labels: list[LabelT]
	transform: Callable[..., Any]

	def __init__(
		self,
		data: list[Path] | list[Tensor],
		labels: list[LabelT],
		transform: Callable[..., Any],
		backend: Literal['pil', 'tensor'] = 'tensor',
	):
		if len(data) == 0:
			self.tensors = []
		elif isinstance(data[0], Path):
			if any(not isinstance(i, Path) for i in data):
				raise ValueError('data should all either be Path or Tensor')
			self.tensors = [
				decode_image(
					str(path),
					ImageReadMode.RGB,
				)
				for path in data
			]
		else:
			if any(not isinstance(i, Tensor) for i in data):
				raise ValueError('data should all either be Path or Tensor')
			self.tensors = cast(list[Tensor], data)
		self.labels = labels
		self.transform = transform

	def __len__(self) -> int:
		return len(self.tensors)

	def __getitem__(self, index: int) -> tuple[Tensor, LabelT]:
		tensor = self.tensors[index]
		label = self.labels[index]
		return self.transform(tensor), label

	def getRawData(self) -> tuple[list[Tensor], list[LabelT]]:
		return self.tensors, self.labels

	def setTransform(self, transform: Callable[..., Any]) -> None:
		self.transform = transform

	def applyTransform(self, transform: Callable[..., Any]) -> None:
		self.tensors = [transform(tensor) for tensor in self.tensors]

	def take(self, k: int) -> 'MemoryDataset[LabelT]':
		return MemoryDataset(self.tensors[:k], self.labels[:k], self.transform)

	def subset(self, indices: list[int]) -> 'MemoryDataset[LabelT]':
		return MemoryDataset(
			[self.tensors[i] for i in indices],
			[self.labels[i] for i in indices],
			self.transform,
		)
