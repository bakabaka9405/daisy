from collections.abc import Callable

from torch import Tensor
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from typing import cast

from .index_dataset import IndexDataset


class MemoryDataset[ImageT, LabelT](IndexDataset[ImageT, LabelT]):
	tensors: list[Tensor]
	labels: list[LabelT] | None
	transform: Callable[..., ImageT]

	def __init__(
		self,
		data: list[Path] | list[Tensor],
		labels: list[LabelT] | None,
		transform: Callable[..., ImageT],
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

		if labels is not None and len(labels) != len(self.tensors):
			raise ValueError('labels length must match data length')
		self.labels = labels
		self.transform = transform

	def __len__(self) -> int:
		return len(self.tensors)

	def __getitem__(self, index: int) -> ImageT | tuple[ImageT, LabelT]:
		tensor = self.tensors[index]

		# 无标签时仅返回图像，有标签时返回 (图像, 标签)
		if self.labels is None:
			return self.transform(tensor)
		return (self.transform(tensor), self.labels[index])

	def getRawData(self) -> tuple[list[Tensor], list[LabelT] | None]:
		return self.tensors, self.labels

	def setTransform(self, transform: Callable[..., ImageT]) -> None:
		self.transform = transform

	def applyTransform(self, transform: Callable[[Tensor], Tensor]) -> None:
		self.tensors = [transform(tensor) for tensor in self.tensors]

	def take(self, k: int) -> 'MemoryDataset[ImageT, LabelT]':
		labels = None if self.labels is None else self.labels[:k]
		return MemoryDataset(self.tensors[:k], labels, self.transform)

	def subset(self, indices: list[int]) -> 'MemoryDataset[ImageT, LabelT]':
		labels = None if self.labels is None else [self.labels[i] for i in indices]
		return MemoryDataset(
			[self.tensors[i] for i in indices],
			labels,
			self.transform,
		)
