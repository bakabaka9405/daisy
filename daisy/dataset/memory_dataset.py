from collections.abc import Callable

import icontract
from torch import Tensor
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from typing import cast

from .index_dataset import IndexDataset


class MemoryDataset[ImageT, LabelT](IndexDataset[ImageT, LabelT]):
	tensors: list[Tensor]
	labels: list[LabelT] | None
	transform: Callable[..., ImageT]

	@icontract.require(lambda data, labels: labels is None or len(labels) == len(data), 'labels length must match data length')
	@icontract.require(
		lambda data: all(isinstance(i, Path) for i in data) or all(isinstance(i, Tensor) for i in data),
		'data should all either be Path or Tensor',
	)
	def __init__(
		self,
		data: list[Path] | list[Tensor],
		labels: list[LabelT] | None,
		transform: Callable[..., ImageT],
	):
		if len(data) == 0:
			self.tensors = []
		elif isinstance(data[0], Path):
			self.tensors = [
				decode_image(
					str(path),
					ImageReadMode.RGB,
				)
				for path in data
			]
		else:
			self.tensors = cast(list[Tensor], data)

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

	@icontract.require(lambda k: k >= 0)
	@icontract.ensure(lambda self, k, result: len(result) == min(k, len(self)))
	def take(self, k: int) -> 'MemoryDataset[ImageT, LabelT]':
		labels = None if self.labels is None else self.labels[:k]
		return MemoryDataset(self.tensors[:k], labels, self.transform)

	@icontract.require(lambda self, indices: all(0 <= i < len(self) for i in indices))
	@icontract.ensure(lambda indices, result: len(result) == len(indices))
	def subset(self, indices: list[int]) -> 'MemoryDataset[ImageT, LabelT]':
		labels = None if self.labels is None else [self.labels[i] for i in indices]
		return MemoryDataset(
			[self.tensors[i] for i in indices],
			labels,
			self.transform,
		)
