from torch import Tensor
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from .index_dataset import IndexDataset
from typing import cast


class MemoryDataset(IndexDataset):
	tensors: list[Tensor]
	labels: list[int]
	transform: transforms.Compose | None

	def __init__(
		self,
		data: list[Path] | list[Tensor],
		labels: list[int],
		transform: transforms.Compose | None = None,
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

	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		tensor = self.tensors[index]
		label = self.labels[index]

		if self.transform:
			tensor = self.transform(tensor)

		return tensor, label

	def getRawData(self) -> tuple[list, list[int]]:
		return self.tensors, self.labels

	def setTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform

	def applyTransform(self, transform: transforms.Compose) -> None:
		self.tensors = [transform(tensor) for tensor in self.tensors]

	def take(self, k: int) -> 'MemoryDataset':
		return MemoryDataset(self.tensors[:k], self.labels[:k], self.transform)
