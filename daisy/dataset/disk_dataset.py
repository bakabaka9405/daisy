from collections.abc import Callable
from torch import Tensor
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from typing import Literal, Any
from PIL import Image

from .index_dataset import IndexDataset


class DiskDataset[LabelT](IndexDataset[LabelT]):
	file_paths: list[Path]
	labels: list[LabelT]
	transform: Callable[..., Any] | None
	backend: Literal['pil', 'tensor']

	def __init__(
		self,
		file_paths: list[Path],
		labels: list[LabelT],
		transform: Callable[..., Any] | None = None,
		backend: Literal['pil', 'tensor'] = 'tensor',
	):
		self.file_paths = file_paths
		self.labels = labels
		self.transform = transform
		self.backend = backend

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> tuple[Any, LabelT]:
		if self.backend == 'pil':
			img = Image.open(self.file_paths[index])
		else:
			img = decode_image(str(self.file_paths[index]), ImageReadMode.RGB)
		label = self.labels[index]

		if self.transform:
			img = self.transform(img)

		return img, label

	def getRawData(self) -> tuple[list, list[LabelT]]:
		return self.file_paths, self.labels

	def setTransform(self, transform: Callable[..., Any]) -> None:
		self.transform = transform

	def applyTransform(self, transform: Callable[..., Any]) -> None:
		self.transform = transform

	def take(self, k: int) -> 'DiskDataset[LabelT]':
		return DiskDataset(self.file_paths[:k], self.labels[:k], self.transform)

	def subset(self, indices: list[int]) -> 'DiskDataset[LabelT]':
		sub_file_paths = [self.file_paths[i] for i in indices]
		sub_labels = [self.labels[i] for i in indices]
		return DiskDataset(sub_file_paths, sub_labels, self.transform, self.backend)
