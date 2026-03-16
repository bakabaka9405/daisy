from torch import Tensor
from torchvision.transforms import v2 as transforms
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from typing import Literal, Any
from PIL import Image

from .index_dataset import IndexDataset


class DiskDataset(IndexDataset):
	def __init__(
		self,
		file_paths: list[Path],
		labels: list[int],
		transform: transforms.Compose | None = None,
		backend: Literal['pil', 'tensor'] = 'tensor',
	):
		self.file_paths = file_paths
		self.labels = labels
		self.transform = transform
		self.backend = backend

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> tuple[Any, int]:
		if self.backend == 'pil':
			img = Image.open(self.file_paths[index])
		else:
			img = decode_image(str(self.file_paths[index]), ImageReadMode.RGB)
		label = self.labels[index]

		if self.transform:
			img = self.transform(img)

		return img, label

	def getRawData(self) -> tuple[list, list[int]]:
		return self.file_paths, self.labels

	def setTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform

	def applyTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform

	def take(self, k: int) -> 'DiskDataset':
		return DiskDataset(self.file_paths[:k], self.labels[:k], self.transform)
