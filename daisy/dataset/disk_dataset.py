from torch import Tensor
from torchvision.transforms import v2 as transforms
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path

import albumentations as A
import numpy as np
import cv2

from .index_dataset import IndexDataset


class DiskDataset(IndexDataset):
	def __init__(
		self,
		file_paths: list[Path],
		labels: list[int],
		transform: transforms.Compose | None = None,
	):
		self.file_paths = file_paths
		self.labels = labels
		self.transform = transform

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		tensor = decode_image(str(self.file_paths[index]), ImageReadMode.RGB)
		label = self.labels[index]

		if self.transform:
			tensor = self.transform(tensor)

		return tensor, label

	def getRawData(self) -> tuple[list, list[int]]:
		return self.file_paths, self.labels

	def setTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform

	def applyTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform


class DiskDatasetA(IndexDataset):
	def __init__(
		self,
		file_paths: list[Path],
		labels: list[int],
		transform: transforms.Compose | None = None,
	):
		self.file_paths = file_paths
		self.labels = labels
		self.transform = transform

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		img = cv2.imdecode(np.fromfile(self.file_paths[index], dtype=np.uint8), cv2.IMREAD_COLOR_BGR)
		label = self.labels[index]

		assert self.transform is not None, 'Transform must be set before applying.'
		img = self.transform(image=img)['image']

		return img, label

	def getRawData(self) -> tuple[list, list[int]]:
		return self.file_paths, self.labels

	def setTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform

	def applyTransform(self, transform: transforms.Compose) -> None:
		self.transform = transform
