from torch import Tensor
from torchvision.transforms import v2 as transforms
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path

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
