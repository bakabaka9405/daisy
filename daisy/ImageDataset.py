import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy
from util import DecodeImageFromFile


class ImageDataset(Dataset):
	prefetchLevel: int
	device: torch.device | None
	labels: list[int]
	img_paths: list[str]
	images: list[Tensor]

	def __init__(self, prefetchLevel: int = 1, device: torch.device | None = None) -> None:
		self.prefetchLevel = prefetchLevel
		self.device = device
		self.labels = []
		self.images = []
		self.img_paths = []

		if prefetchLevel == 2 and device is None:
			raise ValueError('device must be specified when prefetchLevel is 2')
		if prefetchLevel != 2 and device is not None:
			raise ValueError('device must be None when prefetchLevel is not 2')

	def __len__(self) -> int:
		return len(self.labels)

	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		img: Tensor
		if self.prefetchLevel > 0:
			img = self.images[index]
		else:
			img = DecodeImageFromFile(self.img_paths[index], self.device)
		return img, self.labels[index]

	def append(self, label: int, path: str):
		self.labels.append(label)
		self.img_paths.append(path)
		if self.prefetchLevel > 0:
			self.images.append(DecodeImageFromFile(path, self.device))

	def split(
		self,
		*,
		ratio: float | None = None,
		index: list[int] | numpy.ndarray | None = None,
	) -> tuple['ImageDataset', 'ImageDataset']:
		if (ratio is None) ^ (index is None):
			raise ValueError('Either ratio or index must be specified')

		cnt = len(self)
		res1 = ImageDataset(prefetchLevel=self.prefetchLevel, device=self.device)
		res2 = ImageDataset(prefetchLevel=self.prefetchLevel, device=self.device)

		if ratio is not None:
			index = numpy.random.permutation(cnt)[: int(cnt * ratio)]

		assert index is not None

		mask = [0] * cnt
		for i in index:
			mask[i] = 1
		for i in range(cnt):
			if mask[i]:
				res1.img_paths.append(self.img_paths[i])
				res1.labels.append(self.labels[i])
				if self.prefetchLevel > 0:
					res1.images.append(self.images[i])
			else:
				res2.img_paths.append(self.img_paths[i])
				res2.labels.append(self.labels[i])
				if self.prefetchLevel > 0:
					res2.images.append(self.images[i])
		return res1, res2
