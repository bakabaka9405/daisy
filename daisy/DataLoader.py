from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy
from collections.abc import Callable
import torch
from torch import Tensor


class DataLoader:
	dataset: Dataset
	dataset_size: int
	batch_size: int
	num_batch: int
	transform: Callable[[Tensor], Tensor]
	shuffled: bool
	trunc_end: bool
	current_batch: int
	indexes: list[int] | numpy.ndarray

	def __init__(self, dataset: Dataset, batch_size: int, transform: Compose | None = None, shuffled: bool = True, trunc_end: bool = True):
		self.dataset = dataset
		self.dataset_size = len(dataset)  # type: ignore
		self.batch_size = batch_size
		self.num_batch = self.dataset_size // batch_size if trunc_end else (self.dataset_size + batch_size - 1) // batch_size
		if transform is not None:
			self.transform = transform
		else:
			self.transform = lambda x: x
		self.shuffled = shuffled
		self.trunc_end = trunc_end
		self.current_batch = 0
		self.indexes = []
		self.reset()

	def reset(self):
		self.current_batch = 0
		if self.shuffled:
			self.indexes = numpy.random.permutation(self.dataset_size)
		else:
			self.indexes = [i for i in range(self.dataset_size)]

	def __iter__(self):
		while True:
			if self.current_batch == self.num_batch:
				self.reset()
				break
			data = [
				self.dataset[self.indexes[i]]
				for i in range(
					self.current_batch * self.batch_size,
					min(self.dataset_size, (self.current_batch + 1) * self.batch_size),
				)
			]
			yield torch.stack([self.transform(i[0]) for i in data], dim=0), torch.as_tensor([i[1] for i in data])
			self.current_batch += 1
