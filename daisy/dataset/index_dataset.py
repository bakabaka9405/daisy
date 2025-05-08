from torch import Tensor
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import Any


class IndexDataset(Dataset):
	@abstractmethod
	def __init__(
		self,
		data: list,
		labels: list,
		**kwargs,
	):
		pass

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		pass

	@abstractmethod
	def getRawData(self) -> tuple[list, list[int]]:
		pass

	@abstractmethod
	def setTransform(self, transform: Any) -> None:
		pass

	@abstractmethod
	def applyTransform(self, transform: Any) -> None:
		pass
