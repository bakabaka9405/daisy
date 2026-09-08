from torch import Tensor
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import Any, Self


class IndexDataset[LabelT](Dataset):
	@abstractmethod
	def __init__(
		self,
		data: list,
		labels: list[LabelT],
		**kwargs,
	):
		pass

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def __getitem__(self, index: int) -> tuple[Tensor, LabelT]:
		pass

	@abstractmethod
	def getRawData(self) -> tuple[list, list[LabelT]]:
		pass

	@abstractmethod
	def setTransform(self, transform: Any) -> None:
		pass

	@abstractmethod
	def applyTransform(self, transform: Any) -> None:
		pass

	@abstractmethod
	def take(self, k: int) -> Self:
		pass

	@abstractmethod
	def subset(self, indices: list[int]) -> Self:
		pass
