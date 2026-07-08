from torch import Tensor
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import Any, Generic, TypeVar

LabelT = TypeVar('LabelT')
T = TypeVar('T', bound='IndexDataset')


class IndexDataset(Dataset, Generic[LabelT]):
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
	def take(self: T, k: int) -> T:
		pass

	@abstractmethod
	def subset(self: T, indices: list[int]) -> T:
		pass
