from collections.abc import Callable
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import Any, Self


class IndexDataset[SampleT](Dataset[SampleT]):
	transform: Callable[..., Any]

	@abstractmethod
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		pass

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def __getitem__(self, index: int) -> SampleT:
		pass

	@abstractmethod
	def getRawData(self) -> Any:
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
