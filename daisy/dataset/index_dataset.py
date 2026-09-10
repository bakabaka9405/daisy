from collections.abc import Callable, Sequence
from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import Any, Self


class IndexDataset[ImageT, LabelT](Dataset[ImageT | tuple[ImageT, LabelT]]):
	"""索引数据集基类：图像样本可选携带标签。"""

	transform: Callable[..., ImageT]

	@abstractmethod
	def __init__(self, *args: Any, **kwargs: Any) -> None:
		pass

	@abstractmethod
	def __len__(self) -> int:
		pass

	@abstractmethod
	def __getitem__(self, index: int) -> ImageT | tuple[ImageT, LabelT]:
		pass

	@abstractmethod
	def getRawData(self) -> tuple[Sequence[Path | Tensor], list[LabelT] | None]:
		pass

	@abstractmethod
	def setTransform(self, transform: Callable[..., ImageT]) -> None:
		pass

	@abstractmethod
	def applyTransform(self, transform: Callable[..., Any]) -> None:
		"""就地应用 transform：Disk 更新图像变换，Memory 直接变换底层张量。"""
		pass

	@abstractmethod
	def take(self, k: int) -> Self:
		pass

	@abstractmethod
	def subset(self, indices: list[int]) -> Self:
		pass
