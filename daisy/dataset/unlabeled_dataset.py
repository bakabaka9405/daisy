"""无标签数据集 (用于自监督学习)"""

from collections.abc import Callable
from pathlib import Path
from typing import Self

from torch import Tensor
from torchvision.io import decode_image, ImageReadMode

from .index_dataset import IndexDataset


class UnlabeledDiskDataset[SampleT](IndexDataset[SampleT]):
	"""无标签数据集，用于 MAE、MoCo 等自监督学习任务。"""

	file_paths: list[Path]
	transform: Callable[[Tensor], SampleT]

	def __init__(
		self,
		file_paths: list[Path],
		transform: Callable[[Tensor], SampleT],
	):
		self.file_paths = file_paths
		self.transform = transform

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> SampleT:
		return self.transform(decode_image(str(self.file_paths[index]), ImageReadMode.RGB))

	def getRawData(self) -> list[Path]:
		return self.file_paths

	def setTransform(self, transform: Callable[[Tensor], SampleT]) -> None:
		self.transform = transform

	def applyTransform(self, transform: Callable[[Tensor], SampleT]) -> None:
		self.transform = transform

	def take(self, k: int) -> Self:
		return type(self)(self.file_paths[:k], self.transform)

	def subset(self, indices: list[int]) -> Self:
		return type(self)([self.file_paths[i] for i in indices], self.transform)


def load_files_from_folder(root: Path | str, extensions: tuple[str, ...] | None = None) -> list[Path]:
	"""从文件夹递归加载所有图像文件

	Args:
		root: 数据根目录
		extensions: 允许的文件扩展名，默认为常见图像格式

	Returns:
		所有图像文件的路径列表
	"""
	if isinstance(root, str):
		root = Path(root)

	if extensions is None:
		extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif')

	files = []
	for ext in extensions:
		files.extend(root.rglob(f'*{ext}'))
		# files.extend(root.rglob(f'*{ext.upper()}'))

	return sorted(files)
