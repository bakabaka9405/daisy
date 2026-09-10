from collections.abc import Callable
from torchvision.io import decode_image, ImageReadMode
from pathlib import Path
from typing import Literal
from PIL import Image

from .index_dataset import IndexDataset


class DiskDataset[ImageT, LabelT](IndexDataset[ImageT, LabelT]):
	"""磁盘图像数据集，可选携带标签"""

	file_paths: list[Path]
	labels: list[LabelT] | None
	transform: Callable[..., ImageT]
	backend: Literal['pil', 'tensor']

	def __init__(
		self,
		file_paths: list[Path],
		labels: list[LabelT] | None,
		transform: Callable[..., ImageT],
		*,
		backend: Literal['pil', 'tensor'] = 'tensor',
	):
		if labels is not None and len(labels) != len(file_paths):
			raise ValueError('labels length must match file_paths length')
		self.file_paths = file_paths
		self.labels = labels
		self.transform = transform
		self.backend = backend

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> ImageT | tuple[ImageT, LabelT]:
		# 按 backend 解码图像
		if self.backend == 'pil':
			img = Image.open(self.file_paths[index])
		else:
			img = decode_image(str(self.file_paths[index]), ImageReadMode.RGB)

		# 无标签时仅返回图像，有标签时返回 (图像, 标签)
		if self.labels is None:
			return self.transform(img)
		return (self.transform(img), self.labels[index])

	def getRawData(self) -> tuple[list[Path], list[LabelT] | None]:
		return self.file_paths, self.labels

	def setTransform(self, transform: Callable[..., ImageT]) -> None:
		self.transform = transform

	def applyTransform(self, transform: Callable[..., ImageT]) -> None:
		self.transform = transform

	def take(self, k: int) -> 'DiskDataset[ImageT, LabelT]':
		labels = None if self.labels is None else self.labels[:k]
		return DiskDataset(self.file_paths[:k], labels, self.transform, backend=self.backend)

	def subset(self, indices: list[int]) -> 'DiskDataset[ImageT, LabelT]':
		sub_file_paths = [self.file_paths[i] for i in indices]
		sub_labels = None if self.labels is None else [self.labels[i] for i in indices]
		return DiskDataset(sub_file_paths, sub_labels, self.transform, backend=self.backend)


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
