"""无标签数据集 (用于自监督学习)"""

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
from torchvision.io import decode_image, ImageReadMode


class UnlabeledDiskDataset(Dataset):
	"""无标签数据集，用于 MAE 等自监督学习任务

	仅加载图像，不处理标签。为兼容 DataLoader 返回虚拟标签 0。
	"""

	def __init__(
		self,
		file_paths: list[Path],
		transform: transforms.Compose | None = None,
	):
		self.file_paths = file_paths
		self.transform = transform

	def __len__(self) -> int:
		return len(self.file_paths)

	def __getitem__(self, index: int) -> tuple[Tensor, int]:
		tensor = decode_image(str(self.file_paths[index]), ImageReadMode.RGB)

		if self.transform:
			tensor = self.transform(tensor)

		return tensor, 0

	def set_transform(self, transform: transforms.Compose) -> None:
		"""设置 transform"""
		self.transform = transform


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
		files.extend(root.rglob(f'*{ext.upper()}'))

	return sorted(files)
