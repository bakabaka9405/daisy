"""无竞态的 CUDA 数据预取器。"""

from collections.abc import Iterator
from typing import Any

import torch
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import DataLoader, Dataset
from torch.utils._pytree import tree_leaves, tree_map
from torchvision.transforms.v2 import functional as F

from .transform import LazyNormalizedTensor


class Prefetcher:
	"""在独立 CUDA 流上搬运 batch，对 LazyNormalizedTensor 做设备端归一化。

	普通 Tensor 只搬设备并保留 dtype 与数值；标记张量在设备上转 float32 后归一化，
	uint8 缩放到 [0, 1]，float 输入保持原值。pytree 容器与非 Tensor 元数据原样保留，
	输出中的标记已消费为普通 Tensor。CUDA 路径对预取张量调用 record_stream，避免
	缓存分配器在消费流读取完成前复用其显存。
	"""

	def __init__(self, loader: DataLoader, device: torch.device):
		self.loader = loader
		self.device = device

	def __len__(self) -> int:
		return len(self.loader)

	def _map_leaf(self, leaf: Any) -> Any:
		if isinstance(leaf, LazyNormalizedTensor):
			tensor = leaf.tensor.to(self.device, non_blocking=True)
			tensor = F.to_dtype(tensor, dtype=torch.float32, scale=True)
			return F.normalize(tensor, list(leaf.policy.mean), list(leaf.policy.std))
		if isinstance(leaf, torch.Tensor):
			return leaf.to(self.device, non_blocking=True)
		return leaf

	def __iter__(self) -> Iterator[Any]:
		if self.device.type != 'cuda':
			for batch in self.loader:
				yield tree_map(self._map_leaf, batch)
			return

		stream = torch.cuda.Stream(device=self.device)
		current = torch.cuda.current_stream(self.device)
		for batch in self.loader:
			with torch.cuda.stream(stream):
				out = tree_map(self._map_leaf, batch)
			current.wait_stream(stream)
			for leaf in tree_leaves(out):
				if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
					leaf.record_stream(current)
			yield out


def make_dataloader(
	dataset: Dataset,
	batch_size: int,
	device: torch.device,
	*,
	shuffle: bool = True,
	drop_last: bool = True,
	num_workers: int = 0,
	pin_memory: bool = True,
) -> Prefetcher:
	return Prefetcher(
		MultiEpochsDataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_workers,
			pin_memory=pin_memory,
			drop_last=drop_last,
		),
		device,
	)
