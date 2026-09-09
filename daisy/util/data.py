"""无竞态的 CUDA 数据预取器。"""

from collections.abc import Iterator

import torch
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import DataLoader, Dataset


class Prefetcher:
	"""在独立 CUDA 流上把 batch 搬运到设备、转换为 fp32 并归一化。

	输入假定为 uint8 [0, 255]，归一化等价于 (x / 255 - mean) / std。
	预取张量在预取流上分配，消费流使用前先等待预取流，并对张量登记
	record_stream，使缓存分配器在消费流读取完成前不会复用其显存。
	"""

	def __init__(
		self,
		loader: DataLoader,
		device: torch.device,
		mean: tuple[float, ...] = (0.485, 0.456, 0.406),
		std: tuple[float, ...] = (0.229, 0.224, 0.225),
	):
		self.loader = loader
		self.device = device
		# mean/std 乘以 255，在设备上对 uint8 输入做原地归一化
		self.mean = torch.tensor([m * 255 for m in mean], device=device).view(-1, 1, 1)
		self.std = torch.tensor([s * 255 for s in std], device=device).view(-1, 1, 1)

	def __len__(self) -> int:
		return len(self.loader)

	def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
		if self.device.type != 'cuda':
			for data, target in self.loader:
				data = data.to(self.device, dtype=torch.float32).sub_(self.mean).div_(self.std)
				yield data, target.to(self.device)
			return

		stream = torch.cuda.Stream(device=self.device)
		current = torch.cuda.current_stream(self.device)

		for data, target in self.loader:
			with torch.cuda.stream(stream):
				data = data.to(self.device, non_blocking=True)
				target = target.to(self.device, non_blocking=True)
				data = data.to(torch.float32).sub_(self.mean).div_(self.std)

			# 消费流等待预取完成；record_stream 保证显存在消费流读取完成前不被复用
			current.wait_stream(stream)
			data.record_stream(current)
			target.record_stream(current)

			yield data, target


def make_dataloader(
	dataset: Dataset,
	batch_size: int,
	device: torch.device,
	*,
	shuffle: bool = True,
	drop_last: bool = True,
	num_workers: int = 0,
	pin_memory: bool = True,
	mean: tuple[float, ...] = (0.485, 0.456, 0.406),
	std: tuple[float, ...] = (0.229, 0.224, 0.225),
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
		mean=mean,
		std=std,
	)
