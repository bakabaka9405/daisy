from pathlib import Path
from ImageDataset import ImageDataset
import torch


def LoadImages(path: Path, prefetchLevel: int, device: torch.device | None) -> ImageDataset:
	dataset = ImageDataset(prefetchLevel=prefetchLevel, device=device)
	for label, subpath in enumerate(path.iterdir()):
		for imgpath in subpath.iterdir():
			dataset.append(label, str(imgpath))
	return dataset
