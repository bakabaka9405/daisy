"""框架内置 feeder"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from daisy.dataset.dataset_sample import random_subsample
from daisy.dataset.disk_dataset import DiskDataset, load_files_from_folder
from .io import DEFAULT_IMAGE_EXTENSIONS
from .registry import FeederSplits, register_feeder


class ImageFolderParams(BaseModel):
	"""无标签图像文件夹参数"""

	model_config = ConfigDict(extra='forbid')

	root: Path | list[Path] = Field(default_factory=Path)
	extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_IMAGE_EXTENSIONS))
	sample_ratio: float = Field(default=1.0, gt=0.0, le=1.0)
	sample_seed: int | None = None


@register_feeder('image_folder')
def image_folder(params: ImageFolderParams, *, seed: int | None) -> FeederSplits:
	"""递归收集文件夹内全部图像作为无标签 train 划分"""
	roots = params.root if isinstance(params.root, list) else [params.root]
	files: list[Path] = []
	for root in roots:
		files.extend(load_files_from_folder(root, extensions=params.extensions))

	dataset = DiskDataset(files, None, nn.Identity())
	if params.sample_ratio < 1.0:
		dataset = random_subsample(dataset, params.sample_ratio, params.sample_seed if params.sample_seed is not None else seed)
	return {'train': dataset}
