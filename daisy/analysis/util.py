"""分析与可视化通用工具函数。"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from daisy.typing import ImageF32, MatrixF32


def show_cam_on_image(
	image: torch.Tensor,
	heatmap: torch.Tensor | MatrixF32,
	input_size: int | tuple[int, int] = 224,
) -> tuple[MatrixF32, ImageF32]:
	"""将二维热力图双线性插值到显示尺寸并与图像进行 JET 伪彩色叠加。

	Args:
		image: 输入图像 tensor，shape (3, H, W)
		heatmap: 二维热力图 tensor 或 ndarray，shape (H, W)
		input_size: 输出尺寸，整数或 (height, width) 元组

	Returns:
		(heatmap, overlay):
			- heatmap: 归一化到 [0, 1] 的 2D ndarray，shape (H, W)
			- overlay: JET 颜色映射叠加后的图像 ndarray，shape (H, W, 3)，范围 [0, 1]
	"""
	# 阶段 1: 校验输入维度并预处理背景图像
	height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
	if image.ndim != 3 or image.shape[0] != 3:
		raise ValueError(f'image must have shape (3, H, W), got {tuple(image.shape)}')

	img = image.detach().cpu().numpy().transpose(1, 2, 0)
	img = np.float32(img)
	img = (img - img.min()) / (img.max() - img.min() + 1e-8)
	img = cv2.resize(img, (width, height))

	# 阶段 2: 校验二维热力图并执行双线性空间插值
	heatmap_tensor = torch.as_tensor(heatmap, dtype=torch.float32)
	if heatmap_tensor.ndim != 2:
		raise ValueError(f'heatmap must have shape (H, W), got {tuple(heatmap_tensor.shape)}')

	if heatmap_tensor.shape != (height, width):
		rescaled = (
			F.interpolate(
				heatmap_tensor.unsqueeze(0).unsqueeze(0),
				size=(height, width),
				mode='bilinear',
				align_corners=False,
			)
			.squeeze(0)
			.squeeze(0)
		)
	else:
		rescaled = heatmap_tensor

	# 阶段 3: 非负最大值安全归一化
	nonnegative = rescaled.clamp_min(0)
	maximum = nonnegative.max()
	normalized = nonnegative / maximum if maximum.item() > 0 else torch.zeros_like(nonnegative)
	heatmap_out: MatrixF32 = normalized.detach().cpu().numpy().astype(np.float32, copy=False)

	# 阶段 4: JET 色彩映射与半透明图像融合
	heatmap_uint8 = (255 * heatmap_out).astype(np.uint8)
	colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
	colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
	colormap = np.float32(colormap) / 255

	overlay = 0.5 * colormap + 0.5 * img
	overlay_max = overlay.max()
	overlay = overlay / overlay_max if overlay_max > 0 else overlay
	overlay_out: ImageF32 = overlay.astype(np.float32, copy=False)

	return heatmap_out, overlay_out
