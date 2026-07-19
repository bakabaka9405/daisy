"""ViT token diagnostic 与 avg-pool Chefer attention visualization。

官方部分使用 Chefer 的 ``R_A × target gradient`` 与无 row-normalization rollout；
Daisy 新增 MeanPool relprop 与多 query ``uᵀ(J-I)`` readout。这是 attention
visualization，不是 input attribution 或因果解释。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from timm.layers.attention import Attention

from .lrp import ViTAttentionRelevanceExplainer

if TYPE_CHECKING:
	from timm.models.vision_transformer import VisionTransformer


@contextmanager
def _hook_attention_weights(model: VisionTransformer):
	"""临时关闭 fused attention 并通过 attn_drop 的 forward hook 捕获注意力权重和梯度。

	对所有 timm Attention 模块：
	1. 设置 fused_attn = False，使 attention 走非 SDPA 路径（显式计算 attn 矩阵）
	2. 在 attn_drop (nn.Dropout) 上注册 forward hook，捕获其输入（= softmax 后的注意力权重）
	3. 在捕获的 attn tensor 上注册梯度 hook

	Yields:
		(all_attentions, all_gradients): 两个 list，分别存储各层的注意力权重和梯度。
	"""
	attn_modules: list[Attention] = []
	for module in model.modules():
		if isinstance(module, Attention):
			attn_modules.append(module)

	all_attentions: list[torch.Tensor] = []
	all_gradients: list[torch.Tensor] = []

	# 保存原始 fused_attn 值并关闭
	orig_fused = {}
	for mod in attn_modules:
		orig_fused[id(mod)] = mod.fused_attn
		mod.fused_attn = False  # type: ignore[misc]  # Final 仅 TorchScript 强制

	# 在 attn_drop 上注册 forward hook
	handles = []
	for mod in attn_modules:

		def make_hook():
			def hook_fn(_module, args, _output):
				attn = args[0]  # attn_drop 的输入就是 softmax 后的 attention
				all_attentions.append(attn)
				attn.register_hook(lambda grad: all_gradients.append(grad))

			return hook_fn

		h = mod.attn_drop.register_forward_hook(make_hook(), with_kwargs=False)
		handles.append(h)

	try:
		yield all_attentions, all_gradients
	finally:
		for h in handles:
			h.remove()
		for mod in attn_modules:
			mod.fused_attn = orig_fused[id(mod)]  # type: ignore[misc]


def _compute_rollout(
	all_layer_matrices: list[torch.Tensor],
	start_layer: int = 0,
) -> torch.Tensor:
	"""Attention Rollout 算法。

	对每层注意力矩阵加 Identity（残差连接），行归一化，然后逐层矩阵乘法。

	Args:
		all_layer_matrices: 每层的注意力矩阵，shape (batch, num_tokens, num_tokens)
		start_layer: 从哪一层开始 rollout

	Returns:
		rollout 矩阵，shape (batch, num_tokens, num_tokens)
	"""
	matrices = all_layer_matrices[start_layer:]
	for i, matrix in enumerate(matrices):
		eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
		matrix = matrix + eye
		matrix = matrix / matrix.sum(dim=-1, keepdim=True)
		matrices[i] = matrix

	rollout = matrices[0]
	for i in range(1, len(matrices)):
		# 按层从后向前左乘，保持与 Transformer rollout 文献实现一致
		rollout = torch.bmm(matrices[i], rollout)

	return rollout


def _compute_grad_rollout(
	attentions: list[torch.Tensor],
	gradients: list[torch.Tensor],
	start_layer: int = 0,
) -> torch.Tensor:
	"""Gradient-weighted Attention Rollout。

	每层: grad * attn -> clamp(min=0) -> average over heads，然后送入 rollout。

	Args:
		attentions: 每层注意力权重，shape (batch, num_heads, num_tokens, num_tokens)
		gradients: 每层注意力梯度（已反转为正序），shape 同上
		start_layer: 从哪一层开始 rollout

	Returns:
		rollout 矩阵，shape (batch, num_tokens, num_tokens)
	"""
	avg_matrices = []
	for attn, grad in zip(attentions, gradients):
		grad_weighted = (attn * grad).clamp(min=0).mean(dim=1)
		avg_matrices.append(grad_weighted)

	return _compute_rollout(avg_matrices, start_layer)


def show_cam_on_image(
	image: torch.Tensor,
	heatmap: np.ndarray,
	input_size: int,
) -> np.ndarray:
	"""将热力图叠加到原始图像上。

	Args:
		image: 输入图像 tensor，shape (3, H, W)
		heatmap: 热力图 ndarray，shape (input_size, input_size)，范围 [0, 1]
		input_size: 输出尺寸

	Returns:
		叠加后的图像 ndarray，shape (input_size, input_size, 3)，范围 [0, 1]
	"""
	img = image.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
	img = np.float32(img)
	img = (img - img.min()) / (img.max() - img.min() + 1e-8)
	img = cv2.resize(img, (input_size, input_size))

	heatmap_uint8 = (255 * heatmap).astype(np.uint8)
	colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
	colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
	colormap = np.float32(colormap) / 255

	overlay = 0.5 * colormap + 0.5 * img
	overlay = overlay / overlay.max()
	return overlay


def generate_vit_grad_rollout_heatmap(
	model: VisionTransformer,
	image: torch.Tensor,
	target_class: int,
	input_size: int = 224,
	start_layer: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
	"""生成 token-only legacy gradient-weighted attention rollout 诊断图。

	该函数使用原始 softmax attention 与普通 autograd gradient，未执行 attention
	relprop；它不是 Chefer 论文或官方 Transformer-Explainability 的忠实实现。

	Args:
		model: timm VisionTransformer 实例（需处于 eval 模式）
		image: 输入图像 tensor，shape (3, H, W)，已预处理
		target_class: 目标类别索引
		input_size: 输出热力图尺寸
		start_layer: 从哪一层开始 rollout

	Returns:
		(heatmap, overlay):
			- heatmap: ndarray shape (input_size, input_size)，范围 [0, 1]
			- overlay: ndarray shape (input_size, input_size, 3)，范围 [0, 1]
	"""
	if model.global_pool == 'avg':
		raise ValueError("global_pool='avg' 必须使用 ViTAttentionRelevanceExplainer 与 generate_vit_attention_relevance_heatmap。")
	if model.global_pool != 'token':
		raise ValueError(f"Unsupported global_pool={model.global_pool!r}; expected 'token'.")

	device = next(model.parameters()).device
	input_tensor = image.unsqueeze(0).to(device)
	model.zero_grad()
	num_prefix = model.num_prefix_tokens

	with torch.enable_grad(), _hook_attention_weights(model) as (all_attentions, all_gradients):
		tokens = model.forward_features(input_tensor)
		output = model.forward_head(tokens)

		one_hot = torch.zeros_like(output)
		one_hot[0, target_class] = 1
		output.backward(gradient=one_hot, retain_graph=False)

		# 梯度通过 backward hook 收集，顺序为反向（最后一层先到）
		all_gradients = all_gradients[::-1]
		rollout = _compute_grad_rollout(all_attentions, all_gradients, start_layer)
		mask = rollout[0, 0, num_prefix:]

	grid_size = model.patch_embed.grid_size
	mask = mask.reshape(1, 1, grid_size[0], grid_size[1])
	mask = F.interpolate(mask, size=(input_size, input_size), mode='bilinear', align_corners=False)
	mask = mask.squeeze()
	mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
	heatmap = mask.detach().cpu().numpy()

	overlay = show_cam_on_image(image, heatmap, input_size)

	return heatmap, overlay


def generate_vit_attention_relevance_heatmap(
	explainer: ViTAttentionRelevanceExplainer,
	image: torch.Tensor,
	target_class: int,
	start_layer: int = 0,
	input_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
	"""仅以 bilinear 将 normalized attention relevance 对齐到显示尺寸。"""
	mask = explainer.generate(image, target_class, start_layer).heatmap
	if mask.shape != (input_size, input_size):
		mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(input_size, input_size), mode='bilinear', align_corners=False).squeeze()
	maximum = mask.max()
	mask = mask / maximum if maximum.item() > 0 else torch.zeros_like(mask)

	heatmap = mask.detach().cpu().numpy().astype(np.float32, copy=False)
	display_image = image[0] if image.ndim == 4 else image
	overlay = show_cam_on_image(display_image, heatmap, input_size)
	return heatmap, overlay
