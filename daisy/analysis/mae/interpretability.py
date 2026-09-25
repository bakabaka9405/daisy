"""ViT 的 token rollout 诊断与 avg-pool attention relevance 可视化。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import torch
from timm.layers.attention import Attention

from daisy.analysis.util import show_cam_on_image
from daisy.typing import MatrixF32
from .lrp import ViTAttentionRelevanceExplainer

if TYPE_CHECKING:
	from timm.models.vision_transformer import VisionTransformer


@contextmanager
def _hook_attention_weights(model: VisionTransformer):
	"""关闭 fused attention，以捕获 softmax 权重及其梯度，并在退出时恢复模型。"""
	attn_modules = [module for module in model.modules() if isinstance(module, Attention)]

	all_attentions: list[torch.Tensor] = []
	all_gradients: list[torch.Tensor] = []

	orig_fused = [(mod, mod.fused_attn) for mod in attn_modules]
	handles = []

	def capture_attention(_module, args, _output):
		attn = args[0]
		all_attentions.append(attn)
		attn.register_hook(lambda grad: all_gradients.append(grad))

	try:
		for mod in attn_modules:
			mod.fused_attn = False  # type: ignore[misc]
			handles.append(mod.attn_drop.register_forward_hook(capture_attention))

		yield all_attentions, all_gradients
	finally:
		for h in handles:
			h.remove()
		for mod, fused_attn in orig_fused:
			mod.fused_attn = fused_attn  # type: ignore[misc]


def _compute_rollout(
	attention_matrices: list[torch.Tensor],
	start_layer: int = 0,
) -> torch.Tensor:
	"""对带残差的 attention 逐层行归一化并累乘。"""
	matrices = attention_matrices[start_layer:]
	for i, matrix in enumerate(matrices):
		eye = torch.eye(matrix.size(-1), device=matrix.device, dtype=matrix.dtype)
		matrix = matrix + eye
		matrix = matrix / matrix.sum(dim=-1, keepdim=True)
		matrices[i] = matrix

	rollout = matrices[0]
	for i in range(1, len(matrices)):
		# 后层左乘前层，保留沿模型前向路径的传播顺序。
		rollout = torch.bmm(matrices[i], rollout)

	return rollout


def _compute_grad_rollout(
	attentions: list[torch.Tensor],
	gradients: list[torch.Tensor],
	start_layer: int = 0,
) -> torch.Tensor:
	"""逐层取正向梯度贡献的 head 均值并计算 rollout；梯度须按前向层序排列。"""
	avg_matrices = []
	for attn, grad in zip(attentions, gradients, strict=True):
		grad_weighted = (attn * grad).clamp(min=0).mean(dim=1)
		avg_matrices.append(grad_weighted)

	return _compute_rollout(avg_matrices, start_layer)


def generate_vit_grad_rollout_heatmap(
	model: VisionTransformer,
	image: torch.Tensor,
	target_class: int,
	input_size: int = 224,
	start_layer: int = 0,
) -> tuple[MatrixF32, np.ndarray]:
	"""生成基于 attention 梯度的 token rollout 诊断图。

	使用 softmax attention 与 autograd gradient，不执行 Chefer attention relprop。

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
	if model.global_pool != 'token':
		raise ValueError(f"Unsupported global_pool={model.global_pool!r}; expected 'token'.")

	device = next(model.parameters()).device
	input_tensor = image.unsqueeze(0).to(device)
	model.zero_grad()
	num_prefix = model.num_prefix_tokens

	with torch.enable_grad(), _hook_attention_weights(model) as (all_attentions, all_gradients):
		tokens = model.forward_features(input_tensor)
		output = model.forward_head(tokens)
		if not 0 <= target_class < output.shape[1]:
			raise ValueError('target_class is outside the logits range.')

		one_hot = torch.zeros_like(output)
		one_hot[0, target_class] = 1
		output.backward(gradient=one_hot, retain_graph=False)

		# 反向收集梯度传入 rollout
		all_gradients = all_gradients[::-1]
		rollout = _compute_grad_rollout(all_attentions, all_gradients, start_layer)
		mask = rollout[0, 0, num_prefix:]

	grid_size = model.patch_embed.grid_size
	mask = mask.reshape(grid_size[0], grid_size[1])
	mask = mask - mask.min()
	return show_cam_on_image(image, mask, input_size)


def generate_vit_attention_relevance_heatmap(
	explainer: ViTAttentionRelevanceExplainer,
	image: torch.Tensor,
	target_class: int,
	start_layer: int = 0,
	input_size: int | tuple[int, int] = 224,
) -> tuple[MatrixF32, np.ndarray]:
	"""生成注意力相关性热力图与叠加图像。"""
	mask = explainer.generate(image, target_class, start_layer)
	display_image = image[0] if image.ndim == 4 else image
	return show_cam_on_image(display_image, mask, input_size)
