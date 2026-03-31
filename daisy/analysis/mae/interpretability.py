"""ViT Grad Rollout 可解释性

基于 avgpool ViT 的 Grad Rollout 热力图生成，参考 Transformer-Explainability 仓库思路。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
	from timm.models.vision_transformer import VisionTransformer


def compute_rollout_attention(all_layer_matrices: list[torch.Tensor], start_layer: int = 0) -> torch.Tensor:
	"""计算 attention rollout

	Args:
		all_layer_matrices: 每层的 attention 矩阵列表 [B, num_tokens, num_tokens]
		start_layer: 起始层索引

	Returns:
		rollout attention [B, num_tokens, num_tokens]
	"""
	num_tokens = all_layer_matrices[0].shape[1]
	batch_size = all_layer_matrices[0].shape[0]
	eye = torch.eye(num_tokens, device=all_layer_matrices[0].device).expand(batch_size, num_tokens, num_tokens)

	# 加入残差连接
	all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
	matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True) for i in range(len(all_layer_matrices))]

	joint_attention = matrices_aug[start_layer]
	for i in range(start_layer + 1, len(matrices_aug)):
		joint_attention = matrices_aug[i].bmm(joint_attention)

	return joint_attention


class ViTAttentionExtractor:
	"""ViT Attention 提取器，支持 Grad Rollout"""

	def __init__(self, model: VisionTransformer):
		self.model = model
		self.attention_maps: list[torch.Tensor] = []
		self.attention_grads: list[torch.Tensor] = []
		self.hooks: list = []

		# 关闭 fused_attn 以便抓取 attention map
		for blk in self.model.blocks:  # type: ignore
			if hasattr(blk.attn, 'fused_attn'):
				blk.attn.fused_attn = False

	def _save_attention_map(self, module, input, output):
		"""保存 attention map 的 forward hook"""
		# attn_drop 的输入就是 softmax 后的 attention
		# 保留 requires_grad 以便后续注册 backward hook
		self.attention_maps.append(input[0])

	def _save_attention_grad(self, grad):
		"""保存 attention gradient 的 backward hook"""
		self.attention_grads.append(grad)

	def register_hooks(self):
		"""注册 hooks"""
		self.attention_maps.clear()
		self.attention_grads.clear()
		self.hooks.clear()

		for blk in self.model.blocks:  # type: ignore
			# 在 attn_drop 上挂 forward hook
			hook = blk.attn.attn_drop.register_forward_hook(self._save_attention_map)
			self.hooks.append(hook)

	def remove_hooks(self):
		"""移除 hooks"""
		for hook in self.hooks:
			hook.remove()
		self.hooks.clear()

	def forward_and_extract(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
		"""前向并提取 attention

		Args:
			x: 输入图像 [B, C, H, W]
			target_class: 目标类别，None 表示使用预测类别

		Returns:
			grad rollout heatmap [B, H_patch, W_patch]
		"""
		self.register_hooks()
		x.requires_grad_(True)

		# 前向
		output = self.model(x)

		# 确定目标类别
		target_class_tensor: torch.Tensor
		if target_class is None:
			target_class_tensor = output.argmax(dim=1)
		elif isinstance(target_class, int):
			target_class_tensor = torch.tensor([target_class] * x.shape[0], device=x.device)
		else:
			target_class_tensor = target_class

		# 注册 gradient hooks
		self.attention_grads.clear()
		for attn_map in self.attention_maps:
			attn_map.register_hook(self._save_attention_grad)

		# 反向传播
		self.model.zero_grad()
		one_hot = torch.zeros_like(output)
		one_hot.scatter_(1, target_class_tensor.unsqueeze(1), 1.0)
		loss = (one_hot * output).sum()
		loss.backward()

		# 计算 grad rollout
		cams = []
		for attn_map, attn_grad in zip(self.attention_maps, self.attention_grads):
			# attn_map: [B, num_heads, num_tokens, num_tokens]
			# 对每个 head 做 grad * attn，然后平均
			cam = attn_grad * attn_map
			cam = cam.clamp(min=0).mean(dim=1)  # [B, num_tokens, num_tokens]
			cams.append(cam)

		rollout = compute_rollout_attention(cams, start_layer=0)

		# avgpool 模型：对所有 patch token 的输出取平均
		# rollout: [B, num_tokens, num_tokens]
		# num_tokens = 1 (cls) + num_patches
		num_patches = self.model.patch_embed.num_patches
		patch_size = self.model.patch_embed.patch_size[0]
		grid_size = int(num_patches**0.5)

		# 取 patch tokens 对所有 token 的平均 attention
		# [B, num_patches, num_tokens] -> [B, num_patches]
		heatmap = rollout[:, 1:, :].mean(dim=-1)

		# reshape 到 2D
		heatmap = heatmap.reshape(-1, grid_size, grid_size)

		self.remove_hooks()
		return heatmap


def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""将热力图叠加到图像上

	Args:
		img: 原始图像 [H, W, 3]，范围 [0, 1]
		mask: 热力图 [H, W]，范围 [0, 1]

	Returns:
		叠加后的图像 [H, W, 3]，范围 [0, 1]
	"""
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # type: ignore
	heatmap = np.float32(heatmap) / 255
	heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # type: ignore
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	return cam


def generate_vit_grad_rollout_heatmap(
	model: VisionTransformer,
	image: torch.Tensor,
	target_class: int | None = None,
	input_size: int = 224,
) -> tuple[np.ndarray, np.ndarray]:
	"""生成 ViT Grad Rollout 热力图

	Args:
		model: ViT 模型
		image: 输入图像 [C, H, W]，已归一化
		target_class: 目标类别，None 表示使用预测类别
		input_size: 输入尺寸

	Returns:
		(heatmap, overlay): 热力图和叠加图像，均为 [H, W, 3] numpy 数组
	"""
	model.eval()
	device = next(model.parameters()).device

	extractor = ViTAttentionExtractor(model)
	image_batch = image.unsqueeze(0).to(device)

	with torch.enable_grad():
		heatmap = extractor.forward_and_extract(image_batch, target_class)

	# 上采样到输入尺寸
	heatmap = F.interpolate(heatmap.unsqueeze(1), size=(input_size, input_size), mode='bilinear', align_corners=False)
	heatmap = heatmap.squeeze().detach().cpu().numpy()

	# 归一化
	heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

	# 反归一化原图
	mean = np.array([0.485, 0.456, 0.406])
	std = np.array([0.229, 0.224, 0.225])
	image_np = image.cpu().numpy().transpose(1, 2, 0)
	image_np = image_np * std + mean
	image_np = np.clip(image_np, 0, 1)

	# 叠加
	overlay = show_cam_on_image(image_np, heatmap)

	return heatmap, overlay


def save_heatmap_visualization(
	heatmap: np.ndarray,
	overlay: np.ndarray,
	original_image: np.ndarray,
	save_path: Path,
	true_label: int,
	pred_label: int,
):
	"""保存热力图可视化

	Args:
		heatmap: 热力图 [H, W]
		overlay: 叠加图像 [H, W, 3]
		original_image: 原始图像 [H, W, 3]
		save_path: 保存路径
		true_label: 真实标签
		pred_label: 预测标签
	"""
	import matplotlib.pyplot as plt

	fig, axes = plt.subplots(1, 3, figsize=(15, 5))

	axes[0].imshow(original_image)
	axes[0].set_title(f'Original\nTrue: {true_label}, Pred: {pred_label}')
	axes[0].axis('off')

	axes[1].imshow(heatmap, cmap='jet')
	axes[1].set_title('Grad Rollout Heatmap')
	axes[1].axis('off')

	axes[2].imshow(overlay)
	axes[2].set_title('Overlay')
	axes[2].axis('off')

	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
