"""ViT 分类模型

用于 MAE finetune 的 Vision Transformer 分类模型，适配 timm 1.0.24

参考:
- https://github.com/facebookresearch/mae
- timm.models.vision_transformer
"""

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.layers.weight_init import trunc_normal_

from .pos_embed import interpolate_pos_embed


def vit_base_patch16(**kwargs) -> VisionTransformer:
	"""ViT-Base/16 模型"""
	return VisionTransformer(
		patch_size=16,
		embed_dim=768,
		depth=12,
		num_heads=12,
		mlp_ratio=4,
		qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)


def vit_large_patch16(**kwargs) -> VisionTransformer:
	"""ViT-Large/16 模型"""
	return VisionTransformer(
		patch_size=16,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		mlp_ratio=4,
		qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)


def vit_huge_patch14(**kwargs) -> VisionTransformer:
	"""ViT-Huge/14 模型"""
	return VisionTransformer(
		patch_size=14,
		embed_dim=1280,
		depth=32,
		num_heads=16,
		mlp_ratio=4,
		qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)


VIT_MODELS = {
	'vit_base_patch16': vit_base_patch16,
	'vit_large_patch16': vit_large_patch16,
	'vit_huge_patch14': vit_huge_patch14,
}


def create_vit_model(name: str, **kwargs) -> VisionTransformer:
	"""创建 ViT 模型

	Args:
		name: 模型名称 (vit_base_patch16, vit_large_patch16, vit_huge_patch14)
		**kwargs: 传递给模型的参数，常用的有:
			- num_classes: 分类数
			- global_pool: 池化方式 ('avg', 'token', '')
			- drop_path_rate: DropPath 比率
			- img_size: 输入图像大小

	Returns:
		ViT 模型实例
	"""
	if name not in VIT_MODELS:
		raise ValueError(f'Unknown model: {name}. Available: {list(VIT_MODELS.keys())}')
	return VIT_MODELS[name](**kwargs)


def load_mae_pretrained_weights(
	model: VisionTransformer,
	checkpoint_path: str,
	init_head: bool = True,
) -> None:
	"""从 MAE 预训练 checkpoint 加载权重到 ViT 模型

	Args:
		model: ViT 模型
		checkpoint_path: MAE 预训练权重路径
		init_head: 是否重新初始化分类头
	"""
	checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

	# MAE checkpoint 可能包含 'model' key
	if 'model' in checkpoint:
		checkpoint_model = checkpoint['model']
	else:
		checkpoint_model = checkpoint

	state_dict = model.state_dict()

	# 移除分类头（形状可能不匹配）
	for k in ['head.weight', 'head.bias']:
		if k in checkpoint_model and k in state_dict:
			if checkpoint_model[k].shape != state_dict[k].shape:
				print(f'Removing key {k} from pretrained checkpoint (shape mismatch)')
				del checkpoint_model[k]

	# 插值位置编码（如果分辨率不同）
	interpolate_pos_embed(model, checkpoint_model)

	# 加载权重
	msg = model.load_state_dict(checkpoint_model, strict=False)
	print(f'Loaded pretrained weights: {msg}')

	# 重新初始化分类头
	if init_head:
		trunc_normal_(model.head.weight, std=2e-5)  # type: ignore
		if model.head.bias is not None:
			nn.init.zeros_(model.head.bias)  # type: ignore
