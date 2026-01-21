"""MAE (Masked Autoencoder) 模型模块

提供 MAE 预训练模型和相关工具
"""

from .models_mae import (
	MaskedAutoencoderViT,
	mae_vit_base_patch16,
	mae_vit_large_patch16,
	mae_vit_huge_patch14,
	create_mae_model,
	MAE_MODELS,
)
from .models_vit import (
	vit_base_patch16,
	vit_large_patch16,
	vit_huge_patch14,
	create_vit_model,
	VIT_MODELS,
	load_mae_pretrained_weights,
)
from .pos_embed import (
	get_2d_sincos_pos_embed,
	interpolate_pos_embed,
)
from .lr_decay import (
	get_layer_id_for_vit,
	param_groups_lrd,
)

__all__ = [
	# MAE models
	'MaskedAutoencoderViT',
	'mae_vit_base_patch16',
	'mae_vit_large_patch16',
	'mae_vit_huge_patch14',
	'create_mae_model',
	'MAE_MODELS',
	# ViT models for finetune
	'vit_base_patch16',
	'vit_large_patch16',
	'vit_huge_patch14',
	'create_vit_model',
	'VIT_MODELS',
	'load_mae_pretrained_weights',
	# Position embedding
	'get_2d_sincos_pos_embed',
	'interpolate_pos_embed',
	# LR decay
	'get_layer_id_for_vit',
	'param_groups_lrd',
]
