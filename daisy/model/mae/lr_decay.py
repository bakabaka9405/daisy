"""Layer-wise LR Decay 工具

为 ViT 模型不同层分配不同学习率，较浅的层使用较小的学习率。

参考:
- https://github.com/facebookresearch/mae
- BEiT: https://github.com/microsoft/unilm/tree/master/beit
- ELECTRA: https://github.com/google-research/electra
"""

import torch.nn as nn


def get_layer_id_for_vit(name: str, num_layers: int) -> int:
	"""获取参数所属的层 ID

	Args:
		name: 参数名称
		num_layers: 模型总层数（Transformer blocks 数量）

	Returns:
		层 ID，范围 [0, num_layers]
		- 0: cls_token, pos_embed, patch_embed
		- 1 ~ num_layers: transformer blocks
		- num_layers: head (最后一层)
	"""
	if name in ['cls_token', 'pos_embed']:
		return 0
	elif name.startswith('patch_embed'):
		return 0
	elif name.startswith('blocks'):
		return int(name.split('.')[1]) + 1
	else:
		return num_layers


def param_groups_lrd(
	model: nn.Module,
	weight_decay: float = 0.05,
	no_weight_decay_list: list[str] | None = None,
	layer_decay: float = 0.75,
) -> list[dict]:
	"""创建带 layer-wise LR decay 的参数组

	较早的层使用较小的学习率:
		lr_scale = layer_decay^(num_layers - layer_id)

	Args:
		model: ViT 模型
		weight_decay: 权重衰减
		no_weight_decay_list: 不使用权重衰减的参数名列表
		layer_decay: 层间衰减系数（0.65-0.75 推荐）

	Returns:
		参数组列表，每组包含:
		- params: 参数列表
		- lr_scale: 学习率缩放系数
		- weight_decay: 权重衰减
	"""
	if no_weight_decay_list is None:
		no_weight_decay_list = []

	param_groups = {}

	# 获取模型层数
	num_layers = len(model.blocks) + 1  # type: ignore

	# 计算每层的学习率缩放系数
	layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

	for name, param in model.named_parameters():
		if not param.requires_grad:
			continue

		# 判断是否使用 weight decay
		# 1D 参数（bias, LayerNorm）不使用 weight decay
		if param.ndim == 1 or name in no_weight_decay_list:
			g_decay = 'no_decay'
			this_decay = 0.0
		else:
			g_decay = 'decay'
			this_decay = weight_decay

		# 获取层 ID
		layer_id = get_layer_id_for_vit(name, num_layers)
		group_name = f'layer_{layer_id}_{g_decay}'

		if group_name not in param_groups:
			param_groups[group_name] = {
				'lr_scale': layer_scales[layer_id],
				'weight_decay': this_decay,
				'params': [],
			}

		param_groups[group_name]['params'].append(param)

	return list(param_groups.values())
