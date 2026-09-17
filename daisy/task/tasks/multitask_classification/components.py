"""多任务分类模型与优化器构建组件。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from daisy.model.mae import create_vit_model, load_mae_pretrained_weights, param_groups_lrd
from daisy.model.multihead import MultiHeadModel
from .config import MultiTaskModelConfig


def build_backbone(model_cfg: MultiTaskModelConfig) -> tuple[nn.Module, int]:
	"""根据配置构建 backbone 并返回 backbone 与其输出特征维度 feature_dim。

	在外部构造边界处提取 feature_dim，保证 MultiHeadModel 的显式 feature_dim 契约。
	统一支持 global_pool 与 compile_backend；权重加载发生在 compile 之前。
	"""
	if model_cfg.checkpoint_path is not None:
		mae_kwargs: dict[str, Any] = {
			'num_classes': 0,
			'drop_path_rate': model_cfg.drop_path_rate,
		}
		if model_cfg.global_pool is not None:
			mae_kwargs['global_pool'] = model_cfg.global_pool
		if model_cfg.img_size is not None:
			mae_kwargs['img_size'] = model_cfg.img_size
		backbone = create_vit_model(model_cfg.name, **mae_kwargs)

		load_mae_pretrained_weights(
			backbone,
			checkpoint_path=model_cfg.checkpoint_path,
			init_head=False,
			verbose=False,
			source_grid_size=model_cfg.source_grid_size,
		)
	else:
		model_kwargs: dict[str, Any] = {'pretrained': model_cfg.pretrained, 'num_classes': 0}
		if model_cfg.img_size is not None:
			model_kwargs['img_size'] = model_cfg.img_size
		if model_cfg.drop_path_rate > 0:
			model_kwargs['drop_path_rate'] = model_cfg.drop_path_rate
		if model_cfg.global_pool is not None:
			model_kwargs['global_pool'] = model_cfg.global_pool
		backbone = timm.create_model(model_cfg.name, **model_kwargs)

	feature_dim_val = getattr(backbone, 'num_features', None)
	if not isinstance(feature_dim_val, int):
		raise AttributeError(f"Backbone '{model_cfg.name}' does not provide integer 'num_features'")
	feature_dim = feature_dim_val

	if model_cfg.compile_backend is not None:
		backbone.compile(backend=model_cfg.compile_backend)

	return backbone, feature_dim


def create_multihead_model(
	model_cfg: MultiTaskModelConfig,
	num_classes_list: Sequence[int],
) -> MultiHeadModel:
	"""创建包含共享 backbone 与独立头部的完整 MultiHeadModel。"""
	backbone, feature_dim = build_backbone(model_cfg)
	return MultiHeadModel(backbone, num_classes_list, feature_dim=feature_dim)


def build_lrd_optimizer_factory(
	layer_decay: float,
	is_cuda: bool,
) -> Callable[[MultiHeadModel, float, float], Optimizer]:
	"""构建 layer-wise lr decay 优化器工厂。

	工厂接收已构建并在目标设备上的 MultiHeadModel 以及 lr、weight_decay。
	在 model.backbone 上应用 param_groups_lrd，分类头参数以 lr_scale=1.0 分为 decay/no-decay 组。
	"""

	def optimizer_factory(model: MultiHeadModel, lr: float, weight_decay: float) -> Optimizer:
		param_groups = param_groups_lrd(
			model.backbone,
			weight_decay=weight_decay,
			layer_decay=layer_decay,
		)

		head_decay_params: list[torch.nn.Parameter] = []
		head_no_decay_params: list[torch.nn.Parameter] = []

		for p in model.heads.parameters():
			if not p.requires_grad:
				continue
			if p.ndim > 1:
				head_decay_params.append(p)
			else:
				head_no_decay_params.append(p)

		if head_decay_params:
			param_groups.append(
				{
					'params': head_decay_params,
					'weight_decay': weight_decay,
					'lr_scale': 1.0,
				}
			)
		if head_no_decay_params:
			param_groups.append(
				{
					'params': head_no_decay_params,
					'weight_decay': 0.0,
					'lr_scale': 1.0,
				}
			)

		return AdamW(
			param_groups,
			lr=lr,
			betas=(0.9, 0.999),
			fused=is_cuda,
		)

	return optimizer_factory
