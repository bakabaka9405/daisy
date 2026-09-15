"""MAE 预训练与微调算法方法实现。"""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer

from daisy.method.base import Method
from daisy.method.classification import ClassificationMethod
from daisy.model.mae import MaskedAutoencoderViT, param_groups_lrd


class MAEPretrainMethod(Method[MaskedAutoencoderViT]):
	"""MAE 预训练方法。"""

	@override
	def __init__(
		self,
		model: MaskedAutoencoderViT,
		*,
		device: torch.device,
		lr: float,
		weight_decay: float = 0.05,
		mask_ratio: float = 0.75,
		norm_pix_loss: bool = False,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
		optimizer: Optimizer | None = None,
	) -> None:
		self.mask_ratio = mask_ratio
		self.norm_pix_loss = norm_pix_loss
		super().__init__(
			model,
			device=device,
			lr=lr,
			weight_decay=weight_decay,
			use_amp=use_amp,
			accum_iter=accum_iter,
			clip_grad=clip_grad,
			optimizer=optimizer,
		)

	@override
	def build_model(self, backbone: nn.Module) -> MaskedAutoencoderViT:
		"""采用已传入的 MaskedAutoencoderViT 实例。"""
		if not isinstance(backbone, MaskedAutoencoderViT):
			raise TypeError('Expected a MaskedAutoencoderViT model.')
		return backbone

	@override
	def build_optimizer(
		self,
		model: MaskedAutoencoderViT,
		*,
		lr: float,
		weight_decay: float,
	) -> Optimizer:
		"""构建带偏置与归一化层过滤的 AdamW 优化器。"""
		param_groups = [
			{'params': [p for n, p in model.named_parameters() if 'bias' not in n and 'norm' not in n]},
			{'params': [p for n, p in model.named_parameters() if 'bias' in n or 'norm' in n], 'weight_decay': 0.0},
		]
		return AdamW(
			param_groups,
			lr=lr,
			weight_decay=weight_decay,
			betas=(0.9, 0.95),
			fused=True,
		)

	def process(self, batch: Tensor) -> Tensor:
		"""单步前向计算并返回重构损失。"""
		# 前向编码与解码预测
		pred, mask = self.model(batch, mask_ratio=self.mask_ratio)

		# 图像分块与目标特征对齐
		target = self.model.patchify(batch)
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.0e-6) ** 0.5

		# 计算均方误差并对掩码区域求均值
		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)
		return (loss * mask).sum() / mask.sum()
