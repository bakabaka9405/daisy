"""MAE 预训练与微调算法方法实现。"""

from __future__ import annotations

from typing import cast, override

import icontract
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, Optimizer

from daisy.method.base import Method
from daisy.model.mae import MaskedAutoencoderViT
from daisy.training.state import TrainState
from daisy.training.trainer import Trainer


class MAEPretrainMethod(Method[MaskedAutoencoderViT]):
	"""MAE 预训练方法。"""

	@override
	@icontract.require(lambda freeze_encoder_epochs: freeze_encoder_epochs >= 0)
	@icontract.require(lambda mask_ratio: 0 < mask_ratio < 1)
	def __init__(
		self,
		model: MaskedAutoencoderViT,
		*,
		lr: float,
		weight_decay: float = 0.05,
		mask_ratio: float = 0.75,
		norm_pix_loss: bool = False,
		freeze_encoder_epochs: int = 0,
	) -> None:
		self.mask_ratio = mask_ratio
		self.norm_pix_loss = norm_pix_loss
		self.freeze_encoder_epochs = freeze_encoder_epochs
		super().__init__(
			model,
			lr=lr,
			weight_decay=weight_decay,
		)

	@override
	def register(self, trainer: Trainer) -> None:
		"""向 Trainer 安装模型、默认优化器与训练步，并根据配置注册编码器冻结策略。"""
		super().register(trainer)

		if self.freeze_encoder_epochs <= 0:
			return

		# 筛选并锁定注册时具备梯度的编码器参数，保留调用方预先冻结的参数
		encoder_prefixes = ('patch_embed.', 'blocks.', 'norm.')
		managed_params = tuple(
			p for name, p in self.model.named_parameters()
			if (name == 'cls_token' or name.startswith(encoder_prefixes)) and p.requires_grad
		)

		def on_epoch_start(state: TrainState) -> None:
			# 阶段判定：基于 TrainState 绝对轮次决定编码器状态
			should_freeze = state.epoch < self.freeze_encoder_epochs
			requires_grad = not should_freeze

			# 状态更新：配置梯度需求并在冻结阶段清除残余梯度
			for param in managed_params:
				param.requires_grad = requires_grad
				if should_freeze:
					param.grad = None

		trainer.on_epoch_start(on_epoch_start)

	@override
	@icontract.require(lambda backbone: isinstance(backbone, MaskedAutoencoderViT), 'Expected a MaskedAutoencoderViT model.')
	def build_model(self, backbone: nn.Module) -> MaskedAutoencoderViT:
		"""采用已传入的 MaskedAutoencoderViT 实例。"""
		return cast(MaskedAutoencoderViT, backbone)

	@override
	def build_optimizer(
		self,
		model: MaskedAutoencoderViT,
		*,
		lr: float,
		weight_decay: float,
	) -> Optimizer:
		"""构建 bias/norm 不衰减、其余参数衰减并使用 beta2=0.95 的 AdamW。"""
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

	@icontract.require(lambda batch: batch.ndim == 4)
	@icontract.ensure(lambda result: result.ndim == 0)
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
