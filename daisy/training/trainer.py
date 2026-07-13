from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from daisy.training.state import TrainState


SideEffectFn = Callable[[TrainState], None]
BeforeForwardFn = Callable[[TrainState, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
AfterForwardFn = Callable[[TrainState, torch.Tensor], torch.Tensor]


class Trainer:
	"""通过事件回调和 Plugin 扩展固定训练循环。

	每轮依次执行 batch start、输入变换、forward、backward、优化器更新和 batch end；
	每个 epoch 前后分别触发对应回调。
	"""

	def __init__(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer,
		criterion: nn.Module,
		device: torch.device,
		*,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
	):
		self.model = model
		self.optimizer = optimizer
		self.criterion = criterion
		self.device = device
		self.use_amp = use_amp
		self.accum_iter = accum_iter
		self.clip_grad = clip_grad

		self.scaler = torch.GradScaler(device.type, enabled=use_amp)

		self._on_fit_start: list[SideEffectFn] = []
		self._on_fit_end: list[SideEffectFn] = []
		self._on_epoch_start: list[SideEffectFn] = []
		self._on_epoch_end: list[SideEffectFn] = []
		self._on_batch_start: list[SideEffectFn] = []
		self._on_batch_end: list[SideEffectFn] = []
		self._before_forward: list[BeforeForwardFn] = []
		self._after_forward: list[AfterForwardFn] = []

	def on_fit_start(self, fn: SideEffectFn) -> Trainer:
		self._on_fit_start.append(fn)
		return self

	def on_fit_end(self, fn: SideEffectFn) -> Trainer:
		self._on_fit_end.append(fn)
		return self

	def on_epoch_start(self, fn: SideEffectFn) -> Trainer:
		self._on_epoch_start.append(fn)
		return self

	def on_epoch_end(self, fn: SideEffectFn) -> Trainer:
		self._on_epoch_end.append(fn)
		return self

	def on_batch_start(self, fn: SideEffectFn) -> Trainer:
		self._on_batch_start.append(fn)
		return self

	def on_batch_end(self, fn: SideEffectFn) -> Trainer:
		self._on_batch_end.append(fn)
		return self

	def before_forward(self, fn: BeforeForwardFn) -> Trainer:
		"""注册返回 ``(images, targets)`` 的前向输入变换。"""
		self._before_forward.append(fn)
		return self

	def after_forward(self, fn: AfterForwardFn) -> Trainer:
		"""注册返回变换后输出的 forward hook。"""
		self._after_forward.append(fn)
		return self

	def use(self, *plugins: Any) -> Trainer:
		"""注册 Plugin。"""
		for plugin in plugins:
			plugin.register(self)
		return self

	def fit(self, train_loader: DataLoader, epochs: int) -> TrainState:
		"""执行训练循环并返回最终状态。"""
		self.model.to(self.device)

		state = TrainState(
			model=self.model,
			optimizer=self.optimizer,
			device=self.device,
			total_epochs=epochs,
			total_batches=len(train_loader),
		)

		for fn in self._on_fit_start:
			fn(state)

		for epoch in range(epochs):
			state.epoch = epoch
			state.total_batches = len(train_loader)

			for fn in self._on_epoch_start:
				fn(state)

			state.epoch_train_loss = self._train_one_epoch(train_loader, state)

			for fn in self._on_epoch_end:
				fn(state)

			if state.stop_training:
				break

		for fn in self._on_fit_end:
			fn(state)

		return state

	def _train_one_epoch(self, train_loader: DataLoader, state: TrainState) -> float:
		self.model.train()
		total_loss = 0.0
		num_batches = len(train_loader)

		for batch_idx, (images, targets) in enumerate(train_loader):
			state.batch_idx = batch_idx

			for fn in self._on_batch_start:
				fn(state)

			images = images.to(self.device, non_blocking=True)
			targets = targets.to(self.device, non_blocking=True)

			for fn in self._before_forward:
				images, targets = fn(state, images, targets)

			with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
				outputs = self.model(images)
				for fn in self._after_forward:
					outputs = fn(state, outputs)
				# 缩放 loss，使累计梯度与整批平均梯度处于同一尺度。
				loss = self.criterion(outputs, targets) / self.accum_iter

			self.scaler.scale(loss).backward()

			# 最后一个不完整累积窗口也必须提交梯度。
			if (batch_idx + 1) % self.accum_iter == 0 or (batch_idx + 1) == num_batches:
				if self.clip_grad is not None:
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
				self.scaler.step(self.optimizer)
				self.scaler.update()
				self.optimizer.zero_grad(set_to_none=True)
				state.global_step += 1

			batch_loss = loss.item() * self.accum_iter
			state.batch_loss = batch_loss
			total_loss += batch_loss

			for fn in self._on_batch_end:
				fn(state)

		return total_loss / max(num_batches, 1)
