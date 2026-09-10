from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from daisy.typing import GenericDataLoader as DataLoader
from daisy.util.data import Prefetcher

from daisy.training.state import TrainState


SideEffectFn = Callable[[TrainState], None]
ProcessFn = Callable[[Any], torch.Tensor]


class Trainer:
	"""固定训练循环，业务逻辑通过 ``process_fn`` / ``predict_fn`` 闭包注入。

	每个 batch 由 Prefetcher 完整搬运到设备；``process_fn`` 在 autocast 下返回
	保留计算图的标量 loss，Trainer 负责反向、梯度累积、梯度裁剪、scaler 与优化器更新。
	"""

	def __init__(
		self,
		model: nn.Module,
		optimizer: torch.optim.Optimizer | None = None,
		process_fn: ProcessFn | None = None,
		device: torch.device | None = None,
		*,
		use_amp: bool = True,
		accum_iter: int = 1,
		clip_grad: float | None = None,
	):
		self.model = model
		self.optimizer = optimizer
		self.process_fn = process_fn
		self.device = device if device is not None else torch.device('cpu')
		self.use_amp = use_amp
		self.accum_iter = accum_iter
		self.clip_grad = clip_grad

		self.scaler = torch.GradScaler(self.device.type, enabled=use_amp)

		self._on_fit_start: list[SideEffectFn] = []
		self._on_fit_end: list[SideEffectFn] = []
		self._on_epoch_start: list[SideEffectFn] = []
		self._on_epoch_end: list[SideEffectFn] = []
		self._on_batch_start: list[SideEffectFn] = []
		self._on_batch_end: list[SideEffectFn] = []

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

	def use(self, *plugins: Any) -> Trainer:
		"""注册 Plugin。"""
		for plugin in plugins:
			plugin.register(self)
		return self

	def fit(self, train_loader: DataLoader, epochs: int, *, start_epoch: int = 0) -> TrainState:
		"""执行 ``[start_epoch, epochs)`` 区间训练并返回最终状态。"""
		if self.optimizer is None:
			raise RuntimeError('fit 需要 optimizer。')
		if self.process_fn is None:
			raise RuntimeError('fit 需要 process_fn。')

		optimizer = self.optimizer
		self.model.to(self.device)

		loader = train_loader if isinstance(train_loader, Prefetcher) else Prefetcher(train_loader, self.device)

		state = TrainState(
			model=self.model,
			optimizer=optimizer,
			device=self.device,
			total_epochs=epochs,
			total_batches=len(loader),
		)

		optimizer.zero_grad(set_to_none=True)

		for fn in self._on_fit_start:
			fn(state)

		for epoch in range(start_epoch, epochs):
			state.epoch = epoch
			state.total_batches = len(loader)

			for fn in self._on_epoch_start:
				fn(state)

			state.epoch_train_loss = self._train_one_epoch(loader, state)

			for fn in self._on_epoch_end:
				fn(state)

			if state.stop_training:
				break

		for fn in self._on_fit_end:
			fn(state)

		return state

	def _train_one_epoch(self, train_loader: Prefetcher, state: TrainState) -> float:
		if self.optimizer is None:
			raise RuntimeError('Optimizer is required for _train_one_epoch.')
		if self.process_fn is None:
			raise RuntimeError('process_fn is required for _train_one_epoch.')

		optimizer = self.optimizer
		process_fn = self.process_fn
		self.model.train()
		total_loss = torch.zeros((), device=self.device)
		num_batches = len(train_loader)

		for batch_idx, batch in enumerate(train_loader):
			state.batch_idx = batch_idx

			for fn in self._on_batch_start:
				fn(state)

			with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
				raw_loss = process_fn(batch)
				if not isinstance(raw_loss, torch.Tensor) or raw_loss.ndim != 0:
					raise TypeError('process_fn 必须返回标量 Tensor loss。')
				# 缩放 loss，使累计梯度与整批平均梯度处于同一尺度。
				loss = raw_loss / self.accum_iter

			self.scaler.scale(loss).backward()

			# 最后一个不完整累积窗口也必须提交梯度。
			if (batch_idx + 1) % self.accum_iter == 0 or (batch_idx + 1) == num_batches:
				if self.clip_grad is not None:
					self.scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
				self.scaler.step(optimizer)
				self.scaler.update()
				optimizer.zero_grad(set_to_none=True)
				state.global_step += 1

			batch_loss = raw_loss.detach()
			state.batch_loss = batch_loss
			total_loss.add_(batch_loss)

			for fn in self._on_batch_end:
				fn(state)

		return (total_loss / max(num_batches, 1)).item()

	@torch.no_grad()
	def inference(self, data_loader: DataLoader, predict_fn: Callable[[Any], torch.Tensor]) -> torch.Tensor:
		"""按 batch 顺序拼接预测结果，返回 CPU 张量。"""
		loader = data_loader if isinstance(data_loader, Prefetcher) else Prefetcher(data_loader, self.device)
		was_training = self.model.training
		self.model.to(self.device)
		self.model.eval()

		try:
			scores: list[torch.Tensor] = []
			for batch in loader:
				with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
					outputs = predict_fn(batch)
				scores.append(outputs.cpu())

			return torch.cat(scores) if scores else torch.empty(0)
		finally:
			self.model.train(was_training)
