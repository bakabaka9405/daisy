from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from daisy.typing import GenericDataLoader as DataLoader
from daisy.util.data import Prefetcher

from daisy.training.state import TrainState
from torch.utils._pytree import tree_map


SideEffectFn = Callable[[TrainState], None]
ProcessFn = Callable[[Any], torch.Tensor]
FitFn = Callable[[], TrainState]
FitWrapper = Callable[[FitFn], TrainState]


class Trainer:
	"""固定训练循环，业务逻辑通过 ``process_fn`` / ``predict_fn`` 闭包注入。

	每个 batch 由 Prefetcher 完整搬运到设备；``process_fn`` 在 autocast 下返回
	保留计算图的标量 loss，Trainer 负责反向、梯度累积、梯度裁剪、scaler 与优化器更新。
	当前 epoch 的标量指标保存在 ``state.record`` 中，并在下一轮开始前清空。
	"""

	def __init__(
		self,
		model: nn.Module | None = None,
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
		self._process_fn: ProcessFn | None = None
		if process_fn is not None:
			self.set_process_fn(process_fn)
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
		self._around_fit: list[FitWrapper] = []

	@property
	def process_fn(self) -> ProcessFn | None:
		"""获取当前绑定的 process_fn。"""
		return self._process_fn

	def set_process_fn(self, process_fn: ProcessFn) -> Trainer:
		"""设置训练处理闭包，至多允许绑定一次（包含构造函数初始化）。"""
		if self._process_fn is not None:
			raise RuntimeError('process_fn has already been set.')
		if not callable(process_fn):
			raise TypeError('process_fn must be callable.')

		self._process_fn = process_fn
		return self

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

	def around_fit(self, fn: FitWrapper) -> Trainer:
		"""包裹整个训练过程：``fn(run)`` 必须调用 ``run()`` 并返回其结果。先注册者位于最外层。"""
		self._around_fit.append(fn)
		return self

	def use(self, *plugins: Any) -> Trainer:
		"""注册 Plugin。"""
		for plugin in plugins:
			plugin.register(self)
		return self

	def fit(self, train_loader: DataLoader, epochs: int, *, start_epoch: int = 0) -> TrainState:
		"""执行 ``[start_epoch, epochs)`` 区间训练并返回最终状态。"""
		if self.model is None:
			raise RuntimeError('fit 需要 model。')
		if self.optimizer is None:
			raise RuntimeError('fit 需要 optimizer。')
		if self.process_fn is None:
			raise RuntimeError('fit 需要 process_fn。')

		run: FitFn = partial(self._fit, train_loader, epochs, start_epoch)
		for wrapper in reversed(self._around_fit):
			run = partial(wrapper, run)
		return run()

	def _fit(self, train_loader: DataLoader, epochs: int, start_epoch: int) -> TrainState:
		if self.model is None:
			raise RuntimeError('Model is required for _fit.')
		if self.optimizer is None:
			raise RuntimeError('Optimizer is required for _fit.')

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
			state.record.clear()
			state.epoch = epoch
			state.total_batches = len(loader)

			for fn in self._on_epoch_start:
				fn(state)

			mean_loss = self._train_one_epoch(loader, state)
			state.record['train/loss'] = mean_loss
			state.record['optim/lr'] = optimizer.param_groups[0]['lr']

			for fn in self._on_epoch_end:
				fn(state)

			if state.stop_training:
				break

		for fn in self._on_fit_end:
			fn(state)

		return state

	def _train_one_epoch(self, train_loader: Prefetcher, state: TrainState) -> float:
		if self.model is None:
			raise RuntimeError('Model is required for _train_one_epoch.')
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
				# if not isinstance(raw_loss, torch.Tensor) or raw_loss.ndim != 0:
				# 	raise TypeError('process_fn 必须返回标量 Tensor loss。')
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
	def inference[T](self, data_loader: DataLoader, predict_fn: Callable[[Any], T]) -> T:
		"""按 batch 顺序拼接预测结果，返回保持 pytree 结构的 CPU 张量。"""
		if self.model is None:
			raise RuntimeError('inference 需要 model。')

		loader = data_loader if isinstance(data_loader, Prefetcher) else Prefetcher(data_loader, self.device)
		was_training = self.model.training
		self.model.to(self.device)
		self.model.eval()

		try:
			batch_outputs: list[T] = []
			for batch in loader:
				with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
					outputs = predict_fn(batch)
				batch_outputs.append(tree_map(lambda t: t.cpu(), outputs))

			if not batch_outputs:
				raise ValueError('Data loader yielded no batches for inference.')

			return tree_map(lambda *leaves: torch.cat(leaves, dim=0), *batch_outputs)
		finally:
			self.model.train(was_training)
