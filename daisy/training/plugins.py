"""Trainer 内置插件。

插件通过 ``register()`` 注册回调，在自身实例中维护状态，并通过构造器注入依赖。
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import icontract
import torch
from rich.console import Console
from rich.table import Table

from daisy.training._tui import EpochRecorded, Progress, TrainMonitorApp, format_value
from daisy.training.state import TrainState

if TYPE_CHECKING:
	from daisy.training.trainer import Trainer


class Plugin:
	"""通过 ``register()`` 将回调挂载到 Trainer 的插件接口。"""

	def register(self, trainer: Trainer) -> None:
		raise NotImplementedError


class CosineAnnealingLR(Plugin):
	"""Warmup + Cosine Annealing 学习率调度。

	支持 per-epoch 和 per-iteration 两种粒度。
	支持 layer-wise lr scale（通过 param_group['lr_scale']）。
	"""

	@icontract.require(lambda warmup_epochs: warmup_epochs >= 0)
	@icontract.require(lambda lr, min_lr: 0 <= min_lr <= lr)
	@icontract.require(lambda lr: lr > 0)
	def __init__(
		self,
		lr: float,
		min_lr: float = 1e-6,
		warmup_epochs: int = 5,
		per_iteration: bool = False,
	):
		self.lr = lr
		self.min_lr = min_lr
		self.warmup_epochs = warmup_epochs
		self.per_iteration = per_iteration

	def register(self, trainer: Trainer) -> None:
		if self.per_iteration:
			trainer.on_batch_start(self._step)
		else:
			trainer.on_epoch_start(self._step)

	@icontract.require(lambda epoch: epoch >= 0)
	@icontract.ensure(lambda self, result: 0 <= result <= self.lr * (1 + 1e-9))
	def _calc_lr(self, epoch: float, total_epochs: int) -> float:
		if epoch < self.warmup_epochs:
			return self.lr * epoch / max(self.warmup_epochs, 1e-8)
		progress = (epoch - self.warmup_epochs) / max(total_epochs - self.warmup_epochs, 1)
		return self.min_lr + (self.lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

	@icontract.require(lambda state: state.optimizer is not None, 'CosineAnnealingLR 需要 optimizer。')
	def _step(self, state: TrainState) -> None:
		optimizer = state.optimizer
		assert optimizer is not None

		if self.per_iteration:
			frac = state.epoch + state.batch_idx / max(state.total_batches, 1)
		else:
			frac = float(state.epoch)
		lr = self._calc_lr(frac, state.total_epochs)
		for pg in optimizer.param_groups:
			pg['lr'] = lr * pg.get('lr_scale', 1.0)


class Eval(Plugin):
	"""每个 epoch 后调用 ``eval_fn(trainer)`` 执行验证并将指标发布到当前 epoch 记录。"""

	def __init__(
		self,
		eval_fn: Callable[[Trainer], dict[str, Any]],
		namespace: str = 'val',
	):
		self.eval_fn = eval_fn
		self.namespace = namespace

	def register(self, trainer: Trainer) -> None:
		def _eval(state: TrainState) -> None:
			metrics = self.eval_fn(trainer)
			for key, value in metrics.items():
				state.record[f'{self.namespace}/{key}'] = value

		trainer.on_epoch_end(_eval)


class BestModel(Plugin):
	"""根据关注的指标保存最佳模型参数。"""

	def __init__(
		self,
		watch_metric: str,
		mode: Literal['max', 'min'] = 'max',
		save_path: Path | None = None,
	):
		self.watch_metric = watch_metric
		self.mode = mode
		self.save_path = save_path

		self.best_value: float = -float('inf') if mode == 'max' else float('inf')
		self.best_epoch: int = -1
		if self.save_path:
			self.save_path.mkdir(parents=True, exist_ok=True)

	def register(self, trainer: Trainer) -> None:
		trainer.on_epoch_end(self._check)

	def _is_better(self, current: float) -> bool:
		return current > self.best_value if self.mode == 'max' else current < self.best_value

	def _check(self, state: TrainState) -> None:
		val = state.record.get(self.watch_metric)
		if val is None:
			return
		if self._is_better(val):
			self.best_value = val
			self.best_epoch = state.epoch
			if self.save_path:
				torch.save(state.model.state_dict(), self.save_path / 'best_model.pth')


class EarlyStop(Plugin):
	"""指标连续 ``patience`` 个 epoch 未改善时请求终止训练。"""

	@icontract.require(lambda patience: patience >= 1)
	def __init__(
		self,
		watch_metric: str,
		patience: int = 10,
		mode: Literal['max', 'min'] = 'max',
	):
		self.watch_metric = watch_metric
		self.patience = patience
		self.mode = mode

		self._best: float = -float('inf') if mode == 'max' else float('inf')
		self._wait: int = 0

	def register(self, trainer: Trainer) -> None:
		trainer.on_epoch_end(self._check)

	def _is_better(self, current: float) -> bool:
		return current > self._best if self.mode == 'max' else current < self._best

	def _check(self, state: TrainState) -> None:
		val = state.record.get(self.watch_metric)
		if val is None:
			return
		if self._is_better(val):
			self._best = val
			self._wait = 0
		else:
			self._wait += 1
			if self._wait >= self.patience:
				print(f'Early stopping at epoch {state.epoch + 1} ')
				state.stop_training = True


class Checkpoint(Plugin):
	"""按频率保存 checkpoint，并可滚动保留最近若干个。"""

	@icontract.require(lambda keep_recent: keep_recent >= 0)
	@icontract.require(lambda save_freq: save_freq >= 0)
	def __init__(
		self,
		save_dir: Path,
		save_freq: int = 0,
		keep_recent: int = 0,
	):
		self.save_dir = save_dir
		self.save_freq = save_freq
		self.keep_recent = keep_recent

	def register(self, trainer: Trainer) -> None:
		trainer.on_fit_start(self._init_dir)
		trainer.on_epoch_end(self._save)

	def _init_dir(self, state: TrainState) -> None:
		self.save_dir.mkdir(parents=True, exist_ok=True)

	def _save(self, state: TrainState) -> None:
		ep = state.epoch + 1

		if self.save_freq > 0 and ep % self.save_freq == 0:
			torch.save(state.model.state_dict(), self.save_dir / f'checkpoint_{ep:04d}.pth')

		if self.keep_recent > 0:
			torch.save(state.model.state_dict(), self.save_dir / f'checkpoint_{ep:04d}.pth')
			old_ep = ep - self.keep_recent
			if old_ep > 0:
				old_path = self.save_dir / f'checkpoint_{old_ep:04d}.pth'
				if old_path.exists():
					old_path.unlink()


class EpochPrint(Plugin):
	"""每个 epoch 后打印训练进度及关注的指标。"""

	def __init__(self, watch: Iterable[str]):
		self.watch: tuple[str, ...] = tuple(watch)

	def register(self, trainer: Trainer) -> None:
		trainer.on_epoch_end(self._print)

	def _print(self, state: TrainState) -> None:
		parts = [f'Epoch {state.epoch + 1}/{state.total_epochs}']
		record = state.record
		for key in self.watch:
			if key in record:
				val = record[key]
				parts.append(f'{key}={val:.4f}' if isinstance(val, float) else f'{key}={val}')
		print('  '.join(parts))


class RichEpochPrint(Plugin):
	"""Textual 全屏训练监控：上方为逐 epoch 指标表，下方接管训练期间的 print，二者均可滚动回看。

	界面占用主线程，训练在工作线程中执行。训练结束后按 q 退出，随后将输出与指标表回放到终端；
	训练中连按两次 q 中止训练（抛出 ``KeyboardInterrupt``）。
	非交互终端或 TERM=dumb 时自动使用 EpochPrint。
	"""

	PROGRESS_INTERVAL = 0.2

	@icontract.require(lambda max_log_lines: max_log_lines >= 1)
	def __init__(self, watch: Iterable[str], max_log_lines: int = 10000):
		self.watch: tuple[str, ...] = tuple(watch)
		self.max_log_lines = max_log_lines
		self.history: list[dict[str, Any]] = []

		self._app: TrainMonitorApp | None = None
		self._last_progress: float = 0.0

	def register(self, trainer: Trainer) -> None:
		if not (sys.stdin.isatty() and sys.stdout.isatty()) or os.environ.get('TERM') == 'dumb':
			print('RichEpochPrint: non-interactive terminal detected; using EpochPrint.')
			EpochPrint(self.watch).register(trainer)
			return

		trainer.around_fit(self._wrap)
		trainer.on_epoch_start(self._on_epoch_start)
		trainer.on_batch_end(self._on_batch_end)
		trainer.on_epoch_end(self._on_epoch_end)

	@property
	def _columns(self) -> tuple[str, ...]:
		return ('epoch', *self.watch)

	@icontract.require(lambda self: self._app is not None, 'RichEpochPrint 回调只能在 fit 期间触发。')
	def _require_app(self) -> TrainMonitorApp:
		app = self._app
		assert app is not None
		if app.abort.is_set():
			raise KeyboardInterrupt('用户在界面中中止训练。')
		return app

	def _wrap(self, run: Callable[[], TrainState]) -> TrainState:
		self.history = []
		app = TrainMonitorApp(run, self._columns, self.max_log_lines)
		self._app = app
		app.run()
		# 用户中止或界面崩溃时训练线程仍在运行，需等待其在下一个回调处终止
		if app.worker.is_alive():
			app.abort.set()
			app.worker.join()
		self._app = None

		self._replay(app)
		if app.return_code:
			raise RuntimeError(f'RichEpochPrint 界面异常退出（return_code={app.return_code}）。') from app.error
		if app.error is not None:
			raise app.error
		if app.result is None:
			raise RuntimeError('RichEpochPrint 界面退出时训练未执行。')
		return app.result

	def _replay(self, app: TrainMonitorApp) -> None:
		for line, stderr in app.print_history:
			print(line, file=sys.stderr if stderr else sys.stdout)

		table = Table(title='Training metrics')
		for column in self._columns:
			table.add_column(column, justify='right')
		for row in self.history:
			table.add_row(*(format_value(row[column]) for column in self._columns))
		Console().print(table)

	def _on_epoch_start(self, state: TrainState) -> None:
		self._require_app().post_from_worker(Progress(f'Epoch {state.epoch + 1}/{state.total_epochs}'))

	def _on_batch_end(self, state: TrainState) -> None:
		app = self._require_app()
		now = time.monotonic()
		idx = state.batch_idx + 1
		if now - self._last_progress >= self.PROGRESS_INTERVAL or idx == state.total_batches:
			self._last_progress = now
			app.post_from_worker(Progress(f'Epoch {state.epoch + 1}/{state.total_epochs} · batch {idx}/{state.total_batches}'))

	def _on_epoch_end(self, state: TrainState) -> None:
		app = self._require_app()
		row: dict[str, Any] = {
			'epoch': state.epoch + 1,
			**{key: state.record.get(key) for key in self.watch},
		}
		self.history.append(row)
		app.post_from_worker(EpochRecorded([format_value(row[column]) for column in self._columns]))


class BatchPrint(Plugin):
	"""按指定频率打印 batch 进度。"""

	@icontract.require(lambda print_freq: print_freq >= 0)
	def __init__(self, print_freq: int = 50):
		self.print_freq = print_freq

	def register(self, trainer: Trainer) -> None:
		trainer.on_batch_end(self._print)

	def _print(self, state: TrainState) -> None:
		idx = state.batch_idx + 1
		if self.print_freq > 0 and (idx % self.print_freq == 0 or idx == state.total_batches):
			print(f'  [{idx}/{state.total_batches}] loss={state.batch_loss:.4f} lr={state.current_lr:.6f}')


class CSVLog(Plugin):
	"""将关注的指标写入 CSV，并在 ``history`` 中保留独立快照。"""

	def __init__(
		self,
		log_path: Path,
		watch: Iterable[str],
	):
		self.log_path = log_path
		self.watch: tuple[str, ...] = tuple(watch)
		self.history: list[dict[str, Any]] = []

	def register(self, trainer: Trainer) -> None:
		trainer.on_fit_start(self._init_fit)
		trainer.on_epoch_end(self._write)

	def _init_fit(self, state: TrainState) -> None:
		self.history = []
		self.log_path.parent.mkdir(parents=True, exist_ok=True)
		with open(self.log_path, 'w', newline='', encoding='utf-8-sig') as f:
			writer = csv.writer(f)
			writer.writerow(['epoch', *self.watch])

	def _write(self, state: TrainState) -> None:
		row: dict[str, Any] = {
			'epoch': state.epoch + 1,
			**{key: state.record.get(key, None) for key in self.watch},
		}
		self.history.append(row)

		with open(self.log_path, 'a', newline='', encoding='utf-8-sig') as f:
			writer = csv.writer(f)
			writer.writerow([row['epoch'], *[row[key] if row[key] is not None else '' for key in self.watch]])


class Timer(Plugin):
	"""记录每个 epoch 的耗时并发布到当前 epoch 记录。"""

	def __init__(self) -> None:
		self._t0: float = 0.0

	def register(self, trainer: Trainer) -> None:
		trainer.on_epoch_start(self._start)
		trainer.on_epoch_end(self._end)

	def _start(self, state: TrainState) -> None:
		self._t0 = time.perf_counter()

	def _end(self, state: TrainState) -> None:
		state.record['time/epoch'] = time.perf_counter() - self._t0
