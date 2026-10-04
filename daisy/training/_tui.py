"""``RichEpochPrint`` 使用的 Textual 训练监控界面。

界面占用主线程，训练在 ``TrainMonitorApp.worker`` 线程中执行；工作线程只通过
``post_from_worker`` 与界面通信。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import icontract
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, RichLog

from daisy.training.state import TrainState

QUIT_CONFIRM_SECONDS = 3.0


def format_value(value: Any) -> str:
	"""将指标值格式化为表格单元格文本。"""
	if value is None:
		return '-'
	if isinstance(value, float):
		return f'{value:.3e}' if 0 < abs(value) < 1e-3 else f'{value:.4f}'
	return str(value)


class EpochRecorded(Message):
	"""一个 epoch 的指标行（已格式化）。"""

	def __init__(self, row: Sequence[str]) -> None:
		super().__init__()
		self.row = row


class Progress(Message):
	"""训练进度文本，显示在标题栏副标题。"""

	def __init__(self, text: str) -> None:
		super().__init__()
		self.text = text


class TrainingDone(Message):
	"""训练线程结束；``failed`` 表示以异常结束。"""

	def __init__(self, failed: bool) -> None:
		super().__init__()
		self.failed = failed


class MetricTable(DataTable):
	"""逐 epoch 指标表；光标位于末行时自动跟随新行。"""

	def __init__(self, columns: Sequence[str], **kwargs: Any) -> None:
		super().__init__(cursor_type='row', zebra_stripes=True, **kwargs)
		self._column_labels = columns

	def on_mount(self) -> None:
		self.border_title = 'Metrics'
		self.add_columns(*self._column_labels)

	@icontract.require(lambda self, row: len(row) == len(self._column_labels))
	def append_row(self, row: Sequence[str]) -> None:
		follow = self.row_count == 0 or self.cursor_row == self.row_count - 1
		self.add_row(*(Text(cell, justify='right') for cell in row))
		if follow:
			self.move_cursor(row=self.row_count - 1)


class PrintLog(RichLog):
	"""接管 stdout/stderr 的输出面板；按行缓冲，视图位于底部时自动跟随。"""

	def __init__(self, history: list[tuple[str, bool]], max_lines: int, **kwargs: Any) -> None:
		super().__init__(max_lines=max_lines, wrap=True, markup=False, highlight=False, **kwargs)
		self.history = history
		self._partial: dict[bool, str] = {False: '', True: ''}

	def on_mount(self) -> None:
		self.border_title = 'Output'
		self.app.begin_capture_print(self)

	def on_print(self, event: events.Print) -> None:
		*lines, rest = (self._partial[event.stderr] + event.text).split('\n')
		# '\r' 表示回到行首（进度条），仅保留最后一段
		self._partial[event.stderr] = rest.rsplit('\r', 1)[-1]
		for line in lines:
			self._emit(line.rstrip('\r').rsplit('\r', 1)[-1], event.stderr)

	def flush_partial(self) -> None:
		for stderr, rest in self._partial.items():
			if rest:
				self._emit(rest, stderr)
		self._partial = {False: '', True: ''}

	def _emit(self, line: str, stderr: bool) -> None:
		self.history.append((line, stderr))
		self.write(Text(line, style='red' if stderr else ''), scroll_end=self.is_vertical_scroll_end)


class TrainMonitorApp(App[None]):
	"""上方指标表、下方输出日志的训练监控界面。"""

	CSS = """
	#metrics, #output {
		height: 1fr;
		border: round $primary;
	}
	"""

	BINDINGS = [
		Binding('q', 'request_quit', '退出', priority=True),
		Binding('ctrl+c', 'request_quit', '退出', show=False, priority=True),
	]

	def __init__(self, run: Callable[[], TrainState], columns: Sequence[str], max_log_lines: int) -> None:
		super().__init__()
		self._run = run
		self._columns = columns
		self._max_log_lines = max_log_lines

		self.result: TrainState | None = None
		self.error: BaseException | None = None
		self.finished = False
		self.abort = threading.Event()
		self.print_history: list[tuple[str, bool]] = []
		self.worker = threading.Thread(target=self._train, name='RichEpochPrint-fit')

		self._post_lock = threading.Lock()
		self._accepting = True
		self._quit_armed_at = -float('inf')

	def compose(self) -> ComposeResult:
		yield Header(show_clock=True)
		yield MetricTable(self._columns, id='metrics')
		yield PrintLog(self.print_history, self._max_log_lines, id='output')
		yield Footer()

	def on_mount(self) -> None:
		self.title = 'DAISY Training'
		self.sub_title = '启动中…'

	def on_ready(self) -> None:
		# 等所有控件挂载完成（PrintLog 已开始捕获输出）后再启动训练
		self.worker.start()

	def on_unmount(self) -> None:
		with self._post_lock:
			self._accepting = False

	def post_from_worker(self, message: Message) -> None:
		"""工作线程向界面投递消息；界面关闭后静默丢弃。"""
		with self._post_lock:
			if self._accepting:
				self.post_message(message)

	def _train(self) -> None:
		try:
			self.result = self._run()
		except BaseException as e:
			# 不恢复训练：异常交由主线程在界面关闭后原样重新抛出
			self.error = e
		self.post_from_worker(TrainingDone(failed=self.error is not None))

	def on_epoch_recorded(self, message: EpochRecorded) -> None:
		self.query_one(MetricTable).append_row(message.row)

	def on_progress(self, message: Progress) -> None:
		self.sub_title = message.text

	def on_training_done(self, message: TrainingDone) -> None:
		self.query_one(PrintLog).flush_partial()
		self.finished = not message.failed
		self.exit()

	def action_request_quit(self) -> None:
		if self.finished:
			self.exit()
			return
		now = time.monotonic()
		if now - self._quit_armed_at <= QUIT_CONFIRM_SECONDS:
			self.abort.set()
			self.exit()
		else:
			self._quit_armed_at = now
			self.notify(f'训练仍在进行，{QUIT_CONFIRM_SECONDS:.0f} 秒内再按一次以中止训练', severity='warning', timeout=QUIT_CONFIRM_SECONDS)
