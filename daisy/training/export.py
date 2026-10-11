"""最佳参数快照的同步与异步导出策略。"""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from threading import Condition, Thread
from typing import Any, Literal

import icontract
import torch


def _atomic_save(snapshot: dict[str, Any], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
		temporary = Path(file.name)
	try:
		torch.save(snapshot, temporary)
		temporary.replace(path)
	finally:
		temporary.unlink(missing_ok=True)


class InstantExport:
	"""每次提交时同步原子写入。"""

	def __init__(self, path: Path):
		self.path = path

	def submit(self, snapshot: dict[str, Any]) -> None:
		_atomic_save(snapshot, self.path)

	def start(self) -> None:
		pass

	def check(self) -> None:
		pass

	def close(self, *, success: bool) -> None:
		pass


class FinalExport(InstantExport):
	"""成功结束时写入最后一次提交的快照。"""

	def __init__(self, path: Path):
		super().__init__(path)
		self._pending: dict[str, Any] | None = None

	def submit(self, snapshot: dict[str, Any]) -> None:
		self._pending = snapshot

	def close(self, *, success: bool) -> None:
		pending, self._pending = self._pending, None
		if success and pending is not None:
			_atomic_save(pending, self.path)


class DebouncedExport(InstantExport):
	"""单个写入线程以尾随延迟合并最新快照。"""

	@icontract.require(lambda debounce_seconds: debounce_seconds > 0)
	def __init__(self, path: Path, debounce_seconds: float = 30.0):
		super().__init__(path)
		self.debounce_seconds = debounce_seconds
		self._condition = Condition()
		self._pending: dict[str, Any] | None = None
		self._deadline = 0.0
		self._closing = False
		self._worker: Thread | None = None
		self._result: Future[None] = Future()
		self._error_reported = False

	def start(self) -> None:
		# 先结束上一轮资源，再为本轮创建独立的完成状态；线程仍在首次提交时启动。
		if self._worker is not None:
			self.close(success=False)
		with self._condition:
			self._pending = None
			self._deadline = 0.0
			self._closing = False
			self._worker = None
			self._result = Future()
			self._error_reported = False

	def check(self) -> None:
		if self._result.done() and not self._error_reported:
			try:
				self._result.result()
			except BaseException:
				self._error_reported = True
				raise

	def submit(self, snapshot: dict[str, Any]) -> None:
		self.check()
		with self._condition:
			if self._closing:
				raise RuntimeError('Exporter is closed')
			self._pending = snapshot
			self._deadline = time.monotonic() + self.debounce_seconds
			if self._worker is None:
				self._worker = Thread(target=self._run, name='BestModel export')
				self._worker.start()
			self._condition.notify_all()
		self.check()

	def _run(self) -> None:
		try:
			while True:
				with self._condition:
					while True:
						if self._closing and self._pending is None:
							self._result.set_result(None)
							return
						if self._pending is not None:
							delay = self._deadline - time.monotonic()
							if self._closing or delay <= 0:
								snapshot, self._pending = self._pending, None
								break
							self._condition.wait(delay)
						else:
							self._condition.wait()
				_atomic_save(snapshot, self.path)
		except BaseException as error:
			self._result.set_exception(error)

	def close(self, *, success: bool) -> None:
		with self._condition:
			self._closing = True
			if not success:
				self._pending = None
			self._condition.notify_all()
		if self._worker is not None:
			self._worker.join()
			with self._condition:
				self._pending = None
			self.check()


def make_best_export(
	path: Path,
	mode: Literal['instant', 'final', 'debounced'],
	*,
	debounce_seconds: float,
) -> InstantExport:
	constructors = {
		'instant': InstantExport,
		'final': FinalExport,
		'debounced': partial(DebouncedExport, debounce_seconds=debounce_seconds),
	}
	return constructors[mode](path)
