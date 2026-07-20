"""Trainer 内置插件。

插件通过 ``register()`` 注册回调，在自身实例中维护状态，并通过构造器注入依赖。
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

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

    def _calc_lr(self, epoch: float, total_epochs: int) -> float:
        if epoch < self.warmup_epochs:
            return self.lr * epoch / max(self.warmup_epochs, 1e-8)
        progress = (epoch - self.warmup_epochs) / max(total_epochs - self.warmup_epochs, 1)
        return self.min_lr + (self.lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _step(self, state: TrainState) -> None:
        optimizer = state.optimizer
        if optimizer is None:
            raise RuntimeError('CosineAnnealingLR 需要 optimizer。')

        if self.per_iteration:
            frac = state.epoch + state.batch_idx / max(state.total_batches, 1)
        else:
            frac = float(state.epoch)
        lr = self._calc_lr(frac, state.total_epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_scale', 1.0)


class Eval(Plugin):
    """每个 epoch 后调用 ``eval_fn(trainer)`` 执行验证。

    结果保存在 ``metrics`` 中，供依赖此实例的插件读取。

    用法：
        evaluator = Eval(eval_fn=lambda trainer: {'pc': ..., 'mae': ...})
        best = BestModel(evaluator, watch_metric='pc')
    """

    def __init__(self, eval_fn: Callable[[Trainer], dict[str, float]]):
        self.eval_fn = eval_fn
        self.metrics: dict[str, float] = {}
        self._trainer: Trainer | None = None

    def register(self, trainer: Trainer) -> None:
        self._trainer = trainer
        trainer.on_epoch_end(self._eval)

    def _eval(self, state: TrainState) -> None:
        if self._trainer is None:
            raise RuntimeError('Eval 尚未注册到 Trainer。')
        self.metrics = self.eval_fn(self._trainer)


class BestModel(Plugin):
    """根据 Eval 指标保存最佳模型参数。"""

    def __init__(
        self,
        evaluator: Eval,
        watch_metric: str = 'pc',
        mode: Literal['max', 'min'] = 'max',
        save_path: Path | str | None = None,
    ):
        self.evaluator = evaluator
        self.watch_metric = watch_metric
        self.mode = mode
        self.save_path = Path(save_path) if save_path else None

        self.best_value: float = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch: int = -1

    def register(self, trainer: Trainer) -> None:
        trainer.on_epoch_end(self._check)

    def _is_better(self, current: float) -> bool:
        return current > self.best_value if self.mode == 'max' else current < self.best_value

    def _check(self, state: TrainState) -> None:
        val = self.evaluator.metrics.get(self.watch_metric)
        if val is None:
            return
        if self._is_better(val):
            self.best_value = val
            self.best_epoch = state.epoch
            if self.save_path:
                self.save_path.mkdir(parents=True, exist_ok=True)
                torch.save(state.model.state_dict(), self.save_path / 'best_model.pth')


class EarlyStop(Plugin):
    """指标连续 ``patience`` 个 epoch 未改善时请求终止训练。"""

    def __init__(
        self,
        evaluator: Eval,
        patience: int = 10,
        watch_metric: str = 'pc',
        mode: Literal['max', 'min'] = 'max',
    ):
        self.evaluator = evaluator
        self.patience = patience
        self.watch_metric = watch_metric
        self.mode = mode

        self._best: float = -float('inf') if mode == 'max' else float('inf')
        self._wait: int = 0

    def register(self, trainer: Trainer) -> None:
        trainer.on_epoch_end(self._check)

    def _is_better(self, current: float) -> bool:
        return current > self._best if self.mode == 'max' else current < self._best

    def _check(self, state: TrainState) -> None:
        val = self.evaluator.metrics.get(self.watch_metric)
        if val is None:
            return
        if self._is_better(val):
            self._best = val
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                print(
                    f'Early stopping at epoch {state.epoch + 1} '
                )
                state.extras['stop_training'] = True


class Checkpoint(Plugin):
    """按频率保存 checkpoint，并可滚动保留最近若干个。"""

    def __init__(
        self,
        save_dir: Path | str,
        save_freq: int = 0,
        keep_recent: int = 0,
    ):
        self.save_dir = Path(save_dir)
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
    """每个 epoch 后打印训练指标及可选的 Eval 指标。

    ``metric_names`` 为 None 时打印 Eval 提供的全部指标。
    """

    def __init__(
        self,
        evaluator: Eval | None = None,
        metric_names: list[str] | None = None,
    ):
        self.evaluator = evaluator
        self.metric_names = metric_names

    def register(self, trainer: Trainer) -> None:
        trainer.on_epoch_end(self._print)

    def _print(self, state: TrainState) -> None:
        parts = [
            f'Epoch {state.epoch + 1}/{state.total_epochs}',
            f'lr={state.current_lr:.6f}',
            f'train_loss={state.epoch_train_loss:.4f}',
        ]
        if self.evaluator:
            keys = self.metric_names if self.metric_names else list(self.evaluator.metrics.keys())
            for k in keys:
                if k in self.evaluator.metrics:
                    parts.append(f'val_{k}={self.evaluator.metrics[k]:.4f}')
        print('  '.join(parts))


class BatchPrint(Plugin):
    """按指定频率打印 batch 进度。"""

    def __init__(self, print_freq: int = 50):
        self.print_freq = print_freq

    def register(self, trainer: Trainer) -> None:
        trainer.on_batch_end(self._print)

    def _print(self, state: TrainState) -> None:
        idx = state.batch_idx + 1
        if self.print_freq > 0 and (idx % self.print_freq == 0 or idx == state.total_batches):
            print(
                f'  [{idx}/{state.total_batches}] '
                f'loss={state.batch_loss:.4f} lr={state.current_lr:.6f}'
            )


class CSVLog(Plugin):
    """将每个 epoch 的指标写入 CSV，并在 ``history`` 中保留记录。"""

    def __init__(
        self,
        log_path: Path | str,
        evaluator: Eval | None = None,
    ):
        self.log_path = Path(log_path)
        self.evaluator = evaluator
        self.history: list[dict[str, Any]] = []
        self._header_written = False

    def register(self, trainer: Trainer) -> None:
        trainer.on_fit_start(self._init_file)
        trainer.on_epoch_end(self._write)

    def _init_file(self, state: TrainState) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, state: TrainState) -> None:
        record: dict[str, Any] = {
            'epoch': state.epoch + 1,
            'lr': state.current_lr,
            'train_loss': state.epoch_train_loss,
        }
        if self.evaluator:
            record.update({f'val_{k}': v for k, v in self.evaluator.metrics.items()})

        self.history.append(record)

        mode = 'a' if self._header_written else 'w'
        with open(self.log_path, mode, newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(record)


class Timer(Plugin):
    """记录每个 epoch 的耗时，结果保存在 ``elapsed`` 中。"""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def register(self, trainer: Trainer) -> None:
        trainer.on_epoch_start(self._start)
        trainer.on_epoch_end(self._end)

    def _start(self, state: TrainState) -> None:
        self._t0 = time.perf_counter()

    def _end(self, state: TrainState) -> None:
        self.elapsed = time.perf_counter() - self._t0


class Mixup(Plugin):
    """在 forward 前应用 Mixup/CutMix；依赖 ``timm.data.mixup.Mixup``。"""

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        smoothing: float = 0.1,
        num_classes: int = 2,
    ):
        from timm.data.mixup import Mixup as TimmMixup

        self.mixup_fn = TimmMixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=prob,
            switch_prob=switch_prob,
            mode='batch',
            label_smoothing=smoothing,
            num_classes=num_classes,
        )

    def register(self, trainer: Trainer) -> None:
        trainer.before_forward(self._apply)

    def _apply(
        self, state: TrainState, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mixup_fn(images, targets)
