"""Training engine (P072).

`Engine` runs a step function over batches with the mechanics every Stage H run needs and none of the task logic:

* AMP (`torch.autocast` + `GradScaler` on CUDA; a no-op on CPU), gradient accumulation (`accumulate` micro-batches
  per optimizer step), gradient clipping (global norm), an EMA copy of the parameters (`ema_decay`; evaluation
  runs on the EMA weights through `ema_context`), a NaN guard (a non-finite loss skips the step, zeroes the
  gradients and counts; more than `max_nan_steps` in a row raises), checkpoint / resume (model, optimizer,
  scheduler, scaler, EMA, step, epoch, position inside the epoch, RNG states, early-stopping state; written
  atomically), early stopping on a validation metric with a patience, and per-step logging through a `Run`
  (P005) or any `log(step, **metrics)` callable.
* The step function receives `(model, batch, step)` and returns `(loss, metrics)`; `eval_fn(model)` returns a
  metrics dict holding the monitored key. The engine knows nothing about tasks, samplers or heads.

Resume contract (tested): a run interrupted after a checkpoint and resumed in a fresh process ends with the same
parameters and metrics as the uninterrupted run, because the data order is a function of (seed, epoch), the
position inside the epoch is checkpointed and replayed, and the RNG states travel in the checkpoint.
"""

from __future__ import annotations

import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class EngineConfig:
    epochs: int = 1
    accumulate: int = 1
    clip_norm: float | None = 1.0
    amp: bool = True
    ema_decay: float | None = 0.999
    max_nan_steps: int = 10
    monitor: str = "val_loss"
    mode: str = "min"                      # min | max
    patience: int | None = None            # evaluations without improvement before stopping; None: never
    min_delta: float = 0.0
    checkpoint_every: int | None = None    # optimizer steps between 'last' checkpoints; None: once per epoch
    log_every: int = 50
    device: str = "cpu"
    extra: dict = field(default_factory=dict)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


class _Swap:
    """Context manager: the EMA weights in the model for its duration, the live weights restored after."""

    def __init__(self, model: nn.Module, ema: EMA | None):
        self.model, self.ema, self.backup = model, ema, None

    def __enter__(self):
        if self.ema is not None:
            self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items() if k in self.ema.shadow}
            self.model.load_state_dict({**self.model.state_dict(), **self.ema.shadow})
        return self.model

    def __exit__(self, *exc):
        if self.backup is not None:
            self.model.load_state_dict({**self.model.state_dict(), **self.backup})
        return False


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class Engine:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, cfg: EngineConfig,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None, log: Callable | None = None,
                 checkpoint_dir: str | Path | None = None):
        self.model, self.opt, self.cfg, self.sched = model, optimizer, cfg, scheduler
        self.log = log or (lambda step, **m: None)
        self.ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.use_amp = cfg.amp and cfg.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
        self.step = 0                 # optimizer steps taken
        self.epoch = 0                # epochs completed
        self.epoch_pos = 0            # micro-batches consumed in the current epoch
        self.micro = 0                # micro-batches accumulated toward the next optimizer step
        self.nan_streak = 0
        self.nan_total = 0
        self.best: float | None = None
        self.best_step = 0
        self.bad_evals = 0
        self.history: list[dict] = []
        self.stopped = False

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else _NullContext()

    # ------------------------------------------------------------------------------------------ checkpoints
    def state_dict(self) -> dict:
        return {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict() if self.sched else None, "scaler": self.scaler.state_dict(),
                "ema": self.ema.state_dict() if self.ema else None, "step": self.step, "epoch": self.epoch,
                "epoch_pos": self.epoch_pos, "nan_total": self.nan_total, "best": self.best, "best_step": self.best_step,
                "bad_evals": self.bad_evals, "stopped": self.stopped, "history": self.history, "rng": _rng_state(),
                "cfg": self.cfg.__dict__}

    def save(self, path: str | Path | None = None, tag: str = "last") -> Path:
        path = Path(path) if path else self.ckpt_dir / f"{tag}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)
        return path

    def load(self, path: str | Path) -> None:
        state = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["optimizer"])
        if self.sched and state["scheduler"]:
            self.sched.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        if self.ema and state["ema"]:
            self.ema.load_state_dict(state["ema"])
        for k in ("step", "epoch", "epoch_pos", "nan_total", "best", "best_step", "bad_evals", "stopped"):
            setattr(self, k, state[k])
        self.history = list(state["history"])
        self.micro = 0
        _set_rng_state(state["rng"])

    def ema_context(self):
        return _Swap(self.model, self.ema)

    # ------------------------------------------------------------------------------------------ training
    def train_step(self, batch, step_fn: Callable) -> dict | None:
        """One micro-batch; an optimizer step every `accumulate` micro-batches. Returns the step's metrics when an
        optimizer step happened, else None. A non-finite loss is skipped (NaN guard)."""
        self.model.train()
        with self._autocast():
            loss, metrics = step_fn(self.model, batch, self.step)
        if not torch.isfinite(loss):
            self.nan_streak += 1
            self.nan_total += 1
            self.opt.zero_grad(set_to_none=True)
            self.micro = 0
            if self.nan_streak > self.cfg.max_nan_steps:
                raise FloatingPointError(f"{self.nan_streak} non-finite losses in a row at step {self.step}")
            return None
        self.nan_streak = 0
        self.scaler.scale(loss / self.cfg.accumulate).backward()
        self.micro += 1
        if self.micro < self.cfg.accumulate:
            return None
        self.micro = 0
        self.scaler.unscale_(self.opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_norm or math.inf))
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        if self.sched is not None:
            self.sched.step()
        if self.ema is not None:
            self.ema.update(self.model)
        self.step += 1
        out = {"loss": float(loss), "grad_norm": grad_norm, "lr": self.opt.param_groups[0]["lr"],
               **{k: float(v) for k, v in metrics.items()}}
        if self.cfg.log_every and self.step % self.cfg.log_every == 0:
            self.log(self.step, **out)
        return out

    def evaluate(self, eval_fn: Callable, use_ema: bool = True) -> dict:
        with self.ema_context() if use_ema else _NullContext():
            self.model.eval()
            with torch.no_grad():
                metrics = eval_fn(self.model)
        self.history.append({"step": self.step, "epoch": self.epoch, **metrics})
        self.log(self.step, **{f"val/{k}": float(v) for k, v in metrics.items()})
        value = float(metrics[self.cfg.monitor])
        better = (self.best is None or (value < self.best - self.cfg.min_delta if self.cfg.mode == "min"
                                        else value > self.best + self.cfg.min_delta))
        if better:
            self.best, self.best_step, self.bad_evals = value, self.step, 0
            if self.ckpt_dir:
                self.save(tag="best")
        else:
            self.bad_evals += 1
            if self.cfg.patience is not None and self.bad_evals >= self.cfg.patience:
                self.stopped = True
        return metrics

    def fit(self, loader_fn: Callable, step_fn: Callable, eval_fn: Callable | None = None,
            resume: str | Path | None = None) -> dict:
        """loader_fn(epoch) -> an iterable of batches whose order depends only on the epoch (and a fixed seed);
        evaluation at the end of every epoch; 'last' checkpoints every `checkpoint_every` optimizer steps and at
        every epoch end. `resume` continues from a checkpoint, replaying the epoch's batches up to the saved position."""
        if resume:
            self.load(resume)
        t0 = time.perf_counter()
        for epoch in range(self.epoch, self.cfg.epochs):
            if self.stopped:
                break
            self.epoch = epoch
            offset = self.epoch_pos
            for i, batch in enumerate(loader_fn(epoch)):
                if i < offset:
                    continue
                stepped = self.train_step(batch, step_fn)
                self.epoch_pos = i + 1
                if (stepped is not None and self.cfg.checkpoint_every and self.ckpt_dir
                        and self.step % self.cfg.checkpoint_every == 0):
                    self.save(tag="last")
            self.epoch_pos = 0
            if eval_fn is not None:
                self.evaluate(eval_fn)
            self.epoch = epoch + 1
            if self.ckpt_dir:
                self.save(tag="last")
        return {"steps": self.step, "epochs": self.epoch, "best": self.best, "best_step": self.best_step,
                "nan_steps": self.nan_total, "seconds": time.perf_counter() - t0, "stopped_early": self.stopped,
                "history": self.history}
