"""Run directories, metric logging and the run registry.

Every experiment creates `runs/<UTC timestamp>_<name>_<cfghash>/` containing:
  config.yaml   resolved configuration
  env.json      git SHA (+ dirty flag), Python/torch/CUDA versions, GPU, lock-file hash
  metrics.jsonl one JSON object per logged step/epoch
  summary.json  final metrics
  tb/           TensorBoard events (optional)
and appends one row to `runs/registry.csv` when it finishes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .config import config_hash

REGISTRY_FIELDS = ["run_id", "started_utc", "ended_utc", "phase", "name", "git_sha", "git_dirty",
                   "config_hash", "data_hash", "status", "metrics", "run_dir"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment_snapshot() -> dict:
    import torch

    lock = REPO_ROOT / "requirements.lock"
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requirements_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()[:16] if lock.exists() else None,
    }


class Run:
    """A single experiment run. Use as a context manager so the registry row is always written."""

    def __init__(self, cfg: DictConfig, name: str, phase: str, root: str | Path = REPO_ROOT / "runs",
                 data_hash: str = "", tensorboard: bool = False):
        self.cfg, self.name, self.phase, self.data_hash = cfg, name, phase, data_hash
        self.root = Path(root)
        self.cfg_hash = config_hash(cfg)
        self.started = datetime.now(UTC)
        self.run_id = f"{self.started:%Y%m%dT%H%M%SZ}_{name}_{self.cfg_hash}"
        self.dir = self.root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
        self.env = environment_snapshot()
        (self.dir / "env.json").write_text(json.dumps(self.env, indent=2), encoding="utf-8")
        self._metrics = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        self._tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.dir / "tb"))
        self.summary: dict = {}
        self.status = "running"

    def log(self, step: int, **metrics: float) -> None:
        rec = {"step": step, "time": time.time(), **{k: float(v) for k, v in metrics.items()}}
        self._metrics.write(json.dumps(rec) + "\n")
        self._metrics.flush()
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, float(v), step)

    def finish(self, status: str = "done", **summary: float) -> None:
        self.summary.update({k: float(v) for k, v in summary.items()})
        self.status = status
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=2), encoding="utf-8")

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self.status == "running":
            self.status = f"failed: {exc_type.__name__}"
        elif self.status == "running":
            self.status = "done"
        self._metrics.close()
        if self._tb is not None:
            self._tb.close()
        append_registry(self.root / "registry.csv", {
            "run_id": self.run_id, "started_utc": self.started.isoformat(timespec="seconds"),
            "ended_utc": datetime.now(UTC).isoformat(timespec="seconds"), "phase": self.phase,
            "name": self.name, "git_sha": self.env["git_sha"], "git_dirty": self.env["git_dirty"],
            "config_hash": self.cfg_hash, "data_hash": self.data_hash, "status": self.status,
            "metrics": json.dumps(self.summary, sort_keys=True), "run_dir": self.dir.name})


def append_registry(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
