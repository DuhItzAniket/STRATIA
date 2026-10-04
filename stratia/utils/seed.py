"""Global seeding and determinism switches."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA).

    With `deterministic=True`, cuDNN autotuning is disabled and PyTorch is asked for
    deterministic kernels (warning, not error, for operations that have none). Some CUDA
    kernels remain non-deterministic; reproducibility is verified empirically (P004) and
    results are reported as mean ± std over seeds.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding derived from the torch seed of the main process."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)
