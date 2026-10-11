"""Multi-task balancing (P070): Kendall uncertainty weighting and a task-gradient conflict monitor.

`UncertaintyWeighting` holds one learned log-variance s_t per task and combines the task losses as
L = sum_t exp(-s_t) * L_t + s_t (Kendall, Gal & Cipolla 2018); a task that is absent from a batch (loss exactly 0
with no gradient, as every P064 loss returns for an empty mask) contributes nothing, including the s_t term.
`GradConflictMonitor` measures, on a given list of shared parameters, the cosine similarity between the gradients
of pairs of task losses; it is a diagnostic (logged per step, P070's guard against negative transfer), not a
modifier of the update in v1.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn


class UncertaintyWeighting(nn.Module):
    def __init__(self, tasks: tuple[str, ...] | list[str], init_log_var: float = 0.0, min_log_var: float = -6.0):
        super().__init__()
        self.tasks = tuple(tasks)
        self.log_var = nn.ParameterDict({t: nn.Parameter(torch.tensor(float(init_log_var))) for t in self.tasks})
        self.min_log_var = min_log_var

    def weights(self) -> dict[str, float]:
        return {t: float(torch.exp(-self.log_var[t].clamp_min(self.min_log_var))) for t in self.tasks}

    def forward(self, losses: dict[str, torch.Tensor], present: dict[str, bool] | None = None) -> tuple[torch.Tensor, dict]:
        """losses: task -> scalar; present: task -> whether the batch had targets (default: loss requires grad and
        is not exactly 0). Returns (total, {task: weighted contribution})."""
        total = None
        parts = {}
        for t, loss in losses.items():
            if t not in self.log_var:
                raise KeyError(f"unknown task {t!r}; known: {self.tasks}")
            is_present = present[t] if present is not None else bool(loss.requires_grad) and float(loss) != 0.0
            if not is_present:
                parts[t] = loss * 0.0
                continue
            s = self.log_var[t].clamp_min(self.min_log_var)
            part = torch.exp(-s) * loss + s
            parts[t] = part
            total = part if total is None else total + part
        if total is None:
            total = sum(v for v in losses.values()) * 0.0
        return total, parts


class GradConflictMonitor:
    """Cosine similarity between task gradients on shared parameters (a diagnostic, computed on demand)."""

    def __init__(self, shared_params: list[nn.Parameter]):
        self.params = [p for p in shared_params if p.requires_grad]

    def task_gradient(self, loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(loss, self.params, retain_graph=True, allow_unused=True)
        flat = [g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1)
                for g, p in zip(grads, self.params, strict=True)]
        return torch.cat(flat).detach()

    def cosines(self, losses: dict[str, torch.Tensor]) -> dict[str, float]:
        """{'a|b': cos} for every pair of tasks whose loss has a gradient; also '<task>/norm' entries."""
        grads = {t: self.task_gradient(loss) for t, loss in losses.items() if loss.requires_grad}
        out = {f"{t}/norm": float(g.norm()) for t, g in grads.items()}
        for a, b in itertools.combinations(sorted(grads), 2):
            ga, gb = grads[a], grads[b]
            denom = ga.norm() * gb.norm()
            out[f"{a}|{b}"] = float((ga @ gb) / denom) if denom > 0 else float("nan")
        return out

    @staticmethod
    def conflicts(cosines: dict[str, float], threshold: float = -0.1) -> list[str]:
        return [k for k, v in cosines.items() if "|" in k and v == v and v < threshold]
