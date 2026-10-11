"""Reliability score (P068, ADR-005 §5): energy on the genus logits + Mahalanobis distance of the pooled feature to
the training Gaussian, each z-scored with training statistics and averaged into `ood_score` [N, 1] (higher = less
trustworthy). Plus two image flags that need no model: night (dark frame) and glare (saturated disc at the Sun).

`OodScore.fit(features, logits)` estimates the mean and shared covariance (shrinkage 0.1) of the training features
and the mean / std of both scores on the training set; `forward(features, logits)` applies them. The fitted
statistics are buffers (saved with the model; the model card copies the threshold).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """-T * logsumexp(logits / T): low for confident in-distribution inputs, high otherwise."""
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


class OodScore(nn.Module):
    def __init__(self, d: int, shrinkage: float = 0.1):
        super().__init__()
        self.shrinkage = shrinkage
        self.register_buffer("mean", torch.zeros(d))
        self.register_buffer("precision", torch.eye(d))
        self.register_buffer("stats", torch.tensor([0.0, 1.0, 0.0, 1.0]))     # energy mean/std, maha mean/std
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, features: torch.Tensor, logits: torch.Tensor) -> None:
        x = features.float()
        mu = x.mean(0)
        xc = x - mu
        cov = xc.T @ xc / max(len(x) - 1, 1)
        d = cov.shape[0]
        cov = (1 - self.shrinkage) * cov + self.shrinkage * torch.eye(d, device=cov.device) * cov.diagonal().mean()
        self.mean.copy_(mu)
        self.precision.copy_(torch.linalg.inv(cov))
        e = energy(logits.float())
        m = self.mahalanobis(x)
        self.stats.copy_(torch.stack([e.mean(), e.std().clamp_min(1e-6), m.mean(), m.std().clamp_min(1e-6)]))
        self.fitted.fill_(True)

    def mahalanobis(self, features: torch.Tensor) -> torch.Tensor:
        xc = features.float() - self.mean
        return torch.sqrt(((xc @ self.precision) * xc).sum(1).clamp_min(0.0))

    def forward(self, features: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        e = (energy(logits.float()) - self.stats[0]) / self.stats[1]
        m = (self.mahalanobis(features) - self.stats[2]) / self.stats[3]
        return (0.5 * (e + m))[:, None]


def auroc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """AUROC of 'higher score = out-of-domain' (rank statistic, no sklearn needed)."""
    s = np.concatenate([scores_in, scores_out])
    y = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    n_out, n_in = y.sum(), len(y) - y.sum()
    return float((ranks[y == 1].sum() - n_out * (n_out + 1) / 2) / (n_out * n_in))


def night_flag(image_uint8: np.ndarray, threshold: float = 0.08) -> bool:
    """Mean luminance of the frame below `threshold` (of 1.0): night or a covered lens."""
    return float(image_uint8.mean()) / 255.0 < threshold


def glare_flag(image_uint8: np.ndarray, ray_map: np.ndarray | None, sun_zenith_deg: float | None, radius_deg: float = 15.0,
               saturated: int = 250, min_share: float = 0.3) -> bool:
    """Saturated share above `min_share` within `radius_deg` of the Sun (when the geometry is known); else False.
    Works at the ray-map grid: a patch is near the Sun when its direction is within the radius."""
    if ray_map is None or sun_zenith_deg is None or ray_map[3].max() <= 0:
        return False
    sun = np.array([np.sin(np.radians(sun_zenith_deg)), 0.0, np.cos(np.radians(sun_zenith_deg))])   # Sun-aligned frame
    dirs = ray_map[:3].reshape(3, -1).T
    valid = ray_map[3].reshape(-1) > 0.5
    near = valid & (dirs @ sun > np.cos(np.radians(radius_deg)))
    if not near.any():
        return False
    h, w = ray_map.shape[1:]
    ph, pw = image_uint8.shape[0] // h, image_uint8.shape[1] // w
    idx = np.flatnonzero(near)
    shares = []
    for k in idx:
        r, c = divmod(int(k), w)
        patch = image_uint8[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
        shares.append(float((patch.max(axis=-1) >= saturated).mean()) if patch.size else 0.0)
    return float(np.mean(shares)) >= min_share
