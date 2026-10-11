"""Cloud-base-height head (P067, ADR-005 §3): zenith-prior attention over the geometry-fused tokens, CLS, an MLP,
and a distribution over the 24 height bins + "no cloud overhead" (`cbh_probs` [N, 25]).

Attention logit per token = w·token + prior, prior = -(theta / sigma)^2 with theta the zenith angle read from the
ray map's z channel; invalid patches (outside the camera mask, below the horizon, or a null ray map) get the
prior of the horizon, so they are demoted when some patches are valid and the prior is a constant (softmax-
invariant) when none is: an uncalibrated frame falls back to the learned content attention alone. sigma starts
at 20 degrees and is learned through its log.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import cbh_bins as B
from .skyray import zenith_cos


class CbhHead(nn.Module):
    def __init__(self, d_model: int = 384, hidden: int = 256, sigma_deg: float = 20.0, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, 1)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(math.radians(sigma_deg))))
        self.trunk = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, B.N_CLASSES)

    def attention(self, tokens: torch.Tensor, ray_map: torch.Tensor) -> torch.Tensor:
        """[N, L] attention weights over tokens (content + zenith prior)."""
        cosz, valid = zenith_cos(ray_map)
        theta = torch.acos(cosz.clamp(-1.0, 1.0))
        sigma = self.log_sigma.exp()
        prior = -(theta / sigma) ** 2
        horizon = -(math.pi / 2 / sigma) ** 2                  # invalid patches count as 'at the horizon'
        prior = torch.where(valid, prior, torch.full_like(prior, 0.0) + horizon)
        logits = self.attn(self.norm(tokens))[..., 0] + prior
        return torch.softmax(logits, dim=1)

    def forward(self, tokens: torch.Tensor, cls: torch.Tensor, ray_map: torch.Tensor) -> dict[str, torch.Tensor]:
        w = self.attention(tokens, ray_map)
        pooled = (w[..., None] * tokens).sum(1)
        h = self.trunk(torch.cat([pooled, cls], dim=1))
        logits = self.out(h)
        return {"cbh_logits": logits, "cbh_probs": F.softmax(logits, dim=1), "cbh_attention": w}
