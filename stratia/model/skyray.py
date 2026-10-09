"""SkyRay geometry encoder with null tokens (P050; the full encoder and FiLM wiring are P062).

``SkyRayEncoder`` turns the contract's ray map ``[N, 4, H/p, W/p]`` into per-patch tokens ``[N, L, d]``: an MLP on
the unit direction where the valid flag is 1 and a learned **null token** where it is 0, so an all-zero ray map
(uncalibrated camera, or geometry dropout, P050) yields a sequence of null tokens rather than the embedding of a
meaningless zero vector. ``MetaFiLM`` does the same for the Sun metadata ``[N, 3]``: a feature-wise affine
modulation from ``[cos z, sin z]`` when ``valid`` is 1, learned null parameters (identity at initialisation)
otherwise. Both run on CPU; no backbone is needed to exercise them.
"""

from __future__ import annotations

import torch
from torch import nn

from ..contract import META_DIM, RAY_MAP_CHANNELS


class SkyRayEncoder(nn.Module):
    def __init__(self, d_model: int = 384, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(RAY_MAP_CHANNELS - 1, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.null_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.null_token, std=0.02)

    def forward(self, ray_map: torch.Tensor) -> torch.Tensor:
        """ray_map [N, 4, h, w] -> tokens [N, h*w, d]; invalid patches get the null token."""
        n, c, h, w = ray_map.shape
        assert c == RAY_MAP_CHANNELS, f"ray map must have {RAY_MAP_CHANNELS} channels"
        flat = ray_map.permute(0, 2, 3, 1).reshape(n, h * w, c)
        valid = flat[..., 3:4] > 0.5
        emb = self.mlp(flat[..., :3])
        return torch.where(valid, emb, self.null_token.expand_as(emb))


class MetaFiLM(nn.Module):
    def __init__(self, d_model: int = 384, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(META_DIM - 1, hidden), nn.GELU(), nn.Linear(hidden, 2 * d_model))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.null_gamma = nn.Parameter(torch.zeros(d_model))
        self.null_beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, tokens: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """tokens [N, L, d], meta [N, 3] -> tokens * (1 + gamma) + beta; null parameters where meta.valid is 0."""
        n, _, d = tokens.shape
        assert meta.shape == (n, META_DIM)
        gb = self.mlp(meta[:, :2])
        gamma, beta = gb[:, :d], gb[:, d:]
        valid = (meta[:, 2:3] > 0.5)
        gamma = torch.where(valid, gamma, self.null_gamma.expand_as(gamma))
        beta = torch.where(valid, beta, self.null_beta.expand_as(beta))
        return tokens * (1.0 + gamma[:, None, :]) + beta[:, None, :]


class GeometryStub(nn.Module):
    """Tiny stand-in for a full model: SkyRay tokens + FiLM, mean-pooled to one logit. Lets P050 prove that a
    batch runs with and without metadata and that gradients reach the null parameters."""

    def __init__(self, d_model: int = 32):
        super().__init__()
        self.rays = SkyRayEncoder(d_model, hidden=32)
        self.film = MetaFiLM(d_model, hidden=16)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ray_map: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        tokens = self.film(self.rays(ray_map), meta)
        return self.head(tokens.mean(dim=1))
