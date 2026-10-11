"""Global heads (P063, ADR-005 §1): attention pooling over the geometry-fused tokens, CLS concatenation, a shared
MLP trunk, and the three contract outputs `genus_logits` [N, 12], `etage_logits` [N, 3], `oktas_probs` [N, 9].

`CoralHead` is the ordinal head (Cao et al., CORAL): one linear projection shared by all thresholds and K-1 ordered
biases give cumulative logits P(y > k); the class probabilities are their differences, clamped at 0 and
renormalised so they sum to 1 whatever the biases do during training (the loss, ADR-006 §3, works on the
cumulative logits). No BatchNorm anywhere (batch 16 at 512 px); LayerNorm and dropout only.
"""

from __future__ import annotations

import torch
from torch import nn

from ..contract import ETAGE_CLASSES, GENUS_CLASSES, N_OKTAS


class AttentionPool(nn.Module):
    """One learned query attends over the tokens (multi-head); returns [N, d]."""

    def __init__(self, d_model: int = 384, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm(tokens)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        return out[:, 0]

    def weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Attention weights averaged over heads, [N, L] (for inspection and the P093 saliency figures)."""
        x = self.norm(tokens)
        q = self.query.expand(x.shape[0], -1, -1)
        _, w = self.attn(q, x, x, need_weights=True, average_attn_weights=True)
        return w[:, 0]


class CoralHead(nn.Module):
    """Ordinal output over K classes: shared weight vector, K-1 biases (sorted at forward time)."""

    def __init__(self, d_in: int, n_classes: int = N_OKTAS):
        super().__init__()
        self.n_classes = n_classes
        self.weight = nn.Linear(d_in, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, n_classes - 1))   # P(y > k) decreasing in k at init

    def cumulative_logits(self, x: torch.Tensor) -> torch.Tensor:
        """[N, K-1] logits of P(y > k), k = 0..K-2; the biases are kept in decreasing order by sorting."""
        b, _ = torch.sort(self.bias, descending=True)
        return self.weight(x) + b[None, :]

    @staticmethod
    def probs_from_cumulative(cum_logits: torch.Tensor) -> torch.Tensor:
        """[N, K-1] cumulative logits -> [N, K] class probabilities (differences, clamped, renormalised)."""
        c = torch.sigmoid(cum_logits)
        ones = torch.ones_like(c[:, :1])
        zeros = torch.zeros_like(c[:, :1])
        upper = torch.cat([ones, c], dim=1)          # P(y > k-1), with P(y > -1) = 1
        lower = torch.cat([c, zeros], dim=1)         # P(y > k), with P(y > K-1) = 0
        p = (upper - lower).clamp_min(0.0)
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cum = self.cumulative_logits(x)
        return self.probs_from_cumulative(cum), cum


class GlobalHeads(nn.Module):
    """Attention pool ⊕ cls -> MLP trunk -> genus logits, étage logits, oktas (CORAL)."""

    def __init__(self, d_model: int = 384, hidden: int = 512, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pool = AttentionPool(d_model, n_heads)
        self.trunk = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, hidden), nn.GELU(),
                                   nn.Dropout(dropout))
        self.genus = nn.Linear(hidden, len(GENUS_CLASSES))
        self.etage = nn.Linear(hidden, len(ETAGE_CLASSES))
        self.oktas = CoralHead(hidden, N_OKTAS)

    def features(self, tokens: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        return self.trunk(torch.cat([self.pool(tokens), cls], dim=1))

    def forward(self, tokens: torch.Tensor, cls: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.features(tokens, cls)
        oktas_probs, oktas_cum = self.oktas(h)
        return {"genus_logits": self.genus(h), "etage_logits": self.etage(h), "oktas_probs": oktas_probs,
                "oktas_cum_logits": oktas_cum, "pooled": h}
