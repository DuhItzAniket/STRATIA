"""The STRATIA model (P071): backbone + geometry encoder + heads, producing the contract's seven outputs.

Two entry points share everything after the backbone:
    forward(image, ray_map, meta)              image [N, 3, H, W] uint8 or normalised float -> outputs (backbone in the graph)
    forward_tokens(tokens, cls, ray_map, meta) cached tokens [N, L, d] + cls [N, d] -> outputs (Stages G-H)
Dense heads need the 32 x 32 grid; the global and CBH heads work on any grid. `outputs(...)` returns only the
contract names in the contract order (for the export, P096); the heads' extra tensors (cumulative oktas logits,
cbh logits, attention maps, pooled features) are available from `forward*` for the losses and the diagnostics.

Backbones: `FrozenBackbone` (stratia.features.backbones) or any module with `.dim` and
`forward(x_uint8, patches=True) -> {"cls", "patches"}`; `FakeBackbone` (tests) is a deterministic stand-in.
"""

from __future__ import annotations

import torch
from torch import nn

from ..contract import OUTPUT_NAMES
from .cbh import CbhHead
from .dense import DenseHeads
from .heads import GlobalHeads
from .ood import OodScore
from .skyray import GeometryEncoder


class FakeBackbone(nn.Module):
    """Deterministic stand-in: tokens = per-patch mean colour projected to d; cls = image mean projected."""

    def __init__(self, d: int = 32, patch: int = 16):
        super().__init__()
        self.dim, self.patch = d, patch
        g = torch.Generator().manual_seed(0)
        self.register_buffer("proj", torch.randn(3, d, generator=g))

    def forward(self, x, patches: bool = True) -> dict[str, torch.Tensor]:
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        n, _, h, w = x.shape
        s = h // self.patch
        grid = x.reshape(n, 3, s, self.patch, w // self.patch, self.patch).mean(dim=(3, 5)).permute(0, 2, 3, 1).reshape(n, -1, 3)
        return {"cls": x.mean(dim=(2, 3)) @ self.proj, "patches": grid @ self.proj if patches else None}


class STRATIA(nn.Module):
    def __init__(self, backbone: nn.Module | None, d_model: int = 384, dense: bool = True, cbh: bool = True,
                 ood: bool = True, global_hidden: int = 512):
        super().__init__()
        self.backbone = backbone
        self.geo = GeometryEncoder(d_model)
        self.global_heads = GlobalHeads(d_model, hidden=global_hidden)
        self.dense_heads = DenseHeads(d_model) if dense else None
        self.cbh_head = CbhHead(d_model) if cbh else None
        self.ood = OodScore(global_hidden) if ood else None

    def forward_tokens(self, tokens: torch.Tensor, cls: torch.Tensor, ray_map: torch.Tensor,
                       meta: torch.Tensor) -> dict[str, torch.Tensor]:
        t, c = self.geo(tokens, cls, ray_map, meta)
        out = self.global_heads(t, c)
        if self.dense_heads is not None:
            out.update(self.dense_heads(t))
        if self.cbh_head is not None:
            out.update(self.cbh_head(t, c, ray_map))
        if self.ood is not None:
            out["ood_score"] = self.ood(out["pooled"], out["genus_logits"])
        return out

    def forward(self, image: torch.Tensor, ray_map: torch.Tensor, meta: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.backbone is None:
            raise RuntimeError("no backbone: use forward_tokens with cached features")
        feats = (self.backbone(image, patches=True) if isinstance(self.backbone, FakeBackbone)
                 else self.backbone.forward(image, patches=True))
        return self.forward_tokens(feats["patches"].float(), feats["cls"].float(), ray_map, meta)

    def outputs(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The contract's outputs only, in the contract's order (missing heads are left out)."""
        return {k: out[k] for k in OUTPUT_NAMES if k in out}

    def head_parameters(self) -> dict[str, list[nn.Parameter]]:
        groups = {"geo": list(self.geo.parameters()), "global": list(self.global_heads.parameters())}
        if self.dense_heads is not None:
            groups["dense"] = list(self.dense_heads.parameters())
        if self.cbh_head is not None:
            groups["cbh"] = list(self.cbh_head.parameters())
        return groups
