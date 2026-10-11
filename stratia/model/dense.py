"""Dense heads (P065 sky parsing, P066 layers; ADR-005 §2): a light DPT-style decoder shared by both.

`DenseDecoder` reassembles the token sequence [N, L, d] into a [N, d, s, s] grid (s = sqrt(L)), projects to
`width` channels and runs two fusion blocks (3 x 3 conv, GroupNorm, GELU, twice) each followed by bilinear x2
upsampling, so a 32 x 32 grid becomes a [N, out, 128, 128] feature map (H/4 for a 512 input, the contract's dense
resolution). `SkyParseHead` (4 classes) and `LayerHead` (3 classes) are 1 x 1 convolutions on that map. No
BatchNorm (P010: batch 16); ~0.6 M parameters at d = 384.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..contract import LAYER_CLASSES, SKY_PARSE_CLASSES


class FusionBlock(nn.Module):
    def __init__(self, cin: int, cout: int, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.skip = nn.Conv2d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(y + self.skip(x))


def tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    """[N, L, d] -> [N, d, s, s] with s = sqrt(L)."""
    n, L, d = tokens.shape
    s = int(math.isqrt(L))
    assert s * s == L, f"{L} tokens do not form a square grid"
    return tokens.transpose(1, 2).reshape(n, d, s, s)


class DenseDecoder(nn.Module):
    def __init__(self, d_model: int = 384, width: int = 128, mid: int = 96, out: int = 64, upsample_steps: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Conv2d(d_model, width, 1)
        chans = [width, mid, out][: upsample_steps + 1]
        self.blocks = nn.ModuleList(FusionBlock(chans[i], chans[i + 1]) for i in range(upsample_steps))
        self.out_channels = chans[-1]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.proj(tokens_to_grid(self.norm(tokens)))
        for block in self.blocks:
            x = F.interpolate(block(x), scale_factor=2.0, mode="bilinear", align_corners=False)
        return x


class SkyParseHead(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, len(SKY_PARSE_CLASSES), 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)


class LayerHead(nn.Module):
    def __init__(self, in_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, len(LAYER_CLASSES), 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.conv(feat)


class DenseHeads(nn.Module):
    """Decoder + the two 1 x 1 heads; returns the contract's `sky_parse_logits` and `layer_logits` at H/4."""

    def __init__(self, d_model: int = 384, width: int = 128, mid: int = 96, out: int = 64):
        super().__init__()
        self.decoder = DenseDecoder(d_model, width, mid, out)
        self.sky_parse = SkyParseHead(out)
        self.layer = LayerHead(out)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.decoder(tokens)
        return {"sky_parse_logits": self.sky_parse(feat), "layer_logits": self.layer(feat), "dense_feat": feat}
