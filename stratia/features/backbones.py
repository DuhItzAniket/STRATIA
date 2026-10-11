"""Frozen backbones behind one interface (P051, P054, P055).

``FrozenBackbone(key)`` loads a Hugging Face model and turns a uint8 image batch (N, 3, H, W) into
``{"cls": (N, D), "mean": (N, D), "patches": (N, L, D) | None}``:

* ``cls``: the model's own pooled image embedding (DINO: the CLS token after the final norm; CLIP: the projected
  image embedding used for zero-shot; SigLIP 2: the attention-pooled embedding), so probes and zero-shot share it;
* ``mean``: the mean of the patch tokens (a second, cheap global descriptor);
* ``patches``: the patch-token grid (registers and CLS removed) for dense probes, in the backbone's own width.

Normalisation constants belong to the backbone, not the caller. Images are resized full-frame (no crop) by the
caller to ``size`` x ``size`` (contract rule); CLIP and the fixed-resolution SigLIP 2 take 224 only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class BackboneSpec:
    key: str
    hf_id: str
    family: str                  # dinov3 | dinov2 | clip | siglip2
    patch: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    fixed_size: int | None = None  # None: any multiple of `patch`; else only this input size
    gated: bool = False
    compute_dtype: str = "float16"  # DINOv3 ViT-L overflows in fp16 (every CLS row NaN, P051): it runs in bfloat16


_IN = (IMAGENET_MEAN, IMAGENET_STD)
BACKBONES: dict[str, BackboneSpec] = {
    "dinov3_vits16": BackboneSpec("dinov3_vits16", "facebook/dinov3-vits16-pretrain-lvd1689m", "dinov3", 16, *_IN, gated=True),
    "dinov3_vitb16": BackboneSpec("dinov3_vitb16", "facebook/dinov3-vitb16-pretrain-lvd1689m", "dinov3", 16, *_IN, gated=True),
    "dinov3_vitl16": BackboneSpec("dinov3_vitl16", "facebook/dinov3-vitl16-pretrain-lvd1689m", "dinov3", 16, *_IN, gated=True,
                                  compute_dtype="bfloat16"),
    "dinov2_small": BackboneSpec("dinov2_small", "facebook/dinov2-small", "dinov2", 14, IMAGENET_MEAN, IMAGENET_STD),
    "dinov2_base": BackboneSpec("dinov2_base", "facebook/dinov2-base", "dinov2", 14, IMAGENET_MEAN, IMAGENET_STD),
    "clip_vitb16": BackboneSpec("clip_vitb16", "openai/clip-vit-base-patch16", "clip", 16, CLIP_MEAN, CLIP_STD, fixed_size=224),
    "siglip2_b16": BackboneSpec("siglip2_b16", "google/siglip2-base-patch16-224", "siglip2", 16, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                fixed_size=224),
}


class FrozenBackbone:
    def __init__(self, key: str, device: str | None = None, dtype: torch.dtype | None = None):
        self.spec = BACKBONES[key]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        wanted = dtype or getattr(torch, self.spec.compute_dtype)
        self.dtype = wanted if self.device == "cuda" else torch.float32
        self.model = self._load().to(self.device).eval().to(self.dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._mean = torch.tensor(self.spec.mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(self.spec.std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.dim, self.mean_dim = self._probe_dims()

    # -- loading ------------------------------------------------------------------------------------------------
    def _load(self):
        from transformers import AutoModel

        if self.spec.family == "clip":
            from transformers import CLIPModel

            try:
                return CLIPModel.from_pretrained(self.spec.hf_id)
            except ValueError as err:                     # .bin weights refused by transformers >= 5 on torch < 2.6
                if "torch.load" not in str(err):
                    raise
                return CLIPModel.from_pretrained(converted_checkpoint(self.spec.hf_id))
        return AutoModel.from_pretrained(self.spec.hf_id)

    def _probe_dims(self) -> tuple[int, int]:
        """(width of `cls`, width of `mean` / patches): they differ for CLIP (projected 512 vs 768 tokens)."""
        size = self.spec.fixed_size or 224
        x = torch.zeros(1, 3, size, size, dtype=torch.uint8, device=self.device)
        out = self.forward(x, patches=False)
        return int(out["cls"].shape[1]), int(out["mean"].shape[1])

    # -- images ----------------------------------------------------------------------------------------------------
    def normalise(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.to(self.device, non_blocking=True).to(self.dtype) / 255.0
        return (x - self._mean) / self._std

    @torch.no_grad()
    def forward(self, x_uint8: torch.Tensor, patches: bool = True) -> dict[str, torch.Tensor | None]:
        x = self.normalise(x_uint8)
        if self.spec.fixed_size and x.shape[-1] != self.spec.fixed_size:
            raise ValueError(f"{self.spec.key} takes {self.spec.fixed_size} px inputs, got {x.shape[-1]}")
        fam = self.spec.family
        if fam in ("dinov3", "dinov2"):
            out = self.model(pixel_values=x)
            n_skip = 1 + (int(getattr(self.model.config, "num_register_tokens", 0)) if fam == "dinov3" else 0)
            tokens = out.last_hidden_state[:, n_skip:, :]
            cls = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        elif fam == "clip":
            vis = self.model.vision_model(pixel_values=x)
            cls = self.model.visual_projection(vis.pooler_output)
            tokens = vis.last_hidden_state[:, 1:, :]
        elif fam == "siglip2":
            vis = self.model.vision_model(pixel_values=x)
            cls = vis.pooler_output
            tokens = vis.last_hidden_state
        else:
            raise ValueError(fam)
        return {"cls": cls.float(), "mean": tokens.float().mean(dim=1), "patches": tokens if patches else None}

    # -- text (zero-shot) ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def text_features(self, prompts: list[str]) -> torch.Tensor:
        """Normalised text embeddings (n, D) for CLIP / SigLIP 2; the image `cls` must be normalised the same way."""
        from transformers import AutoTokenizer

        if self.spec.family not in ("clip", "siglip2"):
            raise ValueError(f"{self.spec.key} has no text tower")
        tok = AutoTokenizer.from_pretrained(self.spec.hf_id)
        if self.spec.family == "clip":
            enc = tok(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**enc)
        else:
            enc = tok(prompts, padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**enc)
        feats = getattr(feats, "pooler_output", feats)          # transformers >= 5 returns the output object
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    @property
    def logit_scale(self) -> float:
        scale = getattr(self.model, "logit_scale", None)
        return float(scale.exp().item()) if scale is not None else 100.0

    @property
    def logit_bias(self) -> float:
        bias = getattr(self.model, "logit_bias", None)
        return float(bias.item()) if bias is not None else 0.0


def grid_side(size: int, patch: int) -> int:
    return size // patch


def to_grid(patches: np.ndarray | torch.Tensor, size: int, patch: int):
    """(N, L, D) -> (N, side, side, D)."""
    side = grid_side(size, patch)
    return patches.reshape(patches.shape[0], side, side, patches.shape[-1])



def converted_checkpoint(hf_id: str):
    """A safetensors copy of a Hub checkpoint that only ships pytorch_model.bin (OpenAI CLIP), made once under
    <cache_root>/hf_converted/<id>. torch.load(weights_only=True) reads the pickle; save_pretrained writes safetensors."""
    from pathlib import Path

    from ..data.registry import load_paths

    target = Path(load_paths()["cache_root"]) / "hf_converted" / hf_id.replace("/", "--")
    if (target / "model.safetensors").exists():
        return target
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer, CLIPModel

    local = Path(snapshot_download(hf_id, allow_patterns=["*.json", "*.bin", "*.txt"]))
    state = torch.load(local / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model = CLIPModel(AutoConfig.from_pretrained(hf_id))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"{hf_id}: converted checkpoint misses {len(missing)} tensors, e.g. {missing[:3]}")
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(hf_id).save_pretrained(target)
    return target

