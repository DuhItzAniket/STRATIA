"""Model sanity suite (P071): the checks every training configuration must pass before real training, runnable on
a tiny fake backbone in CI and on the real model before a Stage H run.

* `overfit_one_batch`: a model that cannot drive one batch's loss to ~0 has a pipeline bug (labels, masks, shapes).
* `shuffled_labels_check`: with labels permuted, the held-out score must stay at chance; a model that "learns"
  shuffled labels on held-out data is leaking inputs into targets.
* `constant_input_check`: identical inputs give identical outputs (no batch-dependent normalisation: no BatchNorm).
* `determinism_check`: the same seed and data give the same parameters after a few steps.
* `toy_batch`: a synthetic batch with every target kind and partial masks (the shapes of the contract).
"""

from __future__ import annotations

import numpy as np
import torch

from ..contract import ETAGE_CLASSES, GENUS_CLASSES, N_OKTAS, PATCH_SIZE, RAY_MAP_CHANNELS
from . import losses as L


def toy_batch(n: int = 8, size: int = 128, seed: int = 0) -> dict[str, torch.Tensor]:
    """Images whose colour encodes the genus, with every target type and masks that leave some heads unlabelled."""
    g = torch.Generator().manual_seed(seed)
    side = size // PATCH_SIZE
    genus = torch.randint(0, len(GENUS_CLASSES), (n,), generator=g)
    image = torch.zeros(n, 3, size, size, dtype=torch.uint8)
    for i in range(n):
        base = torch.tensor([40 + 15 * int(genus[i]), 200 - 12 * int(genus[i]), 90 + 7 * int(genus[i])])
        image[i] = base[:, None, None].expand(3, size, size).to(torch.uint8)
        image[i, :, : size // 2] = (image[i, :, : size // 2].float() * 0.7).to(torch.uint8)       # a darker lower half
    noise = torch.randint(-10, 10, (n, 3, size, size), generator=g)
    image = (image.int() + noise).clamp(0, 255).to(torch.uint8)
    ray_map = torch.zeros(n, RAY_MAP_CHANNELS, side, side)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, side), torch.linspace(-1, 1, side), indexing="ij")
    r = torch.sqrt(xs ** 2 + ys ** 2).clamp(max=1.0)
    theta = r * (np.pi / 2)
    phi = torch.atan2(ys, xs)
    ray_map[:, 0] = torch.sin(theta) * torch.cos(phi)
    ray_map[:, 1] = torch.sin(theta) * torch.sin(phi)
    ray_map[:, 2] = torch.cos(theta)
    ray_map[:, 3] = (r < 0.98).float()
    ray_map[n // 2:] = 0.0                                                                  # half the batch uncalibrated
    meta = torch.zeros(n, 3)
    meta[: n // 2] = torch.tensor([np.cos(np.radians(40.0)), np.sin(np.radians(40.0)), 1.0])
    genus_set = torch.nn.functional.one_hot(genus, len(GENUS_CLASSES)).float()
    etage_m = L.genus_etage_matrix()
    etage = genus_set @ etage_m
    oktas = torch.randint(0, N_OKTAS, (n,), generator=g)
    sky = torch.full((n, size, size), 255, dtype=torch.long)
    sky[:, size // 2:, :] = 1
    sky[:, : size // 2, :] = 2
    heights = torch.where(torch.rand(n, generator=g) < 0.3, torch.full((n,), float("nan")),
                          torch.exp(torch.rand(n, generator=g) * (np.log(9000.0) - np.log(200.0)) + np.log(200.0)))
    return {"image": image, "ray_map": ray_map, "meta": meta, "genus": genus, "genus_set": genus_set,
            "genus_alt": torch.zeros(n, dtype=torch.bool), "mask_genus": torch.ones(n, dtype=torch.bool),
            "etage": etage, "mask_etage": torch.arange(n) % 2 == 0, "oktas": oktas, "mask_oktas": torch.arange(n) % 3 == 0,
            "sky_mask": sky, "height_m": heights, "mask_cbh": torch.ones(n, dtype=torch.bool)}


def total_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(batch["image"], batch["ray_map"], batch["meta"])
    genus = L.set_valued_bce(out["genus_logits"], batch["genus_set"], batch["genus_alt"], batch["mask_genus"], 1.0)
    etage = L.soft_bce(out["etage_logits"], batch["etage"], batch["mask_etage"])
    oktas = L.coral_loss(out["oktas_cum_logits"], batch["oktas"], batch["mask_oktas"])
    sky = L.dense_ce(out["sky_parse_logits"], batch["sky_mask"])
    cbh, _ = L.cbh_loss(out["cbh_logits"], batch["height_m"], batch["mask_cbh"])
    return genus + etage + oktas + sky + cbh


def overfit_one_batch(model, batch: dict[str, torch.Tensor], steps: int = 200, lr: float = 3e-3) -> list[float]:
    """Loss history of training on one batch; a healthy pipeline reaches a small fraction of the first value."""
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    hist = []
    for _ in range(steps):
        opt.zero_grad()
        loss = total_loss(model, batch)
        loss.backward()
        opt.step()
        hist.append(float(loss))
    return hist


def chance_level_genus() -> float:
    return 1.0 / len(GENUS_CLASSES)


def shuffled_labels_check(model, batch: dict[str, torch.Tensor], steps: int = 50, lr: float = 1e-3, seed: int = 0) -> float:
    """Train on the first half with permuted genus labels, score genus accuracy on the second half with its own
    permuted labels: must stay near chance."""
    g = torch.Generator().manual_seed(seed)
    n = len(batch["genus"])
    perm = torch.randperm(n, generator=g)
    shuffled = dict(batch)
    shuffled["genus"] = batch["genus"][perm]
    shuffled["genus_set"] = batch["genus_set"][perm]
    half = n // 2
    train = {k: v[:half] for k, v in shuffled.items()}
    held = {k: v[half:] for k, v in shuffled.items()}
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = L.set_valued_bce(model(train["image"], train["ray_map"], train["meta"])["genus_logits"], train["genus_set"],
                                train["genus_alt"], train["mask_genus"], 1.0)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(held["image"], held["ray_map"], held["meta"])["genus_logits"].argmax(1)
    return float((pred == held["genus"]).float().mean())


def constant_input_check(model, size: int = 128, n: int = 4) -> float:
    """Max spread of the outputs over n identical inputs (0 for a model without batch-coupled layers)."""
    model.eval()
    image = torch.full((n, 3, size, size), 120, dtype=torch.uint8)
    side = size // PATCH_SIZE
    with torch.no_grad():
        out = model(image, torch.zeros(n, RAY_MAP_CHANNELS, side, side), torch.zeros(n, 3))
    spread = 0.0
    for v in out.values():
        if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
            spread = max(spread, float((v - v[:1]).abs().max()))
    return spread


def determinism_check(make_model, batch: dict[str, torch.Tensor], steps: int = 5, lr: float = 1e-3) -> np.ndarray:
    """Flattened parameters after `steps` of training from `make_model()` (compare two calls for equality)."""
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = make_model()
    overfit_one_batch(model, batch, steps=steps, lr=lr)
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).numpy()


__all__ = ["toy_batch", "total_loss", "overfit_one_batch", "shuffled_labels_check", "constant_input_check",
           "determinism_check", "chance_level_genus", "ETAGE_CLASSES"]
