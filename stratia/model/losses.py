"""Losses for the heads (P064, P067; ADR-006). Every function takes per-sample targets with a mask and returns a
scalar that is 0 (with a zero gradient) when the mask selects nothing, so any batch composition is valid.

Genus: set-valued BCE with partial negatives. Soft labels: BCE against expert shares. Ordinal: CORAL on cumulative
logits with hard or soft cumulative targets. Hierarchy: noisy-OR of the genus probabilities per étage against the
étage head. Dense: cross-entropy with ignore. CBH: cross-entropy against a log-height Gaussian target + CRPS over
the ordered classes, and the layer-consistency KL from the CBH-implied lowest étage to the étage head.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..contract import ETAGE_CLASSES, GENUS_CLASSES
from ..labels.ontology import load_ontology
from . import cbh_bins as B

IGNORE = 255
EPS = 1e-6


def _masked_mean(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of `per_sample` over the rows where `mask` is true; exact 0 with gradient when none is."""
    m = mask.to(per_sample.dtype)
    return (per_sample * m).sum() / m.sum().clamp_min(1.0)


# ----------------------------------------------------------------------------------------------- genus / étage
def genus_etage_matrix() -> torch.Tensor:
    """[12, 3] 0/1: the primary étage of each genus class (clear and contrail belong to none)."""
    ont = load_ontology()
    m = torch.zeros(len(GENUS_CLASSES), len(ETAGE_CLASSES))
    for i, g in enumerate(GENUS_CLASSES):
        if g in ont.genera:
            m[i, ETAGE_CLASSES.index(ont.etage_of(g))] = 1.0
    return m


def set_valued_bce(logits: torch.Tensor, target_set: torch.Tensor, alternatives: torch.Tensor, mask: torch.Tensor,
                   negative_weight: torch.Tensor | float = 1.0) -> torch.Tensor:
    """logits [N, C]; target_set [N, C] 0/1 (the known set S); alternatives [N] bool (S names alternatives);
    mask [N] bool (sample has a genus set); negative_weight [N] or float (weight of BCE(p_c, 0) for c not in S).

    Co-present sets: BCE(p_c, 1) summed over S. Alternatives: -log(1 - prod_{c in S}(1 - p_c)). Negatives: weighted
    BCE(p_c, 0) over the complement, averaged over the complement size. Per sample: positives + negatives."""
    logp = F.logsigmoid(logits)                     # log p
    log1mp = F.logsigmoid(-logits)                  # log (1 - p)
    s = target_set.to(logits.dtype)
    n_pos = s.sum(1).clamp_min(1.0)
    copresent = -(logp * s).sum(1) / n_pos
    log_none = (log1mp * s).sum(1)                  # log prod (1 - p_c) over S
    alt = -torch.log1p(-torch.exp(log_none).clamp_max(1.0 - EPS))
    positives = torch.where(alternatives, alt, copresent)
    comp = 1.0 - s
    n_neg = comp.sum(1).clamp_min(1.0)
    negatives = -(log1mp * comp).sum(1) / n_neg
    nw = negative_weight if torch.is_tensor(negative_weight) else torch.full_like(negatives, float(negative_weight))
    return _masked_mean(positives + nw * negatives, mask)


def soft_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE against soft targets in [0, 1] (expert shares), averaged over classes, masked over samples."""
    per = F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype), reduction="none").mean(1)
    return _masked_mean(per, mask)


def hierarchy_consistency(genus_logits: torch.Tensor, etage_logits: torch.Tensor, mask: torch.Tensor,
                          matrix: torch.Tensor | None = None) -> torch.Tensor:
    """mean_e (p_e - p̂_e)^2 with p̂_e = 1 - prod_{g in e}(1 - p_g) (noisy-OR over the genera of étage e)."""
    m = (genus_etage_matrix() if matrix is None else matrix).to(genus_logits.device, genus_logits.dtype)
    log1mp = F.logsigmoid(-genus_logits)                       # [N, 12]
    log_none = log1mp @ m                                      # [N, 3]: log prod (1 - p_g) over genera of e
    implied = 1.0 - torch.exp(log_none)
    p_e = torch.sigmoid(etage_logits)
    return _masked_mean(((p_e - implied) ** 2).mean(1), mask)


# ----------------------------------------------------------------------------------------------- ordinal (CORAL)
def cumulative_targets(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Hard labels [N] (long) or distributions [N, K] -> cumulative targets P(y > k) [N, K-1]."""
    if target.dim() == 1:
        ks = torch.arange(n_classes - 1, device=target.device)
        return (target[:, None] > ks[None, :]).to(torch.float32)
    tail = torch.flip(torch.cumsum(torch.flip(target, [1]), 1), [1])    # P(y >= k)
    return tail[:, 1:].to(torch.float32)                                # P(y > k) = P(y >= k+1)


def coral_loss(cum_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sum over thresholds of BCE(sigmoid(cum_logit_k), P(y > k)); target hard [N] or soft [N, K]."""
    t = cumulative_targets(target, cum_logits.shape[1] + 1).to(cum_logits.dtype)
    per = F.binary_cross_entropy_with_logits(cum_logits, t, reduction="none").sum(1)
    return _masked_mean(per, mask)


def kl_to_distribution(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(target || probs) per sample, masked mean (the agreement metric for annotator distributions)."""
    t = target.to(probs.dtype).clamp_min(EPS)
    per = (t * (torch.log(t) - torch.log(probs.clamp_min(EPS)))).sum(1)
    return _masked_mean(per, mask)


# ----------------------------------------------------------------------------------------------- dense
def dense_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """Cross-entropy with ignore 255; logits [N, C, h, w] are upsampled to the target's [N, H, W] if needed."""
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    valid = target != IGNORE
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target.long(), weight=weight, ignore_index=IGNORE)


# ----------------------------------------------------------------------------------------------- CBH
def cbh_soft_targets(height_m: torch.Tensor, sigma_log: float = 0.15) -> torch.Tensor:
    """height_m [N] (NaN = no cloud) -> [N, K+1] soft targets (log-height Gaussian over bins; one-hot no-cloud)."""
    return torch.from_numpy(B.soft_target(height_m.detach().cpu().numpy(), sigma_log)).to(height_m.device)


def crps_ordered(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared difference of cumulative distributions over ordered classes, per sample [N]."""
    return ((torch.cumsum(probs, 1) - torch.cumsum(target, 1)) ** 2).mean(1)


def cbh_loss(logits: torch.Tensor, height_m: torch.Tensor, mask: torch.Tensor, sigma_log: float = 0.15,
             lambda_crps: float = 1.0) -> tuple[torch.Tensor, dict]:
    """logits [N, K+1]; height_m [N] with NaN for "no cloud overhead"; mask [N].
    CE against the soft target + lambda * CRPS (over the height bins for cloudy targets, over the cloud / no-cloud
    split for clear targets). Returns (loss, {"ce", "crps"})."""
    target = cbh_soft_targets(height_m, sigma_log).to(logits.dtype)
    logq = F.log_softmax(logits, dim=1)
    ce = -(target * logq).sum(1)
    q = logq.exp()
    cloudy = ~torch.isnan(height_m)
    qc = q[:, :B.K_BINS] / q[:, :B.K_BINS].sum(1, keepdim=True).clamp_min(EPS)
    crps_cloud = crps_ordered(qc, target[:, :B.K_BINS] / target[:, :B.K_BINS].sum(1, keepdim=True).clamp_min(EPS))
    crps_clear = (q[:, B.NO_CLOUD] - target[:, B.NO_CLOUD]) ** 2
    crps = torch.where(cloudy, crps_cloud, crps_clear)
    loss = _masked_mean(ce + lambda_crps * crps, mask)
    return loss, {"ce": _masked_mean(ce, mask).detach(), "crps": _masked_mean(crps, mask).detach()}


def lowest_etage_from_etage_logits(etage_logits: torch.Tensor) -> torch.Tensor:
    """[N, 3] sigmoid outputs (low, mid, high present) -> [N, 4] distribution of the lowest étage (+ none)."""
    p = torch.sigmoid(etage_logits)
    low, mid, high = p[:, 0], p[:, 1], p[:, 2]
    d = torch.stack([low, (1 - low) * mid, (1 - low) * (1 - mid) * high, (1 - low) * (1 - mid) * (1 - high)], 1)
    return d / d.sum(1, keepdim=True).clamp_min(EPS)


def cbh_implied_etage(cbh_probs: torch.Tensor) -> torch.Tensor:
    """[N, K+1] -> [N, 4]: mass per étage of the lowest base (low, mid, high) + no cloud."""
    m = torch.zeros(B.N_CLASSES, 4, device=cbh_probs.device, dtype=cbh_probs.dtype)
    m[torch.arange(B.K_BINS), torch.from_numpy(B.ETAGE_OF_BIN).to(cbh_probs.device)] = 1.0
    m[B.NO_CLOUD, 3] = 1.0
    return cbh_probs @ m


def cbh_etage_consistency(cbh_probs: torch.Tensor, etage_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(étage-head lowest étage || CBH-implied lowest étage); the étage head is detached (gradient to CBH only)."""
    ref = lowest_etage_from_etage_logits(etage_logits).detach()
    implied = cbh_implied_etage(cbh_probs).clamp_min(EPS)
    per = (ref * (torch.log(ref.clamp_min(EPS)) - torch.log(implied))).sum(1)
    return _masked_mean(per, mask)


def etage_from_bins_numpy(probs: np.ndarray) -> np.ndarray:
    """numpy twin of `cbh_implied_etage` for evaluation code."""
    return cbh_implied_etage(torch.from_numpy(np.asarray(probs, dtype=np.float32))).numpy()
