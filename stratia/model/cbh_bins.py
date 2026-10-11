"""Cloud-base-height bins (ADR-005, docs/architecture.md §3): the single source of the contract's `cbh_probs` axis.

24 height bins whose edges contain the ontology's étage thresholds (2,000 m, 6,000 m), log-spaced inside each
étage (10 low, 8 mid, 6 high), plus the final "no cloud overhead" class (index 24). The model card copies
``EDGES_M``; the layer-consistency loss sums bins per étage through ``ETAGE_OF_BIN``.
"""

from __future__ import annotations

import numpy as np

ETAGE_EDGES_M = (0.0, 2000.0, 6000.0, 12000.0)
BINS_PER_ETAGE = (10, 8, 6)
FIRST_EDGE_M = 100.0        # the lowest bin is [0, 100): fog and bases on the ground


def _edges() -> np.ndarray:
    edges = [0.0]
    for (lo, hi), n in zip(zip(ETAGE_EDGES_M[:-1], ETAGE_EDGES_M[1:], strict=True), BINS_PER_ETAGE, strict=True):
        start = FIRST_EDGE_M if lo == 0.0 else lo
        k = n - 1 if lo == 0.0 else n                      # the low étage spends one bin on [0, 100)
        if lo == 0.0:
            edges.append(start)
        edges.extend(start * (hi / start) ** (np.arange(1, k + 1) / k))
    return np.round(np.asarray(edges, dtype=np.float64), 0)


EDGES_M = _edges()                                   # 25 edges
K_BINS = len(EDGES_M) - 1                            # 24
N_CLASSES = K_BINS + 1                               # + no cloud overhead
NO_CLOUD = K_BINS
ETAGE_OF_BIN = np.repeat(np.arange(3), BINS_PER_ETAGE)     # 0 low, 1 mid, 2 high, per height bin


def centres_m() -> np.ndarray:
    """Representative height per bin: the geometric mean of the edges (50 m for the [0, 100) bin)."""
    lo, hi = EDGES_M[:-1], EDGES_M[1:]
    c = np.sqrt(np.where(lo > 0, lo, 25.0) * hi)
    c[0] = 50.0
    return c


def bin_of(height_m: np.ndarray) -> np.ndarray:
    """Height in metres -> bin index 0..K-1 (heights above the last edge go to the last bin); NaN -> NO_CLOUD."""
    h = np.asarray(height_m, dtype=np.float64)
    idx = np.clip(np.searchsorted(EDGES_M, h, side="right") - 1, 0, K_BINS - 1)
    return np.where(np.isnan(h), NO_CLOUD, idx).astype(np.int64)


def soft_target(height_m: np.ndarray, sigma_log: float = 0.15) -> np.ndarray:
    """(n, K+1) targets: a Gaussian in log height over the 24 bins (renormalised) for a measured base, one-hot
    NO_CLOUD for NaN."""
    h = np.asarray(height_m, dtype=np.float64)
    out = np.zeros((len(h), N_CLASSES), dtype=np.float32)
    logc = np.log(centres_m())
    cloudy = ~np.isnan(h)
    if cloudy.any():
        d = (np.log(np.clip(h[cloudy], 1.0, None))[:, None] - logc[None, :]) / sigma_log
        w = np.exp(-0.5 * d ** 2)
        out[cloudy, :K_BINS] = (w / w.sum(1, keepdims=True)).astype(np.float32)
    out[~cloudy, NO_CLOUD] = 1.0
    return out


def median_height_m(probs: np.ndarray) -> np.ndarray:
    """Median of the height distribution (over the 24 bins, renormalised) in metres; NaN rows where the no-cloud
    class holds the majority."""
    p = np.asarray(probs, dtype=np.float64)
    cloud = p[:, :K_BINS]
    mass = cloud.sum(1, keepdims=True)
    cdf = np.cumsum(cloud / np.clip(mass, 1e-12, None), axis=1)
    idx = np.clip((cdf >= 0.5).argmax(1), 0, K_BINS - 1)
    # interpolate inside the bin in log height
    lo, hi = np.log(np.clip(EDGES_M[idx], 1.0, None)), np.log(EDGES_M[idx + 1])
    prev = np.where(idx > 0, cdf[np.arange(len(p)), np.clip(idx - 1, 0, None)], 0.0)
    frac = np.clip((0.5 - prev) / np.clip(cdf[np.arange(len(p)), idx] - prev, 1e-12, None), 0.0, 1.0)
    med = np.exp(lo + frac * (hi - lo))
    return np.where(p[:, NO_CLOUD] > 0.5, np.nan, med)


def interval_m(probs: np.ndarray, lo_q: float = 0.1, hi_q: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) height quantiles of the cloud part of the distribution, in metres."""
    p = np.asarray(probs, dtype=np.float64)[:, :K_BINS]
    cdf = np.cumsum(p / np.clip(p.sum(1, keepdims=True), 1e-12, None), axis=1)
    lo = EDGES_M[np.clip((cdf >= lo_q).argmax(1), 0, K_BINS - 1)]
    hi = EDGES_M[np.clip((cdf >= hi_q).argmax(1), 0, K_BINS - 1) + 1]
    return lo, hi
