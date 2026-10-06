"""Near-duplicates (P025): perceptual hashes and DINOv3 embeddings.

Two views of "the same picture, changed a little":
    perceptual hashes   dHash (9 x 8 gradients) and pHash (DCT of a 32 x 32 grey image), 64 bits each, computed for
                        all eight flips and rotations so that an augmented copy matches its source (SWINySEG was
                        built from SWIMSEG and SWINSEG with augmentation); distance = minimum Hamming distance
    embeddings          DINOv3 ViT-S/16 CLS vectors at 224 px, L2-normalised; cosine similarity; k nearest neighbours
Candidate pairs come from the embedding neighbours; each pair carries both measures. Pairs above the chosen
thresholds are joined into groups (connected components); every member of a group shares one group_id at split
time (P038). Eye2Sky pairs within the same station are a question of time, not of copies (frames 30 s apart look
alike): they are reported but kept out of the groups, and handled by temporal blocking (P027, P037).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.fft import dct

HASH_BITS = 64
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------------------------------------------------------------------------------ perceptual hashes


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.ravel():
        value = (value << 1) | int(b)
    return value


def dhash(grey: np.ndarray) -> int:
    """Difference hash: 8 rows of 8 'brighter than the right neighbour' bits from a 9 x 8 resample."""
    small = cv2.resize(grey, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _bits_to_int(small[:, 1:] > small[:, :-1])


def phash(grey: np.ndarray) -> int:
    """Perceptual hash: the 8 x 8 lowest DCT frequencies of a 32 x 32 resample, above their median."""
    small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = low.ravel()[1:]  # the DC term is the mean brightness, not structure
    return _bits_to_int(low.ravel() > np.median(flat))


def dihedral(grey: np.ndarray) -> list[np.ndarray]:
    """The eight images one can make of `grey` with flips and right-angle rotations (the first is `grey`)."""
    out = []
    for k in range(4):
        rotated = np.rot90(grey, k)
        out.append(rotated)
        out.append(np.fliplr(rotated))
    return out


def hashes_of(path: str | Path) -> dict:
    """dHash and pHash of the image and of its seven flips and rotations, or `error`."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    grey = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if grey is None:
        return {"dhash": [None] * 8, "phash": [None] * 8, "error": f"cannot decode {path}"}
    variants = dihedral(grey)
    return {"dhash": [dhash(v) for v in variants], "phash": [phash(v) for v in variants], "error": None}


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def min_hamming(hashes_a: list[int], hash_b: int) -> int:
    """Smallest Hamming distance between an image's eight variant hashes and another image's plain hash."""
    return min(hamming(h, hash_b) for h in hashes_a)


_BYTE_POPCOUNT = np.array([bin(b).count("1") for b in range(256)], dtype=np.uint8)


def popcount(values: np.ndarray) -> np.ndarray:
    """Number of set bits of each uint64 in an array of any shape."""
    as_bytes = np.ascontiguousarray(values, dtype=np.uint64).view(np.uint8).reshape(*values.shape, 8)
    return _BYTE_POPCOUNT[as_bytes].sum(axis=-1).astype(np.int64)


def pair_min_hamming(variants: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray) -> np.ndarray:
    """For each pair: the smallest Hamming distance between any of image i's variant hashes and image j's plain
    hash. `variants` is uint64 of shape (n, 8); the plain hash is column 0."""
    distances = popcount(variants[pairs_i] ^ variants[pairs_j][:, :1])
    return distances.min(axis=1)


def exhaustive_matches(variants_a: np.ndarray, plain_b: np.ndarray, max_distance: int, chunk: int = 512) -> pd.DataFrame:
    """Every (a, b) whose smallest Hamming distance over a's eight variants to b's plain hash is <= max_distance.
    All pairs are tried: for a few thousand images on each side this takes seconds and needs no embedding."""
    rows = []
    for start in range(0, len(variants_a), chunk):
        block = variants_a[start:start + chunk]                      # (c, 8)
        d = popcount(block[:, None, :] ^ plain_b[None, :, None])      # (c, m, 8)
        best = d.min(axis=2)                                          # (c, m)
        a_idx, b_idx = np.nonzero(best <= max_distance)
        rows.append(pd.DataFrame({"a": a_idx + start, "b": b_idx, "phash": best[a_idx, b_idx]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["a", "b", "phash"])


# ------------------------------------------------------------------------------------------------ embeddings


def normalise_batch(images_uint8_chw: np.ndarray) -> np.ndarray:
    """uint8 [B, 3, H, W] -> float32 ImageNet-normalised, as DINOv3 expects."""
    x = images_uint8_chw.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN[None, :, None, None]) / IMAGENET_STD[None, :, None, None]


def l2_normalise(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return (features / np.maximum(norms, 1e-12)).astype(np.float32)


def nearest_neighbours(embeddings: np.ndarray, k: int = 10, chunk: int = 2048, device: str | None = None):
    """For every row, the indices and cosine similarities of its k most similar other rows (sorted, best first)."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    emb = torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32)).to(device)
    n = emb.shape[0]
    k = min(k, n - 1)
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, chunk):
            rows = emb[start:start + chunk]
            scores = rows @ emb.T
            ar = torch.arange(start, start + rows.shape[0], device=device)
            scores[torch.arange(rows.shape[0], device=device), ar] = -2.0  # not oneself
            top = torch.topk(scores, k, dim=1)
            idx[start:start + rows.shape[0]] = top.indices.cpu().numpy()
            sim[start:start + rows.shape[0]] = top.values.cpu().numpy()
    return idx, sim


def candidate_pairs(idx: np.ndarray, sim: np.ndarray, min_sim: float) -> pd.DataFrame:
    """Unique pairs i < j among the neighbour lists with cosine >= min_sim."""
    n, k = idx.shape
    i = np.repeat(np.arange(n), k)
    j = idx.ravel()
    s = sim.ravel()
    keep = s >= min_sim
    a, b = np.minimum(i[keep], j[keep]), np.maximum(i[keep], j[keep])
    pairs = pd.DataFrame({"i": a, "j": b, "cosine": s[keep]})
    return pairs.sort_values("cosine", ascending=False).drop_duplicates(["i", "j"]).reset_index(drop=True)


# -------------------------------------------------------------------------------------------------- grouping


def connected_components(n: int, edges_i: np.ndarray, edges_j: np.ndarray) -> np.ndarray:
    """Union-find. Returns a group label per node; nodes without an edge get label -1."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(edges_i.tolist(), edges_j.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(x) for x in range(n)])
    has_edge = np.zeros(n, dtype=bool)
    has_edge[edges_i] = True
    has_edge[edges_j] = True
    labels = np.where(has_edge, roots, -1)
    return labels


# --------------------------------------------------------------------------------------------- contact sheets


def contact_sheet(pairs: pd.DataFrame, files: list[str], cache_root: str | Path, out: str | Path,
                  caption_cols: tuple[str, ...] = ("cosine", "phash"), tile: int = 160, columns: int = 4) -> Path:
    """Pairs side by side (left i, right j) with their measures printed, `columns` pairs per row."""
    from stratia.data.image_cache import cache_path, read_rgb

    rows = int(np.ceil(len(pairs) / columns))
    sheet = np.full((max(rows, 1) * (tile + 24), columns * (2 * tile + 12), 3), 255, dtype=np.uint8)
    for n, (_, p) in enumerate(pairs.iterrows()):
        r, c = divmod(n, columns)
        y, x = r * (tile + 24), c * (2 * tile + 12)
        for side, index in enumerate((int(p["i"]), int(p["j"]))):
            try:
                img = cv2.resize(read_rgb(cache_path(cache_root, files[index])), (tile, tile), interpolation=cv2.INTER_AREA)
            except OSError:
                img = np.zeros((tile, tile, 3), dtype=np.uint8)
            sheet[y:y + tile, x + side * (tile + 4):x + side * (tile + 4) + tile] = img
        text = "  ".join(f"{col[:3]} {p[col]:.3f}" if isinstance(p[col], float) else f"{col[:3]} {p[col]}"
                         for col in caption_cols)
        cv2.putText(sheet, text, (x, y + tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise OSError(f"cannot encode {out}")
    enc.tofile(str(out))
    return out
