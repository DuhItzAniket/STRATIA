"""Label-noise estimation (P040): confident-learning style flags on frozen features.

Out-of-fold class probabilities from a logistic-regression probe (folds by split unit, so near-duplicates never
inform each other's probability) give, for each image, how confidently the probe believes a class other than the
given one. Following Northcutt et al.'s confident joint, a per-class threshold is the mean probability the probe
gives class k to images labelled k; an image labelled y is flagged when some other class k reaches its threshold
and is the most probable of those. Flags are suggestions for review, never deletions or relabels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STRONG_MIN_PROB = 0.5        # the suggested class must be the probe's majority belief ...
STRONG_MIN_MARGIN = 0.25     # ... and clearly ahead of the given class


def out_of_fold_probs(X: np.ndarray, y: np.ndarray, units: np.ndarray, k: int = 5, seed: int = 0, max_iter: int = 2000,
                      C: float = 0.1):
    """Out-of-fold probabilities (n x classes) from a standardised logistic-regression probe with folds by unit.
    C = 0.1 (stronger regularisation than the P029 probes) keeps the probabilities from saturating at 1.0 on a few
    thousand images in 384 dimensions, which would make every flag look certain."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(y)
    rng = np.random.default_rng(seed)
    unique_units = np.unique(units)
    fold_of_unit = dict(zip(unique_units, rng.integers(0, k, size=len(unique_units)), strict=True))
    folds = np.array([fold_of_unit[u] for u in units])
    probs = np.zeros((len(y), len(classes)), dtype=np.float32)
    for f in range(k):
        test = folds == f
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        scaler = StandardScaler().fit(X[~test])
        clf = LogisticRegression(max_iter=max_iter, C=C).fit(scaler.transform(X[~test]), y[~test])
        p = clf.predict_proba(scaler.transform(X[test]))
        for j, c in enumerate(clf.classes_):
            probs[test, np.searchsorted(classes, c)] = p[:, j]
    return probs, classes


def confident_flags(probs: np.ndarray, y: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
    """Per image: given class, its out-of-fold probability, the most confident other class that reaches its own
    threshold (or None), that probability, and the margin; `flagged` when such a class exists."""
    idx = {c: j for j, c in enumerate(classes)}
    given = np.array([idx[v] for v in y])
    thresholds = np.array([probs[given == j, j].mean() if (given == j).any() else np.inf for j in range(len(classes))])
    above = probs >= thresholds[None, :]
    above[np.arange(len(y)), given] = False
    masked = np.where(above, probs, -1.0)
    best = masked.argmax(axis=1)
    flagged = masked[np.arange(len(y)), best] > -1.0
    p_given = probs[np.arange(len(y)), given]
    p_best = probs[np.arange(len(y)), best]
    margin = np.where(flagged, p_best - p_given, np.nan)
    # A weak probe (CCSN's eleven genera reach 46 % with a linear probe, P029) has low thresholds and flags half the
    # images; the strong tier keeps the flags a reviewer should look at first.
    strong = flagged & (p_best >= STRONG_MIN_PROB) & (margin >= STRONG_MIN_MARGIN)
    return pd.DataFrame({"given": classes[given], "p_given": p_given, "suggested": np.where(flagged, classes[best], None),
                         "p_suggested": np.where(flagged, p_best, np.nan), "margin": margin, "flagged": flagged,
                         "strong": strong})


def summary(flags: pd.DataFrame) -> dict:
    f = flags[flags.flagged]
    pairs = (f.groupby(["given", "suggested"]).size().sort_values(ascending=False).head(10)
             .rename("n").reset_index()) if len(f) else pd.DataFrame(columns=["given", "suggested", "n"])
    strong = int(flags.strong.sum()) if "strong" in flags else 0
    return {"images": int(len(flags)), "flagged": int(len(f)), "share": float(len(f) / max(len(flags), 1)),
            "strong": strong, "strong_share": float(strong / max(len(flags), 1)),
            "margin_median": float(f.margin.median()) if len(f) else np.nan, "top_pairs": pairs}


def contact_sheet(flags: pd.DataFrame, files: list[str], cache_root: str | Path, out: str | Path, n: int = 16,
                  tile: int = 160, columns: int = 4) -> Path:
    """The `n` most confident flags, given -> suggested in the caption."""
    import cv2

    from stratia.data.image_cache import cache_path, read_rgb

    top = flags[flags.flagged].sort_values("margin", ascending=False).head(n)
    rows = int(np.ceil(len(top) / columns))
    sheet = np.full((max(rows, 1) * (tile + 24), columns * (tile + 8), 3), 255, dtype=np.uint8)
    for k, (i, r) in enumerate(top.iterrows()):
        row, col = divmod(k, columns)
        y0, x0 = row * (tile + 24), col * (tile + 8)
        try:
            img = cv2.resize(read_rgb(cache_path(cache_root, files[int(i)])), (tile, tile), interpolation=cv2.INTER_AREA)
        except OSError:
            img = np.zeros((tile, tile, 3), dtype=np.uint8)
        sheet[y0:y0 + tile, x0:x0 + tile] = img
        caption = f"{str(r.given)[:11]} -> {str(r.suggested)[:11]} ({r.p_suggested:.2f})"     # long class names overlap
        cv2.putText(sheet, caption, (x0, y0 + tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise OSError(f"cannot encode {out}")
    enc.tofile(str(out))
    return out


def report_markdown(summaries: dict[str, dict], decisions: list[str], sheets: dict[str, str]) -> str:
    lines = ["# Label-noise estimation report (P040)", "",
             "Confident-learning style flags from out-of-fold logistic-regression probes on frozen DINOv3 ViT-S/16 "
             "features (folds by split unit). A flag means the probe is more confident in another class than the "
             "class's own typical confidence; it is a suggestion for review. Nothing is deleted or relabelled. "
             "Flags: `data/label_noise_flags.parquet`.", "",
             f"Strong flags: the suggested class has probability >= {STRONG_MIN_PROB} and leads the given class by "
             f">= {STRONG_MIN_MARGIN}.", "",
             "| Dataset | Images | Flagged | Share | Strong flags | Strong share | Median margin |",
             "|---|---|---|---|---|---|---|"]
    for ds, s in summaries.items():
        lines.append(f"| {ds} | {s['images']:,} | {s['flagged']:,} | {s['share']:.1%} | {s['strong']:,} | "
                     f"{s['strong_share']:.1%} | {s['margin_median']:.2f} |")
    for ds, s in summaries.items():
        lines += ["", f"## {ds}: most frequent given → suggested", "", "| Given | Suggested | Flags |", "|---|---|---|"]
        for _, r in s["top_pairs"].iterrows():
            lines.append(f"| {r.given} | {r.suggested} | {int(r.n):,} |")
        if ds in sheets:
            lines += ["", f"![top flags]({sheets[ds]})"]
    lines += ["", "## Review and decisions", ""] + [f"- {d}" for d in decisions]
    return "\n".join(lines) + "\n"
