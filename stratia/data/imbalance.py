"""Imbalance report (P030): what each dataset's labels look like, and how training will sample them.

Distributions measured: native classes per dataset (and per official split where one exists), Montenegro's
majority oktas and height codes, cloud fraction of the segmentation masks, sun-zenith bins where the sun position
is known, and the ceilometer's cloud-base bins (the Paper B targets). Each distribution carries three imbalance
numbers: the ratio of the largest to the smallest class, the normalised entropy, and the effective number of
classes exp(H).

Sampling decided here (see the report): sources are drawn in proportion to the square root of their size
(`source_weights`), classes inside a source are weighted by the class-balanced rule of Cui et al. (2019)
(`class_balanced_weights`), soft labels are never resampled, and macro metrics are reported alongside micro ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stratia.data.segmentation import CLOUD, IGNORE, SKY, load_masks

CBH_EDGES_M = (0.0, 2000.0, 6000.0, np.inf)            # weak étage thresholds of the lowest base (P036 ablates them)
CBH_BINS = ("low", "mid", "high")
SUN_EDGES = (0.0, 30.0, 50.0, 70.0, 85.0, 180.0)
SUN_BINS = ("0-30", "30-50", "50-70", "70-85", "85+")
CLOUD_FRACTION_EDGES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0001)
CLOUD_FRACTION_BINS = ("clear <5%", "5-25%", "25-50%", "50-75%", "75-95%", "overcast >95%")


# ------------------------------------------------------------------------------------------ distributions


def distribution(values: pd.Series, order: list | None = None) -> pd.DataFrame:
    """Counts and shares of the values (NaN dropped), as rows value / count / share."""
    counts = values.dropna().value_counts()
    if order is not None:
        counts = counts.reindex(order, fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame({"value": counts.index.astype(str), "count": counts.values.astype(int),
                         "share": counts.values / total if total else 0.0})


def imbalance_metrics(counts: np.ndarray) -> dict:
    """Largest/smallest ratio, normalised entropy (1 = uniform) and effective number of classes exp(H)."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if len(c) == 0:
        return {"classes": 0, "ratio": np.nan, "entropy": np.nan, "effective_classes": np.nan}
    p = c / c.sum()
    h = float(-(p * np.log(p)).sum())
    return {"classes": int(len(c)), "ratio": float(c.max() / c.min()),
            "entropy": h / np.log(len(c)) if len(c) > 1 else 1.0, "effective_classes": float(np.exp(h))}


def cbh_bin(cbh_m: pd.Series | np.ndarray, edges: tuple = CBH_EDGES_M) -> pd.Series:
    """'none' where there is no cloud base, else low / mid / high by the weak thresholds."""
    s = pd.Series(np.asarray(cbh_m, dtype=float))
    out = pd.Series(pd.cut(s, bins=list(edges), labels=list(CBH_BINS), right=False).astype(object))
    return out.where(s.notna(), "none")


def sun_bin(zenith_deg: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(np.asarray(zenith_deg, dtype=float))
    return pd.Series(pd.cut(s, bins=list(SUN_EDGES), labels=list(SUN_BINS), right=False).astype(object))


def cloud_fraction_bin(fraction: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(np.asarray(fraction, dtype=float))
    return pd.Series(pd.cut(s, bins=list(CLOUD_FRACTION_EDGES), labels=list(CLOUD_FRACTION_BINS), right=False).astype(object))


def mask_cloud_fraction(args: tuple[str, str]) -> dict:
    """Cloud share of the labelled sky pixels of one mask, and the share of each layer class where there are layers."""
    dataset, path = args
    try:
        sky, layer = load_masks(dataset, path)
    except (OSError, ValueError) as e:
        return {"cloud_fraction": np.nan, "low": np.nan, "mid": np.nan, "high": np.nan, "error": str(e)}
    labelled = (sky == SKY) | (sky == CLOUD)
    n = int(labelled.sum())
    out = {"cloud_fraction": float((sky == CLOUD).sum() / n) if n else np.nan, "error": None}
    valid = layer != IGNORE
    for k, name in enumerate(("low", "mid", "high")):
        out[name] = float((layer == k).sum() / n) if (n and valid.any()) else np.nan
    return out


# ------------------------------------------------------------------------------------------ sampling


def source_weights(counts: dict[str, int], power: float = 0.5) -> pd.DataFrame:
    """Sampling shares per source under three rules: natural (proportional to size), power (size ** power, the
    rule chosen: 0.5 = square root) and uniform. `oversampling` is the power share over the natural share."""
    names = list(counts)
    n = np.array([counts[k] for k in names], dtype=float)
    natural = n / n.sum()
    powered = n ** power / (n ** power).sum()
    uniform = np.full(len(n), 1.0 / len(n))
    return pd.DataFrame({"source": names, "images": n.astype(int), "natural": natural, "power": powered, "uniform": uniform,
                         "oversampling": powered / natural})


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Loss weights (1 - beta) / (1 - beta ** n_c) of Cui et al. (2019), normalised to a mean of one over classes."""
    n = np.asarray(counts, dtype=float)
    w = (1.0 - beta) / (1.0 - beta ** np.maximum(n, 1.0))
    return w / w.mean()


# ------------------------------------------------------------------------------------------ report


def table_markdown(dist: pd.DataFrame) -> list[str]:
    lines = ["| Value | Count | Share |", "|---|---|---|"]
    for _, r in dist.iterrows():
        lines.append(f"| {r.value} | {int(r['count']):,} | {r.share:.1%} |")
    return lines


def metrics_line(dist: pd.DataFrame) -> str:
    m = imbalance_metrics(dist["count"].values)
    if m["classes"] == 0:
        return "no data"
    return (f"{m['classes']} classes, largest/smallest {m['ratio']:.1f}, normalised entropy "
            f"{m['entropy']:.2f}, effective classes {m['effective_classes']:.1f}")


def report_markdown(sections: list[tuple[str, str, pd.DataFrame]], sources: pd.DataFrame,
                    cb_examples: dict[str, pd.DataFrame], decision: list[str], figure: str) -> str:
    lines = ["# Imbalance report (P030)", "",
             "Label distributions per dataset (native classes, official splits where they exist, Montenegro majority "
             "codes, segmentation cloud fraction, sun-zenith bins, ceilometer cloud-base bins) with three imbalance "
             "numbers each: the largest/smallest class ratio, the normalised entropy (1 = uniform) and the effective "
             "number of classes exp(H). The sampling rules chosen for training are at the end.", "",
             f"![distributions]({figure})", ""]
    for title, note, dist in sections:
        lines += [f"## {title}", "", f"{note} {metrics_line(dist)}.", ""] + table_markdown(dist) + [""]
    lines += ["## Sources and the sampling rule", "",
              "Share of each source under natural (proportional), square-root (chosen) and uniform sampling.", "",
              "| Source | Images | Natural | Square root | Uniform | Oversampling (sqrt / natural) |",
              "|---|---|---|---|---|---|"]
    for _, r in sources.iterrows():
        lines.append(f"| {r.source} | {int(r.images):,} | {r.natural:.1%} | {r.power:.1%} | {r.uniform:.1%} | "
                     f"{r.oversampling:.2f}x |")
    for name, t in cb_examples.items():
        lines += ["", f"### Class-balanced loss weights, {name} (beta = 0.999, mean 1)", "",
                  "| Class | Images | Weight |", "|---|---|---|"]
        for _, r in t.iterrows():
            lines.append(f"| {r['class']} | {int(r.images):,} | {r.weight:.2f} |")
    lines += ["", "## Decision", ""] + [f"- {d}" for d in decision]
    return "\n".join(lines) + "\n"


def figure(panels: list[tuple[str, pd.DataFrame]], out: str | Path, columns: int = 3) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, dist) in zip(axes.ravel(), panels, strict=False):
        ax.axis("on")
        ax.bar(range(len(dist)), dist.share.values, color="#4C78A8")
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels(dist.value.values, rotation=40, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_ylabel("share", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
