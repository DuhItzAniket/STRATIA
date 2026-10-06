"""Temporal autocorrelation (P027): how long two frames of one camera stay "the same scene".

For a camera's time series, pairs of frames are sampled at time gaps in doubling bins (30 s, 1 min, 2 min, ...,
weeks) and measured with the P025 instruments: DINOv3 cosine similarity and the pHash Hamming distance. The curve
of similarity against gap is compared with two baselines from the same camera: *different day* (any pair more
than a day apart: what "independent frames" look like for this camera) and *same hour, other day* (pairs less
than 30 minutes apart in time of day but on different days: what the sun's position alone makes alike). The
minimum gap for a split is the smallest gap at which the curve has come down to the different-day baseline, by
two rules (mean-cosine excess, and the share of pairs that P025 would call the same scene). Whole-day blocks are
the unit of splitting whenever the data has enough days; the gap says how much buffer two blocks need.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stratia.data.near_duplicates import popcount

DAY = 86_400.0
DEFAULT_EDGES = 30.0 * 2.0 ** np.arange(0, 17)          # 30 s ... 22.8 days, doubling
CROSS_EDGES = np.concatenate([[0.0], DEFAULT_EDGES])      # cross-camera: a "same moment" bin [0, 30 s) in front
SAME_SCENE = 0.97                                          # P025's same-scene cosine


def fmt_gap(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "none within the data"
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 2 * DAY:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / DAY:.1f} d"


# ------------------------------------------------------------------------------------------------- times


def times_from_filenames(files: pd.Series, pattern: str = r"(\d{12,14})(?:_\d+)?\.jpg$") -> pd.Series:
    """UTC-naive timestamps parsed from file names such as `asi_001_170328164030.jpg` (yymmddHHMMSS) or
    `20230126135101_00160.jpg` (yyyymmddHHMMSS); NaT where the name has none."""
    digits = files.str.extract(pattern)[0]
    out = pd.Series(pd.NaT, index=files.index, dtype="datetime64[ns]")
    twelve, fourteen = digits.str.len() == 12, digits.str.len() == 14
    out[twelve] = pd.to_datetime(digits[twelve], format="%y%m%d%H%M%S", errors="coerce")
    out[fourteen] = pd.to_datetime(digits[fourteen], format="%Y%m%d%H%M%S", errors="coerce")
    return out


def seconds(times: pd.Series) -> np.ndarray:
    """Seconds since the first timestamp, as float64 (NaT -> NaN)."""
    t = pd.to_datetime(times)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    return ((t - t.min()) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)


# ------------------------------------------------------------------------------------------------- pairs


def sample_pairs_by_gap(t_a: np.ndarray, edges: np.ndarray, per_bin: int, rng: np.random.Generator,
                        t_b: np.ndarray | None = None) -> pd.DataFrame:
    """Up to `per_bin` random pairs for each gap bin [edges[b], edges[b+1]). Within one series (`t_b` None) the
    second frame comes after the first; across two series (`t_b` given, both sorted ascending) the absolute gap
    is used. Columns: i (index into t_a), j (index into t_b or t_a), gap_s, bin. Bins without pairs are absent."""
    t_a = np.asarray(t_a, dtype=np.float64)
    cross = t_b is not None
    t_b = np.asarray(t_b, dtype=np.float64) if cross else t_a
    parts = []
    for b in range(len(edges) - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        i = rng.integers(0, len(t_a), size=4 * per_bin)
        if cross:
            forward = rng.random(len(i)) < 0.5
            start = np.where(forward, np.searchsorted(t_b, t_a[i] + lo, "left"), np.searchsorted(t_b, t_a[i] - hi, "right"))
            stop = np.where(forward, np.searchsorted(t_b, t_a[i] + hi, "left"), np.searchsorted(t_b, t_a[i] - lo, "right"))
        else:
            start, stop = np.searchsorted(t_b, t_a[i] + lo, "left"), np.searchsorted(t_b, t_a[i] + hi, "left")
        ok = stop > start
        i, start, stop = i[ok], start[ok], stop[ok]
        j = start + (rng.random(len(start)) * (stop - start)).astype(np.int64)
        j = np.minimum(j, stop - 1)
        keep = (j != i) if not cross else np.ones(len(j), dtype=bool)
        i, j = i[keep][:per_bin], j[keep][:per_bin]
        if len(i):
            parts.append(pd.DataFrame({"i": i, "j": j, "gap_s": np.abs(t_b[j] - t_a[i]), "bin": b}))
    cols = ["i", "j", "gap_s", "bin"]
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)


def sample_same_hour_other_days(t: np.ndarray, per: int, rng: np.random.Generator, window_s: float = 1800.0) -> pd.DataFrame:
    """Pairs on different days whose times of day differ by less than `window_s`: the sun's position is the same,
    the weather is not. Columns i, j, gap_s, bin = -1."""
    t = np.asarray(t, dtype=np.float64)
    span_days = int(np.ceil((t.max() - t.min()) / DAY)) if len(t) > 1 else 0
    if span_days < 1:
        return pd.DataFrame(columns=["i", "j", "gap_s", "bin"])
    i = rng.integers(0, len(t), size=6 * per)
    k = rng.integers(1, span_days + 1, size=len(i)) * np.where(rng.random(len(i)) < 0.5, -1, 1)
    centre = t[i] + k * DAY
    start, stop = np.searchsorted(t, centre - window_s, "right"), np.searchsorted(t, centre + window_s, "left")   # open window
    ok = stop > start
    i, start, stop = i[ok], start[ok], stop[ok]
    j = np.minimum(start + (rng.random(len(start)) * (stop - start)).astype(np.int64), stop - 1)
    i, j = i[:per], j[:per]
    return pd.DataFrame({"i": i, "j": j, "gap_s": np.abs(t[j] - t[i]), "bin": -1})


def pair_measures(pairs: pd.DataFrame, emb_a: np.ndarray, ph_a: np.ndarray, emb_b: np.ndarray | None = None,
                  ph_b: np.ndarray | None = None) -> pd.DataFrame:
    """Add cosine (embeddings are L2-normalised) and pHash Hamming distance (plain hashes, uint64) to the pairs."""
    emb_b = emb_a if emb_b is None else emb_b
    ph_b = ph_a if ph_b is None else ph_b
    out = pairs.copy()
    i, j = pairs.i.to_numpy(dtype=np.int64), pairs.j.to_numpy(dtype=np.int64)
    out["cosine"] = np.einsum("nd,nd->n", emb_a[i].astype(np.float32), emb_b[j].astype(np.float32))
    out["phash"] = popcount(ph_a[i] ^ ph_b[j]) if len(i) else np.array([], dtype=np.int64)
    return out


# ------------------------------------------------------------------------------------------------- curves


def curve(pairs: pd.DataFrame, edges: np.ndarray, min_pairs: int = 50, same_scene: float = SAME_SCENE) -> pd.DataFrame:
    """One row per gap bin with enough pairs: gap edges, median gap, mean/median/p10 cosine, share of same-scene
    pairs, mean pHash distance."""
    rows = []
    for b, part in pairs.groupby("bin", sort=True):
        if b < 0 or len(part) < min_pairs:
            continue
        rows.append({"bin": int(b), "gap_lo": float(edges[b]), "gap_hi": float(edges[b + 1]), "n": len(part),
                     "gap_median_s": float(part.gap_s.median()), "cos_mean": float(part.cosine.mean()),
                     "cos_median": float(part.cosine.median()), "cos_p10": float(part.cosine.quantile(0.1)),
                     "same_scene_frac": float((part.cosine >= same_scene).mean()), "phash_mean": float(part.phash.mean())})
    cols = ["bin", "gap_lo", "gap_hi", "n", "gap_median_s", "cos_mean", "cos_median", "cos_p10", "same_scene_frac", "phash_mean"]
    return pd.DataFrame(rows, columns=cols)


def baseline_row(pairs: pd.DataFrame, name: str, same_scene: float = SAME_SCENE) -> dict:
    return {"baseline": name, "n": len(pairs), "cos_mean": float(pairs.cosine.mean()) if len(pairs) else np.nan,
            "cos_p10": float(pairs.cosine.quantile(0.1)) if len(pairs) else np.nan,
            "same_scene_frac": float((pairs.cosine >= same_scene).mean()) if len(pairs) else np.nan,
            "phash_mean": float(pairs.phash.mean()) if len(pairs) else np.nan}


def decide_gap(curve_df: pd.DataFrame, base_cos: float, base_same_scene: float, excess_tol: float = 0.1,
               same_scene_tol: float = 0.01) -> dict:
    """The smallest gap (upper edge of the first qualifying bin) at which (a) the mean-cosine excess over the
    different-day baseline, relative to the first bin's excess, is <= excess_tol, and (b) the same-scene share is
    within same_scene_tol of the baseline. `gap_s` is the larger of the two (NaN if either never qualifies)."""
    if len(curve_df) == 0:
        return {"gap_by_excess_s": np.nan, "gap_by_same_scene_s": np.nan, "gap_s": np.nan, "excess": pd.Series(dtype=float)}
    adjacent = float(curve_df.cos_mean.iloc[0])
    excess = (curve_df.cos_mean - base_cos) / max(adjacent - base_cos, 1e-9)
    ok_a = excess <= excess_tol
    ok_b = curve_df.same_scene_frac <= base_same_scene + same_scene_tol
    gap_a = float(curve_df.gap_hi[ok_a].iloc[0]) if ok_a.any() else np.nan
    gap_b = float(curve_df.gap_hi[ok_b].iloc[0]) if ok_b.any() else np.nan
    gap = np.nan if (np.isnan(gap_a) or np.isnan(gap_b)) else max(gap_a, gap_b)
    return {"gap_by_excess_s": gap_a, "gap_by_same_scene_s": gap_b, "gap_s": gap, "excess": excess}


def day_structure(times: pd.Series) -> dict:
    """How many whole-day blocks a series offers and how full they are."""
    t = pd.to_datetime(times).dropna()
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    per_day = t.groupby(t.dt.date).size()
    steps = t.sort_values().diff().dt.total_seconds().dropna()
    return {"frames": int(len(t)), "days": int(len(per_day)),
            "frames_per_day_median": float(per_day.median()) if len(per_day) else np.nan,
            "step_median_s": float(steps.median()) if len(steps) else np.nan,
            "first": str(t.min().date()) if len(t) else "", "last": str(t.max().date()) if len(t) else ""}


# ------------------------------------------------------------------------------------------------- figure, report


def plot_curves(results: dict[str, dict], out: str | Path) -> Path:
    """Two panels per camera row: mean cosine (with p10) and same-scene share against gap, with the two baselines
    as horizontal lines. `results[name]` has keys curve, baselines (DataFrame with column baseline), decision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 2.9 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        r = results[name]
        c, b = r["curve"], r["baselines"].set_index("baseline")
        x = c.gap_median_s / 3600.0
        ax = axes[row, 0]
        ax.plot(x, c.cos_mean, "o-", label="mean cosine")
        ax.plot(x, c.cos_p10, "s--", color="grey", label="10th percentile")
        for key, colour in (("different_day", "C3"), ("same_hour_other_day", "C1")):
            if key in b.index:
                ax.axhline(b.loc[key, "cos_mean"], color=colour, ls=":", label=key.replace("_", " "))
        if np.isfinite(r["decision"]["gap_s"]):
            ax.axvline(r["decision"]["gap_s"] / 3600.0, color="k", lw=0.8, label=f"gap {fmt_gap(r['decision']['gap_s'])}")
        ax.set_xscale("log")
        ax.set_ylabel("DINOv3 cosine")
        ax.set_title(name, loc="left", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
        ax = axes[row, 1]
        ax.plot(x, c.same_scene_frac, "o-", color="C2", label=f"share with cosine >= {SAME_SCENE}")
        if "different_day" in b.index:
            ax.axhline(b.loc["different_day", "same_scene_frac"], color="C3", ls=":", label="different day")
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("same-scene share")
        ax.legend(fontsize=7, loc="upper right")
        for ax in axes[row]:
            ax.set_xlabel("time gap (hours)")
            ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def report_markdown(results: dict[str, dict], structures: dict[str, dict], decisions_text: list[str],
                    figure: str) -> str:
    lines = ["# Temporal autocorrelation report (P027)", "",
             "Pairs of frames from one camera, sampled at time gaps in doubling bins, measured with the P025 instruments "
             "(DINOv3 ViT-S/16 cosine, pHash Hamming distance) and compared with two baselines from the same camera: "
             "*different day* (any pair more than a day apart) and *same hour, other day* (less than 30 minutes apart "
             "in time of day, on different days: the sun's position alone). The minimum gap for a split is where the "
             "curve reaches the different-day baseline (mean-cosine excess <= 10 % of the adjacent-frame excess, and "
             "same-scene share within 1 point of the baseline).", "",
             f"![similarity against gap]({figure})", "",
             "## Series", "", "| Camera | Frames | Days | Frames per day (median) | Cadence (median step) | From | To |",
             "|---|---|---|---|---|---|---|"]
    for name, s in structures.items():
        lines.append(f"| {name} | {s['frames']:,} | {s['days']} | {s['frames_per_day_median']:.0f} | "
                     f"{fmt_gap(s['step_median_s'])} | {s['first']} | {s['last']} |")
    for name, r in results.items():
        c, b, d = r["curve"], r["baselines"], r["decision"]
        lines += ["", f"## {name}", "",
                  "| Gap | Pairs | Mean cosine | 10th pct | Same-scene share | Mean pHash distance | Excess |",
                  "|---|---|---|---|---|---|---|"]
        excess = d["excess"].to_numpy() if len(d["excess"]) else np.full(len(c), np.nan)
        for (_, row), e in zip(c.iterrows(), excess, strict=True):
            gap = f"{fmt_gap(row.gap_lo)} – {fmt_gap(row.gap_hi)}"
            lines.append(f"| {gap} | {int(row.n):,} | {row.cos_mean:.3f} | {row.cos_p10:.3f} | "
                         f"{row.same_scene_frac:.1%} | {row.phash_mean:.1f} | {e:.2f} |")
        for _, row in b.iterrows():
            lines.append(f"| baseline: {row.baseline.replace('_', ' ')} | {int(row.n):,} | {row.cos_mean:.3f} | "
                         f"{row.cos_p10:.3f} | {row.same_scene_frac:.1%} | {row.phash_mean:.1f} | |")
        lines += ["", f"Minimum gap: by cosine excess {fmt_gap(d['gap_by_excess_s'])}; by same-scene share "
                  f"{fmt_gap(d['gap_by_same_scene_s'])}; **decided {fmt_gap(d['gap_s'])}**."]
    lines += ["", "## Decisions", ""] + [f"- {t}" for t in decisions_text]
    return "\n".join(lines) + "\n"
