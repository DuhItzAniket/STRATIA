"""Ceilometer QC and pairing tolerance (P031).

QC: completeness of the 15 s record stream (records per day, duplicate times, gaps), physical range of the heights,
layer ordering, the sky-condition index (rain, fog, snow, particles on the window), optics, laser and error bits.

Pairing tolerance: a camera frame taken at time t gets its cloud-base label from the ceilometer records within
±w of t. Two measurements say how large w may be: how often two records Δ seconds apart agree on cloud presence,
on the étage of the lowest base and on the base height itself (`agreement_vs_offset`), and how often all records
inside a ±w window agree (`window_consistency`). The tolerance is the largest window whose within-window agreement
stays within a small drop of the tightest window's (`choose_tolerance`). The two sites 15 km apart are compared at
the same instants (`cross_site`) to show how local a cloud base is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stratia.data.imbalance import CBH_BINS, cbh_bin

RECORDS_PER_DAY = 5760          # one record every 15 s
MAX_HEIGHT_M = 15000.0
MATCH_TOL_S = 5.0               # a record counts as "Δ later" if it is within this of t + Δ


# ------------------------------------------------------------------------------------------ QC


def qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ceilometer with completeness, range, ordering and flag statistics."""
    rows = []
    for code, part in df.groupby("ceilometer", sort=True):
        t = part.time
        days = t.dt.strftime("%Y-%m-%d")
        per_day = days.value_counts()
        steps = t.diff().dt.total_seconds().dropna()
        layer_order_bad = 0
        for k in (1, 2, 3):
            lo, hi = part[f"cbh{k}"], part[f"cbh{k + 1}"]
            layer_order_bad += int(((hi < lo) & lo.notna() & hi.notna()).sum())
        rows.append({
            "ceilometer": code, "records": len(part), "days": int(per_day.size),
            "records_per_day_min": int(per_day.min()), "records_per_day_median": float(per_day.median()),
            "records_per_day_max": int(per_day.max()), "completeness": float(len(part) / (per_day.size * RECORDS_PER_DAY)),
            "duplicate_times": int(t.duplicated().sum()), "gaps_over_60s": int((steps > 60).sum()),
            "longest_gap_s": float(steps.max()) if len(steps) else np.nan,
            "cbh1_over_max": int((part.cbh1 > MAX_HEIGHT_M).sum()), "layer_order_violations": layer_order_bad,
            "cloud_present": float(part.cbh1.notna().mean()),
            "layers_2plus": float((part.n_layers >= 2).mean()),
            "rain": float(part.qc_rain.mean()), "fog": float(part.qc_fog.mean()), "snow": float(part.qc_snow.mean()),
            "window": float(part.qc_window.mean()), "optics_low": float(part.qc_optics.mean()),
            "laser_warning": float(part.qc_laser_warning.mean()), "error": float(part.qc_error.mean()),
            "qc_any": float(part.qc_any.mean()),
            "cbh1_median_clean_m": float(part.loc[~part.qc_any, "cbh1"].median()),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ pairing tolerance


def _seconds(t: pd.Series) -> np.ndarray:
    tt = pd.to_datetime(t)
    if getattr(tt.dt, "tz", None) is not None:
        tt = tt.dt.tz_convert(None)
    return (tt - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy()


def agreement_vs_offset(df: pd.DataFrame, offsets_s: tuple[float, ...] = (15, 30, 60, 120, 300, 600, 1800)) -> pd.DataFrame:
    """For records Δ seconds apart (both QC-clean): agreement on cloud presence, on the étage bin of the lowest base
    (both cloudy), the median absolute change of the base, and the share of changes under 200 m and under 10 %."""
    part = df[~df.qc_any].sort_values("time")
    t = _seconds(part.time)
    cbh = part.cbh1.to_numpy(dtype=float)
    present = ~np.isnan(cbh)
    bins = cbh_bin(cbh).to_numpy()
    rows = []
    for d in offsets_s:
        j = np.searchsorted(t, t + d)
        ok = j < len(t)
        i = np.flatnonzero(ok)
        j = j[ok]
        close = np.abs(t[j] - (t[i] + d)) <= MATCH_TOL_S
        i, j = i[close], j[close]
        both = present[i] & present[j]
        dh = np.abs(cbh[i][both] - cbh[j][both])
        rel = dh / np.maximum((cbh[i][both] + cbh[j][both]) / 2, 1.0)
        rows.append({"offset_s": float(d), "pairs": int(len(i)),
                     "presence_agreement": float((present[i] == present[j]).mean()) if len(i) else np.nan,
                     "etage_agreement": float((bins[i][both] == bins[j][both]).mean()) if both.any() else np.nan,
                     "abs_change_median_m": float(np.median(dh)) if both.any() else np.nan,
                     "within_200m": float((dh <= 200).mean()) if both.any() else np.nan,
                     "within_10pct": float((rel <= 0.10).mean()) if both.any() else np.nan})
    return pd.DataFrame(rows)


def window_consistency(df: pd.DataFrame, half_windows_s: tuple[float, ...] = (30, 60, 120, 300, 600),
                       grid_s: float = 60.0) -> pd.DataFrame:
    """Windows ±w around every point of a `grid_s` grid: share of windows with at least one clean record, mean
    records per window, share where all records agree on presence, and share (of windows with any cloud) where all
    cloudy records fall in one étage bin."""
    part = df[~df.qc_any].sort_values("time")
    t = _seconds(part.time)
    present = (~part.cbh1.isna()).to_numpy()
    bins = cbh_bin(part.cbh1.to_numpy(dtype=float)).to_numpy()
    cum = {"n": np.concatenate([[0], np.cumsum(np.ones(len(t)))]),
           "cloud": np.concatenate([[0], np.cumsum(present)])}
    for b in CBH_BINS:
        cum[b] = np.concatenate([[0], np.cumsum(bins == b)])
    centres = np.arange(np.floor(t.min() / grid_s) * grid_s, t.max() + grid_s, grid_s)
    rows = []
    for w in half_windows_s:
        lo, hi = np.searchsorted(t, centres - w, "left"), np.searchsorted(t, centres + w, "right")
        n = cum["n"][hi] - cum["n"][lo]
        has = n > 0
        cloud = cum["cloud"][hi] - cum["cloud"][lo]
        unanimous = (cloud == 0) | (cloud == n)
        top = np.max([cum[b][hi] - cum[b][lo] for b in CBH_BINS], axis=0)
        cloudy = has & (cloud > 0)
        rows.append({"half_window_s": float(w), "windows": int(has.sum()), "coverage": float(has.mean()),
                     "records_per_window": float(n[has].mean()) if has.any() else np.nan,
                     "presence_unanimous": float(unanimous[has].mean()) if has.any() else np.nan,
                     "etage_unanimous": float((top[cloudy] == cloud[cloudy]).mean()) if cloudy.any() else np.nan})
    return pd.DataFrame(rows)


def choose_tolerance(windows: pd.DataFrame, max_drop: float = 0.02) -> dict:
    """The largest half-window whose presence and étage unanimity are within `max_drop` of the tightest window's."""
    w = windows.sort_values("half_window_s")
    base_p, base_e = float(w.presence_unanimous.iloc[0]), float(w.etage_unanimous.iloc[0])
    ok = (w.presence_unanimous >= base_p - max_drop) & (w.etage_unanimous >= base_e - max_drop)
    chosen = w[ok].iloc[-1] if ok.any() else w.iloc[0]
    return {"half_window_s": float(chosen.half_window_s), "presence_unanimous": float(chosen.presence_unanimous),
            "etage_unanimous": float(chosen.etage_unanimous), "records_per_window": float(chosen.records_per_window)}


def cross_site(df: pd.DataFrame, a: str = "CDLRA", b: str = "CDLRB", tol_s: float = 10.0) -> dict:
    """Agreement of the two sites at the same instants (clean records): presence, étage of the lowest base (both
    cloudy), median absolute difference of the base, share within 500 m."""
    pa = df[(df.ceilometer == a) & ~df.qc_any].sort_values("time")[["time", "cbh1"]]
    pb = df[(df.ceilometer == b) & ~df.qc_any].sort_values("time")[["time", "cbh1"]].assign(matched=True)
    merged = pd.merge_asof(pa, pb, on="time", suffixes=("_a", "_b"), tolerance=pd.Timedelta(seconds=tol_s), direction="nearest")
    merged = merged[merged.matched.fillna(False).astype(bool)]          # a NaN base is "no cloud", not "no match"
    present_a, present_b = merged.cbh1_a.notna(), merged.cbh1_b.notna()
    both = present_a & present_b
    diff = (merged.cbh1_a - merged.cbh1_b).abs()[both]
    bins_a, bins_b = cbh_bin(merged.cbh1_a.values).to_numpy(), cbh_bin(merged.cbh1_b.values).to_numpy()
    return {"pairs": int(len(merged)), "presence_agreement": float((present_a == present_b).mean()) if len(merged) else np.nan,
            "etage_agreement": float((bins_a[both.values] == bins_b[both.values]).mean()) if both.any() else np.nan,
            "abs_difference_median_m": float(diff.median()) if both.any() else np.nan,
            "within_500m": float((diff <= 500).mean()) if both.any() else np.nan}


# ------------------------------------------------------------------------------------------ figure and report


def figure(offsets: dict[str, pd.DataFrame], windows: dict[str, pd.DataFrame], chosen_s: float, out: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for code, t in offsets.items():
        axes[0].plot(t.offset_s / 60, t.presence_agreement, "o-", label=f"{code} presence")
        axes[0].plot(t.offset_s / 60, t.etage_agreement, "s--", label=f"{code} étage (both cloudy)")
    axes[0].set(xscale="log", xlabel="offset between two records (min)", ylabel="agreement", title="Two records Δ apart")
    for code, t in windows.items():
        axes[1].plot(t.half_window_s / 60, t.presence_unanimous, "o-", label=f"{code} presence unanimous")
        axes[1].plot(t.half_window_s / 60, t.etage_unanimous, "s--", label=f"{code} étage unanimous")
    chosen_label = f"±{chosen_s:.0f} s" if chosen_s < 60 else f"±{chosen_s / 60:.0f} min"
    axes[1].axvline(chosen_s / 60, color="k", lw=0.8, label=f"chosen {chosen_label}")
    axes[1].set(xscale="log", xlabel="half window (min)", ylabel="share of windows", title="All records inside ±w")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def report_markdown(qc: pd.DataFrame, offsets: dict[str, pd.DataFrame], windows: dict[str, pd.DataFrame], chosen: dict,
                    cross: dict, decision: list[str], figure_path: str) -> str:
    def pct(x: float) -> str:
        return "n/a" if pd.isna(x) else f"{x:.1%}"

    lines = ["# Ceilometer QC and pairing tolerance (P031)", "",
             "Lufft CHM15k records of the two Eye2Sky ceilometers (CDLRA, CDLRB; April to July 2022; one record every "
             "15 s), checked for completeness, physical range, layer order and quality flags, then used to decide how "
             "far in time a camera frame may be from the records that label it.", "", f"![pairing]({figure_path})", "",
             "## QC summary", "", "| Quantity | " + " | ".join(qc.ceilometer) + " |", "|---|" + "---|" * len(qc)]
    fmt = {"records": "{:,}", "days": "{}", "records_per_day_min": "{:,}", "records_per_day_median": "{:,.0f}",
           "records_per_day_max": "{:,}", "completeness": "{:.1%}", "duplicate_times": "{:,}", "gaps_over_60s": "{:,}",
           "longest_gap_s": "{:,.0f} s", "cbh1_over_max": "{:,}", "layer_order_violations": "{:,}", "cloud_present": "{:.1%}",
           "layers_2plus": "{:.1%}", "rain": "{:.2%}", "fog": "{:.2%}", "snow": "{:.2%}", "window": "{:.2%}",
           "optics_low": "{:.2%}", "laser_warning": "{:.1%}", "error": "{:.2%}", "qc_any": "{:.1%}",
           "cbh1_median_clean_m": "{:,.0f} m"}
    for col, f in fmt.items():
        lines.append(f"| {col.replace('_', ' ')} | " + " | ".join(f.format(v) for v in qc[col]) + " |")
    for code, t in offsets.items():
        lines += ["", f"## {code}: two records Δ apart (QC-clean)", "",
                  "| Δ | Pairs | Presence agreement | Étage agreement (both cloudy) | Median |Δ base| | Within 200 m "
                  "| Within 10 % |",
                  "|---|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            lines.append(f"| {r.offset_s:.0f} s | {int(r.pairs):,} | {pct(r.presence_agreement)} | {pct(r.etage_agreement)} | "
                         f"{r.abs_change_median_m:,.0f} m | {pct(r.within_200m)} | {pct(r.within_10pct)} |")
    for code, t in windows.items():
        lines += ["", f"## {code}: all records inside ±w (1-minute grid, QC-clean)", "",
                  "| ±w | Windows with records | Coverage | Records per window | Presence unanimous | Étage unanimous |",
                  "|---|---|---|---|---|---|"]
        for _, r in t.iterrows():
            lines.append(f"| {r.half_window_s:.0f} s | {int(r.windows):,} | {pct(r.coverage)} | "
                         f"{r.records_per_window:.1f} | {pct(r.presence_unanimous)} | {pct(r.etage_unanimous)} |")
    lines += ["", "## The two sites at the same instants", "",
              f"- Pairs within 10 s: {cross['pairs']:,}; presence agreement {pct(cross['presence_agreement'])}; étage agreement "
              f"(both cloudy) {pct(cross['etage_agreement'])}; median |difference| of the base "
              f"{cross['abs_difference_median_m']:,.0f} m; within 500 m {pct(cross['within_500m'])}.",
              "", "## Pairing tolerance", "",
              f"Chosen half-window: **±{chosen['half_window_s']:.0f} s** ({chosen['records_per_window']:.1f} records per window; "
              f"presence unanimous {pct(chosen['presence_unanimous'])}, étage unanimous {pct(chosen['etage_unanimous'])}).", ""]
    lines += [f"- {d}" for d in decision]
    return "\n".join(lines) + "\n"
