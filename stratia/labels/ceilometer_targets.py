"""Ceilometer → targets (P036): the cloud-base labels a camera frame receives from the ceilometer at its site.

For a frame at time t the records within ±30 s (P031) give: the share of clean records that see cloud (the label's
confidence), the median lowest base of those that do, its spread inside the window, the median layer count, the
second layer where there is one (multi-layer ambiguity is recorded, not discarded), the étage of the lowest base
under the weak thresholds (2 / 6 km) and under the WMO mid-latitude boundary (2 / 7 km) for the ablation, and a
"no cloud overhead" label when no clean record sees cloud. Windows without a clean record give no label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WEAK_EDGES_M = (0.0, 2000.0, 6000.0)       # low < 2 km, mid 2-6 km, high >= 6 km (configs/ontology.yaml)
WMO_EDGES_M = (0.0, 2000.0, 7000.0)        # ablation: WMO mid-latitude middle/high boundary at 7 km
LABELS = ("none", "low", "mid", "high")


def etage_of(cbh_m: np.ndarray, edges: tuple = WEAK_EDGES_M) -> np.ndarray:
    """'low' / 'mid' / 'high' for a base height, 'none' for NaN."""
    cbh = np.asarray(cbh_m, dtype=float)
    out = np.full(cbh.shape, "none", dtype=object)
    out[(cbh >= edges[0]) & (cbh < edges[1])] = "low"
    out[(cbh >= edges[1]) & (cbh < edges[2])] = "mid"
    out[cbh >= edges[2]] = "high"
    return out


def _seconds(t) -> np.ndarray:
    tt = pd.to_datetime(pd.Series(t))
    if getattr(tt.dt, "tz", None) is not None:
        tt = tt.dt.tz_convert(None)
    return (tt - pd.Timestamp("2000-01-01")).dt.total_seconds().to_numpy()


def targets_at(records: pd.DataFrame, times, half_window_s: float = 30.0, min_cloud_share: float = 0.5) -> pd.DataFrame:
    """One row per query time (records of ONE ceilometer, any order). Label = 'none' when no clean record sees cloud,
    an étage when at least `min_cloud_share` of them do, else 'mixed'; `confidence` is the share behind the label."""
    rec = records.sort_values("time")
    t_rec = _seconds(rec.time)
    clean = (~rec.qc_any.to_numpy()).astype(float)
    cbh1 = rec.cbh1.to_numpy(dtype=float)
    cbh2 = rec.cbh2.to_numpy(dtype=float) if "cbh2" in rec else np.full(len(rec), np.nan)
    layers = rec.n_layers.to_numpy(dtype=float)
    cloudy_clean = clean * (~np.isnan(cbh1))
    cum = lambda x: np.concatenate([[0.0], np.cumsum(x)])  # noqa: E731
    c_all, c_clean, c_cloud = cum(np.ones(len(rec))), cum(clean), cum(cloudy_clean)
    c_layers = cum(np.where(clean > 0, layers, 0.0))
    t_q = _seconds(times)
    lo, hi = np.searchsorted(t_rec, t_q - half_window_s, "left"), np.searchsorted(t_rec, t_q + half_window_s, "right")
    n_all, n_clean, n_cloud = c_all[hi] - c_all[lo], c_clean[hi] - c_clean[lo], c_cloud[hi] - c_cloud[lo]
    cloud_share = np.where(n_clean > 0, n_cloud / np.maximum(n_clean, 1), np.nan)
    layers_mean = np.where(n_clean > 0, (c_layers[hi] - c_layers[lo]) / np.maximum(n_clean, 1), np.nan)
    cbh_m = np.full(len(t_q), np.nan)
    cbh_min, cbh_max, cbh2_m = cbh_m.copy(), cbh_m.copy(), cbh_m.copy()
    for k in np.flatnonzero(n_cloud > 0):
        seg = slice(lo[k], hi[k])
        vals = cbh1[seg][(clean[seg] > 0) & ~np.isnan(cbh1[seg])]
        cbh_m[k], cbh_min[k], cbh_max[k] = np.median(vals), vals.min(), vals.max()
        second = cbh2[seg][(clean[seg] > 0) & ~np.isnan(cbh2[seg])]
        if len(second):
            cbh2_m[k] = np.median(second)
    label = np.full(len(t_q), None, dtype=object)
    valid = n_clean > 0
    label[valid & (n_cloud == 0)] = "none"
    cloudy = valid & (cloud_share >= min_cloud_share)
    label[cloudy] = etage_of(np.where(cloudy, cbh_m, np.nan))[cloudy]
    label[valid & (n_cloud > 0) & (cloud_share < min_cloud_share)] = "mixed"
    etage_wmo = np.full(len(t_q), None, dtype=object)
    etage_wmo[cloudy] = etage_of(np.where(cloudy, cbh_m, np.nan), WMO_EDGES_M)[cloudy]
    confidence = np.where(valid, np.where(n_cloud == 0, 1.0, np.maximum(cloud_share, 1 - cloud_share)), np.nan)
    qc_share = np.where(n_all > 0, 1 - n_clean / np.maximum(n_all, 1), np.nan)
    return pd.DataFrame({"time": pd.to_datetime(pd.Series(times)).to_numpy(), "n_records": n_all.astype(int),
                         "n_clean": n_clean.astype(int), "qc_share": qc_share,
                         "cloud_share": cloud_share, "cbh_m": cbh_m, "cbh_min_m": cbh_min, "cbh_max_m": cbh_max,
                         "cbh2_m": cbh2_m, "layers": layers_mean, "label": label, "etage_wmo": etage_wmo,
                         "confidence": confidence, "valid": valid, "mixed": valid & (n_cloud > 0) & (n_cloud < n_clean)})


def time_grid(records: pd.DataFrame, step_s: float = 30.0) -> pd.DatetimeIndex:
    """Every `step_s` from the first to the last record of each day present in the records (UTC)."""
    t = pd.to_datetime(records.time)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(None)
    days = sorted(set(t.dt.normalize()))
    end = pd.Timedelta(days=1) - pd.Timedelta(seconds=step_s)
    parts = [pd.date_range(d, d + end, freq=f"{int(step_s)}s") for d in days]
    return pd.DatetimeIndex(np.concatenate([p.to_numpy() for p in parts])).tz_localize("UTC")


def daytime(times: pd.DatetimeIndex, latitude: float, longitude: float, min_elevation_deg: float = 0.0) -> np.ndarray:
    import pvlib

    sp = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(times), latitude, longitude)
    return (sp.elevation.to_numpy() > min_elevation_deg)


def summary(targets: pd.DataFrame) -> dict:
    """Coverage and label shares (all grid points and daytime ones) for one ceilometer."""
    out = {}
    for name, part in (("all", targets), ("daytime", targets[targets.daytime])):
        valid = part[part.valid]
        lab = valid.label.value_counts(normalize=True)
        wmo = valid.etage_wmo.dropna().value_counts(normalize=True)
        out[name] = {"grid_points": len(part), "valid_share": float(part.valid.mean()) if len(part) else np.nan,
                     "mixed_share": float(valid.mixed.mean()) if len(valid) else np.nan,
                     "labels": {k: float(lab.get(k, 0.0)) for k in (*LABELS, "mixed")},
                     "etage_wmo": {k: float(wmo.get(k, 0.0)) for k in LABELS[1:]},
                     "layers_2plus_share": float((valid.layers >= 1.5).mean()) if len(valid) else np.nan,
                     "spread_median_m": (float((valid.cbh_max_m - valid.cbh_min_m).median())
                                         if valid.cbh_m.notna().any() else np.nan),
                     "confidence_median": float(valid.confidence.median()) if len(valid) else np.nan}
    return out


def report_markdown(summaries: dict[str, dict], half_window_s: float, decision: list[str]) -> str:
    lines = ["# Ceilometer targets (P036)", "",
             f"Targets on a 30 s grid for every day of each ceilometer, from the clean records within ±{half_window_s:.0f} s "
             "(P031): no cloud overhead, or the étage of the median lowest base (weak thresholds 2 / 6 km; WMO 2 / 7 km "
             "for the ablation), with the cloudy share as confidence, the base spread inside the window, the mean layer "
             "count and the second layer where there is one. Table `data/ceilometer_targets.parquet`.", ""]
    for code, s in summaries.items():
        lines += [f"## {code}", "", "| Subset | Grid points | With a label | Mixed windows | none | low | mid | high | mixed | "
                  "WMO: low / mid / high | >= 2 layers | Base spread (median) | Confidence (median) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, v in s.items():
            lab, wmo = v["labels"], v["etage_wmo"]
            lines.append(f"| {name} | {v['grid_points']:,} | {v['valid_share']:.1%} | {v['mixed_share']:.1%} | "
                         f"{lab['none']:.1%} | {lab['low']:.1%} | {lab['mid']:.1%} | {lab['high']:.1%} | {lab['mixed']:.1%} | "
                         f"{wmo['low']:.1%} / {wmo['mid']:.1%} / {wmo['high']:.1%} | {v['layers_2plus_share']:.1%} | "
                         f"{v['spread_median_m']:,.0f} m | {v['confidence_median']:.2f} |")
        lines.append("")
    lines += ["## Decisions", ""] + [f"- {d}" for d in decision]
    return "\n".join(lines) + "\n"
