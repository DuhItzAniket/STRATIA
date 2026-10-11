"""Image-ceilometer pairing (P047): the cloud-base-height training table.

A pair is one all-sky frame at a ceilometer site (OLDLR <-> CDLRA, WESTE <-> CDLRB) joined to the ceilometer's
target at the same time (P036 table: 30 s grid, +-30 s window, the P031 tolerance), with the geometry needed to read
the label off the image:

* frames come straight from disk on a fixed cadence (every `cadence_s`, default 120 s; P027: adjacent 30 s frames
  are 82-92 % the same scene, so denser sampling adds little and inflates autocorrelation);
* the join takes the nearest grid point within `max_gap_s` (15 s) and keeps the target's label, base height,
  second layer, layer count, confidence, `mixed` and `daytime`;
* the frame's calibration must have a Sun-fitted pose with status ok (P044); the **zenith pixel** (where the
  ceilometer looks) and the radius in pixels of a 10-degree cone around it (the zenith ROI) are recorded, and
  the pixel must be inside the camera mask;
* night frames (Sun below the horizon) are kept in the table with ``daytime = False`` and excluded from the splits:
  the cameras switch to an 80 ms exposure at night and the model is a daytime model (data card);
* splits are by calendar day, shared across the two stations (P027: same-moment frames at nearby stations are
  correlated through the weather), inheriting the in-domain Eye2Sky assignment for days it already covers and
  assigning the remaining days 70/10/20 with the day's dominant label as the stratum.

The guard the plan names is over-representation of slow-changing overcast days: ``day_label_share`` (the share of
the day's pairs that carry the same label as this pair) travels with every row so P075 can down-weight them, and
the report shows the label mix per split and per month.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.eye2sky import parse_image_name
from ..geometry.pose import Pose

SITE_CEILOMETER = {"OLDLR": "CDLRA", "OLUOL": "CDLRA", "OLWIN": "CDLRA", "WESTE": "CDLRB"}
LABEL_ORDER = ("none", "low", "mid", "high", "mixed")
ROI_DEG = 10.0


# ----------------------------------------------------------------------------------------------- frames
def frames_on_disk(root: str | Path, station: str, cadence_s: int = 120) -> pd.DataFrame:
    """Every frame of `station` under <root>/2022/MM/DD/ASI_<date>_<station>/ whose time sits on the cadence grid.
    Columns: station, image_file (relative to the data root's parent of 'Eye2Sky'), utc."""
    root = Path(root)
    rows = []
    for day in sorted(p for p in root.glob(f"2022/*/*/ASI_*_{station}") if p.is_dir()):
        for f in day.rglob("*.jpg"):
            stamp = f.name[:14]
            if len(stamp) != 14 or not stamp.isdigit():
                continue
            sec = int(stamp[8:10]) * 3600 + int(stamp[10:12]) * 60 + int(stamp[12:14])
            if sec % cadence_s:
                continue
            name = parse_image_name(f)
            rows.append({"station": station, "image_file": "Eye2Sky/" + f.relative_to(root).as_posix(), "utc": name.utc})
    df = pd.DataFrame(rows, columns=["station", "image_file", "utc"])
    if len(df):
        df["utc"] = pd.to_datetime(df.utc, utc=True)
        df = df.sort_values("utc").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------------------- join
def join_targets(frames: pd.DataFrame, targets: pd.DataFrame, max_gap_s: float = 15.0) -> pd.DataFrame:
    """Nearest ceilometer grid point per frame (within max_gap_s); frames without one get label NaN, paired=False."""
    if not len(frames):
        return frames.assign(paired=False)
    t = targets.sort_values("time").reset_index(drop=True)
    f = frames.sort_values("utc").reset_index(drop=True)
    merged = pd.merge_asof(f, t, left_on="utc", right_on="time", direction="nearest",
                           tolerance=pd.Timedelta(seconds=max_gap_s))
    merged["gap_s"] = (merged.utc - merged.time).dt.total_seconds().abs()
    merged["paired"] = merged.time.notna() & merged.valid.fillna(False).astype(bool)
    return merged


# ----------------------------------------------------------------------------------------------- geometry
@dataclass(frozen=True)
class ZenithGeometry:
    u: float
    v: float
    roi_radius_px: float
    inside_mask: bool


def zenith_geometry(camera, pose: Pose, mask: np.ndarray | None, roi_deg: float = ROI_DEG) -> ZenithGeometry:
    """Where the zenith projects, the pixel radius of a `roi_deg` cone around it, and whether it is inside the mask."""
    up = np.array([0.0, 0.0, 1.0])
    u, v = camera.ray_to_pixel(pose.enu_to_cam(up))
    u, v = float(u), float(v)
    # a ring of directions roi_deg from the zenith, projected: the mean distance is the ROI radius
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    s, c = np.sin(np.radians(roi_deg)), np.cos(np.radians(roi_deg))
    ring = np.stack([s * np.cos(ang), s * np.sin(ang), np.full_like(ang, c)], axis=1)
    ru, rv = camera.ray_to_pixel(pose.enu_to_cam(ring))
    radius = float(np.nanmean(np.hypot(ru - u, rv - v)))
    inside = bool(np.isfinite(u) and np.isfinite(v) and 0 <= u < camera.width and 0 <= v < camera.height)
    if inside and mask is not None:
        inside = bool(mask[int(round(v)), int(round(u))])
    return ZenithGeometry(u, v, radius, inside)


# ----------------------------------------------------------------------------------------------- splits
def day_strata(pairs: pd.DataFrame) -> pd.Series:
    """Dominant label per day over the daytime pairs of all stations (the stratum for the day assignment)."""
    day = pairs[pairs.paired & pairs.daytime].assign(day=lambda d: d.utc.dt.date)
    return day.groupby("day").label.agg(lambda s: s.value_counts().idxmax())


def assign_days(days: list, inherited: dict, fractions: dict, seed: int, strata: dict | None = None) -> dict:
    """Day -> split. Days in `inherited` keep their split; the others are assigned greedily to reach `fractions`
    inside every stratum (round-robin over a seeded shuffle, largest deficit first)."""
    out = {d: inherited[d] for d in days if d in inherited}
    rest = [d for d in days if d not in inherited]
    rng = np.random.default_rng(seed)
    rng.shuffle(rest)
    strata = strata or {}
    by_stratum: dict[str, list] = {}
    for d in rest:
        by_stratum.setdefault(str(strata.get(d, "unknown")), []).append(d)
    names = list(fractions)
    for _, group in sorted(by_stratum.items()):
        counts = dict.fromkeys(names, 0)
        for d in group:
            total = sum(counts.values()) + 1
            deficit = {n: fractions[n] * total - counts[n] for n in names}
            pick = max(names, key=lambda n: deficit[n])
            counts[pick] += 1
            out[d] = pick
    return out


def split_table(pairs: pd.DataFrame, day_split: dict) -> pd.DataFrame:
    """The protocol file rows: daytime, paired, inside-mask pairs with their day's split."""
    p = pairs[pairs.paired & pairs.daytime & pairs.zenith_inside_mask].copy()
    p["day"] = p.utc.dt.date
    p["split"] = p.day.map(day_split)
    p["sample_id"] = "eye2sky:" + p.image_file
    return p[["sample_id", "station", "day", "label", "split"]].reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- summaries
def add_day_label_share(pairs: pd.DataFrame) -> pd.DataFrame:
    p = pairs.copy()
    p["day"] = p.utc.dt.date
    paired = p[p.paired]
    same = paired.groupby(["station", "day", "label"]).utc.transform("size")
    total = paired.groupby(["station", "day"]).utc.transform("size")
    p["day_label_share"] = same / total
    return p


def summary(pairs: pd.DataFrame, splits: pd.DataFrame) -> dict:
    d = pairs[pairs.paired & pairs.daytime]
    per_station = {}
    for st, g in d.groupby("station"):
        counts = g.label.value_counts().reindex(LABEL_ORDER, fill_value=0)
        cbh = g.cbh_m.dropna()
        bins = pd.cut(cbh, [0, 500, 1000, 2000, 4000, 6000, 8000, 12000], right=False)
        at_station = pairs.station == st
        per_station[st] = {"frames": int(at_station.sum()), "daytime_frames": int((at_station & pairs.daytime).sum()),
                           "pairs": int(len(g)), "inside_mask": int(g.zenith_inside_mask.sum()), "labels": counts.to_dict(),
                           "cbh_bins": bins.value_counts(sort=False).to_dict(), "days": int(g.utc.dt.date.nunique()),
                           "mixed_share": float(g.mixed.mean()) if len(g) else 0.0,
                           "confidence_mean": float(g.confidence.mean()) if len(g) else 0.0,
                           "gap_max_s": float(g.gap_s.max()) if len(g) else 0.0}
    by_split = splits.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=LABEL_ORDER, fill_value=0)
    days_by_split = splits.groupby("split").day.nunique().to_dict()
    month = (d.assign(month=d.utc.dt.month).groupby(["month", "label"]).size().unstack(fill_value=0)
             .reindex(columns=LABEL_ORDER, fill_value=0))
    return {"per_station": per_station, "by_split": by_split, "days_by_split": days_by_split, "by_month": month}


def report_markdown(s: dict, cadence_s: int, max_gap_s: float, decisions: list[str], figures: list[str]) -> str:
    lines = ["# Image-ceilometer pairs (P047)", "",
             f"Generated by `scripts/make_cbh_pairs.py`: frames every {cadence_s} s at the ceilometer sites, joined to "
             f"the nearest 30 s grid point of `data/ceilometer_targets.parquet` within {max_gap_s:.0f} s (P031 tolerance, "
             "P036 targets).", "",
             "## Per station (daytime, paired)", "",
             "| Station | Frames | Daytime | Pairs | Zenith in mask | Days | none | low | mid | high | mixed | mixed share | "
             "Confidence | Max gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st, r in s["per_station"].items():
        lab = r["labels"]
        lines.append(f"| {st} | {r['frames']:,} | {r['daytime_frames']:,} | {r['pairs']:,} | {r['inside_mask']:,} | "
                     f"{r['days']} | {lab['none']:,} | {lab['low']:,} | {lab['mid']:,} | {lab['high']:,} | {lab['mixed']:,} | "
                     f"{r['mixed_share']:.1%} | {r['confidence_mean']:.2f} | {r['gap_max_s']:.0f} s |")
    lines += ["", "## Cloud-base height bins (lowest base, metres; daytime pairs with a base)", "",
              "| Station | " + " | ".join(str(b) for b in next(iter(s["per_station"].values()))["cbh_bins"]) + " |",
              "|---|" + "---|" * len(next(iter(s["per_station"].values()))["cbh_bins"])]
    for st, r in s["per_station"].items():
        lines.append(f"| {st} | " + " | ".join(f"{int(v):,}" for v in r["cbh_bins"].values()) + " |")
    lines += ["", "## Splits (calendar days shared across stations; night frames excluded)", "",
              "| Split | Days | " + " | ".join(LABEL_ORDER) + " | Total |", "|---|---|" + "---|" * (len(LABEL_ORDER) + 1)]
    for split in ("train", "val", "test"):
        if split in s["by_split"].index:
            row = s["by_split"].loc[split]
            cells = " | ".join(f"{int(row[c]):,}" for c in LABEL_ORDER)
            lines.append(f"| {split} | {s['days_by_split'].get(split, 0)} | {cells} | {int(row.sum()):,} |")
    lines += ["", "## Label mix by month (daytime pairs, both stations)", "",
              "| Month | " + " | ".join(LABEL_ORDER) + " |", "|---|" + "---|" * len(LABEL_ORDER)]
    for month, row in s["by_month"].iterrows():
        lines.append(f"| {month} | " + " | ".join(f"{int(row[c]):,}" for c in LABEL_ORDER) + " |")
    lines += ["", "## Decisions", ""] + [f"- {d}" for d in decisions] + ["", "## Figures", ""] + [f"- `{f}`" for f in figures]
    return "\n".join(lines) + "\n"
