"""Image-ceilometer pair table and its day splits (P047).

    python scripts/make_cbh_pairs.py [--cadence 120] [--max-gap 15]

Reads the frames on disk at the ceilometer sites (configs/splits.yaml `cbh_pairs`), the P036 target table, the
calibrations, masks and Sun-fitted poses (P044), and writes:
  data/cbh_pairs.parquet          every frame on the cadence with its target, geometry and day split (ignored)
  data/splits/cbh_pairs.parquet   the protocol file (sample_id, station, day, label, split) the lock covers
  docs/data/cbh_pairs.md          counts per height bin, per split and per month; decisions
  docs/data/figures/cbh_pairs_*.png
Then re-lock the test sets: python scripts/lock_test_sets.py --force (explain in the commit).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.eye2sky import load_mask, mask_path, select_calibration, station_calibrations  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.data.splits import load_split_config  # noqa: E402
from stratia.geometry.pose import load_pose_records, pose_for  # noqa: E402
from stratia.labels import cbh_pairs as cp  # noqa: E402

FIG_DIR = Path("docs/data/figures")

DECISIONS = [
    "**One frame every 2 min** (P027: adjacent 30 s frames are 82-92 % the same scene); the full-rate frames stay on disk "
    "for P075 to densify if a curve asks for it.",
    "**Nearest 30 s grid point within 15 s** of the P036 table carries the label; the +-30 s window behind each grid point is "
    "the P031 tolerance. Frames on the grid (gap 0 s) are the rule, not the exception.",
    "**The ceilometer looks at the zenith; the model must too.** Every pair records the zenith pixel from the Sun-fitted "
    "pose and the radius of a 10-degree cone around it; pairs whose zenith falls outside the camera mask are excluded "
    "from the splits. Calibrations without a fitted pose (status ok) are refused.",
    "**Night frames are not in the splits** (80 ms exposure regime; daytime model), but stay in `data/cbh_pairs.parquet` "
    "with daytime = False.",
    "**Days are the units, shared across OLDLR and WESTE** (P027/P037); days the in-domain Eye2Sky split already assigned "
    "keep that split; the others are assigned 70/10/20 with the day's dominant label as the stratum.",
    "**Overcast-day guard:** `day_label_share` (share of the day's pairs with the same label) travels with every pair; "
    "P075 down-weights by it or caps per day. The per-month table shows where the low-cloud mass sits.",
    "**Synthetic views (P049) adopt this rule:** `inherit_label` is replaced by a join to this table's pairs by source frame.",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cadence", type=int, default=None)
    ap.add_argument("--max-gap", type=float, default=None)
    a = ap.parse_args()
    t0 = time.perf_counter()
    cfg = load_split_config()
    cb = cfg["cbh_pairs"]
    cadence = a.cadence or int(cb["cadence_s"])
    max_gap = a.max_gap or float(cb["max_gap_s"])
    root = Path(load_paths()["data_root"]) / "Eye2Sky"
    meta = root / "asi_meta"
    targets = pd.read_parquet("data/ceilometer_targets.parquet")
    records = load_pose_records()

    parts = []
    for station, ceilometer in cb["stations"].items():
        frames = cp.frames_on_disk(root, station, cadence)
        if not len(frames):
            print(f"{station}: no frames on disk")
            continue
        joined = cp.join_targets(frames, targets[targets.ceilometer == ceilometer], max_gap)
        cals = station_calibrations(meta, station)
        geo = {}
        joined["calib_id"] = [select_calibration(cals, t).file if select_calibration(cals, t) else None for t in joined.utc]
        for cid in joined.calib_id.dropna().unique():
            cal = next(c for c in cals if c.file == cid)
            pose = pose_for(cid, None, records)                          # fitted poses only
            if pose is None:
                print(f"  {station} {cid}: no fitted pose (status ok); its frames are excluded")
                geo[cid] = None
                continue
            geo[cid] = cp.zenith_geometry(cal.model, pose, load_mask(mask_path(meta, cal)), float(cb["roi_deg"]))
        have = {cid: g for cid, g in geo.items() if g is not None}
        joined["pose_ok"] = joined.calib_id.isin(have)
        joined["zenith_u"] = joined.calib_id.map({cid: g.u for cid, g in have.items()})
        joined["zenith_v"] = joined.calib_id.map({cid: g.v for cid, g in have.items()})
        joined["roi_radius_px"] = joined.calib_id.map({cid: g.roi_radius_px for cid, g in have.items()})
        joined["zenith_inside_mask"] = joined.calib_id.map({cid: bool(g.inside_mask) for cid, g in have.items()})
        joined["zenith_inside_mask"] = joined.zenith_inside_mask.fillna(False).astype(bool)
        joined["paired"] = joined.paired & joined.pose_ok
        joined["daytime"] = joined.daytime.fillna(False).astype(bool)
        parts.append(joined)
        print(f"{station}: {len(frames):,} frames on the {cadence} s cadence, {int(joined.paired.sum()):,} paired, "
              f"{int((joined.paired & joined.daytime).sum()):,} daytime pairs", flush=True)
    pairs = pd.concat(parts, ignore_index=True)
    pairs = cp.add_day_label_share(pairs)

    # day splits: inherit the in-domain Eye2Sky assignment, assign the rest
    inherited = {}
    in_domain = Path("data/splits/in_domain.parquet")
    if cb.get("inherit_from") == "in_domain" and in_domain.exists():
        s = pd.read_parquet(in_domain)
        m = pd.read_parquet("data/manifest.parquet", columns=["sample_id", "dataset", "utc"])
        e = s[s.source == "eye2sky"].merge(m, on="sample_id")
        inherited = e.assign(day=e.utc.dt.date).groupby("day").split.agg(lambda x: x.value_counts().idxmax()).to_dict()
    strata = cp.day_strata(pairs).to_dict()
    days = sorted(pairs[pairs.paired & pairs.daytime].utc.dt.date.unique())
    day_split = cp.assign_days(days, inherited, cfg["fractions"], int(cfg["seed"]), strata)
    pairs["split"] = pairs.utc.dt.date.map(day_split)
    pairs.loc[~(pairs.paired & pairs.daytime & pairs.zenith_inside_mask), "split"] = None
    splits = cp.split_table(pairs, day_split)

    Path("data/splits").mkdir(parents=True, exist_ok=True)
    pairs.to_parquet("data/cbh_pairs.parquet", index=False)
    splits.to_parquet("data/splits/cbh_pairs.parquet", index=False)
    s = cp.summary(pairs, splits)
    figs = figures(pairs, splits)
    report = cp.report_markdown(s, cadence, max_gap, DECISIONS, figs)
    Path("docs/data/cbh_pairs.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"inherited {len(inherited)} day assignments; {len(days)} paired days; splits: {s['days_by_split']}")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/cbh_pairs.parquet ({len(pairs):,} rows), "
          f"data/splits/cbh_pairs.parquet ({len(splits):,} rows), docs/data/cbh_pairs.md")
    return 0


def figures(pairs: pd.DataFrame, splits: pd.DataFrame) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    d = pairs[pairs.paired & pairs.daytime]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for st, g in d.groupby("station"):
        axes[0].hist(g.cbh_m.dropna(), bins=np.logspace(np.log10(100), np.log10(12000), 40), alpha=0.6, label=st)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lowest cloud base (m)")
    axes[0].set_ylabel("daytime pairs")
    axes[0].legend()
    axes[0].set_title("cloud-base height of the pairs")
    counts = d.groupby(["station", "label"]).size().unstack(fill_value=0).reindex(columns=cp.LABEL_ORDER, fill_value=0)
    counts.plot.bar(ax=axes[1])
    axes[1].set_title("labels per station (daytime pairs)")
    axes[1].set_ylabel("pairs")
    fig.tight_layout()
    f1 = FIG_DIR / "cbh_pairs_histogram.png"
    fig.savefig(f1, dpi=80)
    plt.close(fig)
    out.append(f1.as_posix())
    # timeline: pairs per day coloured by split
    fig, ax = plt.subplots(figsize=(13, 3.5))
    colours = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    for split, g in splits.groupby("split"):
        per_day = g.groupby("day").size()
        ax.bar(pd.to_datetime(per_day.index), per_day.values, width=1.0, color=colours.get(split, "grey"), label=split)
    ax.set_ylabel("pairs per day (both stations)")
    ax.set_title("day splits of the cloud-base-height pairs")
    ax.legend()
    fig.tight_layout()
    f2 = FIG_DIR / "cbh_pairs_days.png"
    fig.savefig(f2, dpi=80)
    plt.close(fig)
    out.append(f2.as_posix())
    return out


if __name__ == "__main__":
    sys.exit(main())
