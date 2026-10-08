"""Ceilometer target table (P036).

    python scripts/ceilometer_targets.py [--half-window 30]

Reads data/ceilometer_records.parquet (P030/P031 cache) and the Eye2Sky station list (coordinates of CDLRA at OLDLR
and CDLRB at WESTE for the daytime flag); writes data/ceilometer_targets.parquet (30 s grid, both sites) and
docs/data/ceilometer_targets.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.ceilometer import load_all  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.labels import ceilometer_targets as ct  # noqa: E402

DECISION = [
    "**Label = the ceilometer's view in the frame's ±30 s window:** 'none' when no clean record sees cloud, the étage of the "
    "median lowest base when at least half of them do, 'mixed' otherwise; the share behind the label is the confidence and "
    "goes into the loss as a weight (P030: soft labels are distributions). Windows without a clean record (rain, window "
    "particles, optics, error bits, or no data) give no label and become the camera's 'obscured / unlabelled' cases.",
    "**Multi-layer skies are kept:** the second layer's median height and the mean layer count travel with the label, so a "
    "model's lowest-base error can be analysed by layer count (P087) and nothing is discarded for being ambiguous.",
    "**Two étage thresholds are stored** (weak 2 / 6 km from the ontology; WMO 2 / 7 km): the ablation planned in P036 is a "
    "column switch, not a rebuild.",
    "Pairing with frames (P047) joins a frame's time to the nearest grid point (15 s away at most) or recomputes the window "
    "at the exact frame time with `targets_at`; both use the same rule.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half-window", type=float, default=30.0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(load_paths()["data_root"])
    records = load_all(root / "Eye2Sky" / "ceilometer" / "ceil" / "data", "data/ceilometer_records.parquet")
    stations = pd.read_excel(root / "Eye2Sky" / "Eye2Sky_Station_List.xlsx").set_index("Station ID")
    parts, summaries = [], {}
    for code, rec in records.groupby("ceilometer"):
        grid = ct.time_grid(rec)
        t = ct.targets_at(rec, grid, a.half_window)
        lat, lon = float(stations.loc[code, "Latitude"]), float(stations.loc[code, "Longitude"])
        t.insert(0, "ceilometer", code)
        t["daytime"] = ct.daytime(grid, lat, lon)
        parts.append(t)
        summaries[code] = ct.summary(t)
        s = summaries[code]["daytime"]
        print(f"{code}: {len(t):,} grid points, {s['valid_share']:.1%} labelled by day, labels {s['labels']}, "
              f"mixed {s['mixed_share']:.1%}, spread median {s['spread_median_m']:.0f} m", flush=True)
    targets = pd.concat(parts, ignore_index=True)
    targets.to_parquet("data/ceilometer_targets.parquet", index=False)
    Path("docs/data/ceilometer_targets.md").write_text(ct.report_markdown(summaries, a.half_window, DECISION),
                                                       encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/ceilometer_targets.parquet ({len(targets):,} rows), "
          "docs/data/ceilometer_targets.md", flush=True)


if __name__ == "__main__":
    main()
