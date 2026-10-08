"""Ceilometer QC and pairing tolerance (P031).

    python scripts/ceilometer_qc.py [--max-drop 0.02]

Reads data/ceilometer_records.parquet (built by P030 from the CHM15k files, rebuilt here if missing) and writes
data/ceilometer_qc.parquet (QC, offset and window tables), docs/data/ceilometer_qc.md and
docs/data/figures/ceilometer_pairing.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import ceilometer_qc as cq  # noqa: E402
from stratia.data.ceilometer import load_all  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-drop", type=float, default=0.02, help="allowed loss of within-window unanimity against ±30 s")
    a = ap.parse_args()
    t0 = time.perf_counter()
    base = Path(load_paths()["data_root"]) / "Eye2Sky" / "ceilometer" / "ceil" / "data"
    cl = load_all(base, "data/ceilometer_records.parquet")
    qc = cq.qc_summary(cl)
    print(qc.T.to_string(), flush=True)
    offsets, windows = {}, {}
    for code, part in cl.groupby("ceilometer"):
        offsets[code] = cq.agreement_vs_offset(part)
        windows[code] = cq.window_consistency(part)
        print(f"{code} offsets:\n{offsets[code].to_string(index=False)}", flush=True)
        print(f"{code} windows:\n{windows[code].to_string(index=False)}", flush=True)
    both = pd.concat(windows.values()).groupby("half_window_s", as_index=False).mean(numeric_only=True)
    chosen = cq.choose_tolerance(both, a.max_drop)
    cross = cq.cross_site(cl)
    print(f"chosen {chosen}\ncross-site {cross}", flush=True)
    om = pd.concat(offsets.values()).groupby("offset_s").mean(numeric_only=True)

    def agree(offset: float) -> str:
        return f"{om.loc[offset, 'presence_agreement']:.1%} / {om.loc[offset, 'etage_agreement']:.1%}"

    decision = [
        f"Chosen: **±{chosen['half_window_s']:.0f} s**. Two records 30 s apart agree on cloud presence / on the étage of "
        f"the lowest base in {agree(30)} of cases (the instrument's own 15 s consistency is {agree(15)}); at 2 min the "
        f"disagreement roughly doubles ({agree(120)}) and at 5 min triples ({agree(300)}). Inside ±30 s a window holds "
        f"{chosen['records_per_window']:.1f} clean records on average, all of which agree in {chosen['presence_unanimous']:.1%} "
        f"(presence) and {chosen['etage_unanimous']:.1%} (étage) of windows. The label of a frame is the median lowest base "
        "of the clean records in its window, the share of cloudy records is kept as the label's confidence, and a window "
        "without a clean record gives no label (P036).",
        "Rejected: ±2 min and ±5 min, which would label 1 and 1.5 points more of the minutes (coverage 92-93 % against "
        "93-96 %) at twice and three times the time-mismatch noise; and ±15 s, where one dropout decides the label.",
        "The two sites 15 km apart agree on presence and étage far less often than two records a few minutes apart at "
        "one site: a cloud base is local, which is why CDLRB is a genuine held-out site (criterion C3) and why a model "
        "cannot be scored against a distant ceilometer.",
    ]
    tables = pd.concat([qc.assign(table="qc")]
                       + [t.assign(table="offset", ceilometer=c) for c, t in offsets.items()]
                       + [t.assign(table="window", ceilometer=c) for c, t in windows.items()], ignore_index=True)
    tables.to_parquet("data/ceilometer_qc.parquet", index=False)
    fig = cq.figure(offsets, windows, chosen["half_window_s"], "docs/data/figures/ceilometer_pairing.png")
    report = cq.report_markdown(qc, offsets, windows, chosen, cross, decision, "figures/ceilometer_pairing.png")
    Path("docs/data/ceilometer_qc.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/ceilometer_qc.parquet, docs/data/ceilometer_qc.md, {fig}", flush=True)


if __name__ == "__main__":
    main()
