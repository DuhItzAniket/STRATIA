"""B0268 labels: derived targets and rater agreement (P041).

    python scripts/b0268_labels.py [--labels-dir DATA/B0268/labels] [--raters A B] [--extra C D]

Reads the rater sheets (CSV copies `B0268_labels_<R>.csv` next to `B0268_frames.csv`), derives étage / genus /
oktas / cloud labels per annotation, writes
  data/b0268_targets.parquet        per-frame soft targets from the protocol pair (--raters)
  data/b0268_targets_all.parquet    the same from every rater (--raters + --extra), a sensitivity set
  data/b0268_ceiling.parquet        the agreement table of the pair (and of all raters, column `raters`)
  docs/data/b0268_labels.md         the report
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.registry import load_paths  # noqa: E402
from stratia.labels import b0268_labels as bl  # noqa: E402


def main() -> int:
    t0 = time.perf_counter()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", default=None, help="default: <data_root>/B0268/labels")
    ap.add_argument("--raters", nargs="+", default=["A", "B"], help="the P041 pair")
    ap.add_argument("--extra", nargs="*", default=["C", "D"], help="further raters reported as a sensitivity set")
    a = ap.parse_args()
    labels_dir = Path(a.labels_dir) if a.labels_dir else Path(load_paths()["data_root"]) / "B0268" / "labels"
    raters = tuple(a.raters)
    extra = tuple(r for r in a.extra if (labels_dir / f"B0268_labels_{r}.csv").exists())
    frames = bl.load_frames(labels_dir)

    long = bl.load_raters(labels_dir, raters)
    derived = bl.derive(long)
    targets = bl.soft_targets(derived, frames)
    targets.to_parquet("data/b0268_targets.parquet", index=False)
    derived.assign(day=long.day.values, image_file="B0268/" + long.file).to_parquet("data/b0268_derived.parquet", index=False)
    table = bl.agreement_table(long, derived).assign(raters="".join(raters))
    per = bl.per_rater(derived)

    table_all = None
    if extra:
        long_all = bl.load_raters(labels_dir, raters + extra)
        derived_all = bl.derive(long_all)
        bl.soft_targets(derived_all, frames).to_parquet("data/b0268_targets_all.parquet", index=False)
        derived_all.assign(day=long_all.day.values, image_file="B0268/" + long_all.file).to_parquet(
            "data/b0268_derived_all.parquet", index=False)
        table_all = bl.agreement_table(long_all, derived_all).assign(raters="".join(raters + extra))
    pd.concat([table] + ([table_all] if table_all is not None else []), ignore_index=True).to_parquet(
        "data/b0268_ceiling.parquet", index=False)

    print(table.to_string(index=False), flush=True)
    print(per.to_string(index=False), flush=True)
    c = table.set_index("quantity")
    days = frames.day.value_counts().to_dict()
    e_loo, e_alpha = c.loc["étage set", "leave_one_out"], c.loc["étage set", "alpha"]
    n_loo, n_alpha = c.loc["oktas 0-8", "leave_one_out"], c.loc["oktas 0-8", "alpha"]
    g_alpha = c.loc["genus set", "alpha"]
    usable = float(derived.usable.mean())
    n_obscured = int(derived.obscured.fillna(False).astype(bool).sum())
    quality = (f"**Every annotation is usable and none is obscured** ({len(derived)} annotations): the four days have no rain "
               "drops on the lens; the 2026-10-21 session was paused during rain instead."
               if usable == 1.0 and n_obscured == 0 else
               f"**Usable share {usable:.1%}, {n_obscured} obscured annotation(s)**: unusable rows count for nothing but "
               "the usable share; obscured rows carry no cover estimate.")
    decision = [
        f"**B0268 ceiling for the C2-style comparison (P083):** the two annotators agree on the exact étage set in "
        f"{e_loo:.1%} of frames (alpha {e_alpha:.2f}) and on total cover within ±1 okta in {n_loo:.1%} (interval alpha "
        f"{n_alpha:.2f}). A model's agreement with the pair's majority is compared with these numbers.",
        f"**Genus is the weak signal again** (exact genus set alpha {g_alpha:.2f}; high-cloud genera are where the raters "
        "differ: Ci against Ci+Cs against none on hazy days), so per-genus presence is a soft target (share of raters), "
        "never a hard label; the model is scored on the étage set, total cover and genus-in-the-raters'-union.",
        quality,
        "**Targets per frame:** `data/b0268_targets.parquet` (étage and genus shares, oktas distribution, clear / contrail "
        "/ glare / obstruction shares) from the protocol pair; `data/b0268_targets_all.parquet` from every rater for "
        "sensitivity; `data/b0268_derived.parquet` keeps every rater's own labels, so a model is scored against each rater "
        "and averaged, the way the ceiling is measured. With two raters a 'majority' at threshold 0.5 is the union of the "
        "pair; the hard label table uses that union as a convenience view only. Training uses the shares as "
        "distributions, as for Montenegro (P030, P035).",
        "**The annotators are not the two human labellers the plan names.** Both sheets of the pair were produced by "
        "language-model annotators (sheet A by Claude Fable 5, sheet B by Claude Sonnet 5.5; the further sheets C and D "
        "by Claude Fable 5.1 and Claude Opus 5.5), each from the same contact sheets, independently and without seeing "
        "another sheet. The owner's and a friend's labels were lost in a disk failure (2026-10-22). The single-annotator "
        "guard of the plan is partly met (independent raters), the human guard is not; the report says so wherever the "
        "B0268 numbers appear, and human labels on a subset remain owner items.",
    ]
    report = bl.report_markdown(table, per, table_all, raters, extra, len(frames), days, decision)
    Path("docs/data/b0268_labels.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/b0268_targets.parquet"
          f"{', data/b0268_targets_all.parquet' if extra else ''}, data/b0268_ceiling.parquet, docs/data/b0268_labels.md",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
