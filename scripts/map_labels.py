"""Map every native label to the STRATIA ontology (P034).

    python scripts/map_labels.py

Inputs: data/manifest.parquet, configs/ontology.yaml, data/mask_cloud_fraction.parquet (P030),
data/label_conflicts.parquet (P026). Outputs: data/labels.parquet, docs/data/label_mapping.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.labels import mapping as mp  # noqa: E402
from stratia.labels.ontology import load_ontology  # noqa: E402

DECISION = [
    "**Merged classes become sets of alternatives, never one genus**: MGCD 'altocumulus' is {Ac, Cc}, 'cirrus' {Ci, Cs}, "
    "'stratocumulus' {Sc, St, As}, 'cumulonimbus' {Cb, Ns}; the loss will reward mass on any member (P065). Rejected: picking "
    "the first-named genus, which would plant a known error rate in the training signal.",
    "**'Mixed' and the SWIMCAT patch categories carry cloud presence only** (genus unknown), so they train the cloud / clear "
    "and oktas-type heads and not the genus head. Rejected: dropping them (they are 1,020 and 560 images of real sky).",
    "**Montenegro genera come from the majority C_L / C_M / C_H codes, étages from the altitude classes**; a code '/' (part of "
    "the sky not visible) leaves that part unknown rather than empty. The rater distributions themselves stay in the "
    "manifest for soft targets (P035).",
    "**Exact and copy conflicts (P026) are merged**: the members take the union of their genus sets as alternatives and are "
    "flagged `label_conflict`; the split generator keeps flagged pictures out of every test split (P038). Same-scene "
    "disagreements are not merged: the pictures differ.",
    "**The segmentation datasets contribute masks, cloud presence and (Almería) an étage set from the layer masks**; "
    "no genus. Eye2Sky and B0268 carry no image-level class until P036 and P041.",
    "The manifest's own `genus_set` / `etage_set` columns stay empty; labels live in `data/labels.parquet` and are "
    "joined on `sample_id`, so the manifest (and every table aligned to it) never changes when a mapping rule does.",
]


def main() -> None:
    t0 = time.perf_counter()
    ont = load_ontology()
    m = pd.read_parquet("data/manifest.parquet")
    frac = pd.read_parquet("data/mask_cloud_fraction.parquet") if Path("data/mask_cloud_fraction.parquet").exists() else None
    conflicts = pd.read_parquet("data/label_conflicts.parquet") if Path("data/label_conflicts.parquet").exists() else None
    labels = mp.apply_mapping(m, ont, frac, conflicts)
    labels.to_parquet("data/labels.parquet", index=False)
    summ = mp.summary(labels)
    reasons = labels.excluded_reason.dropna().value_counts()
    report = mp.report_markdown(mp.mapping_table(ont), summ, reasons, DECISION)
    Path("docs/data/label_mapping.md").write_text(report, encoding="utf-8", newline="\n")
    print(summ.to_string(index=False), flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s: data/labels.parquet ({len(labels):,} rows), docs/data/label_mapping.md",
          flush=True)


if __name__ == "__main__":
    main()
