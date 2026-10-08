"""Label-noise estimation (P040): confident-learning flags for the class datasets.

    python scripts/label_noise.py [--folds 5]

Inputs: data/manifest.parquet, the P025 features (cache/features/dinov3_vits16_224_cls.npy), data/splits/in_domain.parquet
(units for the folds). Outputs: data/label_noise_flags.parquet, docs/data/label_noise_report.md,
docs/data/figures/label_noise_<dataset>.jpg.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.registry import load_paths  # noqa: E402
from stratia.labels import noise as ln  # noqa: E402

TASKS = {"ccsn": "source_label", "mgcd": "source_label", "swimcat": "source_label", "montenegro": "primary_class"}

DECISIONS = [
    "**No label is changed or removed.** Flags are stored with their suggested class and margin; the data card lists the "
    "per-dataset flag shares as a label-noise estimate next to the conflict rates of P026 and the rater agreement of P035.",
    "**Flagged images stay in training with their given label**; the training recipe may down-weight them (a `noise_flag` "
    "weight, P065) and the evaluation reports metrics with and without the flagged test images, so that a reviewer can "
    "see how much of an error rate is label noise.",
    "**Review of the contact sheets** (the most confident flags per dataset) is recorded in the phase document; where "
    "the sheet shows a systematic confusion (two classes the dataset itself separates poorly), it is noted as a "
    "candidate for merging at evaluation time, not as a correction of the data.",
    "**The flag rate is a lower bound at the linear-probe level**: the probe cannot see what a fine-tuned model would, "
    "and a confident wrong suggestion is as possible as a confident right one; that is why nothing is automated.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    t0 = time.perf_counter()
    paths = load_paths()
    cache_root = Path(paths["cache_root"])
    m = pd.read_parquet("data/manifest.parquet")
    emb = np.load(cache_root / "features" / "dinov3_vits16_224_cls.npy")
    units = pd.read_parquet("data/splits/in_domain.parquet").set_index("sample_id").unit.reindex(m.sample_id).to_numpy()
    files = m.image_file.tolist()
    parts, summaries, sheets = [], {}, {}
    for ds, col in TASKS.items():
        rows = np.flatnonzero(m.dataset.values == ds)
        y = m.source_label.values[rows]
        if col == "primary_class":
            y = pd.Series(y).str.split(",").str[0].values
        keep = pd.notna(y)
        rows, y = rows[keep], y[keep].astype(str)
        probs, classes = ln.out_of_fold_probs(emb[rows], y, units[rows], k=a.folds)
        flags = ln.confident_flags(probs, y, classes)
        flags.index = rows
        flags.insert(0, "sample_id", m.sample_id.values[rows])
        flags.insert(1, "dataset", ds)
        parts.append(flags)
        summaries[ds] = ln.summary(flags)
        sheet = ln.contact_sheet(flags, files, cache_root, f"docs/data/figures/label_noise_{ds}.jpg")
        sheets[ds] = f"figures/{sheet.name}"
        s = summaries[ds]
        print(f"{ds}: {s['flagged']:,} of {s['images']:,} flagged ({s['share']:.1%}); top pairs:", flush=True)
        print(s["top_pairs"].head(5).to_string(index=False), flush=True)
    pd.concat(parts).to_parquet("data/label_noise_flags.parquet", index=False)
    report = ln.report_markdown(summaries, DECISIONS, sheets)
    Path("docs/data/label_noise_report.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s: data/label_noise_flags.parquet, docs/data/label_noise_report.md",
          flush=True)


if __name__ == "__main__":
    main()
