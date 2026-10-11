"""Baseline leaderboard (P059): one table per head from the Stage F result files, plus the protocol-gap table.

    python scripts/baseline_leaderboard.py

Reads data/baseline_probes.parquet (P054/P055), data/baseline_cnn.parquet (P052/P053), data/baseline_zeroshot.parquet
(P055), data/baseline_segmentation.parquet (P057), data/baseline_cbh.parquet (P058) and runs/registry.csv, and writes
docs/data/baseline_leaderboard.md: recognition (in-domain val, LODO val), the protocol gap (random vs grouped vs
LODO), segmentation, cloud-base height, and the "best baseline" lines that criteria C1 and C3 compare against.
Everything is validation-only; locked test sets are untouched until P090.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import tasks as T  # noqa: E402

DOC = Path("docs/data/baseline_leaderboard.md")
STAGE_PHASES = ["P052", "P053", "P054", "P055", "P056", "P057", "P058"]


def read(name: str) -> pd.DataFrame:
    p = Path("data") / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def recognition_rows(probes: pd.DataFrame, cnn: pd.DataFrame, zs: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, protocol, source) with val macro-F1."""
    rows = []
    for _, r in probes.iterrows():
        tag = f"{r.backbone}@{r['size']}"
        rows.append({"method": f"linear probe {tag}", "family": "frozen probe", "protocol": r.protocol, "source": r.source,
                     "macro_f1": r.linear_macro_f1})
        rows.append({"method": f"kNN {tag}", "family": "frozen kNN", "protocol": r.protocol, "source": r.source,
                     "macro_f1": r.knn_macro_f1})
    if len(cnn):
        grouped = cnn[cnn.protocol == "grouped"]
        g = grouped.groupby(["arch", "source"]).val_macro_f1.agg(["mean", "std", "count"]).reset_index()
        for _, r in g.iterrows():
            rows.append({"method": f"fine-tuned {r.arch}", "family": "fine-tuned CNN", "protocol": "in_domain",
                         "source": r.source, "macro_f1": r["mean"], "std": r["std"], "seeds": int(r["count"])})
    for _, r in zs.iterrows():
        rows.append({"method": f"zero-shot {r.backbone} ({r.prompt_set})", "family": "zero-shot", "protocol": "in_domain",
                     "source": r.source, "macro_f1": r.macro_f1})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame, protocol: str, sources) -> list[str]:
    part = df[df.protocol == protocol] if len(df) else df
    if not len(part):
        return ["(no results yet)", ""]
    piv = part.pivot_table(index="method", columns="source", values="macro_f1", aggfunc="mean")
    cols = [s for s in sources if s in piv.columns]
    piv["mean"] = piv[cols].mean(axis=1, skipna=False)          # a method missing a source has no mean
    piv = piv.sort_values("mean", ascending=False, na_position="last")
    lines = ["| Method | " + " | ".join(cols) + " | Mean |", "|---|" + "---|" * (len(cols) + 1)]
    for method, r in piv.iterrows():
        cells = " | ".join(f"{r[c]:.3f}" if pd.notna(r[c]) else "-" for c in cols)
        mean = f"{r['mean']:.3f}" if pd.notna(r["mean"]) else "-"
        lines.append(f"| {method} | {cells} | {mean} |")
    return lines + [""]


def protocol_gap(cnn: pd.DataFrame, probes: pd.DataFrame) -> list[str]:
    lines = ["| Model | Random val | Random val, leaked part | Random val, clean part | Grouped val | LODO étage val | "
             "Gap random→grouped |", "|---|---|---|---|---|---|---|"]
    if len(cnn):
        leg = cnn[cnn.protocol == "legacy_random"].groupby(["arch", "source"]).agg(
            val=("val_macro_f1", "mean"), leaked=("val_leaked_macro_f1", "mean"), clean=("val_clean_macro_f1", "mean"),
            n_leaked=("n_val_leaked", "mean"), n_val=("n_val", "mean"))
        grp = cnn[cnn.protocol == "grouped"].groupby(["arch", "source"]).val_macro_f1.mean()
        for (arch, source), r in leg.iterrows():
            g = grp.get((arch, source), float("nan"))
            lines.append(f"| {arch} on {source} | {r.val:.3f} | {r.leaked:.3f} (n = {r.n_leaked:.0f} of {r.n_val:.0f}) | "
                         f"{r.clean:.3f} | {g:.3f} | - | {r.val - g:+.3f} |")
    if len(probes):
        best = probes.sort_values("linear_macro_f1", ascending=False)
        for source in T.LODO_FOLDS_WITH_ETAGE:
            ind = best[(best.protocol == "in_domain") & (best.source == source)]
            lodo = best[(best.protocol == "lodo_etage") & (best.source == source)]
            if len(ind) and len(lodo):
                i, lo = ind.iloc[0], lodo.iloc[0]
                lines.append(f"| best probe on {source} ({i.backbone}@{i['size']}) | - | - | - | "
                             f"{i.linear_macro_f1:.3f} (native classes) | "
                             f"{lo.linear_macro_f1:.3f} ({lo.backbone}@{lo['size']}) | - |")
    return lines + [""]


def main() -> int:
    t0 = time.perf_counter()
    probes, cnn, zs = read("baseline_probes"), read("baseline_cnn"), read("baseline_zeroshot")
    seg, cbh = read("baseline_segmentation"), read("baseline_cbh")
    rec = recognition_rows(probes, cnn, zs)
    lines = ["# Baseline leaderboard (P059)", "",
             "Generated by `scripts/baseline_leaderboard.py` from the Stage F result files; validation numbers only (locked test "
             "sets are evaluated once in P090). Macro-F1 unless stated.", "",
             "## Recognition, in-domain grouped val (native classes)", ""]
    lines += table(rec, "in_domain", T.RECOGNITION_SOURCES)
    lines += ["## Recognition, leave-one-dataset-out (étage, LODO val)", ""]
    lines += table(rec, "lodo_etage", T.LODO_FOLDS_WITH_ETAGE)
    if len(rec) and len(rec[rec.protocol == "lodo_etage"]):
        piv = rec[rec.protocol == "lodo_etage"].pivot_table(index="method", columns="source", values="macro_f1").mean(axis=1)
        lines += [f"**Best LODO baseline (criterion C1 reference): {piv.idxmax()}, mean étage macro-F1 {piv.max():.3f} over "
                  f"{', '.join(T.LODO_FOLDS_WITH_ETAGE)}.**", ""]
    lines += ["## Protocol gap", ""] + protocol_gap(cnn, probes)
    if len(seg):
        lines += ["## Segmentation (val mIoU; clear-sky cloud fraction = collapse check)", "",
                  "| Model | Dataset | mIoU | IoU cloud | Clear-sky cloud fraction |", "|---|---|---|---|---|"]
        for _, r in seg.sort_values(["model", "dataset"]).iterrows():
            cloud = f"{r.iou_cloud:.3f}" if "iou_cloud" in r and pd.notna(r.iou_cloud) else "-"
            lines.append(f"| {r.model} | {r.dataset} | {r.miou:.3f} | {cloud} | {r.clear_sky_cloud_fraction:.3f} |")
        lines.append("")
    if len(cbh):
        lines += ["## Cloud-base height (val, cloudy pairs)", "",
                  "| Baseline | MAE (m) | Bias (m) | Skill vs climatology | ≤500 m |", "|---|---|---|---|---|"]
        for _, r in cbh.iterrows():
            lines.append(f"| {r.baseline} | {r.mae_m:.0f} | {r.bias_m:+.0f} | {r.skill:+.3f} | {r.within_500m:.2f} |")
        ref = cbh[cbh.baseline.str.startswith("cnn")]
        if len(ref):
            lines += ["", f"**Ceilometer-supervised CNN (criterion C3 reference): MAE {ref.iloc[0].mae_m:.0f} m, skill "
                      f"{ref.iloc[0].skill:+.3f}.**"]
        lines.append("")
    reg = Path("runs/registry.csv")
    if reg.exists():
        r = pd.read_csv(reg)
        stage = r[r.phase.isin(STAGE_PHASES)]
        counts = ", ".join(f"{p} {n}" for p, n in stage.phase.value_counts().sort_index().items())
        lines += ["## Runs", "", f"{len(stage)} Stage F runs in `runs/registry.csv` ({counts})."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
