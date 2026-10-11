"""Frozen-feature probe baselines (P054: DINOv2 / DINOv3; P055: CLIP / SigLIP 2 linear probes).

    python scripts/baseline_probes.py [--backbones dinov3_vits16 ...] [--sizes 224 512] [--phase P054]

For every (backbone, size) cache present: in-domain linear probe and k-NN on CCSN (11 genera), MGCD (7), SWIMCAT (5)
and Montenegro (5), trained on the grouped train split and scored on the grouped val split (macro-F1, accuracy;
C of the linear probe chosen on val); LODO étage probes (multi-label low/mid/high) trained on the other sources'
train + val rows and scored on the held-out source's LODO val units (CCSN, MGCD, Montenegro, Almería folds; the
SWIM family has no étage labels). Test sets are never touched. Each (backbone, size, protocol) is one run in
runs/registry.csv; the table goes to docs/data/baseline_probes.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import probes as P  # noqa: E402
from stratia.baselines import tasks as T  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.backbones import BACKBONES  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/baseline_probes.md")


def run_in_domain(cache: FeatureCache, tables: T.Tables, kind: str) -> list[dict]:
    rows = []
    for source in T.RECOGNITION_SOURCES:
        tab = T.recognition_table(tables, source)
        tr, va = tab[tab.split == "train"], tab[tab.split == "val"]
        xt, xv = cache.features(tr.sample_id, kind), cache.features(va.sample_id, kind)
        lin = P.linear_probe(xt, tr.label.to_numpy(), xv, va.label.to_numpy())
        knn = P.knn_probe(xt, tr.label.to_numpy(), xv, va.label.to_numpy())
        rows.append({"protocol": "in_domain", "source": source, "n_train": len(tr), "n_val": len(va),
                     "classes": int(tr.label.nunique()),
                     "linear_macro_f1": lin.val_metric, "linear_acc": lin.extra["accuracy"], "linear_C": lin.params["C"],
                     "knn_macro_f1": knn.val_metric, "knn_acc": knn.extra["accuracy"]})
    return rows


def run_lodo(cache: FeatureCache, tables: T.Tables, kind: str) -> list[dict]:
    rows = []
    for fold in T.LODO_FOLDS_WITH_ETAGE:
        tab = T.lodo_table(tables, fold)
        tr, va = tab[tab.role == "train"], tab[tab.role == "val"]
        if not len(va):
            continue
        xt, xv = cache.features(tr.sample_id, kind), cache.features(va.sample_id, kind)
        yt, yv = T.etage_matrix(tr.etage_set), T.etage_matrix(va.etage_set)
        lin = P.linear_probe(xt, yt, xv, yv, multilabel=True)
        knn = P.knn_probe(xt, yt, xv, yv, multilabel=True)
        rows.append({"protocol": "lodo_etage", "source": fold, "n_train": len(tr), "n_val": len(va), "classes": 3,
                     "linear_macro_f1": lin.val_metric, "linear_acc": float("nan"), "linear_C": lin.params["C"],
                     "knn_macro_f1": knn.val_metric, "knn_acc": float("nan"),
                     "linear_unsupported_fp": T.unsupported_positive_rate(yv, lin.val_pred),
                     "knn_unsupported_fp": T.unsupported_positive_rate(yv, knn.val_pred)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", nargs="+",
                    default=["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov2_small", "dinov2_base"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[224, 512])
    ap.add_argument("--kind", default="cls", choices=["cls", "mean"])
    ap.add_argument("--phase", default="P054")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cache_root = Path(load_paths()["cache_root"])
    tables = T.load_tables()
    results = []
    for key in a.backbones:
        for size in a.sizes:
            spec = BACKBONES[key]
            if spec.fixed_size and size != spec.fixed_size:
                continue
            cache = FeatureCache(cache_root, key, size)
            if not cache.complete:
                print(f"{cache.key}: not built, skipped")
                continue
            cfg = OmegaConf.create({"backbone": key, "hf_id": spec.hf_id, "size": size, "kind": a.kind, "probe": "linear+knn",
                                    "c_grid": list(P.C_GRID), "k": 20, "cache_fingerprint": cache.meta()["fingerprint"]})
            with Run(cfg, name=f"probe_{key}_{size}", phase=a.phase, data_hash=cache.meta()["fingerprint"]) as run:
                rows = run_in_domain(cache, tables, a.kind) + run_lodo(cache, tables, a.kind)
                for r in rows:
                    r.update({"backbone": key, "size": size, "run_id": run.run_id})
                results += rows
                summary = {f"{r['protocol']}/{r['source']}/linear_f1": round(r["linear_macro_f1"], 4) for r in rows}
                summary.update({f"{r['protocol']}/{r['source']}/knn_f1": round(r["knn_macro_f1"], 4) for r in rows})
                lodo = [r["linear_macro_f1"] for r in rows if r["protocol"] == "lodo_etage"]
                summary["lodo_etage/mean_linear_f1"] = round(float(np.mean(lodo)), 4) if lodo else None
                run.finish(**{k: v for k, v in summary.items() if v is not None})
            print(f"{cache.key}: " + ", ".join(f"{r['source']} {r['linear_macro_f1']:.3f}" for r in rows), flush=True)
    df = pd.DataFrame(results)
    out = Path("data/baseline_probes.parquet")
    if out.exists():
        old = pd.read_parquet(out)
        df = pd.concat([old[~old.set_index(["backbone", "size", "protocol", "source"]).index.isin(
            df.set_index(["backbone", "size", "protocol", "source"]).index)], df], ignore_index=True)
    df.to_parquet(out, index=False)
    write_doc(df)
    print(f"done in {time.perf_counter() - t0:.0f} s: {len(results)} rows -> {DOC}")
    return 0


def write_doc(df: pd.DataFrame) -> None:
    lines = ["# Frozen-feature probe baselines (P054 / P055)", "",
             "Linear probe (standardised features, multinomial logistic regression, C chosen on val) and cosine k-NN (k = 20) "
             "on the "
             "pooled embedding of each frozen backbone. In-domain: grouped train -> grouped val, macro-F1 over native classes. "
             "LODO étage: train on the other sources' train + val rows, score on the held-out source's LODO val units, macro-F1 "
             "over the low / mid / high binary labels restricted to the columns with val support (MGCD's val has no "
             "mid-étage rows: scored on low and high; false positives in unsupported columns are in the parquet as "
             "`*_unsupported_fp`). Test sets untouched.", ""]
    for protocol, title in (("in_domain", "In-domain (val)"), ("lodo_etage", "Leave-one-dataset-out, étage (LODO val)")):
        part = df[df.protocol == protocol]
        if not len(part):
            continue
        piv_l = part.pivot_table(index=["backbone", "size"], columns="source", values="linear_macro_f1")
        piv_k = part.pivot_table(index=["backbone", "size"], columns="source", values="knn_macro_f1")
        cols = [c for c in (T.RECOGNITION_SOURCES if protocol == "in_domain" else T.LODO_FOLDS_WITH_ETAGE) if c in piv_l.columns]
        lines += [f"## {title}", "", "| Backbone | Size | " + " | ".join(f"{c} lin / kNN" for c in cols) + " | Mean lin |",
                  "|---|---|" + "---|" * (len(cols) + 1)]
        for (bb, size), row in piv_l.iterrows():
            cells = " | ".join(f"{row[c]:.3f} / {piv_k.loc[(bb, size), c]:.3f}" for c in cols)
            lines.append(f"| {bb} | {size} | {cells} | {row[cols].mean():.3f} |")
        lines.append("")
    lines += ["Run ids are in `data/baseline_probes.parquet` and `runs/registry.csv` (phase P054/P055)."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    json.dump(df.to_dict("records"), open("data/baseline_probes.json", "w", encoding="utf-8"), indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
