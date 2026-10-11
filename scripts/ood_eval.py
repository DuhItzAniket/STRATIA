"""OOD / quality score evaluation on cached features (P068, deferrable phase).

    python scripts/ood_eval.py [--backbone dinov3_vits16 --size 512]

Leave-one-source-out: for each held-out source among CCSN, MGCD, Montenegro, a genus probe (logistic regression,
as P054) is fitted on the other sources' train rows; `OodScore` (energy on the probe's logits + Mahalanobis on the
pooled feature, z-scored on the training rows) is fitted on the same rows; AUROC is reported for the held-out
source's val rows (out) against the training sources' val rows (in). Two more "out" sets that are not sources of
genus labels: Eye2Sky all-sky frames (the CBH frames in the cache) and the SWIM patches. Night and glare flags are
measured on the Eye2Sky frames (fraction flagged). Results: data/ood_eval.parquet, docs/data/ood_eval.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import tasks as T  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.features.cache import FeatureCache  # noqa: E402
from stratia.model.ood import OodScore, auroc  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/ood_eval.md")
OUT = Path("data/ood_eval.parquet")
SOURCES = ("ccsn", "mgcd", "montenegro")


def probe_logits(x_train: np.ndarray, y_train: np.ndarray, x: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    clf = LogisticRegression(C=0.1, max_iter=2000).fit((x_train - mu) / sd, y_train)
    return clf.decision_function((x - mu) / sd)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="dinov3_vits16")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--phase", default="P068")
    a = ap.parse_args()
    t0 = time.perf_counter()
    cache = FeatureCache(Path(load_paths()["cache_root"]), a.backbone, a.size)
    tables = T.load_tables()
    recog = {s: T.recognition_table(tables, s) for s in SOURCES}
    index = cache.index()
    eye = index[index.sample_id.str.startswith("eye2sky:")].sample_id.to_numpy()
    swim = index[index.sample_id.str.startswith(("swimseg:", "swinyseg:"))].sample_id.to_numpy()
    rng = np.random.default_rng(0)
    eye = rng.choice(eye, min(2000, len(eye)), replace=False)
    swim = rng.choice(swim, min(2000, len(swim)), replace=False)
    rows = []
    with Run(OmegaConf.create({"backbone": a.backbone, "size": a.size, "shrinkage": 0.1}), name="ood_eval", phase=a.phase) as run:
        for held in SOURCES:
            train_parts = [recog[s][recog[s].split == "train"] for s in SOURCES if s != held]
            val_in = pd.concat([recog[s][recog[s].split == "val"] for s in SOURCES if s != held])
            val_out = recog[held][recog[held].split == "val"]
            tr = pd.concat(train_parts)
            x_tr = cache.features(tr.sample_id, "cls")
            x_in, x_out = cache.features(val_in.sample_id, "cls"), cache.features(val_out.sample_id, "cls")
            logits_tr = probe_logits(x_tr, tr.label.to_numpy(), x_tr)
            logits_in = probe_logits(x_tr, tr.label.to_numpy(), x_in)
            logits_out = probe_logits(x_tr, tr.label.to_numpy(), x_out)
            ood = OodScore(x_tr.shape[1])
            ood.fit(torch.from_numpy(x_tr), torch.from_numpy(logits_tr))
            s_in = ood(torch.from_numpy(x_in), torch.from_numpy(logits_in))[:, 0].numpy()
            s_out = ood(torch.from_numpy(x_out), torch.from_numpy(logits_out))[:, 0].numpy()
            e_in = -torch.logsumexp(torch.from_numpy(logits_in), 1).numpy()
            e_out = -torch.logsumexp(torch.from_numpy(logits_out), 1).numpy()
            m_in = ood.mahalanobis(torch.from_numpy(x_in)).numpy()
            m_out = ood.mahalanobis(torch.from_numpy(x_out)).numpy()
            row = {"held_out": held, "n_in": len(s_in), "n_out": len(s_out), "auroc_combined": auroc(s_in, s_out),
                   "auroc_energy": auroc(e_in, e_out), "auroc_mahalanobis": auroc(m_in, m_out)}
            for name, ids in (("eye2sky", eye), ("swim_patches", swim)):
                x_o = cache.features(ids, "cls")
                s_o = ood(torch.from_numpy(x_o), torch.from_numpy(probe_logits(x_tr, tr.label.to_numpy(), x_o)))[:, 0].numpy()
                row[f"auroc_vs_{name}"] = auroc(s_in, s_o)
            rows.append(row)
            print(f"held-out {held}: AUROC combined {row['auroc_combined']:.3f} (energy {row['auroc_energy']:.3f}, "
                  f"Mahalanobis {row['auroc_mahalanobis']:.3f}); vs Eye2Sky {row['auroc_vs_eye2sky']:.3f}, "
                  f"vs SWIM {row['auroc_vs_swim_patches']:.3f}", flush=True)
        df = pd.DataFrame(rows)
        run.finish(mean_auroc_combined=float(df.auroc_combined.mean()), mean_auroc_eye2sky=float(df.auroc_vs_eye2sky.mean()))
    df.to_parquet(OUT, index=False)
    lines = ["# OOD score evaluation (P068)", "",
             f"`OodScore` (energy on a genus probe's logits + Mahalanobis on the pooled {a.backbone} @ {a.size} feature, "
             "z-scored on the training rows, averaged) in a leave-one-source-out design: fitted on two sources' train rows, "
             "AUROC of the held-out source's val rows (out) against the training sources' val rows (in). The last two columns "
             "score the same in-domain rows against 2,000 Eye2Sky all-sky frames and 2,000 SWIM patches.", "",
             "| Held-out source | n in / out | AUROC combined | energy only | Mahalanobis only | vs Eye2Sky | vs SWIM patches |",
             "|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.held_out} | {r.n_in} / {r.n_out} | {r.auroc_combined:.3f} | {r.auroc_energy:.3f} | "
                     f"{r.auroc_mahalanobis:.3f} | {r.auroc_vs_eye2sky:.3f} | {r.auroc_vs_swim_patches:.3f} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
