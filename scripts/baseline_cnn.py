"""CNN baselines (P052 legacy MobileNetV3-L; P053 ResNet-50 and ConvNeXt-T, 3 seeds).

    python scripts/baseline_cnn.py --phase P052 --arch mobilenetv3_large --recipe legacy --sources ccsn \
        --protocols legacy_random grouped --seeds 0
    python scripts/baseline_cnn.py --phase P053 --arch resnet50 convnext_tiny --sources ccsn mgcd swimcat montenegro \
        --seeds 0 1 2

Protocols: `grouped` = the P038 in-domain split (train -> val; test locked); `legacy_random` = the prototype's
per-image random split, drawn with the run's seed from the same unlocked pool (grouped train + val rows) with the
grouped split's per-class val counts, so the two protocols differ only in whether near-duplicate units are kept
together. The random val is scored whole and in two parts: *leaked* images (their P038 unit has a member in the
random train set) and *clean* ones. The locked test rows are never read: the prototype's exact 70/15/15 split over
the whole dataset would need them. Every (arch, source, protocol, seed) is a run with per-epoch curves; results go
to data/baseline_cnn.parquet and docs/data/baseline_cnn.md.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.baselines import tasks as T  # noqa: E402
from stratia.baselines.train_cnn import TrainConfig, fit_verdict, train_classifier  # noqa: E402
from stratia.data.registry import load_paths  # noqa: E402
from stratia.utils.run import Run  # noqa: E402

DOC = Path("docs/data/baseline_cnn.md")
OUT = Path("data/baseline_cnn.parquet")
KEY = ["arch", "source", "protocol", "seed", "recipe"]


def random_split_like(grouped: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A per-image random split of the grouped train + val rows with the grouped split's per-class val counts."""
    rng = random.Random(seed)
    out = grouped[grouped.split.isin(["train", "val"])].copy()
    n_val = out[out.split == "val"].groupby("label").size()
    out["split"] = "train"
    for label, g in out.groupby("label"):
        files = sorted(g.image_file.tolist())
        rng.shuffle(files)
        val_files = set(files[:int(n_val.get(label, 0))])
        out.loc[g.index, "split"] = ["val" if f in val_files else "train" for f in g.image_file]
    return out


def leak_split(tr: pd.DataFrame, va: pd.DataFrame, units: pd.Series) -> np.ndarray:
    """True for val rows whose P038 unit (near-duplicate group) has a member in train."""
    train_units = set(units.reindex(tr.sample_id).dropna())
    return units.reindex(va.sample_id).isin(train_units).to_numpy()


def recipe(name: str, arch: str, epochs: int | None, seed: int) -> TrainConfig:
    if name == "legacy":
        return TrainConfig(arch=arch, epochs=epochs or 20, batch_size=32, lr=1e-3, weight_decay=1e-4, label_smoothing=0.0,
                           schedule="plateau", patience=5, seed=seed)
    return TrainConfig(arch=arch, epochs=epochs or 12, batch_size=32, lr=3e-4, weight_decay=0.05, label_smoothing=0.1,
                       schedule="cosine", seed=seed)


def one_run(a, source: str, protocol: str, arch: str, seed: int, tr, va, units: pd.Series, cache_root: Path) -> dict:
    cfg = recipe(a.recipe, arch, a.epochs, seed)
    cfg.workers = a.workers
    name = f"cnn_{arch}_{source}_{protocol}_s{seed}"
    plain = {k: v for k, v in cfg.__dict__.items() if k != "extra"}
    oc = OmegaConf.create({**plain, "source": source, "protocol": protocol, "recipe": a.recipe,
                           "crop_scale": list(cfg.crop_scale)})
    print(f"{name}: {len(tr)} train / {len(va)} val images, {tr.label.nunique()} classes", flush=True)
    with Run(oc, name=name, phase=a.phase) as run:
        res = train_classifier(tr.image_file.tolist(), tr.label.to_numpy(), va.image_file.tolist(), va.label.to_numpy(),
                               cache_root, cfg, run=run)
        best = res["best"]
        row = {"phase": a.phase, "arch": arch, "source": source, "protocol": protocol, "seed": seed, "recipe": a.recipe,
               "n_train": len(tr), "n_val": len(va), "classes": int(tr.label.nunique()), "best_epoch": best["epoch"],
               "epochs_run": res["epochs_run"], "val_macro_f1": best["macro_f1"], "val_acc": best["acc"],
               "train_val_gap": res["train_val_gap"], "verdict": fit_verdict(res["curves"], best["epoch"]),
               "seconds": res["seconds"], "run_id": run.run_id}
        leaked = leak_split(tr, va, units)
        pred, true = np.asarray(best["pred"]), np.asarray(best["true"])
        row["n_val_leaked"] = int(leaked.sum())
        row["val_leaked_macro_f1"] = T.macro_f1(true[leaked], pred[leaked]) if leaked.any() else np.nan
        row["val_leaked_acc"] = T.accuracy(true[leaked], pred[leaked]) if leaked.any() else np.nan
        row["val_clean_macro_f1"] = T.macro_f1(true[~leaked], pred[~leaked]) if (~leaked).any() else np.nan
        row["val_clean_acc"] = T.accuracy(true[~leaked], pred[~leaked]) if (~leaked).any() else np.nan
        summary = {"val_macro_f1": row["val_macro_f1"], "val_acc": row["val_acc"], "best_epoch": row["best_epoch"],
                   "train_val_gap": row["train_val_gap"], "n_val_leaked": row["n_val_leaked"],
                   "val_clean_macro_f1": row["val_clean_macro_f1"]}
        run.finish(**{k: v for k, v in summary.items() if not (isinstance(v, float) and np.isnan(v))})
    test_txt = (f"; leaked {row['n_val_leaked']} of {len(va)}: F1 {row['val_leaked_macro_f1']:.3f} / clean "
                f"{row['val_clean_macro_f1']:.3f}") if row["n_val_leaked"] else ""
    print(f"  -> val macro-F1 {row['val_macro_f1']:.3f} acc {row['val_acc']:.3f} (epoch {row['best_epoch']}, "
          f"{row['verdict']}){test_txt}", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", default="P053")
    ap.add_argument("--arch", nargs="+", default=["resnet50", "convnext_tiny"])
    ap.add_argument("--sources", nargs="+", default=list(T.RECOGNITION_SOURCES))
    ap.add_argument("--protocols", nargs="+", default=["grouped"], choices=["grouped", "legacy_random"])
    ap.add_argument("--recipe", default="modern", choices=["modern", "legacy"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t0 = time.perf_counter()
    cache_root = Path(load_paths()["cache_root"])
    tables = T.load_tables()
    results = []
    units = tables.in_domain.set_index("sample_id").unit
    for source in a.sources:
        grouped = T.recognition_table(tables, source)                        # train + val rows; test locked
        for protocol in a.protocols:
            for arch in a.arch:
                for seed in a.seeds:
                    tab = random_split_like(grouped, seed) if protocol == "legacy_random" else grouped
                    tr, va = tab[tab.split == "train"], tab[tab.split == "val"]
                    results.append(one_run(a, source, protocol, arch, seed, tr, va, units, cache_root))
    df = pd.DataFrame(results)
    if OUT.exists():
        old = pd.read_parquet(OUT)
        df = pd.concat([old[~old.set_index(KEY).index.isin(df.set_index(KEY).index)], df], ignore_index=True)
    df.to_parquet(OUT, index=False)
    write_doc(df)
    print(f"done in {time.perf_counter() - t0:.0f} s: {len(results)} runs -> {DOC}")
    return 0


def write_doc(df: pd.DataFrame) -> None:
    lines = ["# CNN baselines (P052 / P053)", "",
             "ImageNet-pretrained timm models at 224 px on the P022 image cache; `grouped` = P038 in-domain split (train -> val, "
             "test locked); `legacy_random` = a per-image random split of the same pool with the same per-class val counts "
             "(one draw per seed), its val scored whole and split into *leaked* images (a near-duplicate unit member is in "
             "train) and *clean* ones. Mean ± std over seeds where more than one seed ran.", ""]
    g = df.groupby(["phase", "arch", "source", "protocol", "recipe"])
    summary = g.agg(seeds=("seed", "count"), val_f1=("val_macro_f1", "mean"), val_f1_std=("val_macro_f1", "std"),
                    val_acc=("val_acc", "mean"), n_leaked=("n_val_leaked", "mean"),
                    leaked_f1=("val_leaked_macro_f1", "mean"), clean_f1=("val_clean_macro_f1", "mean"),
                    gap=("train_val_gap", "mean"), epochs=("best_epoch", "mean")).reset_index()
    lines += ["| Phase | Arch | Source | Protocol | Recipe | Seeds | Val macro-F1 | Val acc | Leaked val: n, F1 | "
              "Clean val F1 | Train-val gap | Best epoch |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        std = f" ± {r.val_f1_std:.3f}" if r.seeds > 1 and pd.notna(r.val_f1_std) else ""
        leaked = f"{r.n_leaked:.0f}, {r.leaked_f1:.3f}" if pd.notna(r.leaked_f1) else "0"
        clean = f"{r.clean_f1:.3f}" if pd.notna(r.clean_f1) else "-"
        lines.append(f"| {r.phase} | {r.arch} | {r.source} | {r.protocol} | {r.recipe} | {int(r.seeds)} | {r.val_f1:.3f}{std} | "
                     f"{r.val_acc:.3f} | {leaked} | {clean} | {r.gap:+.3f} | {r.epochs:.1f} |")
    lines += ["", "Verdicts per run (PLAN §4 fit reading) are in `data/baseline_cnn.parquet`; curves in each run's "
              "`metrics.jsonl`."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
