"""Budgeted hyper-parameter search for the global heads on cached features (P078).

    python scripts/hpo_global.py --trials 20 [--epochs 12]

Random search (seeded) over learning rate, weight decay, trunk dropout, hidden width, sampler temperature, geometry
dropout and the hierarchy weight; every trial is one `scripts/train_global_heads.py` run (name `hpo_t<i>`) scored by
its best `val/mean_etage_f1` on the in-domain validation split (Eye2Sky étage rows included). Validation only: the
locked test sets are not read. Output: data/hpo_global.parquet (one row per trial), docs/data/hpo_global.md (the
trials, the best configuration and a one-at-a-time sensitivity read from the sampled grid: Spearman rank
correlation of each parameter with the score). Optuna is not in the environment lock; the search is a plain
random search, which the budget (≤ 30 trials, minutes each) does not need more than.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("data/hpo_global.parquet")
DOC = Path("docs/data/hpo_global.md")
SPACE = {
    "lr": ("log", 2e-4, 3e-3),
    "weight_decay": ("log", 1e-3, 0.3),
    "dropout": ("lin", 0.0, 0.4),
    "hidden": ("choice", [256, 512, 1024]),
    "tau": ("lin", 0.3, 1.0),
    "p_drop": ("choice", [0.0, 0.15, 0.3]),
    "lambda_h": ("choice", [0.0, 0.1, 0.3]),
}


def sample(rng: np.random.Generator) -> dict:
    cfg = {}
    for k, spec in SPACE.items():
        if spec[0] == "log":
            cfg[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif spec[0] == "lin":
            cfg[k] = float(rng.uniform(spec[1], spec[2]))
        else:
            cfg[k] = spec[1][int(rng.integers(len(spec[1])))]
    return cfg


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phase", default="P078")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    rows = []
    t0 = time.perf_counter()
    for i in range(a.trials):
        cfg = sample(rng)
        name = f"hpo_t{i:02d}"
        cmd = [sys.executable, "-u", "-I", "scripts/train_global_heads.py", "--name", name, "--phase", a.phase, "--eye2sky-etage",
               "--epochs", str(a.epochs), "--lr", f"{cfg['lr']:.5g}", "--weight-decay", f"{cfg['weight_decay']:.5g}",
               "--dropout", f"{cfg['dropout']:.3f}", "--hidden", str(cfg["hidden"]), "--tau", f"{cfg['tau']:.3f}",
               "--p-drop", str(cfg["p_drop"]), "--lambda-h", str(cfg["lambda_h"]), "--patience", "4"]
        t1 = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        score = float("nan")
        if res.returncode == 0:
            df = pd.read_parquet("data/stage1_global.parquet")
            r = df[(df.name == name) & (df.split == "val") & (df.dataset == "all") & (df.metric == "mean_etage_f1")]
            score = float(r.value.iloc[0]) if len(r) else float("nan")
        rows.append({"trial": i, "name": name, **cfg, "score": score, "returncode": res.returncode,
                     "seconds": time.perf_counter() - t1})
        tail = "" if res.returncode == 0 else f" FAILED: {res.stderr[-300:]}"
        print(f"trial {i:02d}: {cfg} -> {score:.3f} ({rows[-1]['seconds']:.0f} s){tail}", flush=True)
        pd.DataFrame(rows).to_parquet(OUT, index=False)
    df = pd.DataFrame(rows)
    write_doc(df, a)
    print(f"done in {time.perf_counter() - t0:.0f} s -> {DOC}", flush=True)
    return 0


def write_doc(df: pd.DataFrame, a) -> None:
    ok = df[df.score.notna()].sort_values("score", ascending=False)
    lines = ["# Budgeted hyper-parameter search, global heads (P078)", "",
             f"Random search, {len(df)} trials of {a.epochs} epochs each on cached features (in-domain split with Eye2Sky "
             "étage rows), scored by the best val mean étage macro-F1; the locked test sets are not read. Sensitivity = "
             "Spearman rank correlation of each parameter with the score over the trials (a one-shot read of the sampled "
             "space, not a response surface).", "",
             "| Trial | lr | weight decay | dropout | hidden | tau | p_drop | lambda_h | Score | s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        sc = f"{r.score:.3f}" if pd.notna(r.score) else "failed"
        lines.append(f"| {r.trial} | {r.lr:.2e} | {r.weight_decay:.3f} | {r.dropout:.2f} | {r.hidden} | {r.tau:.2f} | "
                     f"{r.p_drop} | "
                     f"{r.lambda_h} | {sc} | {r.seconds:.0f} |")
    if len(ok):
        best = ok.iloc[0]
        lines += ["", f"**Best:** trial {best.trial} — score {best.score:.3f}: lr {best.lr:.2e}, "
                      f"weight decay {best.weight_decay:.3f}, dropout {best.dropout:.2f}, hidden {best.hidden}, "
                      f"tau {best.tau:.2f}, "
                      f"p_drop {best.p_drop}, lambda_h {best.lambda_h}. "
                      f"Spread over trials: {ok.score.min():.3f}–{ok.score.max():.3f} (median {ok.score.median():.3f}).", "",
                  "| Parameter | Spearman with score |", "|---|---|"]
        for k in SPACE:
            lines.append(f"| {k} | {spearman(ok[k].to_numpy().astype(float), ok.score.to_numpy()):+.2f} |")
    lines += ["", "Rows in `data/hpo_global.parquet`; each trial is a run of `scripts/train_global_heads.py` (phase P078)."]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
