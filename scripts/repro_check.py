"""Reproducibility check (P004/P005): train a tiny model twice with one config and compare.

    python scripts/repro_check.py --config configs/repro_check.yaml [key=value ...]

Each repetition is a registered run (runs/registry.csv). Exit code 0 when the two runs'
per-epoch metrics are bit-identical, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.utils.config import load_config  # noqa: E402
from stratia.utils.run import Run  # noqa: E402
from stratia.utils.seed import set_seed  # noqa: E402


def train_once(cfg, tag: str) -> list[dict]:
    set_seed(cfg.seed, deterministic=cfg.deterministic)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(cfg.seed)
    x = torch.randn(cfg.n_samples, cfg.in_dim, generator=g)
    w_true = torch.randn(cfg.in_dim, cfg.n_classes, generator=g)
    y = (x @ w_true + 0.5 * torch.randn(cfg.n_samples, cfg.n_classes, generator=g)).argmax(1)
    ds = torch.utils.data.TensorDataset(x, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    model = nn.Sequential(nn.Linear(cfg.in_dim, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.n_classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = []
    with Run(cfg, name=f"repro_check_{tag}", phase="P004") as run:
        for epoch in range(cfg.epochs):
            model.train()
            tot, correct, n = 0.0, 0, 0
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=cfg.amp and dev.type == "cuda"):
                    loss = nn.functional.cross_entropy(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item() * len(yb)
                correct += (model(xb).argmax(1) == yb).sum().item()
                n += len(yb)
            rec = {"epoch": epoch, "loss": tot / n, "acc": correct / n}
            run.log(epoch, loss=rec["loss"], acc=rec["acc"])
            history.append(rec)
        run.finish(final_loss=history[-1]["loss"], final_acc=history[-1]["acc"])
        history.append({"run_id": run.run_id})
    return history


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/repro_check.yaml")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    h1, h2 = train_once(cfg, "a"), train_once(cfg, "b")
    m1, m2 = h1[:-1], h2[:-1]
    identical = m1 == m2
    print(json.dumps({"run_a": h1[-1]["run_id"], "run_b": h2[-1]["run_id"], "identical": identical,
                      "epochs_a": m1, "epochs_b": m2}, indent=2))
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
