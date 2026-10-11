"""Global-head losses, prediction and per-dataset scoring on cached features (P073, P076, P077)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..baselines import tasks as T
from ..model import losses as L
from ..model.balance import UncertaintyWeighting
from ..model.stratia import STRATIA
from . import data as D

CEILING = {"etage_set": 0.852, "oktas_within1": 0.873}        # P035 leave-one-out human ceiling (Montenegro)


class GlobalModel(torch.nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float, kendall: bool):
        super().__init__()
        self.net = STRATIA(None, d_model=d_model, dense=False, cbh=False, ood=False, global_hidden=hidden)
        self.net.global_heads.trunk[3].p = dropout
        self.weighting = UncertaintyWeighting(["genus", "etage", "oktas"]) if kendall else None

    def forward(self, tokens, cls, ray_map, meta):
        return self.net.forward_tokens(tokens, cls, ray_map, meta)


def losses_for(out: dict, b: dict, lambda_h: float) -> dict[str, torch.Tensor]:
    genus = L.set_valued_bce(out["genus_logits"], b["genus_set"], b["genus_alt"], b["mask_genus"], b["genus_neg"])
    etage = L.soft_bce(out["etage_logits"], b["etage"], b["mask_etage"], b["etage_w"])
    oktas = L.coral_loss(out["oktas_cum_logits"], b["oktas"], b["mask_oktas"])
    hier = L.hierarchy_consistency(out["genus_logits"], out["etage_logits"], b["mask_genus"] & b["mask_etage"])
    return {"genus": genus, "etage": etage, "oktas": oktas, "hier": lambda_h * hier}


@torch.no_grad()
def predict(model: GlobalModel, ds: D.GlobalDataset, idx: np.ndarray, device: str, batch: int = 256) -> dict[str, np.ndarray]:
    model.eval()
    dl = DataLoader(torch.utils.data.Subset(ds, idx.tolist()), batch_size=batch, shuffle=False, collate_fn=D.collate)
    gen, eta, okt = [], [], []
    for b in dl:
        out = model(b["tokens"].to(device), b["cls"].to(device), b["ray_map"].to(device), b["meta"].to(device))
        gen.append(torch.sigmoid(out["genus_logits"]).cpu().numpy())
        eta.append(torch.sigmoid(out["etage_logits"]).cpu().numpy())
        okt.append(out["oktas_probs"].cpu().numpy())
    return {"genus": np.concatenate(gen), "etage": np.concatenate(eta), "oktas": np.concatenate(okt)}


def score_dataset(rows: pd.DataFrame, pred: dict[str, np.ndarray]) -> dict[str, float]:
    """Metrics for the rows of one dataset (aligned with `pred`)."""
    out = {}
    src = rows.source.iloc[0]
    # étage macro-F1 on rows with a full étage target (majority set for shares)
    full = (np.stack(rows.etage_w.to_numpy()) >= 1.0).all(1)          # exhaustive étage labels only (not Eye2Sky partials)
    if full.any() and (np.stack(rows.etage.to_numpy())[full] > 0.5).any():   # at least one étage with support
        truth = (np.stack(rows.etage.to_numpy())[full] > 0.5).astype(int)
        hat = (pred["etage"][full] > 0.5).astype(int)
        out["etage_f1"] = T.multilabel_macro_f1(truth, hat)
        out["etage_unsupported_fp"] = T.unsupported_positive_rate(truth, hat)
        out["etage_f1_tuned"] = T.multilabel_macro_f1(truth, T.tuned_thresholds(pred["etage"][full], truth))
        out["n_etage"] = int(full.sum())
    elif rows.mask_etage.any() and (np.stack(rows.etage.to_numpy()) > 0.5).any():   # partial étage (Eye2Sky)
        m = rows.mask_etage.to_numpy() & (np.stack(rows.etage.to_numpy()) > 0.5).any(1)
        truth = np.stack(rows.etage.to_numpy())[m].argmax(1)
        hat = pred["etage"][m] > 0.5
        out["etage_recall"] = float(hat[np.arange(m.sum()), truth].mean())
        lowest = np.where(hat.any(1), hat.argmax(1), -1)                      # first étage predicted present
        out["lowest_etage_acc"] = float((lowest == truth).mean())
        out["n_etage"] = int(m.sum())
    # genus over native classes
    if src in ("ccsn", "mgcd", "swim"):
        labels = rows.native_label.dropna().unique().tolist()
        sets = D.native_class_sets(src, labels)
        scorable = rows.native_label.isin([k for k, (idx, _) in sets.items() if idx]).to_numpy()
        if scorable.sum() > 10:
            scores, names = D.native_scores(pred["genus"][scorable], sets)
            hat = np.array(names)[scores.argmax(1)]
            truth = rows.native_label.to_numpy()[scorable]
            out["genus_f1"] = T.macro_f1(truth, hat, names)
            out["genus_acc"] = T.accuracy(truth, hat)
            out["n_genus"] = int(scorable.sum())
    if src == "montenegro":
        m = rows.mask_oktas.to_numpy()
        if m.any():
            dist = np.stack(rows.oktas.to_numpy())[m]
            maj = dist.argmax(1)
            hat = pred["oktas"][m].argmax(1)
            out["oktas_mae"] = float(np.abs(hat - maj).mean())
            out["oktas_within1"] = float((np.abs(hat - maj) <= 1).mean())
            out["oktas_kl"] = float(np.mean(np.sum(dist * (np.log(dist + 1e-9) - np.log(pred["oktas"][m] + 1e-9)), 1)))
        truth_set = (np.stack(rows.etage.to_numpy()) > 0.5).astype(int)
        hat_set = (pred["etage"] > 0.5).astype(int)
        out["etage_set_agreement"] = float((truth_set == hat_set).all(1).mean())
        out["c2_etage_ratio"] = out["etage_set_agreement"] / CEILING["etage_set"]
        if "oktas_within1" in out:
            out["c2_oktas_ratio"] = out["oktas_within1"] / CEILING["oktas_within1"]
    return out
