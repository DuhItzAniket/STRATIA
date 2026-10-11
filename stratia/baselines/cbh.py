"""Cloud-base-height baselines (P058) on the P047 pairs.

Targets: the lowest cloud base (metres) of the ceilometer at the frame's time for cloudy pairs, and "no cloud
overhead" for label `none`. Baselines predict a height for cloudy pairs (scored by MAE of the predicted median,
bias, RMSD, within-tolerance rates, by height bin) and, where they can, cloud presence (balanced accuracy):

* climatology: the train-split median base (the skill reference), and a conditional climatology by month and
  hour-of-day bin (a stronger, still image-free reference);
* étage lookup: an étage classifier (frozen-feature probe trained on the étage-labelled sources) applied to the
  frame, mapped to the train-split median base of that étage;
* frozen-feature regressor: ridge regression of log height on the DINOv3 pooled embedding (train pairs);
* ceilometer-supervised CNN: ResNet-18 regressing log height on the image (train pairs), the plan's "ceilometer-only
  CNN" and the reference of criterion C3.

Skill = 1 - MAE / MAE_climatology (research design §4). Heights are compared in metres; models that predict
log height are evaluated after exponentiation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HEIGHT_BINS = ((0, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 20000))
TOLERANCES_M = (250, 500, 1000)
ETAGE_FROM_LABEL = {"low": "low", "mid": "mid", "high": "high"}


@dataclass
class CbhScores:
    n: int
    mae: float
    bias: float
    rmsd: float
    within: dict[int, float]
    by_bin: dict[str, dict[str, float]]
    skill: float | None = None

    def row(self) -> dict:
        out = {"n": self.n, "mae_m": self.mae, "bias_m": self.bias, "rmsd_m": self.rmsd, "skill": self.skill}
        out.update({f"within_{t}m": v for t, v in self.within.items()})
        out.update({f"mae_{k}": v["mae"] for k, v in self.by_bin.items()})
        return out


def score_heights(pred_m: np.ndarray, true_m: np.ndarray, mae_clim: float | None = None) -> CbhScores:
    pred_m, true_m = np.asarray(pred_m, dtype=float), np.asarray(true_m, dtype=float)
    err = pred_m - true_m
    ae = np.abs(err)
    by_bin = {}
    for lo, hi in HEIGHT_BINS:
        sel = (true_m >= lo) & (true_m < hi)
        if sel.any():
            by_bin[f"{lo}-{hi}"] = {"n": int(sel.sum()), "mae": float(ae[sel].mean()), "bias": float(err[sel].mean())}
    mae = float(ae.mean()) if len(ae) else float("nan")
    return CbhScores(n=int(len(ae)), mae=mae, bias=float(err.mean()) if len(err) else float("nan"),
                     rmsd=float(np.sqrt((err**2).mean())) if len(err) else float("nan"),
                     within={t: float((ae <= t).mean()) for t in TOLERANCES_M}, by_bin=by_bin,
                     skill=None if mae_clim is None else float(1.0 - mae / mae_clim))


def balanced_accuracy(true_cloud: np.ndarray, pred_cloud: np.ndarray) -> float:
    t, p = np.asarray(true_cloud, bool), np.asarray(pred_cloud, bool)
    tpr = p[t].mean() if t.any() else 0.0
    tnr = (~p[~t]).mean() if (~t).any() else 0.0
    return float((tpr + tnr) / 2)


# ----------------------------------------------------------------------------------------------- climatology
def climatology(train: pd.DataFrame) -> float:
    """Median base of the cloudy train pairs."""
    return float(train.loc[train.cbh_m.notna(), "cbh_m"].median())


def conditional_climatology(train: pd.DataFrame, hour_bin_h: int = 3) -> pd.Series:
    """Median base by (month, hour-of-day bin) of the cloudy train pairs; falls back to the overall median."""
    t = train[train.cbh_m.notna()]
    key = list(zip(t.utc.dt.month, t.utc.dt.hour // hour_bin_h, strict=True))
    series = pd.Series(t.cbh_m.to_numpy(), index=pd.MultiIndex.from_tuples(key, names=["month", "hour_bin"]))
    return series.groupby(level=[0, 1]).median()


def apply_conditional(table: pd.Series, frames: pd.DataFrame, fallback: float, hour_bin_h: int = 3) -> np.ndarray:
    keys = list(zip(frames.utc.dt.month, frames.utc.dt.hour // hour_bin_h, strict=True))
    return np.array([float(table.get(k, fallback)) for k in keys])


# ----------------------------------------------------------------------------------------------- étage lookup
def etage_medians(train: pd.DataFrame) -> dict[str, float]:
    """Median base per ceilometer étage label on the cloudy train pairs."""
    t = train[train.cbh_m.notna() & train.label.isin(ETAGE_FROM_LABEL)]
    return {k: float(v) for k, v in t.groupby("label").cbh_m.median().items()}


def lookup_heights(pred_etage: np.ndarray, medians: dict[str, float], fallback: float) -> np.ndarray:
    return np.array([medians.get(e, fallback) for e in pred_etage])


# ----------------------------------------------------------------------------------------------- ridge regressor
def fit_ridge_log_height(x: np.ndarray, cbh_m: np.ndarray, alpha_grid=(1.0, 10.0, 100.0, 1000.0), x_val=None, cbh_val=None):
    """Ridge on standardised features predicting log10 height; alpha chosen on val MAE in metres when val is given."""
    from sklearn.linear_model import Ridge

    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True) + 1e-6
    xs = (x - mu) / sd
    y = np.log10(np.asarray(cbh_m, dtype=float))
    best = None
    for alpha in alpha_grid:
        model = Ridge(alpha=alpha).fit(xs, y)
        if x_val is None:
            return model, mu, sd, alpha
        pred = 10 ** model.predict((x_val - mu) / sd)
        mae = float(np.abs(pred - cbh_val).mean())
        if best is None or mae < best[0]:
            best = (mae, model, alpha)
    return best[1], mu, sd, best[2]


def predict_ridge(model, mu, sd, x: np.ndarray) -> np.ndarray:
    return 10 ** model.predict((x - mu) / sd)
