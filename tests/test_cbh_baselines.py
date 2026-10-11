"""P058: height scoring, climatologies, étage lookup and the ridge regressor on synthetic pairs."""

import numpy as np
import pandas as pd

from stratia.baselines import cbh as C


def _pairs(n=400, seed=0):
    rng = np.random.default_rng(seed)
    utc = pd.date_range("2022-04-01", periods=n, freq="6h", tz="UTC")
    label = rng.choice(["none", "low", "mid", "high"], n, p=[0.25, 0.4, 0.2, 0.15])
    cbh = np.where(label == "low", rng.uniform(300, 2000, n), np.where(label == "mid", rng.uniform(2000, 6000, n),
                   np.where(label == "high", rng.uniform(6000, 11000, n), np.nan)))
    return pd.DataFrame({"utc": utc, "label": label, "cbh_m": cbh, "split": np.where(np.arange(n) % 5 == 0, "val", "train")})


def test_score_heights_and_skill():
    true = np.array([1000.0, 2000.0, 4000.0, 8000.0])
    pred = np.array([1200.0, 1800.0, 4000.0, 9000.0])
    sc = C.score_heights(pred, true, mae_clim=1000.0)
    assert sc.n == 4 and abs(sc.mae - 350.0) < 1e-9 and abs(sc.bias - 250.0) < 1e-9
    assert sc.within[250] == 0.75 and sc.within[500] == 0.75 and sc.within[1000] == 1.0
    assert abs(sc.skill - 0.65) < 1e-9
    assert set(sc.by_bin) == {"1000-2000", "2000-4000", "4000-8000", "8000-20000"}
    row = sc.row()
    assert row["mae_1000-2000"] == 200.0 and row["within_500m"] == 0.75
    assert C.balanced_accuracy([True, True, False, False], [True, False, False, False]) == 0.75


def test_climatologies_and_lookup():
    p = _pairs()
    train = p[p.split == "train"]
    clim = C.climatology(train)
    assert 500 < clim < 8000
    cond = C.conditional_climatology(train)
    assert isinstance(cond.index, pd.MultiIndex) and len(cond) > 1
    pred = C.apply_conditional(cond, p[p.split == "val"], clim)
    assert pred.shape == ((p.split == "val").sum(),) and np.isfinite(pred).all()
    meds = C.etage_medians(train)
    assert set(meds) == {"low", "mid", "high"} and meds["low"] < meds["mid"] < meds["high"]
    heights = C.lookup_heights(np.array(["low", "none", "high"]), meds, clim)
    assert heights[0] == meds["low"] and heights[1] == clim and heights[2] == meds["high"]


def test_ridge_log_height_learns_a_linear_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8))
    cbh = 10 ** (3.0 + 0.3 * x[:, 0] + 0.05 * rng.normal(size=500))
    model, mu, sd, alpha = C.fit_ridge_log_height(x[:400], cbh[:400], x_val=x[400:], cbh_val=cbh[400:])
    pred = C.predict_ridge(model, mu, sd, x[400:])
    sc = C.score_heights(pred, cbh[400:], mae_clim=float(np.abs(np.median(cbh[:400]) - cbh[400:]).mean()))
    assert sc.skill > 0.5 and alpha in (1.0, 10.0, 100.0, 1000.0)
