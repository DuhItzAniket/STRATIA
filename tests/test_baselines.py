"""P054: label tables, metrics and probes behave on synthetic data."""

import numpy as np
import pandas as pd

from stratia.baselines import probes as P
from stratia.baselines import tasks as T


def test_macro_f1_and_accuracy():
    y = np.array(["a", "a", "b", "b", "c"])
    p = np.array(["a", "b", "b", "b", "c"])
    # a: tp1 fp0 fn1 -> f1 .667; b: tp2 fp1 fn0 -> .8; c: 1.0
    assert abs(T.macro_f1(y, p) - (2 / 3 + 0.8 + 1.0) / 3) < 1e-9
    assert T.accuracy(y, p) == 0.8
    assert T.macro_f1(y, y) == 1.0 and T.macro_f1(np.array([]), np.array([])) == 0.0


def test_multilabel_macro_f1_and_etage_matrix():
    sets = pd.Series(['["low"]', '["low", "mid"]', "[]", '["high"]', None])
    m = T.etage_matrix(sets)
    assert m.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1], [0, 0, 0]]
    pred = m.copy()
    pred[1, 1] = 0                                   # miss one mid
    assert abs(T.multilabel_macro_f1(m, pred) - (1.0 + 0.0 + 1.0) / 3) < 1e-9
    assert T.multilabel_macro_f1(m, m) == 1.0
    # a column with no positives in the truth is left out whatever is predicted there (no 0.995 -> 0.644 jump
    # from a single false positive); its false positives are reported separately
    truth = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.int8)
    clean = truth.copy()
    spur = truth.copy()
    spur[2, 1] = 1
    assert T.multilabel_macro_f1(truth, clean) == 1.0 and T.multilabel_macro_f1(truth, spur) == 1.0
    assert T.unsupported_positive_rate(truth, clean) == 0.0 and abs(T.unsupported_positive_rate(truth, spur) - 0.25) < 1e-9
    assert np.isnan(T.unsupported_positive_rate(m, m))


def _blobs(n_per=60, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(3, dim)) * 3
    x = np.concatenate([centres[k] + rng.normal(size=(n_per, dim)) for k in range(3)])
    y = np.repeat(np.array(["a", "b", "c"]), n_per)
    idx = rng.permutation(len(y))
    x, y = x[idx], y[idx]
    return x[:120], y[:120], x[120:], y[120:]


def test_linear_and_knn_probes_on_separable_blobs():
    xt, yt, xv, yv = _blobs()
    lin = P.linear_probe(xt, yt, xv, yv)
    knn = P.knn_probe(xt, yt, xv, yv, k=5)
    assert lin.kind == "linear" and lin.val_metric > 0.95 and lin.extra["accuracy"] > 0.95 and lin.params["C"] in P.C_GRID
    assert knn.kind == "knn" and knn.val_metric > 0.95 and len(knn.val_pred) == len(yv)


def test_multilabel_probes():
    xt, yt, xv, yv = _blobs()
    to_ml = {"a": [1, 0, 0], "b": [1, 1, 0], "c": [0, 0, 1]}
    mt = np.array([to_ml[v] for v in yt], dtype=np.int8)
    mv = np.array([to_ml[v] for v in yv], dtype=np.int8)
    lin = P.linear_probe(xt, mt, xv, mv, multilabel=True)
    knn = P.knn_probe(xt, mt, xv, mv, k=5, multilabel=True)
    assert lin.val_metric > 0.95 and knn.val_metric > 0.95 and lin.val_pred.shape == mv.shape
    constant = np.zeros_like(mt)
    constant[:, 0] = 1                                # a column with a single class is handled, not crashed
    out = P.linear_probe(xt, constant, xv, mv, multilabel=True)
    assert out.val_pred[:, 0].all() and not out.val_pred[:, 1].any()


def test_recognition_and_lodo_tables_from_synthetic_tables():
    manifest = pd.DataFrame({"sample_id": [f"s{i}" for i in range(8)], "dataset": ["ccsn"] * 4 + ["mgcd"] * 4,
                             "image_file": [f"f{i}.jpg" for i in range(8)], "source_label": ["Ac", "Ci", "Ac", "Cu"] + ["x"] * 4,
                             "site": None, "utc": pd.NaT})
    labels = pd.DataFrame({"sample_id": manifest.sample_id, "dataset": manifest.dataset,
                           "etage_set": ['["mid"]', '["high"]', '["mid"]', '["low"]', "[]", '["low"]', '["high"]', '["low"]']})
    in_domain = pd.DataFrame({"sample_id": manifest.sample_id, "dataset": manifest.dataset, "source": manifest.dataset,
                              "unit": range(8), "stratum": manifest.source_label, "test_ok": True,
                              "split": ["train", "val", "test", "train", "train", "val", "train", "test"]})
    lodo = pd.DataFrame({"fold": ["ccsn"] * 8, "sample_id": manifest.sample_id,
                         "role": ["test", "val", "test", "test", "train", "train", "train", "train"]})
    t = T.Tables(manifest, labels, in_domain, lodo)
    rec = T.recognition_table(t, "ccsn")
    assert list(rec.split) == ["train", "val", "train"] and list(rec.label) == ["Ac", "Ci", "Cu"]
    assert len(T.recognition_table(t, "ccsn", include_test=True)) == 4
    lo = T.lodo_table(t, "ccsn")
    assert set(lo.role) == {"train", "val"} and len(lo) == 5
    assert len(T.lodo_table(t, "ccsn", include_test=True)) == 8
