"""Frozen-feature probes (P054, P055): linear probe and k-nearest-neighbour on cached embeddings.

* ``linear_probe``: L2-regularised multinomial logistic regression (scikit-learn, lbfgs) on standardised features;
  `C` chosen on the validation split from a small grid, never on test. Multi-label targets (étage sets) use one
  binary probe per column with a 0.5 threshold.
* ``knn_probe``: cosine k-NN with distance-weighted votes.
Both return predictions plus metrics through ``tasks``; the caller registers the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import tasks as T

C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass
class ProbeResult:
    kind: str
    params: dict
    val_metric: float
    val_pred: np.ndarray
    extra: dict


def _standardise(train: np.ndarray, *others: np.ndarray):
    mu, sd = train.mean(0, keepdims=True), train.std(0, keepdims=True) + 1e-6
    return [(x - mu) / sd for x in (train, *others)]


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def linear_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, multilabel: bool = False,
                 c_grid=C_GRID, seed: int = 0, max_iter: int = 2000) -> ProbeResult:
    from sklearn.linear_model import LogisticRegression

    xt, xv = _standardise(x_train, x_val)
    best = None
    for c in c_grid:
        if multilabel:
            pred = np.zeros_like(y_val)
            for j in range(y_train.shape[1]):
                if y_train[:, j].min() == y_train[:, j].max():            # a column with one class only
                    pred[:, j] = y_train[0, j]
                    continue
                clf = LogisticRegression(C=c, max_iter=max_iter, random_state=seed).fit(xt, y_train[:, j])
                pred[:, j] = clf.predict(xv)
            metric = T.multilabel_macro_f1(y_val, pred)
        else:
            clf = LogisticRegression(C=c, max_iter=max_iter, random_state=seed).fit(xt, y_train)
            pred = clf.predict(xv)
            metric = T.macro_f1(y_val, pred)
        if best is None or metric > best.val_metric:
            best = ProbeResult("linear", {"C": c, "seed": seed}, metric, pred, {})
    best.extra = {"accuracy": T.accuracy(y_val, best.val_pred)} if not multilabel else {}
    return best


def knn_probe(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, k: int = 20,
              multilabel: bool = False) -> ProbeResult:
    xt, xv = _l2(x_train.astype(np.float32)), _l2(x_val.astype(np.float32))
    sims = xv @ xt.T
    idx = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
    w = np.take_along_axis(sims, idx, axis=1)
    w = np.clip(w, 0, None) + 1e-6
    if multilabel:
        votes = np.stack([(w[:, :, None] * y_train[idx]).sum(1) / w.sum(1, keepdims=True)], 0)[0]
        pred = (votes >= 0.5).astype(y_val.dtype)
        metric = T.multilabel_macro_f1(y_val, pred)
        extra = {}
    else:
        classes = np.unique(y_train)
        scores = np.zeros((len(xv), len(classes)), dtype=np.float32)
        lab_idx = np.searchsorted(classes, y_train[idx])
        for j in range(k):
            np.add.at(scores, (np.arange(len(xv)), lab_idx[:, j]), w[:, j])
        pred = classes[scores.argmax(1)]
        metric = T.macro_f1(y_val, pred)
        extra = {"accuracy": T.accuracy(y_val, pred)}
    return ProbeResult("knn", {"k": k}, metric, pred, extra)
