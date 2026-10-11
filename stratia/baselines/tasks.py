"""Label tables per protocol for the baselines (P052-P059).

Recognition protocols use the native single-label classes of each dataset (CCSN 11 genera, MGCD 7 types, SWIMCAT 5,
Montenegro 5 altitude classes = the in-domain split's stratum) on the grouped in-domain splits, and the WMO étage
sets of `data/labels.parquet` (multi-label over low / mid / high; an empty set = no cloud) for leave-one-dataset-out.
Everything is keyed by `sample_id`; test rows are returned only when `include_test=True` (P090).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..contract import ETAGE_CLASSES

REPO_ROOT = Path(__file__).resolve().parents[2]
RECOGNITION_SOURCES = ("ccsn", "mgcd", "swimcat", "montenegro")
LODO_FOLDS_WITH_ETAGE = ("ccsn", "mgcd", "montenegro", "almeria")   # the SWIM family has no étage labels


@dataclass
class Tables:
    manifest: pd.DataFrame
    labels: pd.DataFrame
    in_domain: pd.DataFrame
    lodo: pd.DataFrame


def load_tables(root: str | Path = REPO_ROOT) -> Tables:
    root = Path(root)
    manifest = pd.read_parquet(root / "data/manifest.parquet",
                               columns=["sample_id", "dataset", "image_file", "source_label", "site", "utc"])
    labels = pd.read_parquet(root / "data/labels.parquet")
    in_domain = pd.read_parquet(root / "data/splits/in_domain.parquet")
    lodo = pd.read_parquet(root / "data/splits/lodo.parquet")
    return Tables(manifest, labels, in_domain, lodo)


# ----------------------------------------------------------------------------------------------- in-domain
def recognition_table(t: Tables, source: str, include_test: bool = False) -> pd.DataFrame:
    """sample_id, image_file, label (native class), split for one source; test rows only on request."""
    src = "swim" if source == "swimcat" else source                   # SWIMCAT lives in the SWIM-family source
    s = t.in_domain[t.in_domain.source == src]
    if source == "swimcat":
        s = s[s.dataset == "swimcat"]
    m = t.manifest.set_index("sample_id")
    label = s.stratum if source != "swimcat" else m.loc[s.sample_id, "source_label"].values
    out = pd.DataFrame({"sample_id": s.sample_id.values, "image_file": m.loc[s.sample_id, "image_file"].values,
                        "label": np.asarray(label).astype(str), "split": s.split.values})
    out = out[out.label.notna() & (out.label != "nan") & (out.label != "None")]
    if not include_test:
        out = out[out.split != "test"]
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- étage (LODO)
def etage_matrix(etage_sets: pd.Series) -> np.ndarray:
    """(n, 3) 0/1 matrix over ETAGE_CLASSES from JSON-encoded sets."""
    out = np.zeros((len(etage_sets), len(ETAGE_CLASSES)), dtype=np.int8)
    for i, raw in enumerate(etage_sets):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        for e in json.loads(raw):
            out[i, ETAGE_CLASSES.index(e)] = 1
    return out


def etage_table(t: Tables) -> pd.DataFrame:
    """sample_id, image_file, dataset, source, etage (JSON) for every sample with an étage label."""
    lab = t.labels[t.labels.etage_set.notna()][["sample_id", "dataset", "etage_set"]]
    src = t.in_domain[["sample_id", "source"]]
    out = lab.merge(src, on="sample_id").merge(t.manifest[["sample_id", "image_file"]], on="sample_id")
    return out.reset_index(drop=True)


def lodo_table(t: Tables, fold: str, include_test: bool = False) -> pd.DataFrame:
    """Rows of one LODO fold with étage labels: role train (other sources) / val (held-out source's in-domain val) /
    test (held-out, only on request)."""
    e = etage_table(t)
    f = t.lodo[t.lodo.fold == fold]
    if not include_test:
        f = f[f.role != "test"]
    out = f.merge(e, on="sample_id")
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- metrics
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes=None) -> float:
    """Unweighted mean of per-class F1 (classes absent from both truth and prediction are skipped)."""
    classes = np.unique(np.concatenate([y_true, y_pred])) if classes is None else np.asarray(classes)
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


def multilabel_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the columns of 0/1 matrices (n, k) that have at least one positive in `y_true`.

    Columns without support are left out whatever is predicted for them (a fold whose held-out source has no
    mid-étage rows is scored on low and high only); otherwise a single false positive in an unsupported column
    would switch that column's F1 from 'skipped' to 0 and move the mean by a third. False positives in
    unsupported columns are reported separately by the callers that care (`unsupported_positive_rate`).
    """
    f1s = []
    for j in range(y_true.shape[1]):
        tp = np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 1))
        fp = np.sum((y_true[:, j] == 0) & (y_pred[:, j] == 1))
        fn = np.sum((y_true[:, j] == 1) & (y_pred[:, j] == 0))
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1s)) if f1s else 0.0


THRESHOLD_GRID = tuple(np.linspace(0.05, 0.95, 19))


def tuned_thresholds(probs: np.ndarray, truth: np.ndarray, grid=THRESHOLD_GRID) -> np.ndarray:
    """0/1 predictions with a per-column threshold that maximises that column's F1 on the given rows (an optimistic,
    selection-on-val number reported next to the fixed 0.5 one, as the probes' C is selected on val)."""
    hat = np.zeros_like(truth)
    for j in range(truth.shape[1]):
        if truth[:, j].sum() == 0:
            hat[:, j] = probs[:, j] > 0.5
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            p = probs[:, j] > t
            tp = (p & (truth[:, j] == 1)).sum()
            f1 = 2 * tp / max(2 * tp + (p & (truth[:, j] == 0)).sum() + (~p & (truth[:, j] == 1)).sum(), 1)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        hat[:, j] = probs[:, j] > best_t
    return hat.astype(int)


def unsupported_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of rows predicted positive in columns that have no positive in `y_true` (nan when every column has
    support): the error `multilabel_macro_f1` deliberately leaves out."""
    cols = [j for j in range(y_true.shape[1]) if y_true[:, j].sum() == 0]
    if not cols:
        return float("nan")
    return float(np.mean([np.mean(y_pred[:, j] == 1) for j in cols]))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0
