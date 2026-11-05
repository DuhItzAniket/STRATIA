"""Data-free tests of the Stage J pieces: the P091 hard-target ablation and the ablation assembler's row reader."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_harden_targets_collapses_sets_and_shares():
    mod = _load("train_global_heads")
    rows = pd.DataFrame(
        {
            "genus_set": [np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.2, 0.5, 0.3, 0.0]), np.zeros(4)],
            "etage": [np.array([0.6, 0.6, 0.0]), np.array([0.4, 0.3, 0.3]), np.zeros(3)],
            "oktas": [np.array([0.0, 0.2, 0.8]), np.array([0.5, 0.5, 0.0]), np.zeros(3)],
            "mask_genus": [True, True, False],
            "mask_etage": [True, True, False],
            "mask_oktas": [True, True, False],
            "genus_alt": [True, False, False],
            "weight": [1.0, 2.0, 1.0],
        }
    )
    out = mod.harden_targets(rows)
    # a set keeps its first listed genus only; a share vector keeps its argmax
    assert out.genus_set.iloc[0].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert out.genus_set.iloc[1].tolist() == [0.0, 1.0, 0.0, 0.0]
    # étages above 0.5 stay a set; a row with none above 0.5 keeps its single argmax
    assert out.etage.iloc[0].tolist() == [1.0, 1.0, 0.0]
    assert out.etage.iloc[1].tolist() == [1.0, 0.0, 0.0]
    # oktas: one-hot argmax (ties go to the first)
    assert out.oktas.iloc[0].tolist() == [0.0, 0.0, 1.0]
    assert out.oktas.iloc[1].tolist() == [1.0, 0.0, 0.0]
    # masked-out rows stay all zero, alternatives are dropped, weights and masks untouched
    assert out.genus_set.iloc[2].sum() == 0 and out.etage.iloc[2].sum() == 0 and out.oktas.iloc[2].sum() == 0
    assert not out.genus_alt.any()
    assert out.weight.tolist() == [1.0, 2.0, 1.0] and out.mask_genus.tolist() == [True, True, False]
    assert rows.genus_alt.iloc[0]  # the input frame is not modified


def test_global_rows_reads_the_latest_run_only():
    mod = _load("ablations")
    g = pd.DataFrame(
        [
            ("abl_x", 0, "val", "all", "mean_etage_f1", 0.70, "20260101T000000Z_a"),
            ("abl_x", 0, "val", "all", "mean_genus_f1", 0.60, "20260101T000000Z_a"),
            ("abl_x", 0, "val", "all", "mean_etage_f1", 0.80, "20260102T000000Z_b"),
            ("abl_x", 0, "val", "montenegro", "c2_etage_ratio", 0.95, "20260102T000000Z_b"),
            ("abl_x", 1, "val", "all", "mean_etage_f1", 0.10, "20260103T000000Z_c"),
            ("abl_x", 0, "train", "all", "mean_etage_f1", 0.99, "20260102T000000Z_b"),
        ],
        columns=["name", "seed", "split", "dataset", "metric", "value", "run_id"],
    )
    out = mod.global_rows(g, "abl_x")
    assert out["run_id"] == "20260102T000000Z_b"
    assert out["mean_etage_f1"] == 0.80
    assert np.isnan(out["mean_genus_f1"])  # the latest run has no genus row: nan, not the older run's value
    assert out["montenegro/c2_etage_ratio"] == 0.95
    assert mod.global_rows(g, "missing") == {}
