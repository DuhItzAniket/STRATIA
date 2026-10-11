"""P069: temperature-balanced dataset sampling, per-task batches and label masks; observed shares match the config."""

import numpy as np
import pandas as pd

from stratia.data import sampler as SM


def _table():
    rows = []
    sizes = {"big": 8000, "mid": 2000, "small": 200}
    for d, n in sizes.items():
        for _ in range(n):
            rows.append({"dataset": d, "group_mask": {"global"} | ({"dense"} if d != "big" else set()),
                         "has_genus": True, "has_etage": d != "small", "has_oktas": d == "small",
                         "has_sky_parse": d != "big", "has_layer": d == "mid", "has_cbh": False})
    return pd.DataFrame(rows)


def test_dataset_probs_and_temperature():
    counts = pd.Series({"big": 8000, "mid": 2000, "small": 200})
    p1 = SM.dataset_probs(counts, 1.0)
    p0 = SM.dataset_probs(counts, 0.0)
    ph = SM.dataset_probs(counts, 0.5)
    assert abs(p1["big"] - 8000 / 10200) < 1e-9 and abs(p0["small"] - 1 / 3) < 1e-9
    assert p1["small"] < ph["small"] < p0["small"]                     # the square root lifts the small set
    floored = SM.dataset_probs(counts, 1.0, min_share=0.1)
    assert floored["small"] >= 0.1 - 1e-9 and abs(floored.sum() - 1) < 1e-9


def test_observed_shares_match_expected_and_batches_are_single_group():
    table = _table()
    cfg = SM.SamplerConfig(tau=0.5, group_probs={"global": 0.6, "dense": 0.4}, batch_sizes={"global": 32, "dense": 8},
                           batches_per_epoch=400)
    s = SM.TaskBatchSampler(table, cfg, seed=0)
    for g, idx in s:
        assert g in ("global", "dense") and len(idx) == cfg.batch_sizes[g]
        assert all(g in m for m in table.group_mask.to_numpy()[idx])     # every row serves the batch's group
        if g == "dense":
            assert "big" not in set(table.dataset.to_numpy()[idx])      # big has no dense labels
    stats = SM.sampling_stats(s, epochs=3)
    assert len(stats) == 5                                             # (global x 3) + (dense x 2)
    assert np.all(np.abs(stats.observed - stats.share) < 0.02), stats
    assert abs(stats.share.sum() - 1) < 1e-9
    # determinism per (seed, epoch)
    s.set_epoch(0)
    first = [idx for _, idx in s]
    s.set_epoch(0)
    assert [idx for _, idx in s] == first
    s.set_epoch(1)
    assert [idx for _, idx in s] != first


def test_label_masks_and_group_from_tasks():
    table = _table()
    m = SM.label_mask(table.iloc[0])
    assert m.tolist() == [True, True, False, False, False, False]
    small = SM.label_mask(table[table.dataset == "small"].iloc[0])
    assert small.tolist() == [True, False, True, True, False, False]
    assert SM.group_mask_from_tasks(["genus"]) == {"global"}
    assert SM.group_mask_from_tasks(["sky_parse", "layer"]) == {"dense"}
    assert SM.group_mask_from_tasks(["genus_soft", "oktas", "cbh_class", "wmo_codes"]) == {"global"}
    assert SM.group_mask_from_tasks(["cbh"]) == {"cbh"}
