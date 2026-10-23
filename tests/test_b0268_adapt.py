"""Data-free tests of the P084 pieces: B0268 global rows, the B0268 geometry branch and the LP parameter set."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stratia.contract import ETAGE_CLASSES, GENUS_CLASSES
from stratia.train import data as D
from stratia.train.global_eval import GlobalModel

ROOT = Path(__file__).resolve().parents[1]


def _write_fixture_root(tmp_path: Path) -> None:
    (tmp_path / "data" / "splits").mkdir(parents=True)
    utc = pd.to_datetime(["2026-10-18 10:22:54", "2026-10-18 10:23:24", "2026-10-20 10:00:00"], utc=True)
    pd.DataFrame({"sample_id": ["b0268:B0268/BLR01/2026-10-18/f1.jpg", "b0268:B0268/BLR01/2026-10-18/f2.jpg",
                                "b0268:B0268/BLR01/2026-10-20/f3.jpg", "ccsn:x.jpg"],
                  "dataset": ["b0268", "b0268", "b0268", "ccsn"],
                  "image_file": ["B0268/BLR01/2026-10-18/f1.jpg", "B0268/BLR01/2026-10-18/f2.jpg",
                                 "B0268/BLR01/2026-10-20/f3.jpg", "x.jpg"],
                  "utc": list(utc) + [pd.NaT], "sun_azimuth_deg": [251.0, 251.1, 200.0, np.nan],
                  "sun_zenith_deg": [61.0, 61.1, 50.0, np.nan]}).to_parquet(tmp_path / "data/manifest.parquet", index=False)
    pd.DataFrame([
        {"item_id": "f1", "day": "2026-10-18", "image_file": "B0268/BLR01/2026-10-18/f1.jpg",
         "etage_share": json.dumps({"low": 1.0, "mid": 0.5, "high": 0.0}), "genus_share": json.dumps({"Cu": 1.0, "Ac": 0.5}),
         "oktas_dist": json.dumps({"4": 0.5, "5": 0.5}), "clear_share": 0.0, "contrail_share": 0.5},
        {"item_id": "f2", "day": "2026-10-18", "image_file": "B0268/BLR01/2026-10-18/f2.jpg",
         "etage_share": json.dumps({"low": 0.0, "mid": 0.0, "high": 0.0}), "genus_share": json.dumps({}),
         "oktas_dist": json.dumps({"0": 1.0}), "clear_share": 1.0, "contrail_share": 0.0},
        {"item_id": "f3", "day": "2026-10-20", "image_file": "B0268/BLR01/2026-10-20/f3.jpg",
         "etage_share": None, "genus_share": None, "oktas_dist": None, "clear_share": None, "contrail_share": None},
    ]).to_parquet(tmp_path / "data/b0268_targets.parquet", index=False)
    pd.DataFrame({"draw": ["k1_d0"] * 3, "shots": [1] * 3,
                  "sample_id": ["b0268:B0268/BLR01/2026-10-18/f1.jpg", "b0268:B0268/BLR01/2026-10-18/f2.jpg",
                                "b0268:B0268/BLR01/2026-10-20/f3.jpg"],
                  "day": ["2026-10-18", "2026-10-18", "2026-10-20"], "role": ["train", "train", "test"]}).to_parquet(
        tmp_path / "data/splits/few_shot_target.parquet", index=False)


def test_b0268_rows_carry_soft_targets_roles_and_sun(tmp_path):
    _write_fixture_root(tmp_path)
    rows = D.b0268_rows(root=tmp_path, draw="k1_d0")
    assert list(rows.item_id) == ["f1", "f2"]                        # f3 has no usable annotation
    assert list(rows.split) == ["train", "train"] and list(rows.unit) == ["2026-10-18"] * 2
    r = rows.iloc[0]
    g = r.genus_set
    assert g[GENUS_CLASSES.index("Cu")] == 1.0 and g[GENUS_CLASSES.index("Ac")] == 0.5
    assert g[GENUS_CLASSES.index("contrail")] == 0.5 and g[GENUS_CLASSES.index("clear")] == 0.0
    assert r.genus_neg.tolist() == [1.0] * len(GENUS_CLASSES) and r.mask_genus and r.mask_etage and r.mask_oktas
    assert r.etage.tolist() == [1.0, 0.5, 0.0] and r.etage_w.tolist() == [1.0] * len(ETAGE_CLASSES)
    assert r.oktas[4] == 0.5 and r.oktas[5] == 0.5 and abs(r.oktas.sum() - 1.0) < 1e-6
    assert r.sun_zenith_deg == 61.0 and r.sun_azimuth_deg == 251.0 and r.calib_id == "b0268_nominal" and r.has_geometry
    clear = rows.iloc[1]
    assert clear.genus_set[GENUS_CLASSES.index("clear")] == 1.0 and clear.oktas[0] == 1.0
    assert list(D.b0268_rows(root=tmp_path).split) == ["val", "val"]         # no draw: everything is val


def test_geometry_cache_builds_b0268_ray_maps(tmp_path):
    _write_fixture_root(tmp_path)
    rows = D.b0268_rows(root=tmp_path, draw="k1_d0")
    geo = D.GeometryCache.build(rows, tmp_path, None)
    assert geo.ray_maps.shape == (2, 4, D.SIZE // D.POOL_PATCH, D.SIZE // D.POOL_PATCH) and geo.metas.shape == (2, 3)
    valid = geo.ray_maps[0, 3] > 0.5
    assert valid.sum() > 40                                              # a 105 deg fisheye at the zenith sees sky everywhere
    norms = np.linalg.norm(geo.ray_maps[0, :3], axis=0)[valid]
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert np.allclose(geo.metas[0], [np.cos(np.radians(61.0)), np.sin(np.radians(61.0)), 1.0], atol=1e-5)
    assert not np.allclose(geo.ray_maps[0], geo.ray_maps[1])            # the Sun moved, so the Sun-aligned frame did


def test_lp_parameters_are_the_three_output_layers():
    spec = importlib.util.spec_from_file_location("adapt_b0268", ROOT / "scripts" / "adapt_b0268.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = GlobalModel(384, 64, 0.1, False)
    lp = {id(p) for p in mod.lp_parameters(model)}
    named = {n for n, p in model.named_parameters() if id(p) in lp}
    assert named == {"net.global_heads.genus.weight", "net.global_heads.genus.bias", "net.global_heads.etage.weight",
                     "net.global_heads.etage.bias", "net.global_heads.oktas.weight.weight", "net.global_heads.oktas.bias"}
    assert len(lp) < sum(1 for _ in model.parameters())
    path = tmp = ROOT / "runs" / "_test_ckpt.pt"
    try:
        mod.save_checkpoint(model, path, 3)
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert set(state["model"]) == set(model.state_dict()) and state["ema"] is None and state["epoch"] == 3
    finally:
        tmp.unlink(missing_ok=True)


def test_results_from_rows_rebuilds_the_report_records():
    spec = importlib.util.spec_from_file_location("adapt_b0268", ROOT / "scripts" / "adapt_b0268.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []

    def add(draw, shots, stage, scope, group, metric, value):
        rows.append({"draw": draw, "shots": shots, "stage": stage, "scope": scope, "group": group, "metric": metric,
                     "value": value, "run_id": "no_such_run", "tag": "t"})

    for stage, e, o, f_e, f_g in (("zero_shot", 0.1, 0.2, 0.80, 0.70), ("adapted", 0.5, 0.9, 0.79, 0.66)):
        for group in ("all", "2026-10-19"):
            add("k1_d0", 1, stage, "b0268_val", group, "n", 10.0)
            add("k1_d0", 1, stage, "b0268_val", group, "etage_set_agreement", e)
            add("k1_d0", 1, stage, "b0268_val", group, "oktas_within1", o)
        add("k1_d0", 1, stage, "source_val", "all", "mean_etage_f1", f_e)
        add("k1_d0", 1, stage, "source_val", "all", "mean_genus_f1", f_g)
        add("k1_d0", 1, stage, "source_val", "montenegro", "c2_etage_ratio", 1.0)
        add("k0_d0", 3, stage, "source_val", "all", "mean_etage_f1", f_e)
        add("k0_d0", 3, stage, "source_val", "all", "mean_genus_f1", 0.70 if stage == "zero_shot" else 0.69)
    out = mod.results_from_rows(pd.DataFrame(rows), "t", 0.02)
    assert [r["draw"] for r in out] == ["k1_d0", "k0_d0"]          # the final run comes last
    r = out[0]
    assert r["verdict"] == "source regression" and r["n_val"] == 10 and r["train_days"] == "?"
    assert list(r["after_b"].day) == ["all", "2026-10-19"]
    assert r["after_b"].loc[0, "oktas_within1"] == 0.9 and r["before_b"].loc[0, "etage_set_agreement"] == 0.1
    assert r["after_s"]["montenegro/c2_etage_ratio"] == 1.0
    final = out[1]
    assert final["verdict"] == "ok" and len(final["after_b"]) == 0 and final["n_val"] == 0
    table = mod.sweep_table(pd.DataFrame(rows), 0.02)           # the final run is not one of the sweep's draws
    assert table[0].startswith("## Configurations") and "| `t` | 1 |" in table[4]
