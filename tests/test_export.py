"""P096: the ONNX export keeps the contract (names, shapes, probabilities), matches PyTorch, folds the calibration,
pools the ray map faithfully and ships a valid model card. Uses the small untrained dummy graph (no backbone weights)."""

import json

import numpy as np
import pytest
import torch

from stratia import contract as C
from stratia.export import onnx_model as X
from stratia.geometry.cameras import nominal_fisheye
from stratia.geometry.pose import Pose
from stratia.geometry.raymap import ray_map

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def dummy_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    model = X.dummy_model()
    card = X.model_card("0.0.0-test", {"patch_size": C.PATCH_SIZE, "mean": [0, 0, 0], "std": [1, 1, 1]}, None, dummy=True)
    path = X.export_onnx(model, out / "stratia_dummy.onnx", metadata={X.METADATA_KEY: json.dumps(card)})
    return model, path, card


@pytest.mark.slow
def test_exported_graph_keeps_the_contract_and_matches_torch(dummy_export):
    model, path, _ = dummy_export
    session = X.onnx_session(path)
    assert [i.name for i in session.get_inputs()] == list(C.INPUT_NAMES)
    assert [o.name for o in session.get_outputs()] == list(C.OUTPUT_NAMES)
    for n in (1, 3):
        ex = X.example_inputs(n, seed=n)
        out = X.run_onnx(session, ex)
        assert X.check_contract(out, n) == []
        report = X.parity(model, session, ex)
        assert max(v["max_abs"] for v in report.values()) < 1e-4, report
    # the geometry changes the outputs of the heads that read it, and an all-zero geometry is a valid input
    ex = X.example_inputs(2, seed=5)
    with_geo = X.run_onnx(session, ex)
    ex["ray_map"][:] = 0.0
    ex["meta"][:] = 0.0
    without = X.run_onnx(session, ex)
    assert np.isfinite(without["cbh_probs"]).all()
    assert not np.allclose(with_geo["genus_logits"][0], without["genus_logits"][0])


@pytest.mark.slow
def test_model_card_is_embedded_in_the_onnx_metadata(dummy_export):
    import onnx

    _, path, card = dummy_export
    m = onnx.load(str(path))
    props = {p.key: p.value for p in m.metadata_props}
    assert json.loads(props[X.METADATA_KEY]) == card


def test_temperatures_are_folded_into_the_graph():
    hot = X.dummy_model()
    cold = X.dummy_model()
    with torch.no_grad():
        cold.t_genus.fill_(1.0)
        cold.t_cbh.fill_(1.0)
    ex = X.example_inputs(2, seed=2)
    a, b = X.run_torch(hot, ex), X.run_torch(cold, ex)
    assert np.allclose(a["genus_logits"] * 2.0, b["genus_logits"], atol=1e-5)      # dummy temperature 2
    assert np.allclose(a["etage_logits"], b["etage_logits"], atol=1e-6)            # temperature 1
    assert not np.allclose(a["cbh_probs"], b["cbh_probs"])                          # dummy temperature 1.5
    assert np.allclose(a["ood_score"], b["ood_score"], atol=1e-6)                   # fitted on raw logits


def test_pooled_ray_map_matches_a_ray_map_computed_at_the_coarse_patch():
    camera = nominal_fisheye(4656, 3496, 105.0, "b0268", "nominal")
    pose = Pose(R=np.eye(3), source="declared")
    fine = ray_map(camera, pose, 135.0, 45.0, out_size=512, patch=C.PATCH_SIZE)
    coarse = ray_map(camera, pose, 135.0, 45.0, out_size=512, patch=C.PATCH_SIZE * X.GRID_POOL)
    pooled = X.pool_ray_map(torch.from_numpy(fine)[None])[0].numpy()
    assert pooled.shape == coarse.shape == (4, 8, 8)
    both = (coarse[3] > 0.5) & (pooled[3] > 0.5)
    assert both.sum() >= 0.9 * (coarse[3] > 0.5).sum()
    angles = np.degrees(np.arccos(np.clip((coarse[:3] * pooled[:3]).sum(0)[both], -1, 1)))
    assert angles.max() < 1.0, angles.max()
    assert np.allclose(np.linalg.norm(pooled[:3], axis=0)[pooled[3] > 0.5], 1.0, atol=1e-5)
    assert (pooled[:3][:, pooled[3] < 0.5] == 0).all()
    zero = X.pool_ray_map(torch.zeros(1, 4, 32, 32))
    assert (zero == 0).all()


def test_model_card_validates_against_the_schema_and_rejects_a_wrong_contract():
    card = X.model_card("0.0.0-test", {"patch_size": 16, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                        None, dummy=True)
    assert X.validate_model_card(card) == []
    assert card["calibration"]["folded_into_graph"] is True
    assert len(card["cbh_bin_edges_m"]) == 25 and card["cbh_bin_edges_m"][-1] == 12000.0
    bad = dict(card, contract_version="2.0", genus_classes=list(reversed(C.GENUS_CLASSES)))
    problems = X.validate_model_card(bad)
    assert any("contract_version" in p for p in problems) and any("genus_classes" in p for p in problems)
