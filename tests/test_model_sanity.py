"""P071: the model sanity suite on a tiny fake backbone (CPU, seconds): contract shapes, overfit one batch,
shuffled labels at chance, constant input gives constant output, determinism, both geometry states."""

import numpy as np
import torch

from stratia import contract
from stratia.model import cbh_bins as B
from stratia.model import losses as L
from stratia.model.sanity import (
    chance_level_genus,
    constant_input_check,
    determinism_check,
    overfit_one_batch,
    shuffled_labels_check,
    toy_batch,
)
from stratia.model.stratia import STRATIA, FakeBackbone

D = 32


def _model(seed=0):
    torch.manual_seed(seed)
    return STRATIA(FakeBackbone(D), d_model=D, global_hidden=48)


def test_contract_shapes_and_both_geometry_states():
    model = _model().eval()
    batch = toy_batch(n=2, size=128, seed=0)
    out = model(batch["image"], batch["ray_map"], batch["meta"])
    shapes = contract.output_shapes(2, 128, 128, k_cbh=B.K_BINS)
    for name in contract.OUTPUT_NAMES:
        assert tuple(out[name].shape) == shapes[name], (name, out[name].shape, shapes[name])
    sub = model.outputs(out)
    assert list(sub) == list(contract.OUTPUT_NAMES)
    assert torch.allclose(out["cbh_probs"].sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(out["oktas_probs"].sum(1), torch.ones(2), atol=1e-5)
    null = model(batch["image"], torch.zeros_like(batch["ray_map"]), torch.zeros_like(batch["meta"]))
    assert all(torch.isfinite(null[k]).all() for k in contract.OUTPUT_NAMES)
    assert not torch.allclose(null["cbh_probs"], out["cbh_probs"])        # geometry changes the CBH head's view


def test_overfit_one_batch_reaches_near_zero_loss():
    model = _model()
    batch = toy_batch(n=4, size=128, seed=1)
    hist = overfit_one_batch(model, batch, steps=250, lr=5e-3)
    assert hist[-1] < 0.1 * hist[0], (hist[0], hist[-1])          # a 10x drop proves the pipeline; 20x is BLAS-dependent


def test_shuffled_labels_stay_at_chance_and_constant_input_is_constant():
    model = _model(seed=2)
    batch = toy_batch(n=64, size=128, seed=2)
    acc = shuffled_labels_check(model, batch, steps=30, lr=1e-3)
    assert acc < chance_level_genus() + 0.25                      # 32 held-out images: no memorising a shuffled set
    spread = constant_input_check(_model(seed=3), size=128)
    assert spread < 1e-5


def test_determinism_same_seed_same_outputs():
    batch = toy_batch(n=4, size=128, seed=4)
    a = determinism_check(lambda: _model(seed=5), batch, steps=5)
    b = determinism_check(lambda: _model(seed=5), batch, steps=5)
    assert np.allclose(a, b, atol=0.0)
    c = determinism_check(lambda: _model(seed=6), batch, steps=5)
    assert not np.allclose(a, c)


def test_losses_run_on_the_model_outputs_with_partial_masks():
    model = _model(seed=7)
    batch = toy_batch(n=6, size=128, seed=7)
    out = model(batch["image"], batch["ray_map"], batch["meta"])
    genus = L.set_valued_bce(out["genus_logits"], batch["genus_set"], batch["genus_alt"], batch["mask_genus"], 0.3)
    etage = L.soft_bce(out["etage_logits"], batch["etage"], batch["mask_etage"])
    oktas = L.coral_loss(out["oktas_cum_logits"], batch["oktas"], batch["mask_oktas"])
    sky = L.dense_ce(out["sky_parse_logits"], batch["sky_mask"])
    cbh, _ = L.cbh_loss(out["cbh_logits"], batch["height_m"], batch["mask_cbh"])
    cons = L.cbh_etage_consistency(out["cbh_probs"], out["etage_logits"], batch["mask_cbh"])
    total = genus + etage + oktas + sky + cbh + 0.1 * cons
    total.backward()
    assert torch.isfinite(total)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
