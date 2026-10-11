"""P064 / P067: toy cases with known gradient directions for every loss of ADR-006."""

import numpy as np
import torch

from stratia.contract import ETAGE_CLASSES, GENUS_CLASSES
from stratia.model import cbh_bins as B
from stratia.model import losses as L

C = len(GENUS_CLASSES)


def _grad(fn, logits):
    x = logits.clone().requires_grad_(True)
    fn(x).backward()
    return x.grad


def test_set_valued_bce_moves_mass_into_the_set_without_choosing_a_member():
    logits = torch.zeros(1, C)
    s = torch.zeros(1, C)
    s[0, GENUS_CLASSES.index("Ac")] = s[0, GENUS_CLASSES.index("Cc")] = 1          # MGCD "altocumulus" = Ac or Cc
    alt = torch.tensor([True])
    mask = torch.tensor([True])
    g = _grad(lambda x: L.set_valued_bce(x, s, alt, mask, negative_weight=0.3), logits)
    ac, cc, cu = (GENUS_CLASSES.index(k) for k in ("Ac", "Cc", "Cu"))
    assert g[0, ac] < 0 and g[0, cc] < 0 and abs(g[0, ac] - g[0, cc]) < 1e-6    # both pushed up, equally
    assert g[0, cu] > 0                                                          # a non-member pushed down
    # a set already satisfied (one member at probability ~1) gives no positive gradient
    sat = torch.zeros(1, C)
    sat[0, ac] = 30.0
    g2 = _grad(lambda x: L.set_valued_bce(x, s, alt, mask, negative_weight=0.0), sat)
    assert abs(g2[0, ac]) < 1e-6 and abs(g2[0, cc]) < 1e-6
    # co-present set: each member is a positive on its own
    g3 = _grad(lambda x: L.set_valued_bce(x, s, torch.tensor([False]), mask, negative_weight=0.0), sat)
    assert abs(g3[0, ac]) < 1e-6 and g3[0, cc] < -0.1
    # empty mask: zero loss, zero gradient, no NaN
    g4 = _grad(lambda x: L.set_valued_bce(x, s, alt, torch.tensor([False]), 0.3), logits)
    assert torch.all(g4 == 0)
    assert float(L.set_valued_bce(logits, s, alt, torch.tensor([False]), 0.3)) == 0.0


def test_soft_bce_and_hierarchy_consistency():
    logits = torch.zeros(2, 3)
    target = torch.tensor([[0.9, 0.1, 0.0], [0.5, 0.5, 0.5]])
    g = _grad(lambda x: L.soft_bce(x, target, torch.tensor([True, True])), logits)
    assert g[0, 0] < 0 and g[0, 2] > 0 and abs(g[1, 0]) < 1e-6          # toward 0.9, toward 0, flat at 0.5
    m = L.genus_etage_matrix()
    assert m.shape == (C, 3) and m[GENUS_CLASSES.index("Ci"), ETAGE_CLASSES.index("high")] == 1
    assert m[GENUS_CLASSES.index("clear")].sum() == 0 and m.sum() == 10
    genus = torch.full((1, C), -10.0)
    genus[0, GENUS_CLASSES.index("Cu")] = 10.0                           # Cu certain -> low étage implied ~1
    etage = torch.zeros(1, 3)                                            # étage head says 0.5 everywhere
    gg = genus.clone().requires_grad_(True)
    ee = etage.clone().requires_grad_(True)
    L.hierarchy_consistency(gg, ee, torch.tensor([True])).backward()
    assert ee.grad[0, 0] < 0 and ee.grad[0, 1] > 0 and ee.grad[0, 2] > 0   # low pushed up, mid and high down (implied 0)
    assert gg.grad is not None and torch.isfinite(gg.grad).all()


def test_coral_targets_and_loss_rank_behaviour():
    hard = torch.tensor([0, 3, 8])
    t = L.cumulative_targets(hard, 9)
    assert t.shape == (3, 8) and t[0].sum() == 0 and t[1].sum() == 3 and t[2].sum() == 8
    dist = torch.zeros(1, 9)
    dist[0, 4] = dist[0, 5] = 0.5
    ts = L.cumulative_targets(dist, 9)
    assert torch.allclose(ts[0], torch.tensor([1, 1, 1, 1, 0.5, 0, 0, 0.0]))
    cum = torch.zeros(1, 8)
    g = _grad(lambda x: L.coral_loss(x, torch.tensor([6]), torch.tensor([True])), cum)
    assert (g[0, :6] < 0).all() and (g[0, 6:] > 0).all()                   # thresholds below 6 up, above down
    assert float(L.kl_to_distribution(dist, dist, torch.tensor([True]))) < 1e-6


def test_dense_ce_ignores_255_and_upsamples():
    logits = torch.zeros(1, 4, 4, 4)
    target = torch.full((1, 8, 8), 255, dtype=torch.long)
    assert float(L.dense_ce(logits, target)) == 0.0
    target[0, :4, :4] = 2
    g = _grad(lambda x: L.dense_ce(x, target), logits)
    assert g[0, 2].sum() < 0 and g[0, 1].sum() > 0 and torch.isfinite(g).all()


def test_cbh_bins_are_etage_aligned_and_targets_are_sound():
    assert B.K_BINS == 24 and B.N_CLASSES == 25 and len(B.EDGES_M) == 25
    assert B.EDGES_M[0] == 0 and B.EDGES_M[10] == 2000 and B.EDGES_M[18] == 6000 and B.EDGES_M[-1] == 12000
    assert np.all(np.diff(B.EDGES_M) > 0) and list(np.bincount(B.ETAGE_OF_BIN)) == [10, 8, 6]
    assert B.bin_of(np.array([0.0, 99.0, 1999.0, 2000.0, 11999.0, 20000.0])).tolist() == [0, 0, 9, 10, 23, 23]
    assert B.bin_of(np.array([np.nan]))[0] == B.NO_CLOUD
    t = B.soft_target(np.array([1500.0, np.nan]))
    assert t.shape == (2, 25) and abs(t.sum(1) - 1).max() < 1e-5 and t[1, B.NO_CLOUD] == 1 and t[0, B.NO_CLOUD] == 0
    assert t[0].argmax() == B.bin_of(np.array([1500.0]))[0]
    med = B.median_height_m(t)
    assert 1200 < med[0] < 1900 and np.isnan(med[1])
    lo, hi = B.interval_m(t[:1])
    assert lo[0] < med[0] < hi[0]


def test_cbh_loss_prefers_mass_near_the_truth_and_consistency_pulls_toward_the_etage_head():
    h = torch.tensor([1500.0, float("nan")])
    mask = torch.tensor([True, True])
    near = torch.full((2, 25), -5.0)
    far = near.clone()
    near[0, B.bin_of(np.array([1300.0]))[0]] = 5.0                      # one bin below the truth
    far[0, B.bin_of(np.array([8000.0]))[0]] = 5.0                       # high étage
    near[1, B.NO_CLOUD] = 5.0
    far[1, 0] = 5.0
    l_near, parts = L.cbh_loss(near, h, mask)
    l_far, _ = L.cbh_loss(far, h, mask)
    assert l_near < l_far and parts["crps"] >= 0 and torch.isfinite(l_near)
    # CRPS alone: moving mass toward the truth lowers it even at equal CE (both wrong bins get ~0 target mass)
    t = L.cbh_soft_targets(h)[:1]
    q1 = torch.zeros(1, 25)
    q1[0, 5] = 1.0                                                      # a bin far below
    q2 = torch.zeros(1, 25)
    q2[0, B.bin_of(np.array([1300.0]))[0]] = 1.0
    assert L.crps_ordered(q2[:, :24], t[:, :24]) < L.crps_ordered(q1[:, :24], t[:, :24])
    # consistency: the étage head says "low"; CBH mass on high bins is pulled down
    etage = torch.tensor([[6.0, -6.0, -6.0]])
    logits = torch.zeros(1, 25, requires_grad=True)
    L.cbh_etage_consistency(torch.softmax(logits, 1), etage, torch.tensor([True])).backward()
    g = logits.grad[0]
    assert g[:10].mean() < 0 and g[18:24].mean() > 0 and g[B.NO_CLOUD] > 0
    implied = L.cbh_implied_etage(torch.softmax(torch.zeros(1, 25), 1))
    assert torch.allclose(implied.sum(1), torch.ones(1)) and abs(float(implied[0, 0]) - 10 / 25) < 1e-6
