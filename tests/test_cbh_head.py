"""P067: the CBH head's zenith prior concentrates attention near the zenith when geometry is known and degrades
to content attention when it is not; the output is a distribution over the contract's 25 classes."""

import numpy as np
import torch

from stratia import contract
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.geometry.raymap import ray_map
from stratia.model import cbh_bins as B
from stratia.model import losses as L
from stratia.model.cbh import CbhHead
from stratia.model.skyray import zenith_cos

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.0, 0.0, 0.5, "zyx"), source="test")
D = 16


def test_shapes_prior_and_null_fallback():
    rm = ray_map(OLDLR, UP, 200.0, 45.0, out_size=512, patch=64)        # 8 x 8 grid
    rms = torch.from_numpy(np.stack([rm, np.zeros_like(rm)]))
    tokens = torch.zeros(2, 64, D)                                     # no content signal: attention = prior only
    cls = torch.zeros(2, D)
    head = CbhHead(D, hidden=16)
    out = head(tokens, cls, rms)
    assert tuple(out["cbh_probs"].shape) == contract.output_shapes(2, k_cbh=B.K_BINS)["cbh_probs"]
    assert torch.allclose(out["cbh_probs"].sum(1), torch.ones(2), atol=1e-5)
    w = out["cbh_attention"]
    assert w.shape == (2, 64) and torch.allclose(w.sum(1), torch.ones(2), atol=1e-5)
    cosz, valid = zenith_cos(rms[:1])
    zen = int(cosz[0].argmax())
    assert w[0].argmax() == zen and w[0, zen] > 2 * w[0].mean()        # known geometry: mass on the zenith patch
    assert torch.allclose(w[1], torch.full((64,), 1 / 64), atol=1e-6)   # null geometry: uniform (no content)


def test_head_trains_toward_a_target_and_learns_sigma():
    torch.manual_seed(0)
    rm = ray_map(OLDLR, UP, 200.0, 45.0, out_size=512, patch=64)
    rms = torch.from_numpy(np.stack([rm] * 4))
    tokens = torch.randn(4, 64, D)
    cls = torch.randn(4, D)
    heights = torch.tensor([300.0, 1500.0, float("nan"), 8000.0])
    head = CbhHead(D, hidden=32)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    sigma0 = head.log_sigma.item()
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss, _ = L.cbh_loss(head(tokens, cls, rms)["cbh_logits"], heights, torch.ones(4, dtype=torch.bool))
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.5 * first
    probs = head(tokens, cls, rms)["cbh_probs"].detach().numpy()
    pred = B.median_height_m(probs)
    assert np.isnan(pred[2]) and 200 < pred[0] < 500 and 1000 < pred[1] < 2200 and pred[3] > 6000
    assert head.log_sigma.item() != sigma0                             # the prior width is trained
