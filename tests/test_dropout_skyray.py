"""P050: geometry dropout produces the contract's null state and the encoder runs with and without metadata."""

import numpy as np
import torch

from stratia.augment.dropout import GeometryDropout, is_null_meta, is_null_ray_map
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.geometry.raymap import meta_vector, ray_map
from stratia.model.skyray import GeometryStub, MetaFiLM, SkyRayEncoder

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.0, 0.0, 0.5, "zyx"), source="test")


def _sample():
    return ray_map(OLDLR, UP, 200.0, 45.0), meta_vector(45.0)


def test_dropout_rates_and_null_state():
    rm, mt = _sample()
    d = GeometryDropout(p_ray=0.3, p_meta=0.3)
    rng = np.random.default_rng(0)
    n = 4000
    dropped_ray = dropped_meta = both = 0
    for _ in range(n):
        r, m, info = d(rm, mt, rng)
        dropped_ray += info["ray_dropped"]
        dropped_meta += info["meta_dropped"]
        both += info["ray_dropped"] and info["meta_dropped"]
        if info["ray_dropped"]:
            assert is_null_ray_map(r) and r.shape == rm.shape and r.dtype == rm.dtype
        else:
            assert np.array_equal(r, rm)
        if info["meta_dropped"]:
            assert is_null_meta(m)
        else:
            assert np.array_equal(m, mt)
    assert abs(dropped_ray / n - 0.3) < 0.03 and abs(dropped_meta / n - 0.3) < 0.03
    assert abs(both / n - 0.09) < 0.02                                   # independent draws
    tied = GeometryDropout(p_ray=0.5, tie=True)
    for _ in range(50):
        _, _, info = tied(rm, mt, rng)
        assert info["ray_dropped"] == info["meta_dropped"]


def test_batch_dropout():
    rm, mt = _sample()
    rms = np.stack([rm] * 64)
    mts = np.stack([mt] * 64)
    out_rm, out_mt, dr, dm = GeometryDropout(0.5, 0.5).batch(rms, mts, np.random.default_rng(1))
    assert out_rm.shape == rms.shape and dr.shape == (64,) and 10 < dr.sum() < 54
    assert np.all(out_rm[dr] == 0) and np.all(out_rm[~dr] == rm) and np.all(out_mt[dm] == 0)
    assert np.array_equal(rms[0], rm)                                   # inputs untouched


def test_encoder_null_tokens_and_film():
    rm, mt = _sample()
    enc = SkyRayEncoder(d_model=16, hidden=8)
    x = torch.from_numpy(np.stack([rm, np.zeros_like(rm)]))
    tokens = enc(x)
    assert tokens.shape == (2, 32 * 32, 16)
    assert torch.allclose(tokens[1], enc.null_token.expand(32 * 32, 16))  # all-zero map -> all null tokens
    valid = torch.from_numpy(rm[3]).reshape(-1) > 0.5
    assert torch.allclose(tokens[0][~valid], enc.null_token.expand(int((~valid).sum()), 16))
    assert not torch.allclose(tokens[0][valid], enc.null_token.expand(int(valid.sum()), 16))
    film = MetaFiLM(d_model=16, hidden=8)
    m = torch.from_numpy(np.stack([mt, np.zeros(3, dtype=np.float32)]))
    out = film(tokens, m)
    assert out.shape == tokens.shape
    assert torch.allclose(out[1], tokens[1])                            # null FiLM is the identity at init
    assert torch.allclose(out[0], tokens[0])                            # zero-initialised FiLM too


def test_stub_model_runs_with_and_without_metadata_and_trains_null_params():
    rm, mt = _sample()
    rms = np.stack([rm] * 8)
    mts = np.stack([mt] * 8)
    out_rm, out_mt, dr, dm = GeometryDropout(0.5, 0.5).batch(rms, mts, np.random.default_rng(2))
    model = GeometryStub(d_model=32)
    x = torch.from_numpy(out_rm)
    m = torch.from_numpy(out_mt)
    y = torch.linspace(0, 1, 8)[:, None]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.rays.null_token.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        loss = ((model(x, m) - y) ** 2).mean()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss) and not torch.allclose(model.rays.null_token, before)
    # the extremes: everything known, nothing known
    assert model(torch.from_numpy(rms), torch.from_numpy(mts)).shape == (8, 1)
    assert model(torch.zeros_like(x), torch.zeros_like(m)).shape == (8, 1)
