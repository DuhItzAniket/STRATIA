"""P062: the geometry encoder fuses SkyRay tokens and FiLM with backbone tokens; known and null states share one graph."""

import numpy as np
import torch

from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation
from stratia.geometry.raymap import meta_vector, ray_map
from stratia.model.skyray import GeometryEncoder, zenith_cos

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.0, 0.0, 0.5, "zyx"), source="test")
D = 16


def _batch(n=2, size=512):
    rm = ray_map(OLDLR, UP, 200.0, 45.0, out_size=size)
    rms = torch.from_numpy(np.stack([rm, np.zeros_like(rm)]))[:n]
    metas = torch.from_numpy(np.stack([meta_vector(45.0), np.zeros(3, dtype=np.float32)]))[:n]
    g = torch.Generator().manual_seed(0)
    side = size // 16
    tokens = torch.randn(n, side * side, D, generator=g)
    cls = torch.randn(n, D, generator=g)
    return tokens, cls, rms, metas


def test_fusion_shapes_null_state_and_identity_at_init():
    tokens, cls, rms, metas = _batch()
    enc = GeometryEncoder(D, hidden=8)
    out_t, out_c = enc(tokens, cls, rms, metas)
    assert out_t.shape == tokens.shape and out_c.shape == cls.shape
    # null meta -> FiLM is the identity: the unknown sample's cls is untouched
    assert torch.allclose(out_c[1], cls[1])
    # null ray map -> every patch gets the same offset alpha * null_token
    offset = out_t[1] - tokens[1]
    assert torch.allclose(offset, offset[0].expand_as(offset), atol=1e-6)
    assert torch.allclose(offset[0], enc.alpha * enc.rays.null_token, atol=1e-6)
    # known ray map -> valid patches differ from the null offset, invalid patches do not
    valid = rms[0, 3].reshape(-1) > 0.5
    off0 = out_t[0] - tokens[0]
    assert torch.allclose(off0[~valid], offset[0].expand(int((~valid).sum()), D), atol=1e-6)
    assert not torch.allclose(off0[valid], offset[0].expand(int(valid.sum()), D))


def test_no_cross_talk_between_samples_and_gradients_reach_null_parameters():
    tokens, cls, rms, metas = _batch()
    enc = GeometryEncoder(D, hidden=8)
    with torch.no_grad():
        enc.film.mlp[-1].weight.normal_(std=0.1)       # make the FiLM non-trivial
        enc.film.null_gamma.normal_(std=0.1)
    joint_t, joint_c = enc(tokens, cls, rms, metas)
    for i in range(2):
        t_i, c_i = enc(tokens[i:i + 1], cls[i:i + 1], rms[i:i + 1], metas[i:i + 1])
        assert torch.allclose(joint_t[i], t_i[0], atol=1e-6) and torch.allclose(joint_c[i], c_i[0], atol=1e-6)
    loss = joint_t.pow(2).mean() + joint_c.pow(2).mean()
    loss.backward()
    for name in ("alpha", "rays.null_token", "film.null_gamma", "film.null_beta"):
        p = dict(enc.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_zenith_cos_reads_the_contract_frame():
    tokens, cls, rms, metas = _batch()
    cosz, valid = zenith_cos(rms)
    assert cosz.shape == (2, 32 * 32) and valid.shape == (2, 32 * 32)
    assert valid[0].sum() > 100 and not valid[1].any() and torch.all(cosz[1] == 0)
    assert cosz[0][valid[0]].max() > 0.99                       # a patch looks straight up on the zenith-pointing fisheye
    assert cosz[0][valid[0]].min() >= -1e-6                     # nothing below the horizon is valid


def test_traced_graph_serves_both_states():
    tokens, cls, rms, metas = _batch()
    enc = GeometryEncoder(D, hidden=8).eval()
    traced = torch.jit.trace(enc, (tokens[:1], cls[:1], rms[:1], metas[:1]))   # traced with geometry known
    with torch.no_grad():
        for i in range(2):                                                       # ... serves unknown geometry too
            eager = enc(tokens[i:i + 1], cls[i:i + 1], rms[i:i + 1], metas[i:i + 1])
            got = traced(tokens[i:i + 1], cls[i:i + 1], rms[i:i + 1], metas[i:i + 1])
            assert torch.allclose(eager[0], got[0], atol=1e-6) and torch.allclose(eager[1], got[1], atol=1e-6)
