"""P072: the engine's resume reproduces the uninterrupted run; accumulation, clipping, EMA, NaN guard and early
stopping behave as specified (CPU, a tiny model)."""

import numpy as np
import pytest
import torch
from torch import nn

from stratia.train.engine import Engine, EngineConfig

N, D = 64, 8


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, D, generator=g)
    w = torch.randn(D, 1, generator=g)
    y = x @ w + 0.1 * torch.randn(N, 1, generator=g)
    return x, y


def _loader_fn(x, y, batch=8, seed=0):
    def loader(epoch):
        g = torch.Generator().manual_seed(seed * 1000 + epoch)
        order = torch.randperm(len(x), generator=g)
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            yield x[idx], y[idx]
    return loader


def _step_fn(model, batch, step):
    xb, yb = batch
    pred = model(xb)
    loss = ((pred - yb) ** 2).mean()
    return loss, {"mse": loss.detach()}


def _eval_fn(x, y):
    def ev(model):
        with torch.no_grad():
            return {"val_loss": float(((model(x) - y) ** 2).mean())}
    return ev


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D, 16), nn.GELU(), nn.Dropout(0.1), nn.Linear(16, 1))


def _engine(model, cfg, ckpt=None):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    return Engine(model, opt, cfg, scheduler=sched, checkpoint_dir=ckpt)


class _Crash(Exception):
    pass


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    x, y = _data()
    cfg = EngineConfig(epochs=3, accumulate=2, ema_decay=0.9, checkpoint_every=3, log_every=0, device="cpu")
    # uninterrupted
    torch.manual_seed(42)
    eng_a = _engine(_model(1), cfg)
    res_a = eng_a.fit(_loader_fn(x, y), _step_fn, _eval_fn(x, y))
    params_a = torch.cat([p.detach().reshape(-1) for p in eng_a.model.parameters()])
    # interrupted after a checkpoint inside epoch 1, resumed in a 'fresh process'
    torch.manual_seed(42)
    eng_b = _engine(_model(1), cfg, ckpt=tmp_path)
    crash_after = {"n": 0}
    base_loader = _loader_fn(x, y)

    def crashing_loader(epoch):
        for i, b in enumerate(base_loader(epoch)):
            if epoch == 1 and i == 5:
                raise _Crash
            yield b
    with pytest.raises(_Crash):
        eng_b.fit(crashing_loader, _step_fn, _eval_fn(x, y))
    assert (tmp_path / "last.pt").exists() and eng_b.step > 0
    eng_c = _engine(_model(99), cfg, ckpt=tmp_path)             # different init: everything comes from the checkpoint
    res_c = eng_c.fit(_loader_fn(x, y), _step_fn, _eval_fn(x, y), resume=tmp_path / "last.pt")
    params_c = torch.cat([p.detach().reshape(-1) for p in eng_c.model.parameters()])
    assert res_c["steps"] == res_a["steps"] and res_c["epochs"] == 3
    assert torch.equal(params_a, params_c)
    assert [h["val_loss"] for h in res_a["history"]] == [h["val_loss"] for h in res_c["history"]]
    assert torch.equal(torch.cat([v.reshape(-1) for v in eng_a.ema.shadow.values()]),
                       torch.cat([v.reshape(-1) for v in eng_c.ema.shadow.values()]))
    del crash_after


def test_accumulation_matches_a_bigger_batch_and_clipping_bounds_the_norm():
    x, y = _data()
    xb, yb = x[:16], y[:16]
    cfg1 = EngineConfig(epochs=1, accumulate=1, clip_norm=None, ema_decay=None, log_every=0)
    cfg2 = EngineConfig(epochs=1, accumulate=2, clip_norm=None, ema_decay=None, log_every=0)
    m1, m2 = _model(3).eval(), _model(3).eval()                 # eval: no dropout, so the two paths are identical
    e1 = Engine(m1, torch.optim.SGD(m1.parameters(), lr=0.1), cfg1)
    e2 = Engine(m2, torch.optim.SGD(m2.parameters(), lr=0.1), cfg2)
    e1.model.eval()
    assert e1.train_step((xb, yb), lambda m, b, s: (((m.eval()(b[0]) - b[1]) ** 2).mean(), {})) is not None
    assert e2.train_step((xb[:8], yb[:8]), lambda m, b, s: (((m.eval()(b[0]) - b[1]) ** 2).mean(), {})) is None
    assert e2.train_step((xb[8:], yb[8:]), lambda m, b, s: (((m.eval()(b[0]) - b[1]) ** 2).mean(), {})) is not None
    p1 = torch.cat([p.detach().reshape(-1) for p in m1.parameters()])
    p2 = torch.cat([p.detach().reshape(-1) for p in m2.parameters()])
    assert torch.allclose(p1, p2, atol=1e-6)
    # clipping: the reported norm is the pre-clip norm, the applied update is bounded
    m3 = _model(3)
    e3 = Engine(m3, torch.optim.SGD(m3.parameters(), lr=1.0), EngineConfig(clip_norm=0.01, ema_decay=None, log_every=0))
    before = torch.cat([p.detach().reshape(-1) for p in m3.parameters()]).clone()
    out = e3.train_step((xb * 100, yb * 100), _step_fn)
    after = torch.cat([p.detach().reshape(-1) for p in m3.parameters()])
    assert out["grad_norm"] > 0.01 and float((after - before).norm()) <= 0.01 + 1e-6


def test_nan_guard_and_early_stopping(tmp_path):
    x, y = _data()
    m = _model(4)
    eng = Engine(m, torch.optim.SGD(m.parameters(), lr=0.1), EngineConfig(max_nan_steps=2, ema_decay=None, log_every=0))
    bad = lambda model, b, s: (torch.tensor(float("nan"), requires_grad=True), {})  # noqa: E731
    assert eng.train_step((x[:8], y[:8]), bad) is None and eng.nan_total == 1
    assert eng.train_step((x[:8], y[:8]), bad) is None and eng.nan_total == 2
    with pytest.raises(FloatingPointError):
        eng.train_step((x[:8], y[:8]), bad)
    assert eng.train_step((x[:8], y[:8]), _step_fn) is not None    # a good step afterwards resets the streak
    assert eng.nan_streak == 0
    # early stopping: a metric that never improves stops after `patience` evaluations
    m2 = _model(5)
    eng2 = Engine(m2, torch.optim.SGD(m2.parameters(), lr=0.0), EngineConfig(epochs=10, patience=2, ema_decay=None,
                                                                              log_every=0), checkpoint_dir=tmp_path / "es")
    res = eng2.fit(_loader_fn(x, y), _step_fn, _eval_fn(x, y))
    assert res["stopped_early"] and res["epochs"] == 3 and (tmp_path / "es" / "best.pt").exists()
    assert np.isclose(res["best"], res["history"][0]["val_loss"])
