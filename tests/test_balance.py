"""P070: Kendall uncertainty weighting combines present tasks only; the gradient monitor reports conflicts."""

import torch
from torch import nn

from stratia.model.balance import GradConflictMonitor, UncertaintyWeighting


def test_uncertainty_weighting_skips_absent_tasks_and_learns_log_vars():
    uw = UncertaintyWeighting(["genus", "cbh"])
    x = torch.tensor(2.0, requires_grad=True)
    losses = {"genus": x * 1.0, "cbh": (x * 0.0)}               # cbh absent: exactly 0
    total, parts = uw(losses)
    assert torch.isclose(total, torch.tensor(2.0))            # exp(0) * 2 + 0
    assert float(parts["cbh"]) == 0.0
    # explicit presence flags override the heuristic
    total2, _ = uw({"genus": x * 1.0, "cbh": x * 3.0}, present={"genus": True, "cbh": False})
    assert torch.isclose(total2, torch.tensor(2.0))
    # a large loss drives its log-variance up (down-weighting it) when trained
    opt = torch.optim.SGD(uw.parameters(), lr=0.1)
    for _ in range(20):
        opt.zero_grad()
        t, _ = uw({"genus": torch.tensor(10.0, requires_grad=True), "cbh": torch.tensor(0.1, requires_grad=True)})
        t.backward()
        opt.step()
    w = uw.weights()
    assert w["genus"] < w["cbh"]
    assert uw.log_var["genus"].item() > uw.log_var["cbh"].item()


def test_grad_conflict_monitor_detects_opposed_tasks():
    torch.manual_seed(0)
    shared = nn.Linear(4, 4, bias=False)
    x = torch.randn(8, 4)
    h = shared(x)
    target = torch.randn(8, 4)
    losses = {"a": ((h - target) ** 2).mean(), "b": -((h - target) ** 2).mean(), "c": ((h - target) ** 2).mean() * 2}
    mon = GradConflictMonitor(list(shared.parameters()))
    cos = mon.cosines(losses)
    assert cos["a|c"] > 0.99                                   # same direction
    assert cos["a|b"] < -0.5                                   # opposed
    assert cos["a/norm"] > 0 and "b/norm" in cos
    assert set(mon.conflicts(cos)) == {"a|b", "b|c"}
    # the graph survives: the real backward still works afterwards
    sum(losses.values()).backward()
    assert shared.weight.grad is not None
