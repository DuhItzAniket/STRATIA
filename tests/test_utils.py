import csv
import json

import pytest
import torch
from omegaconf import OmegaConf

from stratia.utils.config import config_hash, load_config
from stratia.utils.run import Run
from stratia.utils.seed import set_seed


# ---------------------------------------------------------------- config
def test_config_hash_ignores_key_order():
    a = OmegaConf.create({"lr": 0.1, "model": {"depth": 2, "width": 8}})
    b = OmegaConf.create({"model": {"width": 8, "depth": 2}, "lr": 0.1})
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_values():
    assert config_hash(OmegaConf.create({"lr": 0.1})) != config_hash(OmegaConf.create({"lr": 0.2}))


def test_defaults_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("lr: 0.1\nepochs: 3\nmodel: {width: 8}\n")
    (tmp_path / "exp.yaml").write_text("defaults: [base.yaml]\nepochs: 5\n")
    cfg = load_config(tmp_path / "exp.yaml", ["model.width=16"])
    assert (cfg.lr, cfg.epochs, cfg.model.width) == (0.1, 5, 16)
    assert "defaults" not in cfg


# ---------------------------------------------------------------- seed
def test_seed_reproduces_cpu_random():
    set_seed(7)
    a = torch.randn(5)
    set_seed(7)
    assert torch.equal(a, torch.randn(5))


# ---------------------------------------------------------------- run bookkeeping
def test_run_writes_artifacts_and_registry(tmp_path):
    cfg = OmegaConf.create({"seed": 1, "lr": 0.01})
    with Run(cfg, name="unit", phase="P005", root=tmp_path, data_hash="abc") as run:
        run.log(0, loss=1.5)
        run.log(1, loss=1.0)
        run.finish(final_loss=1.0)
    files = {p.name for p in run.dir.iterdir()}
    assert {"config.yaml", "env.json", "metrics.jsonl", "summary.json"} <= files
    assert len((run.dir / "metrics.jsonl").read_text().splitlines()) == 2
    rows = list(csv.DictReader(open(tmp_path / "registry.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["status"] == "done" and rows[0]["data_hash"] == "abc"
    assert json.loads(rows[0]["metrics"]) == {"final_loss": 1.0}
    assert rows[0]["config_hash"] in rows[0]["run_id"]


def test_run_writes_tensorboard_events(tmp_path):
    pytest.importorskip("tensorboard")
    with Run(OmegaConf.create({"a": 1}), name="tb", phase="P005", root=tmp_path, tensorboard=True) as run:
        run.log(0, loss=2.0)
        run.log(1, loss=1.0)
    assert list((run.dir / "tb").glob("events.out.tfevents.*"))


def test_run_records_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with Run(OmegaConf.create({"x": 1}), name="boom", phase="P005", root=tmp_path):
            raise RuntimeError("simulated crash")
    rows = list(csv.DictReader(open(tmp_path / "registry.csv", encoding="utf-8")))
    assert rows[0]["status"] == "failed: RuntimeError"
