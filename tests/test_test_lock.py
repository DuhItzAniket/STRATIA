import json

import pandas as pd
import pytest

from stratia.data import test_lock as tl


def _splits(tmp_path):
    d = tmp_path / "splits"
    d.mkdir()
    pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "source": ["x", "x", "x", "y"],
                  "split": ["train", "test", "test", "test"]}).to_parquet(d / "in_domain.parquet", index=False)
    pd.DataFrame({"sample_id": ["a", "b", "d"], "fold": ["x", "x", "y"], "role": ["train", "test", "test"]}).to_parquet(
        d / "lodo.parquet", index=False)
    return d


def test_lock_is_built_verified_and_detects_tampering(tmp_path):
    d = _splits(tmp_path)
    lock = tl.build_lock(d, 1, 0, "abcd")
    assert set(lock["tests"]) == {"in_domain/x/test", "in_domain/y/test", "lodo/x/test", "lodo/y/test"}
    assert lock["tests"]["in_domain/x/test"]["n"] == 2 and tl.well_formed(lock) == [] and tl.verify_lock(lock, d) == []
    pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "source": ["x", "x", "x", "y"],
                  "split": ["train", "test", "train", "test"]}).to_parquet(d / "in_domain.parquet", index=False)
    problems = tl.verify_lock(lock, d)
    assert len(problems) == 1 and "in_domain/x/test" in problems[0]
    (d / "lodo.parquet").unlink()
    assert any("locked but not present" in p for p in tl.verify_lock(lock, d))
    bad = json.loads(json.dumps(lock))
    bad["tests"]["lodo/x/test"]["sha256"] = "zz"
    del bad["seed"]
    assert len(tl.well_formed(bad)) == 2


def test_final_reason_is_required_and_logged(tmp_path):
    lock = {"locked_at": "2026-10-08T00:00:00Z"}
    with pytest.raises(ValueError):
        tl.require_final_reason("", lock, tmp_path / "log.md")
    with pytest.raises(ValueError):
        tl.require_final_reason("short", lock, tmp_path / "log.md")
    p = tl.require_final_reason("paper table 2, final numbers after model selection on val", lock, tmp_path / "log.md", run="r1")
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# Final evaluations") and "| r1 | paper table 2" in text
    tl.require_final_reason("second final evaluation for the rebuttal", lock, p)
    assert p.read_text(encoding="utf-8").count("\n| 20") == 2


def test_training_configs_must_not_reference_test_sets(tmp_path):
    (tmp_path / "ok.yaml").write_text("train:\n  split: train\n  files: [data/splits/in_domain.parquet]\nval:\n  split: val\n",
                                      encoding="utf-8")
    (tmp_path / "bad.yaml").write_text("training:\n  data:\n    split: test\n    path: data/splits/lodo/test.parquet\n",
                                       encoding="utf-8")
    (tmp_path / "eval.yaml").write_text("eval:\n  split: test\n", encoding="utf-8")       # evaluation configs may name test
    (tmp_path / "broken.yaml").write_text("train: [unclosed\n", encoding="utf-8")
    problems = tl.scan_training_configs(tmp_path)
    assert any("bad.yaml: training.data.split" in p for p in problems)
    assert any("bad.yaml: training.data.path" in p for p in problems)
    assert any("broken.yaml" in p for p in problems) and not any("ok.yaml" in p or "eval.yaml" in p for p in problems)
