"""Data-free tests of the P090 pieces: the cluster bootstrap, the calibrated CBH coverage, the headline table and
the refusal to read the locked rows without --final."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("locked_test", ROOT / "scripts" / "locked_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cluster_bootstrap_resamples_units_not_rows():
    mod = _load()
    values = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    units = np.array(["a", "a", "b", "b", "c", "c"])
    seen = []

    def fn(idx):
        seen.append(idx)
        return {"mean": float(values[idx].mean()), "skip": float("nan")}

    ci = mod.cluster_bootstrap(units, fn, 40, seed=1)
    assert set(ci) == {"mean"}                                   # nan-valued metrics get no interval
    lo, hi = ci["mean"]
    assert 0.0 <= lo <= hi <= 1.0
    for idx in seen:                                             # whole units: rows come in pairs
        assert len(idx) == 6 and all(np.sum(units[idx] == u) % 2 == 0 for u in "abc")
    assert mod.cluster_bootstrap(units, fn, 40, seed=1) == ci    # deterministic


def test_calibrated_coverage_is_one_when_mass_sits_on_the_true_bin():
    mod = _load()
    B = mod.B
    heights = np.array([150.0, 2500.0, 7000.0, np.nan])
    probs = np.full((4, B.K_BINS + 1), 1e-6)
    for i, h in enumerate(heights[:3]):
        probs[i, B.bin_of(np.array([h]))[0]] = 1.0
    probs /= probs.sum(1, keepdims=True)
    assert mod.calibrated_coverage(probs, heights, temperature=1.0, radius=0.4) == 1.0
    assert mod.calibrated_coverage(probs, heights, temperature=1.2394, radius=0.3265) == 1.0
    flat = np.full((4, B.K_BINS + 1), 1.0 / (B.K_BINS + 1))
    cov = mod.calibrated_coverage(flat, heights, temperature=1.0, radius=0.05)
    assert 0.0 <= cov < 1.0                                      # a narrow interval of a flat predictive misses some


def test_records_and_headline_table(tmp_path, monkeypatch):
    mod = _load()
    rec = mod.Records()
    rec.add("lodo", "ccsn", "lodo_ccsn_e2s", "etage_f1", 0.6, (0.5, 0.7), n=10)
    rec.add("lodo", "ccsn", mod.PROBE, "etage_f1", 0.5, n=10)
    for f in ("mgcd", "montenegro", "almeria"):
        rec.add("lodo", f, f"lodo_{f}_e2s", "etage_f1", 0.6, n=10)
        rec.add("lodo", f, mod.PROBE, "etage_f1", 0.5, n=10)
    for s in ("ccsn", "mgcd"):
        rec.add("in_domain", s, "indomain_e2s_s0", "genus_f1", 0.7, n=10)
        rec.add("in_domain", s, mod.PROBE_IND, "genus_f1", 0.72, n=10)
    rec.add("in_domain", "montenegro", "indomain_e2s_s0", "c2_etage_ratio", 1.0, (0.9, 1.1), n=10)
    rec.add("in_domain", "montenegro", "indomain_e2s_s0", "c2_oktas_ratio", 0.85, n=10)
    rec.add("cbh_pairs", "OLDLR", "cbh_base_s0", "skill", 0.4, (0.3, 0.5), n=10)
    rec.add("cbh_pairs", "WESTE", "cbh_oldlr_only_s0", "skill", 0.05, n=10)
    rec.add("cbh_pairs", "all", "cbh_base_s0", "mae_m", 600.0, (550.0, 650.0), n=10)
    rec.add("cbh_pairs", "all", mod.RIDGE, "mae_m", 700.0, n=10)
    rec.add("cbh_pairs", "all", "cbh_base_s0", "coverage_80_calibrated", 0.79, n=10)
    df = rec.frame()
    assert list(df.columns) == ["protocol", "set", "model", "metric", "value", "ci_lo", "ci_hi", "n", "note"]
    assert np.isnan(df.ci_lo.iloc[1]) and df.n.iloc[0] == 10
    monkeypatch.chdir(tmp_path)                                  # no data/gate_g4.parquet: the val column is NaN
    head = mod.headline(df)
    by = {(r["criterion"], r["quantity"][:12]): r for r in head}
    assert by[("C1", "LODO étage m")]["pass"] is True            # 0.6 vs 0.5: +0.10 >= +0.05
    assert by[("C1", "in-domain ge")]["pass"] is False           # 0.70 vs 0.72: below −0.01
    assert by[("C2", "Montenegro é")]["pass"] is True and by[("C2", "Montenegro o")]["pass"] is False
    assert by[("C3", "CBH skill, O")]["pass"] is True and by[("C3", "CBH skill, W")]["pass"] is False
    assert by[("C3", "CBH MAE (m),")]["pass"] is True            # 600 <= 0.9 * 700
    assert by[("C3", "80 % interva")]["pass"] is True
    assert by[("C3", "CBH MAE (m),")]["ci"] == "[550, 650]"
    assert all(np.isnan(r["val_p082"]) for r in head)
    mod.DOC = tmp_path / "locked_test.md"
    mod.write_doc(df, head, {"lock_prefix": "abc", "run_id": "r", "git_sha": "g", "n_boot": 3}, "val")
    text = mod.DOC.read_text(encoding="utf-8")
    assert "validation check" in text and "| C1 |" in text and "`lodo`" in text


def test_locked_rows_need_final_and_a_reason():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "locked_test.py"), "--split", "test"], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 2 and "--final" in out.stderr
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "locked_test.py"), "--split", "val", "--final"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 2
