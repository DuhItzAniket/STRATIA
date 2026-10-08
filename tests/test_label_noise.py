import numpy as np
import pandas as pd

from stratia.labels import noise as ln


def _data(n_per_class: int = 120, noise: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=np.float32)
    X, y_true = [], []
    for c in range(3):
        X.append(centres[c] + rng.normal(0, 1, size=(n_per_class, 3)).astype(np.float32))
        y_true += [c] * n_per_class
    X, y_true = np.vstack(X), np.array(y_true)
    y = y_true.copy()
    flipped = rng.choice(len(y), noise, replace=False)
    y[flipped] = (y[flipped] + 1) % 3                              # planted wrong labels
    units = np.arange(len(y)) // 2                                 # pairs of near-duplicates share a unit
    return X, np.array([f"c{v}" for v in y]), flipped, units


def test_confident_flags_recover_planted_noise():
    X, y, flipped, units = _data()
    probs, classes = ln.out_of_fold_probs(X, y, units, k=5)
    assert probs.shape == (360, 3) and classes.tolist() == ["c0", "c1", "c2"]
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    flags = ln.confident_flags(probs, y, classes)
    assert flags.flagged.sum() >= 8 and flags.flagged.sum() <= 40
    recovered = flags.flagged.values[flipped].mean()
    assert recovered >= 0.6                                         # most planted flips are flagged
    assert flags.flagged.values[np.setdiff1d(np.arange(360), flipped)].mean() < 0.1
    sugg = flags[flags.flagged]
    assert (sugg.suggested != sugg.given).all() and (sugg.margin > 0).all() and (sugg.p_suggested >= sugg.p_given).all()
    s = ln.summary(flags)
    assert s["images"] == 360 and s["flagged"] == int(flags.flagged.sum()) and len(s["top_pairs"]) >= 1
    empty = ln.summary(flags.assign(flagged=False))
    assert empty["flagged"] == 0 and np.isnan(empty["margin_median"])


def test_report_and_contact_sheet(tmp_path):
    from PIL import Image

    from stratia.data.image_cache import cache_path, write_cached

    files = ["a.png", "b.png"]
    for name in files:
        Image.fromarray(np.full((20, 20, 3), 90, np.uint8)).save(tmp_path / name)
        write_cached(tmp_path / name, cache_path(tmp_path / "cache", name))
    flags = pd.DataFrame({"given": ["x", "y"], "p_given": [0.2, 0.9], "suggested": ["y", None], "p_suggested": [0.7, np.nan],
                          "margin": [0.5, np.nan], "flagged": [True, False]})
    out = ln.contact_sheet(flags, files, tmp_path / "cache", tmp_path / "sheet.jpg", n=4, tile=30, columns=2)
    assert out.exists() and out.stat().st_size > 500
    text = ln.report_markdown({"demo": ln.summary(flags)}, ["decided"], {"demo": "figures/x.jpg"})
    assert "| demo | 2 | 1 | 50.0% | 0 | 0.0% | 0.50 |" in text and "| x | y | 1 |" in text and "- decided" in text
