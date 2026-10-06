import numpy as np
import pandas as pd
from PIL import Image

from stratia.data import shortcuts as sc
from stratia.data.image_cache import cache_path, write_cached


def _frames(tmp_path, n: int = 6, size: int = 32) -> list[str]:
    """Frames with a black fixed corner and a changing sky, written through the image cache."""
    rng = np.random.default_rng(0)
    files = []
    for k in range(n):
        img = rng.integers(60, 200, size=(size, size, 3), dtype=np.uint8)
        img[:8, :8] = 0                                            # fixed corner
        name = f"f{k}.png"
        Image.fromarray(img).save(tmp_path / name)
        write_cached(tmp_path / name, cache_path(tmp_path / "cache", name))
        files.append(name)
    return files


def test_camera_statistics_and_fixed_mask_find_the_corner(tmp_path):
    files = _frames(tmp_path, n=24)
    mean, std = sc.camera_statistics(files, tmp_path / "cache", size=32, batch=8, workers=0)
    assert mean.shape == std.shape == (3, 32, 32)
    assert mean[:, :6, :6].max() < 12 and std[:, :6, :6].max() < 6          # the corner (JPEG cache: not exactly 0)
    assert mean[:, 12:, 12:].mean() > 100 and std[:, 12:, 12:].mean() > 20   # the changing sky
    mask = sc.fixed_mask(std, rel=0.5)
    assert mask[:6, :6].all() and mask[12:, 12:].mean() < 0.02 and mask.dtype == bool   # noise pixels rarely look fixed


def test_fixed_mask_covers_changing_digits_inside_the_fixed_surround():
    std = np.full((3, 40, 40), 30.0, dtype=np.float32)
    std[:, :12, :] = 0.0                                       # a fixed band at the top ...
    std[:, 3:6, 4:10] = 25.0                                   # ... with changing digits inside it
    mask = sc.fixed_mask(std)
    assert mask[3:6, 4:10].all() and mask[:12].all() and not mask[12:].any()


def test_variants_change_the_right_pixels(tmp_path):
    files = _frames(tmp_path, n=2)
    std = np.ones((3, 32, 32), dtype=np.float32)
    std[:, :8, :8] = 0
    masks = {"cam": sc.fixed_mask(std)}
    original = sc.VariantDataset(files, tmp_path / "cache", "original", size=32)[0][0].numpy()
    sky = sc.VariantDataset(files, tmp_path / "cache", "skyonly", size=32, masks=masks, cameras=["cam", "cam"])[0][0].numpy()
    fixed = sc.VariantDataset(files, tmp_path / "cache", "fixedonly", size=32, masks=masks, cameras=["cam", "cam"])[0][0].numpy()
    low = sc.VariantDataset(files, tmp_path / "cache", "lowres", size=32, low=8)[0][0].numpy()
    assert (sky[:, :8, :8] == sc.GREY).all() and (sky[:, 8:, 8:] == original[:, 8:, 8:]).all()
    assert (fixed[:, 8:, :] == sc.GREY).all() and (fixed[:, :8, :8] == original[:, :8, :8]).all()
    assert low.shape == original.shape and low.std() < original.std()      # blurred: less pixel-to-pixel variation
    try:
        sc.VariantDataset(files, tmp_path / "cache", "nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown variant accepted")


def test_holdouts_keep_blocks_and_groups_whole():
    days = pd.Series(["d1", "d1", "d2", "d3", "d3", "d4", "d5"])
    test = sc.holdout_last_blocks(days, frac=0.3)
    assert test.tolist() == [False, False, False, False, False, True, True]     # the last two of five days
    groups = pd.Series(["a", "a", "b", "c", "c", "d", "e", "f"])
    test = sc.holdout_random_groups(groups, frac=0.5, seed=1)
    assert test[0] == test[1] and test[3] == test[4] and 0 < test.sum() < len(groups)
    assert not sc.holdout_last_blocks(pd.Series(["only"] * 3)).any()


def test_balanced_subsample_caps_each_class():
    labels = np.array(["a"] * 10 + ["b"] * 3 + ["c"] * 5)
    idx = sc.balanced_subsample(labels, 4, np.random.default_rng(0))
    counts = pd.Series(labels[idx]).value_counts()
    assert counts["a"] == 4 and counts["b"] == 3 and counts["c"] == 4 and (np.diff(idx) > 0).all()


def test_linear_probe_separates_what_is_separable():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 60)
    X = rng.normal(size=(180, 8)).astype(np.float32)
    X[:, 0] += 6 * y                                                             # class is written in one feature
    test = np.arange(180) % 4 == 0
    r = sc.linear_probe(X, y, test)
    assert r["accuracy"] > 0.95 and r["balanced_accuracy"] > 0.95 and r["classes"] == 3
    assert abs(r["chance"] - 1 / 3) < 1e-9 and abs(r["majority"] - 1 / 3) < 1e-9 and r["n_test"] == 45
    noise = sc.linear_probe(rng.normal(size=(180, 8)).astype(np.float32), y, test)
    assert noise["accuracy"] < 0.7


def test_figure_and_report(tmp_path):
    mean = np.full((3, 16, 16), 100, np.float32)
    std = np.ones((3, 16, 16), np.float32)
    std[:, :4, :4] = 0
    out = sc.statistics_figure({"cam-a": (mean, std, sc.fixed_mask(std)), "cam-b": (mean, std, sc.fixed_mask(std))},
                               tmp_path / "stats.jpg", tile=40)
    assert out.exists() and out.stat().st_size > 1000
    probes = pd.DataFrame([{"task": "t", "variant": "original", "classes": 2, "n_train": 10, "n_test": 5, "accuracy": 0.8,
                            "balanced_accuracy": 0.75, "chance": 0.5, "majority": 0.6}])
    share = pd.DataFrame([{"camera": "cam-a", "frames": 10, "fixed_share": 0.0625}])
    text = sc.report_markdown(probes, share, [{"shortcut": "s", "evidence": "e", "mitigation": "m"}], "fig.jpg")
    assert "| t | original | 2 | 10 | 5 | 80.0% | 75.0% | 50.0% | 60.0% |" in text and "| cam-a | 10 | 6.2% |" in text
    assert "| s | e | m |" in text
