import numpy as np
import pandas as pd
from PIL import Image

from stratia.data import label_conflicts as lc
from stratia.data import near_duplicates as nd


def _scene(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    img = 120 + 60 * np.sin(x / (5 + seed)) * np.cos(y / (7 + seed)) + rng.normal(0, 4, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)], "dataset": ["ccsn"] * 4 + ["mgcd"] * 2,
        "image_file": [f"img{i}.jpg" for i in range(6)], "official_split": ["none"] * 4 + ["train", "test"],
        "source_label": ["Cc", "Cs", "Cc", None, "cumulus", "mixed"],
        "seg_file": [None] * 6, "has_layers": [False] * 6})


def test_pairs_within_groups_and_merge_keep_the_strongest_kind():
    exact = lc.pairs_within_groups(pd.Series([3, 0, 1, 5]), pd.Series(["g1", "g1", "g1", "g2"]))
    assert exact.values.tolist() == [[0, 1], [0, 3], [1, 3]]            # every pair of the triple, i < j; g2 alone
    merged = lc.merge_pairs({"copy": pd.DataFrame({"i": [1, 2, 4], "j": [0, 2, 5]}), "exact": exact,
                             "same_scene": pd.DataFrame({"i": [5], "j": [4]})})
    assert merged[(merged.i == 0) & (merged.j == 1)].kind.item() == "exact"   # exact beats copy for the same pair
    assert merged[(merged.i == 4) & (merged.j == 5)].kind.item() == "copy"    # copy beats same_scene
    assert not ((merged.i == 2) & (merged.j == 2)).any() and len(merged) == 4


def test_compare_labels_flags_conflicts_and_published_split_crossings():
    m = _manifest()
    pairs = pd.DataFrame({"i": [0, 0, 0, 4], "j": [1, 2, 3, 5], "kind": ["exact", "exact", "copy", "same_scene"]})
    c = lc.compare_labels(m, pairs)
    assert c.conflict.tolist() == [True, False, False, True]              # Cc/Cs; Cc/Cc; label missing; cumulus/mixed
    assert c.compared.tolist() == [True, True, False, True]
    assert c.crosses_official_split.tolist() == [False, False, False, True]  # MGCD train vs test; "none" never counts
    s = lc.conflict_summary(c).set_index(["dataset", "kind"])
    assert s.loc[("ccsn", "exact"), "pairs"] == 2 and s.loc[("ccsn", "exact"), "conflicts"] == 1
    assert s.loc[("ccsn", "exact"), "images"] == 2 and s.loc[("mgcd", "same_scene"), "rate"] == 1.0
    conf = lc.label_confusions(c)
    assert conf[conf.dataset == "ccsn"].labels.item() == "Cc / Cs"


def test_best_variant_recovers_the_transform_and_aligns_the_mask():
    a = _scene(21)
    variants = np.array([[nd.phash(v) for v in nd.dihedral(a)]], dtype=np.uint64)
    b = np.fliplr(np.rot90(a, 1))                                          # dihedral index 3
    k, d = lc.best_variant(variants, np.array([nd.phash(b)], dtype=np.uint64))
    assert k.tolist() == [3] and d.tolist() == [0]
    mask = (a > 130).astype(np.uint8) * 2 + (a <= 130).astype(np.uint8)   # 1 sky, 2 cloud, asymmetric
    same = lc.mask_agreement(mask, np.fliplr(np.rot90(mask, 1)), k=3)
    assert same["agreement"] == 1.0 and same["iou_cloud"] == 1.0 and same["n_valid"] == mask.size
    wrong = lc.mask_agreement(mask, np.fliplr(np.rot90(mask, 1)), k=0)
    assert wrong["agreement"] < 0.9
    small = lc.mask_agreement(mask, np.fliplr(np.rot90(mask, 1))[::2, ::2], k=3)   # other size: nearest resampling
    assert small["agreement"] > 0.9 and small["n_valid"] == 48 * 48


def test_mask_agreement_ignores_unlabelled_pixels():
    a = np.array([[1, 2, 255], [2, 2, 1]], dtype=np.uint8)
    b = np.array([[1, 1, 2], [255, 2, 1]], dtype=np.uint8)
    r = lc.mask_agreement(a, b)
    assert r["n_valid"] == 4 and abs(r["agreement"] - 0.75) < 1e-9
    assert abs(r["iou_cloud"] - 1 / 2) < 1e-9                              # cloud: a {(0,1),(1,1)}, b {(1,1)}
    assert lc.mask_agreement(np.full((2, 2), 255, np.uint8), a[:, :2])["n_valid"] == 0


def test_compare_masks_reads_the_files_once_aligned(tmp_path):
    a = _scene(31, 64)
    mask = ((a > 128) * 255).astype(np.uint8)
    Image.fromarray(mask).save(tmp_path / "a_GT.png")
    Image.fromarray(np.rot90(mask, 2)).save(tmp_path / "b_GT.png")
    m = pd.DataFrame({"dataset": ["swimseg", "swimseg", "ccsn"], "image_file": ["a.png", "b.png", "c.jpg"],
                      "seg_file": ["a_GT.png", "b_GT.png", None], "has_layers": [False] * 3})
    pairs = pd.DataFrame({"i": [0, 0], "j": [1, 2]})
    out = lc.compare_masks(m, pairs, tmp_path, np.array([4, 0]))           # k=4 is rot90 twice
    assert len(out) == 1 and out.agreement.item() == 1.0 and out.variant.item() == 4   # the unmasked pair is skipped
    assert lc.mask_summary(out.assign(datasets="swimseg / swimseg", kind="copy"), 0.95).below_threshold.item() == 0


def test_rater_agreement_summarises_majority_shares():
    m = pd.DataFrame({"dataset": ["montenegro"] * 3 + ["ccsn"],
                      "oktas_dist": ['{"6": 0.5, "7": 0.5}', '{"8": 1.0}', '{"1": 0.6, "2": 0.4}', None],
                      "h_dist": ['{"5": 0.8, "6": 0.2}', '{"9": 1.0}', None, None]})
    r = lc.rater_agreement(m, columns=("oktas_dist", "h_dist")).set_index("quantity")
    assert r.loc["oktas", "images"] == 3 and abs(r.loc["oktas", "no_majority"] - 1 / 3) < 1e-9
    assert abs(r.loc["oktas", "unanimous"] - 1 / 3) < 1e-9 and r.loc["h", "images"] == 2
    assert set(r.dataset) == {"montenegro"}


def test_report_builds_from_the_tables():
    m = _manifest()
    c = lc.compare_labels(m, pd.DataFrame({"i": [0, 4], "j": [1, 5], "kind": ["exact", "same_scene"]}))
    counts = {"exact": {"pairs": 1, "same_dataset": 1, "labelled": 1, "masked": 0}}
    masks = lc.mask_summary(pd.DataFrame({"datasets": ["swimseg / swinyseg"], "kind": ["copy"], "agreement": [0.98],
                                          "iou_cloud": [0.97]}), 0.95)
    crossing = pd.DataFrame(columns=["dataset", "kind", "pairs"])
    text = lc.report_markdown(counts, lc.conflict_summary(c), lc.label_confusions(c), crossing, masks, 0.95,
                              lc.rater_agreement(m), c[c.conflict & (c.kind == "exact")], "note here")
    assert "| ccsn | exact | 1 | 1 | 100.0% | 2 |" in text and "Cc / Cs" in text and "note here" in text
    assert "| swimseg / swinyseg | copy | 1 | 0.980 |" in text and "## Policy" in text
