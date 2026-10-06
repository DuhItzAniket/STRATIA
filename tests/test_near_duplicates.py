import numpy as np
import pandas as pd
from PIL import Image

from stratia.data import near_duplicates as nd
from stratia.data.image_cache import cache_path, write_cached


def _scene(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    img = 120 + 60 * np.sin(x / (5 + seed)) * np.cos(y / (7 + seed)) + rng.normal(0, 4, (size, size))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_hashes_are_stable_and_tell_pictures_apart():
    a, b = _scene(1), _scene(2)
    assert nd.hamming(nd.phash(a), nd.phash(a)) == 0 and nd.hamming(nd.dhash(a), nd.dhash(a)) == 0
    assert nd.hamming(nd.phash(a), nd.phash(b)) > 12 and nd.hamming(nd.dhash(a), nd.dhash(b)) > 12
    assert 0 <= nd.phash(a) < 2**64


def test_flipped_or_rotated_copies_match_through_the_dihedral_variants():
    a = _scene(3)
    variants = nd.dihedral(a)
    assert len(variants) == 8 and np.array_equal(variants[0], a)
    phashes = [nd.phash(v) for v in variants]
    flipped = np.fliplr(np.rot90(a, 2))
    assert nd.min_hamming(phashes, nd.phash(flipped)) <= 2  # the rotated-and-flipped copy is one of the eight
    assert nd.min_hamming(phashes, nd.phash(_scene(4))) > 10  # a different picture still is not


def test_hashes_of_reads_a_file_and_reports_a_bad_one(tmp_path):
    Image.fromarray(_scene(5)).save(tmp_path / "a.png")
    h = nd.hashes_of(tmp_path / "a.png")
    assert h["error"] is None and len(h["phash"]) == 8 and h["phash"][0] == nd.phash(_scene(5))
    (tmp_path / "bad.png").write_bytes(b"nope")
    bad = nd.hashes_of(tmp_path / "bad.png")
    assert bad["error"] and bad["phash"] == [None] * 8


def test_nearest_neighbours_and_candidate_pairs():
    rng = np.random.default_rng(0)
    base = nd.l2_normalise(rng.normal(size=(6, 16)).astype(np.float32))
    emb = np.vstack([base, base[0] + 0.01 * rng.normal(size=16).astype(np.float32)])  # row 6 is almost row 0
    emb = nd.l2_normalise(emb)
    idx, sim = nd.nearest_neighbours(emb, k=3, chunk=4, device="cpu")
    assert idx.shape == (7, 3) and idx[0, 0] == 6 and idx[6, 0] == 0 and sim[0, 0] > 0.99
    assert all(idx[i, 0] != i for i in range(7))  # never its own neighbour
    pairs = nd.candidate_pairs(idx, sim, min_sim=0.99)
    assert pairs[["i", "j"]].values.tolist() == [[0, 6]]  # one pair, listed once, i < j


def test_connected_components_group_linked_images_only():
    labels = nd.connected_components(6, np.array([0, 1, 3]), np.array([1, 2, 4]))
    assert labels[0] == labels[1] == labels[2] != labels[3]
    assert labels[3] == labels[4] and labels[5] == -1
    assert nd.connected_components(3, np.array([], dtype=int), np.array([], dtype=int)).tolist() == [-1, -1, -1]


def test_imagenet_normalisation_centres_the_values():
    x = np.full((2, 3, 4, 4), 128, dtype=np.uint8)
    y = nd.normalise_batch(x)
    assert y.shape == x.shape and y.dtype == np.float32
    assert abs(float(y[:, 0].mean()) - (128 / 255 - 0.485) / 0.229) < 1e-6


def test_contact_sheet_writes_a_picture(tmp_path):
    files = ["a.png", "b.png"]
    for name, seed in zip(files, (1, 2), strict=True):
        Image.fromarray(np.stack([_scene(seed)] * 3, axis=-1)).save(tmp_path / name)
        write_cached(tmp_path / name, cache_path(tmp_path / "cache", name))
    pairs = pd.DataFrame({"i": [0], "j": [1], "cosine": [0.95], "phash": [7]})
    out = nd.contact_sheet(pairs, files, tmp_path / "cache", tmp_path / "sheet.jpg", tile=40, columns=2)
    assert out.exists() and out.stat().st_size > 500
    with Image.open(out) as sheet:
        assert sheet.size == (2 * (2 * 40 + 12), 40 + 24)


def test_vectorised_hamming_agrees_with_the_scalar_one():
    rng = np.random.default_rng(1)
    variants = rng.integers(0, 2**63, size=(5, 8), dtype=np.int64).astype(np.uint64) * np.uint64(2) + np.uint64(1)
    i, j = np.array([0, 1, 4]), np.array([2, 3, 0])
    fast = nd.pair_min_hamming(variants, i, j)
    slow = [nd.min_hamming([int(v) for v in variants[a]], int(variants[b][0])) for a, b in zip(i, j, strict=True)]
    assert fast.tolist() == slow
    assert nd.popcount(np.array([0, 1, 2**64 - 1], dtype=np.uint64)).tolist() == [0, 1, 64]


def test_exhaustive_matches_find_the_copies_and_nothing_else():
    a = _scene(11)
    variants_a = np.array([[nd.phash(v) for v in nd.dihedral(img)] for img in (a, _scene(12))], dtype=np.uint64)
    plain_b = np.array([nd.phash(np.rot90(a, 3)), nd.phash(_scene(13)), nd.phash(np.fliplr(a))], dtype=np.uint64)
    hits = nd.exhaustive_matches(variants_a, plain_b, max_distance=2, chunk=1)
    assert sorted(zip(hits.a, hits.b, strict=True)) == [(0, 0), (0, 2)]
    assert (hits.phash <= 2).all()
