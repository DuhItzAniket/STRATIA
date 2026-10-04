import numpy as np
from PIL import Image

from stratia.data.image_cache import CACHE_MAX_SIDE, CachedImageDataset, cache_path, read_rgb, write_cached


def test_cache_resizes_longest_side_and_keeps_aspect(tmp_path):
    src = tmp_path / "data" / "My Sample" / "frame.png"          # space in path, PNG source
    src.parent.mkdir(parents=True)
    Image.new("RGB", (2112, 1056), (10, 200, 30)).save(src)
    dst = cache_path(tmp_path / "cache", "My Sample/frame.png")
    assert dst.suffix == ".jpg" and "img768" in dst.parts
    w, h = write_cached(src, dst)
    assert (w, h) == (CACHE_MAX_SIDE, 384)
    img = read_rgb(dst)
    assert img.shape == (384, 768, 3) and abs(int(img[..., 1].mean()) - 200) < 3   # RGB order preserved


def test_small_images_are_not_upscaled(tmp_path):
    src = tmp_path / "small.jpg"
    Image.new("RGB", (125, 125)).save(src)
    assert write_cached(src, tmp_path / "out.jpg") == (125, 125)


def test_dataset_returns_square_uint8_chw(tmp_path):
    src = tmp_path / "a.jpg"
    Image.new("RGB", (300, 200), (255, 0, 0)).save(src)
    write_cached(src, cache_path(tmp_path, "a.jpg"))
    x, i = CachedImageDataset(["a.jpg"], tmp_path, size=64)[0]
    assert x.shape == (3, 64, 64) and x.dtype.is_floating_point is False and i == 0
    assert x[0].float().mean() > 240 and x[1].float().mean() < 15 and np.isclose(x[2].float().mean().item(), 0, atol=15)
