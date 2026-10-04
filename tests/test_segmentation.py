import numpy as np
from PIL import Image

from stratia.data.segmentation import CLOUD, IGNORE, SKY, SKY_INVALID, load_masks


def test_almeria_mapping(tmp_path):
    m = np.array([[0, 1, 2], [3, 4, 1]], dtype=np.uint8)
    Image.fromarray(m).save(tmp_path / "m.png")
    sky, layer = load_masks("almeria", tmp_path / "m.png")
    assert sky.tolist() == [[SKY_INVALID, SKY, CLOUD], [CLOUD, CLOUD, SKY]]
    assert layer.tolist() == [[IGNORE, IGNORE, 0], [1, 2, IGNORE]]


def test_swim_family_threshold_handles_jpeg_noise(tmp_path):
    m = np.array([[0, 3, 127], [128, 250, 255]], dtype=np.uint8)   # JPEG-like intermediate values
    Image.fromarray(m).save(tmp_path / "m.png")
    sky, layer = load_masks("swinseg", tmp_path / "m.png")
    assert sky.tolist() == [[SKY, SKY, SKY], [CLOUD, CLOUD, CLOUD]]
    assert (layer == IGNORE).all()


def test_unknown_dataset_rejected(tmp_path):
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(tmp_path / "m.png")
    try:
        load_masks("nope", tmp_path / "m.png")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
