"""P048: photometric augmentations stay physical, geometry follows the policy, labels and ray maps stay consistent."""

import numpy as np
import pytest

from stratia.augment import photometric as P
from stratia.augment.policy import AugmentationPolicy, load_config
from stratia.geometry import raymap as RM
from stratia.geometry import transforms as T
from stratia.geometry.ocam import OcamModel
from stratia.geometry.pose import Pose, euler_to_rotation

OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)
UP = Pose(R=euler_to_rotation(0.0, 0.0, 0.3, "zyx"), source="test")


def _sky(h=256, w=320, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    base = np.stack([120 + 0.1 * yy, 150 + 0.1 * yy, 220 - 0.1 * yy], axis=-1)   # a blue gradient sky
    cloud = (np.sin(xx / 17.0) * np.cos(yy / 23.0) > 0.5)[..., None] * 60.0
    return np.clip(base + cloud + rng.normal(0, 2, (h, w, 3)), 0, 255).astype(np.uint8)


def _hue(img):
    import cv2

    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 0].astype(float)


def test_exposure_is_monotonic_and_clips():
    img = _sky()
    rng = np.random.default_rng(1)
    bright = P.exposure(img, rng, low=1.4, high=1.5)
    dark = P.exposure(img, rng, low=0.6, high=0.7)
    assert bright.mean() > img.mean() > dark.mean()
    assert bright.max() == 255 and dark.min() >= 0


def test_white_balance_keeps_hue_within_the_colour_temperature_axis():
    img = _sky()
    out = P.white_balance(img, np.random.default_rng(2), max_gain=0.12)
    # green untouched; red and blue move in opposite directions
    assert np.array_equal(out[..., 1], img[..., 1])
    dr, db = out[..., 0].astype(int).mean() - img[..., 0].mean(), out[..., 2].astype(int).mean() - img[..., 2].mean()
    assert dr * db < 0
    # the sky stays blue: hue change small on average
    assert np.abs(_hue(out) - _hue(img)).mean() < 6.0


def test_no_hue_rotation_anywhere_in_the_policy():
    cfg = load_config()
    names = set(cfg["photometric"].keys())
    assert not any("hue" in n or "hsv" in n or "saturation" in n for n in names)


def test_noise_and_jpeg_change_little_on_average():
    img = _sky()
    rng = np.random.default_rng(3)
    noisy = P.sensor_noise(img, rng)
    assert 0 < np.abs(noisy.astype(int) - img).mean() < 12
    comp = P.jpeg(img, rng, 40, 60)
    assert comp.shape == img.shape and 0 < np.abs(comp.astype(int) - img).mean() < 10


def test_sun_glare_brightens_around_the_sun_and_respects_the_mask():
    img = _sky()
    valid = np.ones(img.shape[:2], dtype=bool)
    valid[:, :60] = False
    out = P.sun_glare(img, np.random.default_rng(4), sun_px=(200.0, 100.0), valid=valid, strength=1.0)
    assert out[100, 200].mean() > img[100, 200].mean() + 20
    assert np.array_equal(out[:, :60], img[:, :60])
    assert out[250, 20:60].mean() <= img[250, 20:60].mean() + 1          # far corner (masked) untouched


def test_dirt_drops_and_obstruction():
    img = _sky()
    rng = np.random.default_rng(5)
    out = P.dirt_and_drops(img, rng, n_dirt=(3, 3), n_drops=(2, 2))
    assert out.shape == img.shape and out.mean() < img.mean()           # dirt darkens
    cut, mask = P.obstruction_cutout(img, np.random.default_rng(6), n=(2, 2))
    assert mask.dtype == bool and 0.005 < mask.mean() < 0.5
    assert cut[mask].mean() < 40 and np.array_equal(cut[~mask], img[~mask])


def test_policy_groups_and_chains():
    pol = AugmentationPolicy()
    assert pol.group("fisheye_asi") == "all_sky" and pol.group("consumer_photo") == "consumer"
    assert pol.group("wsi_crop") == "patch" and pol.group("something_new") == "consumer"
    names = [type(t).__name__ for t in pol.geometric_chain("fisheye_asi").transforms]
    assert names == ["RandomResizedCrop", "Rotate"]
    names = [type(t).__name__ for t in pol.geometric_chain("wide_angle_usb").transforms]
    assert names == ["RandomResizedCrop", "Rotate", "HorizontalFlip"]
    off = AugmentationPolicy(strength=0.0)
    assert [type(t).__name__ for t in off.geometric_chain("fisheye_asi").transforms] == ["Resize"]
    img = _sky()
    out = off(img, "fisheye_asi", np.random.default_rng(0))
    assert out.image.shape == (512, 512, 3) and out.applied == []


def test_policy_keeps_label_and_ray_map_consistent_with_an_obstruction():
    pol = AugmentationPolicy()
    cfg = pol.cfg
    for k in cfg["photometric"]:
        cfg["photometric"][k]["p"] = 0.0
    cfg["photometric"]["obstruction"]["p"] = 1.0
    pol = AugmentationPolicy(cfg)
    img = np.full((2048, 2112, 3), 90, dtype=np.uint8)
    label = np.full((2048, 2112), 2, dtype=np.uint8)                      # all "cloud"
    yy, xx = np.mgrid[:2048, :2112]
    mask = (xx - OLDLR.yc) ** 2 + (yy - OLDLR.xc) ** 2 < 950**2
    rng = np.random.default_rng(7)
    out = pol(img, "fisheye_asi", rng, label=label, valid_source=mask)
    assert out.obstruction is not None and out.obstruction.any() and "obstruction" in out.applied
    assert out.label.shape == (512, 512)
    assert np.all(out.label[out.obstruction] == 0)                        # obstruction -> invalid class
    covered = T.patch_fraction(out.state.coverage) == 1.0
    assert set(np.unique(out.label[np.kron(covered, np.ones((16, 16), bool))])) <= {0, 2}
    rm, mask_out = T.geometry_for(out.state, OLDLR, UP, 180.0, 45.0, mask, obstruction=out.obstruction)
    rm_plain, _ = T.geometry_for(out.state, OLDLR, UP, 180.0, 45.0, mask)
    assert rm[3].sum() < rm_plain[3].sum()                                # obstructed patches lost validity
    assert not mask_out[out.obstruction].any()
    # the label outside the source is IGNORE (255) after the random crop/rotation
    assert set(np.unique(out.label)) <= {0, 2, 255}


def test_strength_scales_ranges():
    weak = AugmentationPolicy(strength=0.5)
    strong = AugmentationPolicy(strength=2.0)
    r_weak = next(t for t in weak.geometric_chain("consumer_photo").transforms if isinstance(t, T.Rotate)).degrees
    r_strong = next(t for t in strong.geometric_chain("consumer_photo").transforms if isinstance(t, T.Rotate)).degrees
    assert r_weak == (-2.5, 2.5) and r_strong == (-10.0, 10.0)


def test_ray_map_through_policy_matches_direct_geometry():
    """A ray map built after the policy's geometric chain equals one built from the recorded state (same thing)."""
    pol = AugmentationPolicy(strength=0.0)
    img = np.zeros((2048, 2112, 3), dtype=np.uint8)
    out = pol(img, "fisheye_asi", np.random.default_rng(0))
    rm, _ = T.geometry_for(out.state, OLDLR, UP, 200.0, 50.0)
    direct = RM.ray_map(OLDLR, UP, 200.0, 50.0)
    assert np.allclose(rm, direct, atol=1e-6)


@pytest.mark.parametrize("camera_type", ["fisheye_asi", "consumer_photo", "wsi_crop"])
def test_policy_runs_for_every_group(camera_type):
    pol = AugmentationPolicy()
    img = _sky(300, 400)
    for seed in range(3):
        out = pol(img, camera_type, np.random.default_rng(seed))
        assert out.image.shape == (512, 512, 3) and out.image.dtype == np.uint8
