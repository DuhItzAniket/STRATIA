from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from stratia.data.eye2sky import (
    load_calibration,
    load_mask,
    mask_path,
    parse_image_name,
    select_calibration,
    station_calibrations,
)
from stratia.geometry.ocam import OcamModel

# Intrinsics of Eye2Sky station OLDLR (calibration valid from 2022-04-22), used as a realistic test model.
OLDLR = OcamModel(ss=(-642.6690459406254, 0.0, 0.0002441927468927244, 4.838462720720562e-07),
                  xc=1028.60, yc=1072.59, width=2112, height=2048)


def _sky_pixels(model, n=4000, radius=900, seed=0):
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0, 1, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return model.yc + r * np.cos(a), model.xc + r * np.sin(a)   # u (col), v (row)


def test_round_trip_below_0p1_pixel():
    u, v = _sky_pixels(OLDLR)
    uu, vv = OLDLR.ray_to_pixel(OLDLR.pixel_to_ray(u, v))
    err = np.hypot(uu - u, vv - v)
    assert np.isfinite(err).all() and err.max() < 0.1, err.max()


def test_rays_are_unit_and_centre_is_optical_axis():
    rays = OLDLR.pixel_to_ray(*_sky_pixels(OLDLR, 200))
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    centre = OLDLR.pixel_to_ray(OLDLR.yc, OLDLR.xc)
    assert np.allclose(centre, [0, 0, 1], atol=1e-9)


def test_standard_frame_handedness():
    right = OLDLR.pixel_to_ray(OLDLR.yc + 300, OLDLR.xc)   # +u (column)
    down = OLDLR.pixel_to_ray(OLDLR.yc, OLDLR.xc + 300)    # +v (row)
    assert right[0] > 0 and abs(right[1]) < 1e-9 and right[2] > 0
    assert down[1] > 0 and abs(down[0]) < 1e-9 and down[2] > 0


def test_fisheye_reaches_about_90_degrees_off_axis():
    # The horizon (ray perpendicular to the optical axis) must project inside the image.
    u, v = OLDLR.ray_to_pixel(np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(u) and 0 < u < OLDLR.width


def test_parse_image_name():
    p = Path("2022/04/01/ASI_20220401_AURIC/AURIC/2022/04/01/04/20220401045530_160.jpg")
    n = parse_image_name(p)
    assert n.station == "AURIC" and n.exposure == 160 and n.utc == datetime(2022, 4, 1, 4, 55, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        parse_image_name("near_horizon.png")


# ---------------------------------------------------------------- real files (skipped where absent, e.g. CI)
def _meta_root():
    try:
        from stratia.data.registry import load_paths

        root = Path(load_paths()["data_root"]) / "Eye2Sky" / "asi_meta"
    except FileNotFoundError:
        return None
    return root if root.exists() else None


@pytest.mark.parametrize("station", ["OLDLR", "WESTE", "AURIC"])
def test_real_calibrations_and_masks(station):
    root = _meta_root()
    if root is None:
        pytest.skip("Eye2Sky metadata not available")
    cals = station_calibrations(root, station)
    assert cals
    cal = select_calibration(cals, datetime(2022, 6, 1, 12, tzinfo=UTC))
    assert cal is not None and cal.valid_at(datetime(2022, 6, 1, 12, tzinfo=UTC))
    mask = load_mask(mask_path(root, cal))
    assert mask.shape == (cal.model.height, cal.model.width) and 0.3 < mask.mean() < 0.8
    u, v = _sky_pixels(cal.model, 2000)
    uu, vv = cal.model.ray_to_pixel(cal.model.pixel_to_ray(u, v))
    assert np.nanmax(np.hypot(uu - u, vv - v)) < 0.1


def test_empty_calibration_file_is_skipped(tmp_path):
    (tmp_path / "X").mkdir()
    (tmp_path / "X" / "X_20200807.yaml").write_text("")
    assert load_calibration(tmp_path / "X" / "X_20200807.yaml") is None
    assert station_calibrations(tmp_path, "X") == []
