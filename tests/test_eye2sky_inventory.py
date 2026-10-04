import sys
from datetime import UTC, date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eye2sky_inventory as e2s  # noqa: E402


def test_filename_patterns():
    m = e2s.IMG_RE.match("20220401045530_160.jpg")
    assert m and m.group(1) == "20220401045530" and m.group(2) == "160"
    assert e2s.IMG_RE.match("near_horizon.png") is None
    c = e2s.CEIL_RE.match("20220401_CHM180102.nc")
    assert c and c.group(2) == "CHM180102"


def test_haversine_known_distance():
    # OLDLR to OLUOL from the Eye2Sky station list: about 0.41 km
    km = e2s.haversine_km(53.151372, 8.167010, 53.153478, 8.161923)
    assert abs(km - 0.41) < 0.01
    assert e2s.haversine_km(0, 0, 0, 1) == __import__("pytest").approx(111.2, abs=0.1)


def test_calibration_validity_windows():
    utc = UTC
    cals = [{"file": "empty.yaml", "status": "empty file"},
            {"file": "a.yaml", "status": "ok", "mounted": datetime(2022, 1, 31, tzinfo=utc),
             "demounted": datetime(2022, 4, 22, tzinfo=utc)},
            {"file": "b.yaml", "status": "ok", "mounted": datetime(2022, 4, 22, tzinfo=utc), "demounted": None}]
    assert e2s.calib_valid_on(cals, date(2022, 4, 21)) == "a.yaml"
    assert e2s.calib_valid_on(cals, date(2022, 4, 22)) == "b.yaml"   # demounted is exclusive
    assert e2s.calib_valid_on(cals, date(2021, 1, 1)) is None
