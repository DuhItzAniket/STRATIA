from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import netcdf_file

from stratia.data.ceilometer import LASER_WARNING_BIT, lowest_cbh_near, read_chm15k


def _write_fake_chm15k(path: Path, n: int = 8) -> None:
    """Minimal CHM15k-like NetCDF3 file with the variables the reader uses."""
    f = netcdf_file(str(path), "w")
    f.device_name = b"CHMTEST"
    f.createDimension("time", n)
    f.createDimension("layer", 4)
    t = f.createVariable("time", "d", ("time",))
    t[:] = 3736886400.0 + 15.0 * np.arange(n)          # 2022-06-01 00:00:00 UTC + 15 s steps
    cbh = np.full((n, 4), -1, dtype=np.int16)
    cbh[0, 0], cbh[1, :2], cbh[2, :2] = 1500, (800, 3000), (820, 3100)
    for name, arr in (("cbh", cbh), ("cbe", np.where(cbh > 0, 10, -1)), ("cdp", np.where(cbh > 0, 100, -1))):
        v = f.createVariable(name, "h", ("time", "layer"))
        v[:] = arr.astype(np.int16)
    vals = {"tcc": [8] * n, "bcc": [8] * n, "sci": [0, 0, 1, 4, 0, 0, 0, 0][:n], "mxd": [7000] * n,
            "state_optics": [90, 90, 90, 90, 30, 90, 90, 90][:n], "error_ext": [0, LASER_WARNING_BIT, 0, 0, 0, 131072, 0, 0][:n],
            "state_laser": [70] * n, "state_detector": [90] * n}
    for name, val in vals.items():
        v = f.createVariable(name, "i", ("time",))
        v[:] = np.asarray(val, dtype=np.int32)
    f.close()


def test_reader_units_layers_and_flags(tmp_path):
    p = tmp_path / "20220601_CHMTEST.nc"
    _write_fake_chm15k(p)
    df = read_chm15k(p)
    assert df.index[0] == pd.Timestamp("2022-06-01 00:00:00", tz="UTC")
    assert (df.index[1] - df.index[0]) == pd.Timedelta(seconds=15)
    assert df.cbh1.iloc[0] == 1500 and np.isnan(df.cbh2.iloc[0]) and df.n_layers.tolist()[:4] == [1, 2, 2, 0]
    assert df.qc_rain.tolist()[:3] == [False, False, True] and df.qc_window.iloc[3] and df.qc_optics.iloc[4]
    # 0x8000 is a laser warning, not an error; other bits are errors
    assert df.qc_laser_warning.iloc[1] and not df.qc_error.iloc[1] and not df.qc_any.iloc[1]
    assert df.qc_error.iloc[5] and df.qc_any.iloc[5]


def test_lowest_cbh_near(tmp_path):
    p = tmp_path / "20220601_CHMTEST.nc"
    _write_fake_chm15k(p)
    df = read_chm15k(p)
    r = lowest_cbh_near(df, pd.Timestamp("2022-06-01 00:00:15", tz="UTC"), window_s=15)
    assert r["n"] == 3 and r["cbh1_median"] == pytest.approx(820) and r["cloud_fraction"] == 1.0 and r["qc_any"]
    empty = lowest_cbh_near(df, pd.Timestamp("2022-06-02", tz="UTC"), window_s=15)
    assert empty["n"] == 0 and np.isnan(empty["cbh1_median"])
