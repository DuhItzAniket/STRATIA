"""Lufft CHM15k ceilometer reader for the Eye2Sky NetCDF3 files (P014).

Facts established from the files (2022-06-01, CDLRA and CDLRB):
* `time` is seconds since 1904-01-01 UTC; records every ~15 s (5,760 per day).
* `cbh`, `cbe`, `cdp` hold up to 4 layers, lowest first; -1 means "no layer". Heights are metres above the
  instrument: the files store `altitude` = 0 and `cho` (cloud height offset) = 0. Per the Eye2Sky station
  list, CDLRA is 21 m a.s.l. (9 m above ground).
* `sci` sky condition index: 0 nothing, 1 rain, 2 fog, 3 snow, 4 precipitation or particles on the window.
* The files' own coordinates are not trusted (CDLRB states latitude 5.325 instead of 53.25); use the station list.
Quality problems are flagged, never silently dropped.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EPOCH_1904 = pd.Timestamp("1904-01-01", tz="UTC")
N_LAYERS = 4
OPTICS_MIN_PERCENT = 50   # transmission of optics below this is flagged as degraded
LASER_WARNING_BIT = 0x8000  # CDLRA: grows from 25% to 88% of records Apr-Jul 2022 with falling laser quality;
                            # CBH statistics unchanged (P014), so it is a warning, not an error


def read_chm15k(path: str | Path) -> pd.DataFrame:
    """One day of CHM15k data as a DataFrame indexed by UTC time (layer heights in metres, NaN = no layer)."""
    from scipy.io import netcdf_file

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # scipy warns about mmap copies on Windows
        f = netcdf_file(str(path), "r", mmap=False)
    try:
        v = f.variables
        t = EPOCH_1904 + pd.to_timedelta(np.asarray(v["time"][:], dtype=float), unit="s")
        out = {}
        for name in ("cbh", "cbe", "cdp"):
            arr = np.asarray(v[name][:], dtype=float)
            arr[arr < 0] = np.nan
            for k in range(N_LAYERS):
                out[f"{name}{k + 1}"] = arr[:, k]
        for name in ("tcc", "bcc", "sci", "mxd", "state_optics", "error_ext", "state_laser", "state_detector"):
            out[name] = np.asarray(v[name][:]).astype(np.int64)
        device = f._attributes.get("device_name", b"")
        device = device.decode("latin-1") if isinstance(device, bytes) else str(device)
    finally:
        f.close()
    df = pd.DataFrame(out, index=pd.DatetimeIndex(t, name="time"))
    df["n_layers"] = df[[f"cbh{k + 1}" for k in range(N_LAYERS)]].notna().sum(axis=1)
    df["qc_rain"] = df.sci == 1
    df["qc_fog"] = df.sci == 2
    df["qc_snow"] = df.sci == 3
    df["qc_window"] = df.sci == 4
    df["qc_optics"] = df.state_optics < OPTICS_MIN_PERCENT
    df["qc_laser_warning"] = (df.error_ext & LASER_WARNING_BIT) != 0
    df["qc_error"] = (df.error_ext & ~LASER_WARNING_BIT) != 0
    df["qc_any"] = df[["qc_rain", "qc_fog", "qc_snow", "qc_window", "qc_optics", "qc_error"]].any(axis=1)
    df.attrs["device"] = device
    df.attrs["file"] = Path(path).name
    return df


def load_all(base: str | Path, cache: str | Path | None = None,
             ceilometers: tuple[str, ...] = ("CDLRA", "CDLRB")) -> pd.DataFrame:
    """Every record of every day file under `base/<ceilometer>/*.nc`, with a `ceilometer` column and `time` as a
    column (UTC), sorted by ceilometer and time; cached as Parquet at `cache` when given."""
    cache = Path(cache) if cache else None
    if cache and cache.exists():
        return pd.read_parquet(cache)
    parts = []
    for code in ceilometers:
        for f in sorted((Path(base) / code).glob("*.nc")):
            df = read_chm15k(f).reset_index()
            df.insert(0, "ceilometer", code)
            parts.append(df)
    out = pd.concat(parts, ignore_index=True).sort_values(["ceilometer", "time"], kind="stable").reset_index(drop=True)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache, index=False)
    return out


def read_backscatter(path: str | Path) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """(time, range in m, normalised range-corrected signal) for plotting."""
    from scipy.io import netcdf_file

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = netcdf_file(str(path), "r", mmap=False)
    try:
        t = EPOCH_1904 + pd.to_timedelta(np.asarray(f.variables["time"][:], dtype=float), unit="s")
        rng = np.asarray(f.variables["range"][:], dtype=float)
        beta = np.asarray(f.variables["beta_raw"][:], dtype=np.float32)
    finally:
        f.close()
    return pd.DatetimeIndex(t), rng, beta


def lowest_cbh_near(df: pd.DataFrame, t: pd.Timestamp, window_s: float) -> dict:
    """Median lowest cloud base within ±window_s of t, with the share of records that had any cloud."""
    sub = df.loc[t - pd.Timedelta(seconds=window_s): t + pd.Timedelta(seconds=window_s)]
    if sub.empty:
        return {"n": 0, "cbh1_median": np.nan, "cloud_fraction": np.nan, "qc_any": True}
    return {"n": len(sub), "cbh1_median": float(sub.cbh1.median()) if sub.cbh1.notna().any() else np.nan,
            "cloud_fraction": float(sub.cbh1.notna().mean()), "qc_any": bool(sub.qc_any.any())}
