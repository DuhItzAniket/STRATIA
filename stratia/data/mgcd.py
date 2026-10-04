"""MGCD (multimodal ground-based cloud database) reader (P017).

Layout: <root>/{train,test}/<k>_<class>/<k>_<class>_<NNNNNN>.jpg, plus <root>/{train,test}/<k>_<class>.xlsx with
columns Number, Name, Temperature(°C), Humidity(%RH), Pressure(hpa), Wind speed(m/s), one row per image.
There are no timestamps; image numbers are sequential per class and the official test split holds the higher
numbers, so it is used as given. Weather values are for analysis only (STRATIA's inputs are images).
"""

from __future__ import annotations

import re
from pathlib import Path

import openpyxl
import pandas as pd

CLASS_RE = re.compile(r"^(\d)_([a-z]+)$")
WEATHER_COLS = {"Temperature(℃)": "temperature_c", "Humidity(%RH)": "humidity_pct",
                "Pressure(hpa)": "pressure_hpa", "Wind speed(m/s)": "wind_ms"}


def _read_xlsx(path: Path) -> pd.DataFrame:
    rows = list(openpyxl.load_workbook(path, read_only=True).active.iter_rows(values_only=True))
    df = pd.DataFrame(rows[1:], columns=[str(c).strip() for c in rows[0]])
    df = df.dropna(subset=["Name"]).rename(columns={"Name": "stem", "Number": "number", **WEATHER_COLS})
    return df[["stem", "number", *WEATHER_COLS.values()]]


def load_mgcd(root: str | Path) -> pd.DataFrame:
    """One row per image: relative path, split, class, number and weather (NaN where the sheet has no row)."""
    root = Path(root)
    frames = []
    for split in ("train", "test"):
        for folder in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            m = CLASS_RE.match(folder.name)
            if not m:
                continue
            imgs = pd.DataFrame({"image_file": [f"{split}/{folder.name}/{f.name}" for f in sorted(folder.glob("*.jpg"))]})
            imgs["stem"] = imgs.image_file.str.rsplit("/", n=1).str[-1].str.removesuffix(".jpg")
            imgs["split"], imgs["class_index"], imgs["class_name"] = split, int(m.group(1)), m.group(2)
            sheet = root / split / f"{folder.name}.xlsx"
            weather = _read_xlsx(sheet) if sheet.exists() else pd.DataFrame(columns=["stem", "number", *WEATHER_COLS.values()])
            frames.append(imgs.merge(weather, on="stem", how="left"))
    return pd.concat(frames, ignore_index=True)
