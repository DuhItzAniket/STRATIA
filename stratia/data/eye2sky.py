"""Eye2Sky readers: calibration YAML, camera masks and image file names (P013).

Image files:   <data_root>/Eye2Sky/2022/MM/DD/ASI_<YYYYMMDD>_<STATION>/<STATION>/YYYY/MM/DD/HH/
               <YYYYmmddHHMMSS>_<exp>.jpg
Calibrations:  <data_root>/Eye2Sky/asi_meta/<STATION>/<STATION>_<YYYYMMDD>.yaml   (validity: mounted <= t < demounted)
Masks:         <data_root>/Eye2Sky/asi_meta/<STATION>/masks/<STATION>_<YYYYMMDD>_mask.{png,mat}
File names are UTC (the calibration files state timezone "GMT+0").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ..geometry.ocam import OcamModel

IMG_RE = re.compile(r"^(\d{14})_(\d+)\.jpe?g$", re.I)


@dataclass(frozen=True)
class ImageName:
    station: str
    utc: datetime
    exposure: int          # file-name suffix; equals the configured day exposure (e.g. 160)


def parse_image_name(path: str | Path) -> ImageName:
    p = Path(path)
    m = IMG_RE.match(p.name)
    if not m:
        raise ValueError(f"not an Eye2Sky image name: {p.name}")
    station = next((part.split("_")[2] for part in p.parts if part.startswith("ASI_") and part.count("_") == 2), None)
    if station is None:
        station = p.parts[-6] if len(p.parts) >= 6 else ""   # .../<STATION>/YYYY/MM/DD/HH/<file>
    utc = datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return ImageName(station=station, utc=utc, exposure=int(m.group(2)))


@dataclass(frozen=True)
class Calibration:
    station: str
    file: str
    mounted: datetime
    demounted: datetime | None
    latitude: float
    longitude: float
    altitude: float
    model: OcamModel
    external_orientation: tuple[float, float, float]   # [roll, pitch, yaw] in radians, as stored (convention: P044)
    mask_hint: str

    def valid_at(self, t: datetime) -> bool:
        return self.mounted <= t and (self.demounted is None or t < self.demounted)


def _utc(x) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, str):
        x = datetime.fromisoformat(x)
    return x if x.tzinfo else x.replace(tzinfo=UTC)


def load_calibration(path: str | Path) -> Calibration | None:
    """Parse one calibration YAML; returns None for empty files (one exists for OLUOL, 2020)."""
    p = Path(path)
    if p.stat().st_size == 0:
        return None
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    ic = d["internal_calibration"]
    model = OcamModel(ss=tuple(float(s) for s in ic["ss"]), xc=float(ic["xc"]), yc=float(ic["yc"]),
                      c=float(ic.get("c", 1.0)), d=float(ic.get("d", 0.0)), e=float(ic.get("e", 0.0)),
                      width=int(ic["width"]), height=int(ic["height"]))
    return Calibration(station=d["camera_name"], file=p.name, mounted=_utc(d["mounted"]),
                       demounted=_utc(d.get("demounted")), latitude=float(d["latitude"]),
                       longitude=float(d["longitude"]), altitude=float(d["altitude"]), model=model,
                       external_orientation=tuple(float(v) for v in d["external_orientation"]),
                       mask_hint=str(d.get("camera_mask_file", "")))


def station_calibrations(meta_root: str | Path, station: str) -> list[Calibration]:
    cals = [load_calibration(f) for f in sorted((Path(meta_root) / station).glob(f"{station}_*.yaml"))]
    return [c for c in cals if c is not None]


def select_calibration(cals: list[Calibration], t: datetime) -> Calibration | None:
    hits = [c for c in cals if c.valid_at(t)]
    return max(hits, key=lambda c: c.mounted) if hits else None


def mask_path(meta_root: str | Path, cal: Calibration) -> Path:
    """Resolve the mask file. The YAML hint uses <DATE>_<STATION>_mask but the files are <STATION>_<DATE>_mask."""
    date = cal.file.rsplit("_", 1)[-1].removesuffix(".yaml")
    folder = Path(meta_root) / cal.station / "masks"
    for name in (f"{cal.station}_{date}_mask.png", Path(cal.mask_hint).name.replace(".mat", ".png"),
                 f"{cal.station}_{date}_mask.mat"):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"no mask for {cal.station} {date} in {folder}")


def load_mask(path: str | Path) -> np.ndarray:
    """Boolean (H, W) mask, True = valid sky pixel. Reads the PNG or the MATLAB struct field 'Mask.BW'."""
    p = Path(path)
    if p.suffix.lower() == ".png":
        return np.array(Image.open(p)) > 0
    import scipy.io

    return scipy.io.loadmat(p)["Mask"][0, 0]["BW"] > 0
