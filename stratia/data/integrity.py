"""Image integrity check (P023): can every manifest image be decoded, and does it look like what the manifest says?

Per image: file size, container format, pixel mode, decoded size, EXIF orientation, grey-level mean and standard
deviation, and a set of flags:
    missing         the file does not exist
    zero_bytes      an empty file
    corrupt         the decoder refused it or it is truncated
    odd_mode        not an 8-bit RGB or greyscale image (RGBA, palette, CMYK, 16-bit, ...)
    exif_rotation   an EXIF orientation other than 1: viewers rotate it, decoders do not
    size_mismatch   decoded size differs from the manifest's width/height
    size_outlier    pixel count outside a quarter to four times the dataset's median
    blank           almost no contrast (grey-level standard deviation below 2): a black or saturated frame
Every flag is explained or resolved in docs/data/integrity_report.md; the exit criterion is 0 unexplained failures.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False  # a truncated file is a finding, not something to paper over

EXIF_ORIENTATION = 274
NORMAL_MODES = {"RGB", "L"}
BLANK_STD = 2.0
OUTLIER_LOW, OUTLIER_HIGH = 0.25, 4.0
FLAG_ORDER = ["missing", "zero_bytes", "corrupt", "odd_mode", "exif_rotation", "size_mismatch", "size_outlier", "blank"]

COLUMNS = ["sample_id", "dataset", "image_file", "bytes", "format", "mode", "width", "height", "exif_orientation",
           "mean", "std", "flags", "error"]


def inspect_image(path: str | Path, expected_width: int | None = None, expected_height: int | None = None) -> dict:
    """Decode one image completely and describe it. Never raises: problems become flags and `error`."""
    out: dict = {"bytes": 0, "format": None, "mode": None, "width": None, "height": None, "exif_orientation": 1,
                 "mean": np.nan, "std": np.nan, "flags": [], "error": None}
    path = Path(path)
    if not path.is_file():
        out["flags"].append("missing")
        return _finish(out)
    out["bytes"] = os.path.getsize(path)
    if out["bytes"] == 0:
        out["flags"].append("zero_bytes")
        return _finish(out)
    try:
        with Image.open(path) as img:
            out["format"], out["mode"] = img.format, img.mode
            out["width"], out["height"] = img.size
            orientation = img.getexif().get(EXIF_ORIENTATION, 1)
            out["exif_orientation"] = int(orientation) if orientation else 1
            img.load()  # the full decode: finds truncation and bad data that the header check misses
            grey = np.asarray(img.convert("L"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, MemoryError) as e:
        out["flags"].append("corrupt")
        out["error"] = f"{type(e).__name__}: {e}"
        return _finish(out)
    out["mean"], out["std"] = float(grey.mean()), float(grey.std())
    if out["mode"] not in NORMAL_MODES:
        out["flags"].append("odd_mode")
    if out["exif_orientation"] != 1:
        out["flags"].append("exif_rotation")
    if expected_width is not None and expected_height is not None and (out["width"], out["height"]) != (
            int(expected_width), int(expected_height)):
        out["flags"].append("size_mismatch")
    if out["std"] < BLANK_STD:
        out["flags"].append("blank")
    return _finish(out)


def _finish(out: dict) -> dict:
    out["flags"] = "|".join(f for f in FLAG_ORDER if f in out["flags"])
    return out


def add_size_outliers(table: pd.DataFrame) -> pd.DataFrame:
    """Flag images whose pixel count is far from their dataset's median (per dataset, decoded sizes only)."""
    table = table.copy()
    area = table["width"] * table["height"]
    median = area.groupby(table["dataset"]).transform("median")
    outlier = area.notna() & ((area < OUTLIER_LOW * median) | (area > OUTLIER_HIGH * median))
    table.loc[outlier, "flags"] = table.loc[outlier, "flags"].map(
        lambda f: "|".join([x for x in f.split("|") if x] + ["size_outlier"]))
    table["flags"] = table["flags"].map(lambda f: "|".join(x for x in FLAG_ORDER if x in f.split("|")))
    return table


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: images, bytes, and how many carry each flag."""
    rows = []
    for dataset, part in table.groupby("dataset", sort=True):
        row = {"dataset": dataset, "images": len(part), "gigabytes": round(part["bytes"].sum() / 1e9, 3),
               "clean": int((part["flags"] == "").sum())}
        for flag in FLAG_ORDER:
            row[flag] = int(part["flags"].str.contains(flag, regex=False).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def report_markdown(table: pd.DataFrame, max_examples: int = 40) -> str:
    summary = summarise(table)
    lines = ["# Image integrity report (P023)", "",
             f"{len(table):,} manifest images decoded completely; {int((table['flags'] == '').sum()):,} without a finding.", "",
             "| Dataset | Images | GB | Clean | " + " | ".join(FLAG_ORDER) + " |",
             "|---|---|---|---|" + "---|" * len(FLAG_ORDER)]
    for _, r in summary.iterrows():
        lines.append(f"| {r['dataset']} | {r['images']:,} | {r['gigabytes']} | {r['clean']:,} | "
                     + " | ".join(str(r[f]) for f in FLAG_ORDER) + " |")
    lines += ["", "## Sizes and modes per dataset", "",
              "| Dataset | Formats | Modes | Most common size | Distinct sizes |", "|---|---|---|---|---|"]
    for dataset, part in table.groupby("dataset", sort=True):
        decoded = part.dropna(subset=["width", "height"])
        sizes = (decoded["width"].astype(int).astype(str) + "×" + decoded["height"].astype(int).astype(str))
        top = sizes.value_counts()
        lines.append(f"| {dataset} | {', '.join(sorted(decoded['format'].dropna().unique()))} | "
                     f"{', '.join(sorted(decoded['mode'].dropna().unique()))} | "
                     f"{top.index[0] if len(top) else '-'} ({top.iloc[0]:,}) | {len(top)} |")
    flagged = table[table["flags"] != ""]
    lines += ["", f"## Flagged images ({len(flagged):,}; first {min(max_examples, len(flagged))} shown)", "",
              "| Dataset | File | Flags | Mode | Size | Std | Error |", "|---|---|---|---|---|---|---|"]
    for _, r in flagged.head(max_examples).iterrows():
        size = f"{int(r['width'])}×{int(r['height'])}" if pd.notna(r["width"]) else "-"
        std = f"{r['std']:.1f}" if pd.notna(r["std"]) else "-"
        lines.append(f"| {r['dataset']} | `{r['image_file']}` | {r['flags']} | {r['mode'] or '-'} | {size} | {std} | "
                     f"{r['error'] or ''} |")
    return "\n".join(lines) + "\n"
