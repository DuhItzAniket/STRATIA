"""Check the data registry against the local disk (P007).

    python scripts/check_registry.py [--count]

Validates configs/datasets.yaml, checks each registered folder exists, and with --count
compares image counts with the expected ones (a full inventory is P011).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data.registry import count_images, dataset_dirs, load_paths, load_registry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true", help="count images (slow for large datasets)")
    a = ap.parse_args()
    reg = load_registry()
    root = Path(load_paths()["data_root"])
    problems = 0
    print(f"{'dataset':12} {'status':15} {'licence':11} {'folder':7} {'images':>8} {'expected':>9}")
    for key, d in reg.items():
        dirs = dataset_dirs(d, root)
        exists = all(p.exists() for p in dirs.values()) if dirs else False
        if d["status"] in {"present", "partial"} and not exists:
            problems += 1
        n, exp = "", d.get("expected_images", "")
        if a.count and exists and key != "eye2sky":
            globs = d.get("images_glob")
            counts = {k: count_images(p, globs.get(k) if isinstance(globs, dict) else globs) for k, p in dirs.items()}
            n = sum(counts.values())
            if isinstance(exp, dict):
                exp = sum(exp.values())
            if d["status"] == "present" and isinstance(exp, int) and n != exp:
                print(f"  ! {key}: counted {n}, expected {exp} ({counts})")
        print(f"{key:12} {d['status']:15} {d['licence_status']:11} {'yes' if exists else 'no':7} {n!s:>8} {exp!s:>9}")
    print("OK" if problems == 0 else f"{problems} registered-as-present dataset(s) missing on disk")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
