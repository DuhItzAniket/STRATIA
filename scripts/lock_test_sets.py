"""Write test_lock.json from the split files (P039).

    python scripts/lock_test_sets.py [--force]

Refuses to overwrite an existing lock unless --force is given (a re-lock is a deliberate act that the commit message
must explain).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stratia.data import test_lock as tl  # noqa: E402
from stratia.data.splits import load_split_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if tl.LOCK_PATH.exists() and not a.force:
        print(f"{tl.LOCK_PATH.name} exists; pass --force to re-lock (and say why in the commit)")
        return 1
    config = load_split_config()
    m = pd.read_parquet("data/manifest.parquet", columns=["sample_id"])
    manifest_hash = hashlib.sha256("\n".join(m.sample_id).encode()).hexdigest()[:16]
    lock = tl.build_lock("data/splits", config["version"], config["seed"], manifest_hash)
    problems = tl.well_formed(lock)
    if problems or not lock["tests"]:
        print("not locking:", problems or ["no test files found under data/splits"])
        return 1
    tl.LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"locked {len(lock['tests'])} test sets ({sum(e['n'] for e in lock['tests'].values()):,} ids) -> {tl.LOCK_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
