"""The STRATIA cloud ontology (P033): genera, étages, WMO code tables and each dataset's native classes, read from
`configs/ontology.yaml` and validated so that every class used anywhere maps to known genera or is explicitly
excluded."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "configs" / "ontology.yaml"
ETAGES = ("low", "mid", "high")
CODE_DIGITS = tuple(str(k) for k in range(10)) + ("/",)


@dataclass
class Ontology:
    genera: dict[str, dict]
    extra_classes: dict[str, dict]
    genus_classes: list[str]
    etages: dict[str, dict]
    wmo_codes: dict[str, dict[str, dict]]
    datasets: dict[str, dict]
    version: int = 1
    path: Path | None = field(default=None, repr=False)

    # ---- genera and étages ----
    def etage_of(self, genus: str) -> str:
        return self.genera[genus]["etage"]

    def etages_of(self, genera) -> set[str]:
        """Primary étages of a set of genera (the étage the base of each genus belongs to)."""
        return {self.etage_of(g) for g in genera}

    def etage_heights_m(self, etage: str) -> tuple[float, float]:
        lo, hi = self.etages[etage]["heights_m"]
        return float(lo), float("inf") if hi is None else float(hi)

    def etage_of_height(self, height_m: float) -> str | None:
        """Étage whose weak height band holds a cloud base; None for NaN."""
        if height_m is None or height_m != height_m:
            return None
        for e in ETAGES:
            lo, hi = self.etage_heights_m(e)
            if lo <= height_m < hi:
                return e
        return ETAGES[-1]

    # ---- WMO codes ----
    def genera_for_code(self, table: str, code: str) -> list[str] | None:
        """Genera named by a C_L / C_M / C_H code; [] for 'no clouds of this kind'; None for 'not visible'."""
        entry = self.wmo_codes[table][str(code)]
        return None if entry.get("genera") is None else list(entry["genera"])

    def h_range_m(self, code: str) -> tuple[float, float] | None:
        rng = self.wmo_codes["h"][str(code)].get("range_m")
        if rng is None:
            return None
        return float(rng[0]), float("inf") if rng[1] is None else float(rng[1])

    def oktas(self, code: str) -> int | None:
        return self.wmo_codes["N"][str(code)].get("oktas")

    # ---- dataset classes ----
    def dataset_class(self, dataset: str, label: str) -> dict:
        return self.datasets[dataset]["classes"][label]


def load_ontology(path: str | Path | None = None) -> Ontology:
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    ont = Ontology(genera=raw["genera"], extra_classes=raw["extra_classes"], genus_classes=list(raw["genus_classes"]),
                   etages=raw["etages"], wmo_codes=raw["wmo_codes"], datasets=raw["datasets"], version=int(raw["version"]),
                   path=p)
    problems = validate(ont)
    if problems:
        raise ValueError("ontology.yaml: " + "; ".join(problems))
    return ont


def validate(ont: Ontology) -> list[str]:
    """Everything that must hold for the mapping phases to be sound; an empty list means the file is consistent."""
    problems = []
    for g, spec in ont.genera.items():
        if spec.get("etage") not in ETAGES:
            problems.append(f"genus {g}: étage {spec.get('etage')!r} is not one of {ETAGES}")
        for e in spec.get("extends", []):
            if e not in ETAGES or e == spec.get("etage"):
                problems.append(f"genus {g}: bad 'extends' entry {e!r}")
    expected = set(ont.genera) | set(ont.extra_classes)
    if set(ont.genus_classes) != expected or len(ont.genus_classes) != len(expected):
        problems.append(f"genus_classes must list every genus and extra class once: {sorted(expected)}")
    for e in ETAGES:
        if e not in ont.etages or len(ont.etages[e].get("heights_m", [])) != 2:
            problems.append(f"étage {e}: missing or malformed heights_m")
    lo_prev = -1.0
    for e in ETAGES:
        lo, hi = ont.etage_heights_m(e)
        if lo != max(lo_prev, 0.0) and lo_prev >= 0:
            problems.append(f"étage {e}: height bands must be contiguous (starts at {lo}, previous ended at {lo_prev})")
        lo_prev = hi
    for table in ("CL", "CM", "CH", "h", "N"):
        codes = ont.wmo_codes.get(table, {})
        missing = [c for c in CODE_DIGITS if c not in codes]
        if missing:
            problems.append(f"code table {table}: missing codes {missing}")
        if table in ("CL", "CM", "CH"):
            for c, entry in codes.items():
                for g in (entry.get("genera") or []) + list(entry.get("may_include", [])):
                    if g not in ont.genera:
                        problems.append(f"code table {table} code {c}: unknown genus {g!r}")
    for ds, spec in ont.datasets.items():
        for label, cls in spec.get("classes", {}).items():
            genera = cls.get("genera")
            if genera is None and not cls.get("cloud"):
                problems.append(f"dataset {ds} class {label!r}: genera null must come with cloud: true")
            for g in genera or []:
                if g not in ont.genera:
                    problems.append(f"dataset {ds} class {label!r}: unknown genus {g!r}")
            if cls.get("extra") and cls["extra"] not in ont.extra_classes:
                problems.append(f"dataset {ds} class {label!r}: unknown extra class {cls['extra']!r}")
            if genera == [] and not cls.get("extra"):
                problems.append(f"dataset {ds} class {label!r}: empty genera without an extra class (clear or contrail)")
        for label, cls in spec.get("altitude_class", {}).items():
            for e in cls.get("etages", []):
                if e not in ETAGES:
                    problems.append(f"dataset {ds} altitude class {label!r}: unknown étage {e!r}")
            for g in cls.get("genera", []):
                if g not in ont.genera:
                    problems.append(f"dataset {ds} altitude class {label!r}: unknown genus {g!r}")
    return problems
