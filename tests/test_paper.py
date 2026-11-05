"""Data-free checks of the paper sources: the table script reads the committed results, and the LaTeX sources are
consistent (every cite key, label, number macro, input and graphic resolves) so that a fresh clone compiles."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def _strip(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in s.splitlines())


def _tex() -> str:
    files = [PAPER / "main.tex", *sorted((PAPER / "sec").glob("*.tex")), *sorted((PAPER / "tables").glob("*.tex"))]
    return _strip("\n".join(f.read_text(encoding="utf-8") for f in files))


def test_result_file_prefers_the_committed_copy():
    spec = importlib.util.spec_from_file_location("reproduce_tables", ROOT / "scripts" / "reproduce_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.result_file("P090_locked_test.json") == ROOT / "results" / "P090_locked_test.json"
    assert mod.result_file("ablations.parquet").parent.name == "results"
    assert mod.result_file("does_not_exist.parquet").parent.name == "data"


def test_paper_sources_resolve():
    tex = _tex()
    bib = (PAPER / "main.bib").read_text(encoding="utf-8")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cites = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", tex) for k in grp.split(",")}
    assert cites and cites <= keys, cites - keys
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = {k.strip() for grp in re.findall(r"\\[cC]?ref\{([^}]*)\}", tex) for k in grp.split(",")}
    assert refs <= labels, refs - labels
    macros = set(re.findall(r"\\newcommand\{\\(n\w+)\}", (PAPER / "numbers.tex").read_text(encoding="utf-8")))
    used = set(re.findall(r"\\(n[A-Z]\w*)", tex))
    assert used <= macros, used - macros
    for inp in re.findall(r"\\input\{([^}]*)\}", tex):
        assert (PAPER / (inp if inp.endswith(".tex") else inp + ".tex")).exists(), inp
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        assert (PAPER / g).exists(), g
    for f in [PAPER / "main.tex", *sorted((PAPER / "sec").glob("*.tex"))]:
        s = _strip(f.read_text(encoding="utf-8"))
        assert s.count("{") == s.count("}"), f.name


def test_numbers_are_generated_not_typed():
    """The prose quotes locked-test numbers only through numbers.tex macros: no three-decimal literal that equals a
    headline value appears typed in the sections."""
    sections = "\n".join(_strip(f.read_text(encoding="utf-8")) for f in sorted((PAPER / "sec").glob("*.tex")))
    numbers = (PAPER / "numbers.tex").read_text(encoding="utf-8")
    headline = re.findall(r"\\newcommand\{\\n(?:LodoModel|LodoProbe|IndGenusModel|IndGenusProbe|SkyMiou|LayerMiou)\}\{([0-9.]+)\}", numbers)  # noqa: E501
    assert headline
    for value in headline:
        assert not re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", sections), value
