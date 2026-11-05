"""Data-free tests of the supplementary generator: the Markdown-to-LaTeX conversion and a build from the committed
result documents into a temporary folder."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = """# Title (dropped)

## A section with **bold**, `code_with_underscore` and a [link](https://example.org)

Text with 5 % and R&D, a ≥ 0.9 bound, a 512 × 512 frame, −3 and the owner's CloudScope camera.

| Dataset | Split | F1 [95 % CI] |
|---|---|---|
| ccsn | val | 0.719 [0.68, 0.76] |
| mgcd | test | 1.000 |

- first item with `x`
- second item

1. one
2. two

![a picture](figures/picture.png)

```
raw $ verbatim & text
```
"""


def _load():
    spec = importlib.util.spec_from_file_location("supplementary", ROOT / "scripts" / "supplementary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_markdown_becomes_latex_with_escapes_tables_lists_and_figures(tmp_path):
    mod = _load()
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "picture.png").write_bytes(b"\x89PNG\r\n")
    conv = mod.Converter(tmp_path / "figures", tmp_path / "fig")
    tex = conv.convert(SAMPLE, "Sample")
    assert r"\section{Sample}" in tex and "Title (dropped)" not in tex
    assert r"\subsection{A section with \textbf{bold}, \texttt{code\_\allowbreak{}with\_\allowbreak{}underscore} and a link}" in tex
    assert r"5 \% and R\&D, a $\geq$ 0.9 bound, a 512 $\times$ 512 frame, $-$3 and the authors' the desktop application camera" in tex  # noqa: E501
    assert r"\begin{tabular}{lll}" in tex and r"ccsn & val & 0.719 [0.68, 0.76] \\" in tex and r"\end{center}" in tex
    assert r"\begin{itemize}" in tex and r"\item first item with \texttt{x}" in tex and r"\begin{enumerate}" in tex
    assert r"\includegraphics[width=0.9\linewidth]{supp/fig/picture.png}" in tex and (tmp_path / "fig" / "picture.png").exists()
    assert r"\begin{verbatim}" in tex and "raw $ verbatim & text" in tex
    assert not conv.unknown
    assert conv.identifying == {}
    body = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", tex, flags=re.S)
    assert body.count("{") == body.count("}")
    for ch in body:
        assert ord(ch) <= 127 or ch in mod.ALLOWED, ch


def test_build_from_the_committed_documents(tmp_path):
    mod = _load()
    conv = mod.build(paper=tmp_path, docs_dir=ROOT / "docs" / "data")
    assert (tmp_path / "supp.tex").exists()
    sections = sorted((tmp_path / "supp").glob("sec_*.tex"))
    assert len(sections) == len(mod.DOCS)
    assert not conv.unknown, conv.unknown  # every character of the documents is one pdflatex knows or is mapped
    main = (tmp_path / "supp.tex").read_text(encoding="utf-8")
    for name, _ in mod.DOCS:
        assert f"\\input{{supp/sec_{name}}}" in main
    for sec in sections:
        text = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", "", sec.read_text(encoding="utf-8"), flags=re.S)
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}"), sec.name
        assert "CloudScope" not in text and "DuhItzAniket" not in text, sec.name
