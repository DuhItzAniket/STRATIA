"""P099 — the supplementary material: paper/supp.tex and paper/supp/sec_*.tex built from the result documents.

    python scripts/supplementary.py          # writes paper/supp.tex, paper/supp/sec_<doc>.tex, paper/supp/fig/*

Every table and number of the supplementary comes from the committed Markdown result documents under docs/data/
(which the phase scripts generate from their parquet tables), converted to LaTeX here: no hand-typed numbers. The
conversion handles headings, paragraphs, pipe tables (longtable), lists, inline code, bold, links and the figures
the documents embed (copied into paper/supp/fig/). A few words are rewritten for review anonymity (the desktop
application's name, "the owner"); the remaining identifying words are listed at the end of the run for the P099
checklist. The document is two-column for the title, as the CVPR kit requires, and one column after it so that the
long tables fit; compile it like the main paper (pdflatex; no bibliography).
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "data"
PAPER = ROOT / "paper"

DOCS = [
    ("locked_test", "Locked-test results in full"),
    ("ablations", "Ablations"),
    ("error_analysis", "Error analysis and shortcut re-check"),
    ("human_ceiling_model", "The model as a sixth annotator"),
    ("calibration", "Calibration"),
    ("robustness", "Robustness"),
    ("efficiency", "Efficiency"),
    ("b0268_sun_track", "The authors' camera: Sun track and pose"),
    ("data_card", "Data card"),
]
ANONYMISE = [
    ("CloudScope", "the desktop application"),
    ("The owner's", "The authors'"),
    ("the owner's", "the authors'"),
    ("The owner", "The authors"),
    ("the owner", "the authors"),
    ("owner", "authors"),
    ("DuhItzAniket", ""),
    ("on LEGION", "on the laptop"),
]
IDENTIFYING = ("Bengaluru", "BLR01", "Aniket", "github.com", "LEGION", "Luikz")
ALLOWED = set("éèêëáàâäãåíìîïóòôöõúùûüçñÉÈÊËÁÀÂÄÍÌÎÏÓÒÔÖÚÙÛÜÇÑ–")
MATH = {
    "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$", "×": r"$\times$", "−": "$-$", "→": r"$\rightarrow$",
    "←": r"$\leftarrow$", "²": "$^2$", "³": "$^3$", "°": r"$^\circ$", "…": r"\ldots{}", "—": "---", "·": r"$\cdot$",
    "σ": r"$\sigma$", "λ": r"$\lambda$", "Δ": r"$\Delta$", "µ": r"$\mu$", "μ": r"$\mu$", "ε": r"$\varepsilon$",
    "α": r"$\alpha$", "τ": r"$\tau$", "≈": r"$\approx$", "✓": "yes", "✗": "no", "′": "'", "’": "'", "‘": "`",
    "“": "``", "”": "''", "\u00a0": "~", "‑": "-", "½": "1/2", "⁻": "$^-$", "¹": "$^1$",
}
_ESCAPE = [("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
           ("_", r"\_"), ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ("<", "$<$"), (">", "$>$")]


class Converter:
    def __init__(self, fig_src: Path, fig_dst: Path, fig_prefix: str = "supp/fig/"):
        self.fig_src, self.fig_dst, self.fig_prefix = fig_src, fig_dst, fig_prefix
        self.unknown: dict[str, int] = {}
        self.identifying: dict[str, int] = {}

    # ---- inline
    def escape(self, s: str) -> str:
        for a, b in _ESCAPE:
            s = s.replace(a, b)
        for a, b in MATH.items():
            s = s.replace(a, b)
        out = []
        for ch in s:
            if ord(ch) > 127 and ch not in ALLOWED:
                self.unknown[f"U+{ord(ch):04X}"] = self.unknown.get(f"U+{ord(ch):04X}", 0) + 1
                out.append("?")
            else:
                out.append(ch)
        return "".join(out)

    def inline(self, s: str) -> str:
        for a, b in ANONYMISE:
            s = s.replace(a, b)
        for word in IDENTIFYING:
            if word in s:
                self.identifying[word] = self.identifying.get(word, 0) + s.count(word)
        s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)  # links -> their text
        codes: list[str] = []

        def stash(m: re.Match) -> str:
            codes.append(r"\texttt{" + self.escape(m.group(1)) + "}")
            return f"\x00{len(codes) - 1}\x00"

        s = re.sub(r"`([^`]*)`", stash, s)
        s = self.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
        s = re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], s)
        return s

    # ---- blocks
    def table(self, rows: list[list[str]]) -> list[str]:
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header, body = rows[0], rows[1:]
        size = r"\tiny" if width > 9 else r"\scriptsize"
        out = ["{" + size + r"\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{" + "l" * width + "}", r"\toprule",
               " & ".join(self.inline(c) for c in header) + r" \\", r"\midrule", r"\endhead"]
        out += [" & ".join(self.inline(c) for c in r) + r" \\" for r in body]
        out += [r"\bottomrule", r"\end{longtable}}", ""]
        return out

    def figure(self, alt: str, src: str) -> list[str]:
        name = Path(src).name
        source = self.fig_src / name
        if not source.exists():
            return [f"% figure {src} not found", ""]
        self.fig_dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, self.fig_dst / name)
        return [r"\begin{figure}[h]", r"\centering", r"\includegraphics[width=0.9\linewidth]{" + self.fig_prefix + name + "}",
                r"\caption{" + self.inline(alt or name) + "}", r"\end{figure}", ""]

    def convert(self, markdown: str, title: str) -> str:
        lines = markdown.replace("\r\n", "\n").split("\n")
        out = [r"\section{" + self.inline(title) + "}", ""]
        i = 0
        para: list[str] = []

        def flush() -> None:
            if para:
                out.append(self.inline(" ".join(para)))
                out.append("")
                para.clear()

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("```"):
                flush()
                block = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    block.append(lines[i])
                    i += 1
                out += [r"\begin{verbatim}"] + block + [r"\end{verbatim}", ""]
            elif stripped.startswith("|"):
                flush()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                        rows.append(cells)
                    i += 1
                if rows:
                    out += self.table(rows)
                continue
            elif stripped.startswith("#"):
                flush()
                level = len(stripped) - len(stripped.lstrip("#"))
                text = stripped[level:].strip()
                if level >= 2:
                    cmd = {2: "subsection", 3: "subsubsection"}.get(level, "paragraph")
                    out += ["\\" + cmd + "{" + self.inline(text) + "}", ""]
            elif stripped.startswith("![") and "](" in stripped:
                flush()
                m = re.match(r"!\[([^\]]*)\]\(([^)]*)\)", stripped)
                out += self.figure(m.group(1), m.group(2))
            elif re.match(r"^(-|\*|\d+\.)\s", stripped):
                flush()
                ordered = bool(re.match(r"^\d+\.", stripped))
                env = "enumerate" if ordered else "itemize"
                out.append(r"\begin{" + env + "}")
                while i < len(lines) and re.match(r"^\s*(-|\*|\d+\.)\s", lines[i]):
                    item = re.sub(r"^\s*(-|\*|\d+\.)\s", "", lines[i])
                    i += 1
                    while i < len(lines) and lines[i].startswith("  ") and not re.match(r"^\s*(-|\*|\d+\.)\s", lines[i]):
                        item += " " + lines[i].strip()
                        i += 1
                    out.append(r"\item " + self.inline(item))
                out += [r"\end{" + env + "}", ""]
                continue
            elif stripped in ("---", "***"):
                flush()
                out.append(r"\medskip")
            elif stripped == "":
                flush()
            else:
                para.append(stripped.lstrip("> "))
            i += 1
        flush()
        return "\n".join(out) + "\n"


def build(paper: Path = PAPER, docs_dir: Path = DOCS_DIR, docs=DOCS) -> Converter:
    conv = Converter(docs_dir / "figures", paper / "supp" / "fig")
    (paper / "supp").mkdir(parents=True, exist_ok=True)
    inputs = []
    for name, title in docs:
        src = docs_dir / f"{name}.md"
        if not src.exists():
            print(f"  missing: {src}")
            continue
        tex = conv.convert(src.read_text(encoding="utf-8"), title)
        (paper / "supp" / f"sec_{name}.tex").write_text(tex, encoding="utf-8", newline="\n")
        inputs.append(f"supp/sec_{name}")
    head = r"""% STRATIA supplementary material, generated by scripts/supplementary.py from docs/data/*.md; do not edit.
% Build: pdflatex supp (no bibliography). Two-column for the title as the kit requires, one column after it.
\documentclass[10pt,twocolumn,letterpaper]{article}
\usepackage[review]{cvpr}
\input{preamble}
\usepackage{longtable}
\definecolor{cvprblue}{rgb}{0.21,0.49,0.74}
\usepackage[pagebackref,breaklinks,colorlinks,allcolors=cvprblue]{hyperref}
\def\confName{CVPR}
\def\confYear{2027}
\title{Supplementary Material for\\STRATIA: Honest Cloud Understanding from a Single Sky Camera}
\author{Anonymous \confName~submission\\}
\begin{document}
\maketitle
\onecolumn
\small
This supplement reproduces, in full, the result tables that the paper summarises: the one logged read of the locked
test sets (every model, split, metric and bootstrap interval), the ablations, the error analysis, the model as a
sixth annotator, the calibration, the robustness suite, the efficiency measurements, the Sun-track fit of the
authors' camera, and the data card of the collection. Every table was generated from the committed result files by
the released scripts; nothing here was typed by hand. The code, the split manifests, the lock file and the locked
results accompany this supplement.

"""
    body = "\n".join(r"\input{" + inp + "}" for inp in inputs)
    (paper / "supp.tex").write_text(head + body + "\n\n\\end{document}\n", encoding="utf-8", newline="\n")
    return conv


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    conv = build()
    print(f"supplementary -> {PAPER / 'supp.tex'} ({len(DOCS)} sections)")
    if conv.unknown:
        print("  characters replaced by '?':", conv.unknown)
    if conv.identifying:
        print("  identifying words left for the P099 checklist:", conv.identifying)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
