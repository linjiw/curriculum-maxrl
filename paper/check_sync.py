#!/usr/bin/env python3
"""Fail when the draft and ICLR manuscript bodies drift."""

from __future__ import annotations

import difflib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def normalized_lines(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]


def between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0]
    except IndexError as exc:
        raise SystemExit(f"missing synchronization marker: {start!r} or {end!r}") from exc


def compare(name: str, left: str, right: str) -> None:
    a, b = normalized_lines(left), normalized_lines(right)
    if a == b:
        return
    diff = "\n".join(
        difflib.unified_diff(a, b, fromfile=f"main:{name}", tofile=f"iclr:{name}", n=2)
    )
    raise SystemExit(f"manuscript {name} drifted:\n{diff}")


main = (ROOT / "main.tex").read_text()
iclr = (ROOT / "main_iclr.tex").read_text()

compare(
    "core",
    between(main, r"\begin{abstract}", r"\appendix"),
    between(iclr, r"\begin{abstract}", r"\begin{thebibliography}"),
)
compare(
    "appendix",
    between(main, r"\appendix", r"\begin{thebibliography}"),
    between(iclr, r"\appendix", r"\end{document}"),
)
compare(
    "bibliography",
    between(main, r"\begin{thebibliography}", r"\end{thebibliography}"),
    between(iclr, r"\begin{thebibliography}", r"\end{thebibliography}"),
)

print("manuscript core, appendix, and bibliography are synchronized")
