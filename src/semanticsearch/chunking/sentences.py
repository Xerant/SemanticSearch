"""Rule-based sentence splitting, with no model download (unlike nltk punkt), so it
works on an air-gapped host.

A boundary is sentence punctuation, optional closing quotes/brackets, whitespace, then
something that can start a sentence (capital, digit, opening quote/bracket). It is
rejected when the word before the period is a known abbreviation ("Ltd.", "No.",
"e.g.") or a single-letter initial ("J. Smith"). Decimals ("3.5") and dotted
numbering ("2.3.1") never match, because no whitespace follows their periods.
"""

from __future__ import annotations

import re
from typing import Iterable

DEFAULT_ABBREVIATIONS = frozenset(
    a.lower()
    for a in (
        # titles and company forms
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
        "Ltd.", "Pty.", "Co.", "Corp.", "Inc.", "Plc.", "Bros.", "Assoc.", "Dept.", "Govt.",
        # references inside documents
        "No.", "Nos.", "Vol.", "Ch.", "Sec.", "Sect.", "Cl.", "Para.", "Paras.", "Pt.",
        "Sch.", "Art.", "Fig.", "Figs.", "Eq.", "Ref.", "Refs.", "App.", "Approx.", "Est.",
        "p.", "pp.", "cf.", "vs.", "viz.", "etc.", "al.", "e.g.", "i.e.", "ibid.",
        # months
        "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.",
        "Nov.", "Dec.",
    )
)

_BOUNDARY = re.compile(r"[.!?]+[\"')\]’”]*\s+(?=[\"'(\[‘“]?[A-Z0-9])")


class SentenceSplitter:
    def __init__(self, extra_abbreviations: Iterable[str] = ()) -> None:
        self.abbreviations = DEFAULT_ABBREVIATIONS | {a.lower() for a in extra_abbreviations}

    def split(self, text: str) -> list[str]:
        out, start = [], 0
        for m in _BOUNDARY.finditer(text):
            if text[m.start()] == "." and self._is_abbreviation(text, m.start()):
                continue
            out.append(text[start : m.end()].strip())
            start = m.end()
        tail = text[start:].strip()
        if tail:
            out.append(tail)
        return [s for s in out if s]

    def _is_abbreviation(self, text: str, dot: int) -> bool:
        word_start = max(text.rfind(" ", 0, dot), text.rfind("\n", 0, dot), text.rfind("(", 0, dot)) + 1
        word = text[word_start : dot + 1]
        bare = word.rstrip(".")
        return word.lower() in self.abbreviations or (len(bare) == 1 and bare.isupper())
