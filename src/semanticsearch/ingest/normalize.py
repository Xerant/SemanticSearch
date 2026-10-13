"""Format-independent cleanup applied to every parser's output.

Cleanup is conservative: it fixes extraction artefacts (ligatures, soft hyphens,
odd whitespace, words hyphenated across a line break) but never rewrites wording.
NFKC is deliberately avoided because it folds characters that carry meaning in
technical text, such as superscripts ("m²" -> "m2") and fractions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from typing import Iterable, Sequence

from semanticsearch.core.types import Element, ElementType
from semanticsearch.ingest.sectioning import SectionRules

_TRANSLATE = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        " ": " ",  # no-break space
        " ": " ",
        " ": " ",
        "­": None,  # soft hyphen
        "​": None,  # zero-width space / joiners
        "‌": None,
        "‍": None,
        "﻿": None,
        "\t": " ",
    }
)
_SPACES = re.compile(r"[ \x0b\x0c]+")
_HYPHEN_BREAK = re.compile(r"(\w)-$")
_KEEP_LINES = (ElementType.TABLE, ElementType.OTHER)


def clean_text(text: str) -> str:
    """Normalise characters and whitespace, keeping line breaks."""
    text = unicodedata.normalize("NFC", text).translate(_TRANSLATE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def join_lines(lines: Iterable[str]) -> str:
    """Join visual lines into running text, rejoining words hyphenated at a break.

    "contrac-" + "tion" -> "contraction", but "risk-" + "Based" keeps the hyphen,
    because a capital after the break usually means a real compound.
    """
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif _HYPHEN_BREAK.search(out) and line[0].islower():
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out


def normalize_elements(elements: Sequence[Element], rules: SectionRules) -> tuple[Element, ...]:
    """Clean every element, drop empty ones, and promote paragraphs that match a pack
    heading regex. Tables and OTHER blocks (code, email headers) keep their line breaks;
    prose is joined onto one line."""
    out: list[Element] = []
    for el in elements:
        text = clean_text(el.text)
        if el.type not in _KEEP_LINES:
            text = join_lines(text.split("\n"))
        if not text:
            continue
        el = replace(el, text=text)
        if el.type is ElementType.PARAGRAPH:
            level = rules.regex_level(text)
            if level is not None:
                el = replace(el, type=ElementType.HEADING, level=level)
        out.append(el)
    return tuple(out)
