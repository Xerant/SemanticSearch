"""Heading detection rules, built once from a pack's `sectioning` config.

Format parsers and the normaliser share one SectionRules instance, so a line is a
heading for the same reasons whether it came from a PDF, a DOCX or plain text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from semanticsearch.packs.manifest import SectioningConfig

_LEADING_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
_STYLE_LEVEL = re.compile(r"(\d+)\s*$")
_BUILTIN_HEADING_STYLE = re.compile(r"^Heading (\d)$")
_SENTENCE_END = (".", ";", ":", ",")
MAX_LEVEL = 6


@dataclass(frozen=True)
class SectionRules:
    heading_patterns: tuple[re.Pattern[str], ...] = ()
    docx_styles: tuple[str, ...] = ()
    pdf_heuristic: str = "font_size_delta"
    max_heading_chars: int = 120

    @classmethod
    def from_config(cls, cfg: SectioningConfig) -> SectionRules:
        return cls(
            heading_patterns=tuple(re.compile(p) for p in cfg.heading_regexes),
            docx_styles=cfg.docx_styles,
            pdf_heuristic=cfg.pdf_heuristic,
            max_heading_chars=cfg.max_heading_chars,
        )

    def could_be_heading(self, text: str) -> bool:
        text = text.strip()
        return (
            0 < len(text) <= self.max_heading_chars
            and "\n" not in text
            and not text.endswith(_SENTENCE_END)
            and any(c.isalpha() for c in text)
        )

    def regex_level(self, text: str) -> int | None:
        """Heading level if `text` matches a pack heading regex, else None.

        Level comes from leading dotted numbering ("2.3 Risk" -> 2), else 1.
        """
        if not self.heading_patterns or not self.could_be_heading(text):
            return None
        text = text.strip()
        if not any(p.search(text) for p in self.heading_patterns):
            return None
        m = _LEADING_NUMBER.match(text)
        return min(m.group(1).count(".") + 1, MAX_LEVEL) if m else 1

    def style_level(self, style: str | None) -> int | None:
        """Heading level for a DOCX paragraph style, else None."""
        if not style:
            return None
        if not self.docx_styles:
            if style == "Title":
                return 1
            m = _BUILTIN_HEADING_STYLE.match(style)
            return int(m.group(1)) if m else None
        if style not in self.docx_styles:
            return None
        m = _STYLE_LEVEL.search(style)
        level = int(m.group(1)) if m else self.docx_styles.index(style) + 1
        return max(1, min(level, MAX_LEVEL))
