"""PDF via pdfplumber (MIT; PyMuPDF is avoided because it is AGPL).

Two passes over the document:
1. Collect every text line with its median font size and position, then drop running
   headers/footers (text repeated in the top or bottom margin of most pages) and bare
   page numbers.
2. Take the most common font size as body text. Short lines noticeably larger than body
   are headings, ranked by size into levels; pack heading regexes apply line by line,
   and their numbering takes precedence for the level. Remaining lines group into
   paragraphs on vertical gaps, font-size changes and bullets.

Paragraphs never span pages, so every element has an exact page; the chunker merges
across pages. Known limits: multi-column layouts can interleave columns, tables are
read as plain lines, and scanned pages yield no text (the document is recorded EMPTY).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from semanticsearch.core.errors import ParseError
from semanticsearch.core.types import Element, ElementType
from semanticsearch.ingest.sectioning import MAX_LEVEL, SectionRules

MARGIN_BAND = 0.08  # top/bottom fraction of the page where running headers live
RUNNING_MIN_PAGES = 3
RUNNING_PAGE_FRACTION = 0.5
PARAGRAPH_GAP = 0.5  # vertical gap, as a fraction of line height, that ends a paragraph

_DIGITS = re.compile(r"\d+")
_PAGE_NUMBER = re.compile(r"^(page\s*)?#(\s*(of|/)\s*#)?$")
_GLYPH_BULLET = re.compile(r"^[•◦▪▫‣⁃●○■□\-–*]\s+")
_ENUM_BULLET = re.compile(r"^\((?:[a-z]{1,3}|[0-9]{1,3})\)\s+")


@dataclass(frozen=True, slots=True)
class _Line:
    page: int
    text: str
    size: float
    top: float
    bottom: float
    in_margin: bool

    @property
    def height(self) -> float:
        return max(self.bottom - self.top, 1.0)


class PdfParser:
    extensions = (".pdf",)

    def __init__(self, rules: SectionRules) -> None:
        self.rules = rules

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> list[Element]:
        import pdfplumber

        try:
            with pdfplumber.open(str(path)) as pdf:
                lines = [ln for i, page in enumerate(pdf.pages, 1) for ln in _page_lines(page, i)]
        except Exception as e:  # pdfminer raises many unrelated types on bad input
            raise ParseError(f"{path}: not a readable PDF: {e}") from e
        lines = _drop_running_lines(lines)
        return _elements(lines, self.rules) if lines else []


def _page_lines(page, page_no: int) -> list[_Line]:
    height = float(page.height)
    out = []
    for ln in page.extract_text_lines(return_chars=True, strip=True):
        sizes = [c["size"] for c in ln["chars"] if c["text"].strip()]
        if not sizes or not ln["text"].strip():
            continue
        top, bottom = float(ln["top"]), float(ln["bottom"])
        out.append(
            _Line(
                page=page_no,
                text=ln["text"].strip(),
                size=round(median(sizes) * 2) / 2,  # 0.5pt buckets absorb rendering jitter
                top=top,
                bottom=bottom,
                in_margin=top < height * MARGIN_BAND or bottom > height * (1 - MARGIN_BAND),
            )
        )
    return out


def _running_key(text: str) -> str:
    return _DIGITS.sub("#", " ".join(text.lower().split()))


def _drop_running_lines(lines: list[_Line]) -> list[_Line]:
    n_pages = len({ln.page for ln in lines})
    pages_with: dict[str, set[int]] = {}
    for ln in lines:
        if ln.in_margin:
            pages_with.setdefault(_running_key(ln.text), set()).add(ln.page)
    threshold = max(RUNNING_MIN_PAGES, math.ceil(n_pages * RUNNING_PAGE_FRACTION))
    running = {k for k, pages in pages_with.items() if len(pages) >= threshold}

    def keep(ln: _Line) -> bool:
        if not ln.in_margin:
            return True
        key = _running_key(ln.text)
        return key not in running and not _PAGE_NUMBER.match(key)

    return [ln for ln in lines if keep(ln)]


def _elements(lines: list[_Line], rules: SectionRules) -> list[Element]:
    by_size = Counter()
    for ln in lines:
        by_size[ln.size] += len(ln.text)
    body = by_size.most_common(1)[0][0]
    min_heading = body + max(1.0, body * 0.15)

    def font_heading(ln: _Line) -> bool:
        return (
            rules.pdf_heuristic == "font_size_delta"
            and ln.size >= min_heading
            and rules.could_be_heading(ln.text)
        )

    heading_sizes = sorted({ln.size for ln in lines if font_heading(ln)}, reverse=True)
    size_level = {s: min(i + 1, MAX_LEVEL) for i, s in enumerate(heading_sizes)}

    out: list[Element] = []
    block: list[_Line] = []
    block_type = ElementType.PARAGRAPH
    prev_heading: _Line | None = None  # last line if it was a font heading, for wrapped titles

    def flush() -> None:
        nonlocal block, block_type
        if block:
            out.append(Element(block_type, "\n".join(ln.text for ln in block), page=block[0].page))
        block, block_type = [], ElementType.PARAGRAPH

    for ln in lines:
        regex_level = rules.regex_level(ln.text)
        if regex_level is not None or font_heading(ln):
            flush()
            continues_title = (
                regex_level is None
                and prev_heading is not None
                and prev_heading.page == ln.page
                and prev_heading.size == ln.size
                and ln.top - prev_heading.bottom < PARAGRAPH_GAP * prev_heading.height
            )
            if continues_title:
                last = out[-1]
                out[-1] = Element(
                    ElementType.HEADING, f"{last.text} {ln.text}", level=last.level, page=last.page
                )
            else:
                level = regex_level if regex_level is not None else size_level[ln.size]
                out.append(Element(ElementType.HEADING, ln.text, level=level, page=ln.page))
            prev_heading = ln if regex_level is None else None
            continue
        prev_heading = None

        text, is_bullet = ln.text, False
        if m := _GLYPH_BULLET.match(text):
            text, is_bullet = text[m.end() :], True
        elif _ENUM_BULLET.match(text):
            is_bullet = True

        if block:
            prev = block[-1]
            if (
                is_bullet
                or prev.page != ln.page
                or abs(prev.size - ln.size) > 0.5
                or ln.top - prev.bottom > PARAGRAPH_GAP * prev.height
            ):
                flush()
        if is_bullet:
            block_type = ElementType.LIST_ITEM
        block.append(_Line(ln.page, text, ln.size, ln.top, ln.bottom, ln.in_margin))
    flush()
    return out
