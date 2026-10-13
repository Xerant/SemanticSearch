"""Plain text and Markdown. Paragraphs are blank-line separated. For Markdown, ATX
headings, bullet lists, pipe tables and code fences are recognised, and front matter
and thematic breaks are dropped. Plain-text headings come from the pack's regexes,
which the normaliser applies."""

from __future__ import annotations

import re
from pathlib import Path

from semanticsearch.core.errors import ParseError
from semanticsearch.core.types import Element, ElementType

_ATX = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE = re.compile(r"^\s*(```|~~~)")
_THEMATIC_BREAK = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_TABLE_ROW = re.compile(r"^\s*\|")
_TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]*-{3,}[\s:|-]*$")
_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")  # latin-1 decodes any byte string


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ParseError(f"{path}: undecodable text")  # unreachable: latin-1 accepts all bytes


class TextParser:
    extensions = (".txt", ".md", ".markdown")

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> list[Element]:
        markdown = path.suffix.lower() != ".txt"
        return list(_parse(read_text(path), markdown))


def _strip_front_matter(text: str) -> str:
    """Drop a leading YAML front-matter block; it is metadata, not body text."""
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], 1):
            if line.strip() in ("---", "..."):
                return "\n".join(lines[i + 1 :])
    return text


def _parse(text: str, markdown: bool):
    block: list[str] = []
    block_type = ElementType.PARAGRAPH
    in_fence = False

    def flush():
        nonlocal block, block_type
        if block:
            yield Element(block_type, "\n".join(block))
        block, block_type = [], ElementType.PARAGRAPH

    if markdown:
        text = _strip_front_matter(text)
    for line in text.splitlines():
        if markdown and _FENCE.match(line):
            yield from flush()
            if in_fence:
                in_fence = False
            else:
                in_fence, block_type = True, ElementType.OTHER
            continue
        if in_fence:
            block.append(line)
            continue
        if not line.strip():
            yield from flush()
            continue
        if markdown:
            if _TABLE_ROW.match(line):
                if block_type is not ElementType.TABLE:
                    yield from flush()
                    block_type = ElementType.TABLE
                if not _TABLE_RULE.match(line):
                    block.append(" | ".join(c.strip() for c in line.strip().strip("|").split("|")))
                continue
            if block_type is ElementType.TABLE:
                yield from flush()
            if _THEMATIC_BREAK.match(line):
                yield from flush()
                continue
            if m := _ATX.match(line):
                yield from flush()
                yield Element(ElementType.HEADING, m.group(2), level=len(m.group(1)))
                continue
            if _BULLET.match(line):
                yield from flush()
                block_type = ElementType.LIST_ITEM
                block.append(_BULLET.sub("", line, count=1))
                continue
        block.append(line)
    yield from flush()
