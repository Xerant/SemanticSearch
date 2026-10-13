"""Word (.docx) via python-docx. Body paragraphs and tables in document order; headings
come from paragraph styles per the pack's `docx_styles`. DOCX has no fixed pages, so
`page` is always None. Headers, footers, footnotes and text boxes are not extracted.
Legacy .doc is not supported: convert with LibreOffice first."""

from __future__ import annotations

from pathlib import Path

from semanticsearch.core.errors import ParseError
from semanticsearch.core.types import Element, ElementType
from semanticsearch.ingest.sectioning import SectionRules


class DocxParser:
    extensions = (".docx",)

    def __init__(self, rules: SectionRules) -> None:
        self.rules = rules

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> list[Element]:
        import docx
        from docx.table import Table

        try:
            document = docx.Document(str(path))
        except Exception as e:  # python-docx raises several unrelated types on bad input
            raise ParseError(f"{path}: not a readable .docx: {e}") from e

        out: list[Element] = []
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                text = _table_text(block)
                if text:
                    out.append(Element(ElementType.TABLE, text))
                continue
            text = block.text
            if not text.strip():
                continue
            style = block.style.name if block.style is not None else None
            level = self.rules.style_level(style)
            if level is not None:
                out.append(Element(ElementType.HEADING, text, level=level, style=style))
            elif _is_list_item(block, style):
                out.append(Element(ElementType.LIST_ITEM, text, style=style))
            else:
                out.append(Element(ElementType.PARAGRAPH, text, style=style))
        return out


def _is_list_item(paragraph, style: str | None) -> bool:
    if style and style.startswith("List"):
        return True
    ppr = paragraph._p.pPr
    return ppr is not None and ppr.numPr is not None


def _table_text(table) -> str:
    """One line per row, cells separated by " | ". Horizontally merged cells are
    reported once per merged cell by python-docx, so repeats are collapsed."""
    rows = []
    for row in table.rows:
        cells, seen = [], set()
        for cell in row.cells:
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)
