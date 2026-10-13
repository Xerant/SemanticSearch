"""Spreadsheets (.xlsx via openpyxl) and CSV.

Each data row becomes one TABLE element rendered as "header: value | header: value",
so a row stays a self-contained, retrievable record. The first non-empty row of each
sheet is taken as the header. Each XLSX sheet starts with a level-1 heading naming it.
Cached formula results are read, not formulas. Legacy .xls is not supported.
"""

from __future__ import annotations

import csv
import io
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable, Sequence

from semanticsearch.core.errors import ParseError
from semanticsearch.core.types import Element, ElementType
from semanticsearch.ingest.parsers.text import read_text

_DELIMITERS = (",", ";", "\t", "|")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds").removesuffix(" 00:00:00")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return " ".join(str(value).split())


def rows_to_elements(rows: Iterable[Sequence[object]]) -> list[Element]:
    header: list[str] | None = None
    out = []
    for raw in rows:
        cells = [_cell(v) for v in raw]
        if not any(cells):
            continue
        if header is None:
            header = [h or f"column {i + 1}" for i, h in enumerate(cells)]
            continue
        pairs = []
        for i, v in enumerate(cells):
            if v:
                name = header[i] if i < len(header) else f"column {i + 1}"
                pairs.append(f"{name}: {v}")
        out.append(Element(ElementType.TABLE, " | ".join(pairs)))
    return out


class XlsxParser:
    extensions = (".xlsx", ".xlsm")

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> list[Element]:
        import openpyxl

        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        except Exception as e:
            raise ParseError(f"{path}: not a readable workbook: {e}") from e
        try:
            out: list[Element] = []
            for ws in wb.worksheets:
                rows = rows_to_elements(ws.iter_rows(values_only=True))
                if rows:
                    out.append(Element(ElementType.HEADING, ws.title, level=1))
                    out.extend(rows)
            return out
        finally:
            wb.close()


class CsvParser:
    extensions = (".csv", ".tsv")

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> list[Element]:
        text = read_text(path)
        delimiter = "\t" if path.suffix.lower() == ".tsv" else _guess_delimiter(text)
        return rows_to_elements(csv.reader(io.StringIO(text), delimiter=delimiter))


def _guess_delimiter(text: str) -> str:
    """The candidate occurring most often in the header line. csv.Sniffer is avoided:
    it gives up on ragged rows and then treats each whole line as one field."""
    header = next((line for line in text.splitlines() if line.strip()), "")
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(counts, key=counts.__getitem__)
    return best if counts[best] else ","
