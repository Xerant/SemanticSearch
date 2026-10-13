"""Format parsers. Each takes a path and returns Elements; none of them builds document
metadata. Parser libraries are imported inside `parse`, so a missing optional library
only breaks its own format.

Parsers must stay picklable (plain attributes only): ingest ships them to worker
processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from semanticsearch.core.protocols import Parser
from semanticsearch.ingest.parsers.mail import EmlParser
from semanticsearch.ingest.parsers.pdf import PdfParser
from semanticsearch.ingest.parsers.tabular import CsvParser, XlsxParser
from semanticsearch.ingest.parsers.text import TextParser
from semanticsearch.ingest.parsers.word import DocxParser
from semanticsearch.ingest.sectioning import SectionRules

# Fixed rather than `mimetypes.guess_type`, which reads the Windows registry and so
# varies by host.
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".eml": "message/rfc822",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}


def mime_for(path: Path) -> str:
    return MIME_TYPES.get(path.suffix.lower(), "application/octet-stream")


def default_parsers(rules: SectionRules) -> list[Parser]:
    return [PdfParser(rules), DocxParser(rules), XlsxParser(), CsvParser(), EmlParser(), TextParser()]


def find_parser(parsers: Sequence[Parser], path: Path) -> Parser | None:
    return next((p for p in parsers if p.supports(path)), None)


__all__ = [
    "MIME_TYPES",
    "CsvParser",
    "DocxParser",
    "EmlParser",
    "PdfParser",
    "TextParser",
    "XlsxParser",
    "default_parsers",
    "find_parser",
    "mime_for",
]
