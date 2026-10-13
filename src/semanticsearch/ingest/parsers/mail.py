"""Email (.eml) via the standard library. The subject becomes a level-1 heading,
From/To/Date an OTHER element, and the body (plain text preferred, else HTML stripped
to text) blank-line paragraphs. Attachments are not extracted yet."""

from __future__ import annotations

import email
import email.policy
import re
from html.parser import HTMLParser
from pathlib import Path

from semanticsearch.core.errors import ParseError
from semanticsearch.core.types import Element, ElementType

_PARA_BREAK = re.compile(r"\r?\n\s*\r?\n")
_HEADERS = ("From", "To", "Cc", "Date")
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol"}


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _HtmlText()
    p.feed(html)
    p.close()
    return "".join(p.parts)


class EmlParser:
    extensions = (".eml",)

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def parse(self, path: Path) -> list[Element]:
        try:
            with path.open("rb") as f:
                msg = email.message_from_binary_file(f, policy=email.policy.default)
        except Exception as e:
            raise ParseError(f"{path}: not a readable email: {e}") from e

        out: list[Element] = []
        if subject := str(msg.get("Subject", "")).strip():
            out.append(Element(ElementType.HEADING, subject, level=1))
        headers = [f"{h}: {msg[h]}" for h in _HEADERS if msg.get(h)]
        if headers:
            out.append(Element(ElementType.OTHER, "\n".join(headers)))

        part = msg.get_body(preferencelist=("plain", "html"))
        if part is not None:
            try:
                body = part.get_content()
            except (LookupError, UnicodeError) as e:
                raise ParseError(f"{path}: undecodable body: {e}") from e
            if part.get_content_subtype() == "html":
                body = html_to_text(body)
            for para in _PARA_BREAK.split(body):
                if para.strip():
                    out.append(Element(ElementType.PARAGRAPH, para))
        return out
