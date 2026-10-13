"""Normalised documents on disk, one JSON file per document.

Parsing is the expensive stage, so its output is kept: re-chunking or re-tagging after
a pack change reads these files instead of re-parsing the corpus. Writes go to a temp
file and are renamed into place, so a crash never leaves a half-written document.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from semanticsearch.core.types import Document, DocumentMeta, Element, ElementType

FORMAT_VERSION = 1


def document_to_dict(doc: Document) -> dict[str, Any]:
    m = doc.meta
    return {
        "format": FORMAT_VERSION,
        "meta": {
            "document_id": m.document_id,
            "source": m.source,
            "mime": m.mime,
            "sha256": m.sha256,
            "modified_ts": m.modified_ts.isoformat(),
            "acl_groups": list(m.acl_groups),
            "pack_version": m.pack_version,
        },
        "elements": [
            {"type": e.type.value, "text": e.text, "level": e.level, "page": e.page, "style": e.style}
            for e in doc.elements
        ],
    }


def document_from_dict(d: dict[str, Any]) -> Document:
    if d.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported document format {d.get('format')!r}")
    m = d["meta"]
    meta = DocumentMeta(
        document_id=m["document_id"],
        source=m["source"],
        mime=m["mime"],
        sha256=m["sha256"],
        modified_ts=datetime.fromisoformat(m["modified_ts"]),
        acl_groups=tuple(m["acl_groups"]),
        pack_version=m["pack_version"],
    )
    elements = tuple(
        Element(ElementType(e["type"]), e["text"], level=e["level"], page=e["page"], style=e["style"])
        for e in d["elements"]
    )
    return Document(meta=meta, elements=elements)


class DocumentStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, document_id: str) -> Path:
        return self.root / f"{document_id}.json"

    def put(self, doc: Document) -> None:
        path = self._path(doc.document_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(document_to_dict(doc), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def get(self, document_id: str) -> Document:
        return document_from_dict(json.loads(self._path(document_id).read_text(encoding="utf-8")))

    def delete(self, document_id: str) -> None:
        self._path(document_id).unlink(missing_ok=True)

    def __contains__(self, document_id: str) -> bool:
        return self._path(document_id).is_file()
