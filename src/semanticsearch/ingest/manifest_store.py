"""SQLite implementation of ManifestStore: one row per document, single node.

Timestamps are stored as ISO-8601 UTC strings and ACL groups as a JSON array. The
schema version lives in `PRAGMA user_version`; an unknown version is refused rather
than silently misread.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator

from semanticsearch.core.errors import SemanticSearchError
from semanticsearch.core.types import DocumentMeta, IngestStatus, ManifestRecord

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    mime             TEXT NOT NULL,
    sha256           TEXT NOT NULL,
    size_bytes       INTEGER NOT NULL,
    modified_ts      TEXT NOT NULL,
    acl_groups       TEXT NOT NULL,
    pack_version     TEXT,
    pipeline_version TEXT NOT NULL,
    status           TEXT NOT NULL,
    n_elements       INTEGER NOT NULL,
    error            TEXT,
    duplicate_of     TEXT,
    ingested_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS ix_documents_sha256 ON documents(sha256);
"""

_COLUMNS = (
    "document_id, source, mime, sha256, size_bytes, modified_ts, acl_groups, pack_version, "
    "pipeline_version, status, n_elements, error, duplicate_of, ingested_at"
)


class SqliteManifestStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path))
        self._db.execute("PRAGMA journal_mode=WAL")
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            with self._db:
                self._db.executescript(_SCHEMA)
                self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        elif version != SCHEMA_VERSION:
            raise SemanticSearchError(
                f"{self.path}: manifest schema v{version}, this build expects v{SCHEMA_VERSION}"
            )

    def get(self, document_id: str) -> ManifestRecord | None:
        row = self._db.execute(
            f"SELECT {_COLUMNS} FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return _from_row(row) if row else None

    def put(self, record: ManifestRecord) -> None:
        m = record.meta
        with self._db:
            self._db.execute(
                f"INSERT OR REPLACE INTO documents ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    m.document_id,
                    m.source,
                    m.mime,
                    m.sha256,
                    record.size_bytes,
                    m.modified_ts.isoformat(),
                    json.dumps(list(m.acl_groups)),
                    m.pack_version,
                    record.pipeline_version,
                    record.status.value,
                    record.n_elements,
                    record.error,
                    record.duplicate_of,
                    record.ingested_at.isoformat(),
                ),
            )

    def records(self, status: IngestStatus | None = None) -> Iterator[ManifestRecord]:
        if status is None:
            cur = self._db.execute(f"SELECT {_COLUMNS} FROM documents ORDER BY source")
        else:
            cur = self._db.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE status = ? ORDER BY source",
                (status.value,),
            )
        for row in cur.fetchall():
            yield _from_row(row)

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> SqliteManifestStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _from_row(row: tuple) -> ManifestRecord:
    (doc_id, source, mime, sha, size, mtime, acl, pack_v, pipe_v, status, n, err, dup, at) = row
    return ManifestRecord(
        meta=DocumentMeta(
            document_id=doc_id,
            source=source,
            mime=mime,
            sha256=sha,
            modified_ts=datetime.fromisoformat(mtime),
            acl_groups=tuple(json.loads(acl)),
            pack_version=pack_v,
        ),
        size_bytes=size,
        status=IngestStatus(status),
        ingested_at=datetime.fromisoformat(at),
        pipeline_version=pipe_v,
        n_elements=n,
        error=err,
        duplicate_of=dup,
    )
