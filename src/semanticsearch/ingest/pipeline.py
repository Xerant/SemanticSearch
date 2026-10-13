"""Incremental ingest: walk a corpus, parse what changed, record every document.

Phase 1 (main process) plans. Each supported file is stat-ed; its hash is reused from
the manifest when size and mtime are unchanged, otherwise computed. Byte-identical
files are de-duplicated, with an already-ingested owner kept in preference to a new
copy so ownership never flips between runs. Each file is then classified as unchanged,
duplicate, or to-parse. Manifest rows whose files have disappeared are marked REMOVED.

Phase 2 parses and normalises in a process pool (CPU-bound), writes each normalised
document to the DocumentStore and then its manifest row, so a row never points at a
missing document. Only the main process touches the manifest.

Use one manifest per corpus root: files under the root that are not found are marked
REMOVED, so pointing a manifest at a sub-folder marks everything else removed.

Embedding is deliberately not done here. Loading a model in every worker is how the
previous build ran out of GPU memory; indexing is a single batched consumer of the
report's `to_index` list.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from semanticsearch.core.errors import ParseError
from semanticsearch.core.protocols import ManifestStore, Parser
from semanticsearch.core.types import Document, DocumentMeta, Element, IngestStatus, ManifestRecord
from semanticsearch.core.versioning import PIPELINE_VERSION, document_id_for, sha256_file
from semanticsearch.ingest.document_store import DocumentStore
from semanticsearch.ingest.normalize import normalize_elements
from semanticsearch.ingest.parsers import find_parser, mime_for
from semanticsearch.ingest.sectioning import SectionRules

log = logging.getLogger(__name__)

AclResolver = Callable[[str], Sequence[str]]
"""Maps a corpus-relative source path to the groups allowed to read it."""


def no_acl(source: str) -> tuple[str, ...]:
    return ()


_LIVE = (IngestStatus.OK, IngestStatus.EMPTY)


@dataclass
class IngestReport:
    to_index: list[str] = field(default_factory=list)  # parsed OK this run
    stale: list[str] = field(default_factory=list)  # had points; delete them before (re)indexing
    unchanged: list[str] = field(default_factory=list)
    empty: list[str] = field(default_factory=list)  # sources with no text; likely need OCR
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (source, duplicate_of)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (source, error)
    unsupported: list[str] = field(default_factory=list)  # sources with no parser

    def summary(self) -> str:
        return (
            f"indexed {len(self.to_index)}, unchanged {len(self.unchanged)}, "
            f"empty {len(self.empty)}, duplicate {len(self.duplicates)}, "
            f"removed {len(self.removed)}, failed {len(self.failed)}, "
            f"unsupported {len(self.unsupported)}"
        )


@dataclass
class _Entry:
    path: Path
    meta: DocumentMeta
    size: int
    parser: Parser
    prev: ManifestRecord | None


def _parse_task(parser: Parser, path: Path, rules: SectionRules) -> tuple[Element, ...]:
    """Runs in a worker process; must stay a module-level function to be picklable."""
    return normalize_elements(parser.parse(path), rules)


class Ingestor:
    def __init__(
        self,
        corpus_root: Path | str,
        *,
        parsers: Sequence[Parser],
        rules: SectionRules,
        manifest: ManifestStore,
        documents: DocumentStore,
        pack_version: str | None = None,
        acl: AclResolver = no_acl,
        workers: int = 1,
    ) -> None:
        self.root = Path(corpus_root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.parsers = list(parsers)
        self.rules = rules
        self.manifest = manifest
        self.documents = documents
        self.pack_version = pack_version
        self.acl = acl
        self.workers = max(1, workers)

    # ------------------------------------------------------------------ plan

    def _walk(self):
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in sorted(filenames):
                if name.startswith((".", "~$")):  # hidden files, Office lock files
                    continue
                yield Path(dirpath) / name

    def _entry(self, path: Path, parser: Parser) -> _Entry:
        source = PurePosixPath(path.relative_to(self.root).as_posix()).as_posix()
        doc_id = document_id_for(source)
        prev = self.manifest.get(doc_id)
        st = path.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        same_file = (
            prev is not None
            and prev.status is not IngestStatus.FAILED
            and prev.size_bytes == st.st_size
            and prev.meta.modified_ts == mtime
        )
        meta = DocumentMeta(
            document_id=doc_id,
            source=source,
            mime=mime_for(path),
            sha256=prev.meta.sha256 if same_file else sha256_file(path),
            modified_ts=mtime,
            acl_groups=tuple(self.acl(source)),
            pack_version=self.pack_version,
        )
        return _Entry(path, meta, st.st_size, parser, prev)

    @staticmethod
    def _was_owner(e: _Entry) -> bool:
        p = e.prev
        return p is not None and p.status in _LIVE and p.meta.sha256 == e.meta.sha256

    def _is_current(self, e: _Entry) -> bool:
        p = e.prev
        return (
            p is not None
            and p.status in _LIVE
            and p.meta.sha256 == e.meta.sha256
            and p.meta.acl_groups == e.meta.acl_groups
            and p.meta.pack_version == self.pack_version
            and p.pipeline_version == PIPELINE_VERSION
            and (p.status is IngestStatus.EMPTY or e.meta.document_id in self.documents)
        )

    # ------------------------------------------------------------------- run

    def run(self) -> IngestReport:
        report = IngestReport()
        now = datetime.now(timezone.utc)

        entries: list[_Entry] = []
        for path in self._walk():
            parser = find_parser(self.parsers, path)
            if parser is None:
                report.unsupported.append(path.relative_to(self.root).as_posix())
                continue
            entries.append(self._entry(path, parser))
        present = {e.meta.document_id for e in entries}

        by_sha: dict[str, list[_Entry]] = {}
        for e in entries:
            by_sha.setdefault(e.meta.sha256, []).append(e)
        owner_of: dict[str, _Entry] = {
            sha: next((e for e in group if self._was_owner(e)), group[0])
            for sha, group in by_sha.items()
        }

        def record(e: _Entry, status: IngestStatus, **kw) -> ManifestRecord:
            return ManifestRecord(meta=e.meta, size_bytes=e.size, status=status, ingested_at=now, **kw)

        def mark_stale(e: _Entry) -> None:
            if e.prev is not None and e.prev.status is IngestStatus.OK:
                report.stale.append(e.meta.document_id)

        to_parse: list[_Entry] = []
        for e in entries:
            doc_id = e.meta.document_id
            owner = owner_of[e.meta.sha256]
            if owner is not e:
                mark_stale(e)
                self.documents.delete(doc_id)
                self.manifest.put(record(e, IngestStatus.DUPLICATE, duplicate_of=owner.meta.document_id))
                report.duplicates.append((e.meta.source, owner.meta.source))
            elif self._is_current(e):
                assert e.prev is not None
                if e.prev.meta != e.meta or e.prev.size_bytes != e.size:  # touched, not edited
                    self.manifest.put(replace(e.prev, meta=e.meta, size_bytes=e.size))
                report.unchanged.append(doc_id)
            else:
                mark_stale(e)
                to_parse.append(e)

        for rec in self.manifest.records():
            if rec.document_id not in present and rec.status is not IngestStatus.REMOVED:
                if rec.status is IngestStatus.OK:
                    report.stale.append(rec.document_id)
                self.documents.delete(rec.document_id)
                self.manifest.put(replace(rec, status=IngestStatus.REMOVED, ingested_at=now))
                report.removed.append(rec.document_id)

        log.info("ingest plan: %d to parse, %d unchanged", len(to_parse), len(report.unchanged))
        for e, outcome in self._parse_all(to_parse):
            doc_id = e.meta.document_id
            if isinstance(outcome, BaseException):
                msg = str(outcome) if isinstance(outcome, ParseError) else f"{type(outcome).__name__}: {outcome}"
                self.documents.delete(doc_id)
                self.manifest.put(record(e, IngestStatus.FAILED, error=msg))
                report.failed.append((e.meta.source, msg))
                log.warning("failed: %s", msg)
            elif not outcome:
                self.documents.delete(doc_id)
                self.manifest.put(record(e, IngestStatus.EMPTY))
                report.empty.append(e.meta.source)
            else:
                self.documents.put(Document(meta=e.meta, elements=outcome))
                self.manifest.put(record(e, IngestStatus.OK, n_elements=len(outcome)))
                report.to_index.append(doc_id)

        log.info("ingest done: %s", report.summary())
        return report

    def _parse_all(self, entries: list[_Entry]):
        """Yield (entry, elements-or-exception), in completion order when parallel."""
        if self.workers == 1 or len(entries) < 2:
            for e in entries:
                try:
                    yield e, _parse_task(e.parser, e.path, self.rules)
                except Exception as exc:
                    yield e, exc
            return

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures: dict[Future, _Entry] = {
                pool.submit(_parse_task, e.parser, e.path, self.rules): e for e in entries
            }
            for i, fut in enumerate(as_completed(futures), 1):
                exc = fut.exception()
                yield futures[fut], exc if exc is not None else fut.result()
                if i % 100 == 0:
                    log.info("parsed %d/%d", i, len(entries))
