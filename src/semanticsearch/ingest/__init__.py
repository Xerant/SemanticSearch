"""Ingest: parse source files into normalised Documents and keep a manifest of them.

    corpus file --Parser--> Elements --normalize--> Document --> DocumentStore (JSON)
                                                            \\-> ManifestStore (one row)

Parsers know formats; the Ingestor alone builds document metadata (ID, hash, mtime,
ACLs), so that logic exists once for every format.
"""

from __future__ import annotations

from pathlib import Path

from semanticsearch.ingest.document_store import DocumentStore, document_from_dict, document_to_dict
from semanticsearch.ingest.manifest_store import SqliteManifestStore
from semanticsearch.ingest.normalize import clean_text, join_lines, normalize_elements
from semanticsearch.ingest.parsers import default_parsers, find_parser, mime_for
from semanticsearch.ingest.pipeline import AclResolver, IngestReport, Ingestor, no_acl
from semanticsearch.ingest.sectioning import SectionRules
from semanticsearch.packs.loader import DomainPack

MANIFEST_FILE = "manifest.sqlite"
DOCUMENTS_DIR = "documents"


def ingestor_for_pack(
    corpus_root: Path | str,
    pack: DomainPack,
    workdir: Path | str,
    *,
    acl: AclResolver = no_acl,
    workers: int = 1,
) -> Ingestor:
    """Wire an Ingestor from a domain pack, storing state under `workdir`:
    `manifest.sqlite` and `documents/<document_id>.json`."""
    workdir = Path(workdir)
    rules = SectionRules.from_config(pack.manifest.sectioning)
    return Ingestor(
        corpus_root,
        parsers=default_parsers(rules),
        rules=rules,
        manifest=SqliteManifestStore(workdir / MANIFEST_FILE),
        documents=DocumentStore(workdir / DOCUMENTS_DIR),
        pack_version=pack.version_stamp,
        acl=acl,
        workers=workers,
    )


__all__ = [
    "AclResolver",
    "DocumentStore",
    "IngestReport",
    "Ingestor",
    "SectionRules",
    "SqliteManifestStore",
    "clean_text",
    "default_parsers",
    "document_from_dict",
    "document_to_dict",
    "find_parser",
    "ingestor_for_pack",
    "join_lines",
    "mime_for",
    "no_acl",
    "normalize_elements",
]
