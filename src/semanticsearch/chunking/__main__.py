"""python -m semanticsearch.chunking --workdir DIR [--pack DIR] [--show SOURCE]

Chunks every OK document from an ingest workdir and prints size statistics, for tuning
`chunking.max_tokens` / `overlap_tokens`. With --show, prints one document's chunks.
Writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import median

from semanticsearch.chunking import chunker_for_pack
from semanticsearch.core.types import IngestStatus
from semanticsearch.ingest import DOCUMENTS_DIR, MANIFEST_FILE, DocumentStore, SqliteManifestStore
from semanticsearch.packs import TEMPLATE_PACK_DIR, load_pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m semanticsearch.chunking", description=__doc__)
    ap.add_argument("--workdir", required=True, help="an ingest workdir")
    ap.add_argument("--pack", default=str(TEMPLATE_PACK_DIR), help="domain pack directory")
    ap.add_argument("--show", metavar="SOURCE", help="print the chunks of this source path")
    args = ap.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    workdir = Path(args.workdir)
    pack = load_pack(args.pack)
    chunker = chunker_for_pack(pack)
    documents = DocumentStore(workdir / DOCUMENTS_DIR)

    sizes: list[int] = []
    with SqliteManifestStore(workdir / MANIFEST_FILE) as manifest:
        for rec in manifest.records(IngestStatus.OK):
            if args.show and rec.meta.source != args.show:
                continue
            chunks = chunker.chunk(documents.get(rec.document_id))
            sizes.extend(chunker.count(c.text) for c in chunks)
            if args.show:
                for c in chunks:
                    pages = f"p.{c.page_start}-{c.page_end}" if c.page_start else "-"
                    print(f"--- #{c.ordinal} {chunker.count(c.text)} tok {pages} § {' > '.join(c.heading_path)}")
                    print(c.text)

    if not sizes:
        print("no chunks")
        return 1
    over = sum(s > chunker.max_tokens for s in sizes)
    print(
        f"{len(sizes)} chunks; tokens min {min(sizes)}, median {median(sizes):.0f}, "
        f"max {max(sizes)}; over max_tokens ({chunker.max_tokens}): {over}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
