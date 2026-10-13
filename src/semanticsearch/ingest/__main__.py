"""python -m semanticsearch.ingest CORPUS --workdir DIR [--pack DIR] [--workers N]

Runs one incremental ingest pass and prints the report. Without --pack the template
pack is used, i.e. baseline sectioning with no domain rules.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from semanticsearch.ingest import ingestor_for_pack
from semanticsearch.packs import TEMPLATE_PACK_DIR, load_pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m semanticsearch.ingest", description=__doc__)
    ap.add_argument("corpus", help="corpus root directory")
    ap.add_argument("--workdir", required=True, help="where manifest.sqlite and documents/ live")
    ap.add_argument("--pack", default=str(TEMPLATE_PACK_DIR), help="domain pack directory")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    pack = load_pack(args.pack)
    report = ingestor_for_pack(args.corpus, pack, args.workdir, workers=args.workers).run()

    print(report.summary())
    for source, error in report.failed:
        print(f"  FAILED  {source}: {error}")
    for source in report.empty:
        print(f"  EMPTY   {source} (no text extracted; scanned PDFs need OCR)")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
