"""Version stamps and stable identifiers.

Every indexed point carries `pack_version`, `embedder_id` and `pipeline_version`
so reindexing is a filter ("everything where embedder_id != current"), not a guess.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path, PurePosixPath

PIPELINE_VERSION = "0.1.0"

# Fixed namespace so IDs are reproducible across machines and runs. Never change it:
# doing so re-keys every point in every index.
_ID_NAMESPACE = uuid.UUID("5d0c7a52-3b8e-4f43-9a61-2c3f1e8b7d10")


def document_id_for(source_key: str) -> str:
    """Stable document ID derived from the document's location, not its content.

    `source_key` should be the path relative to the corpus root, so moving the
    corpus does not re-key it. Content changes keep the same ID; `sha256` on the
    document metadata is what detects them. Separators are normalised explicitly
    so Windows and POSIX ingest hosts produce the same ID. Returns a UUID string,
    which Qdrant accepts directly as a point ID.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, PurePosixPath(source_key.replace("\\", "/")).as_posix()))


def chunk_id_for(document_id: str, ordinal: int) -> str:
    """Deterministic chunk ID, so re-upserting a document overwrites in place.

    A re-chunked document can produce fewer chunks than before, so ingest must
    delete by `document_id` before upserting; ordinals alone don't clear stale tails.
    """
    if ordinal < 0:
        raise ValueError(f"ordinal must be >= 0, got {ordinal}")
    return str(uuid.uuid5(uuid.UUID(document_id), str(ordinal)))


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size):
            digest.update(block)
    return digest.hexdigest()
