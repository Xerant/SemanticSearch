"""Component interfaces. Protocols rather than base classes, so adapters need not
inherit from anything; they are registered by name and resolved from config."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal, Protocol, Sequence, runtime_checkable

from semanticsearch.core.types import (
    Answer,
    Candidate,
    Chunk,
    ConceptMatch,
    Document,
    Element,
    ExpansionPolicy,
    Filter,
    IngestStatus,
    ManifestRecord,
    Point,
    Prompt,
    SearchRequest,
    SparseVector,
    StoreCapabilities,
    Vector,
    WeightedConcept,
)

TextKind = Literal["doc", "query"]


@runtime_checkable
class Parser(Protocol):
    """Format-specific extraction only.

    Parsers return elements; ingest builds DocumentMeta (ID, hash, mtime, ACLs) once
    for every format, so that logic is never duplicated across parsers.
    """

    def supports(self, path: Path) -> bool: ...

    def parse(self, path: Path) -> Sequence[Element]: ...


@runtime_checkable
class ManifestStore(Protocol):
    """One row per document. SQLite today; Postgres for multi-node without touching ingest."""

    def get(self, document_id: str) -> ManifestRecord | None: ...

    def put(self, record: ManifestRecord) -> None: ...

    def records(self, status: IngestStatus | None = None) -> Iterator[ManifestRecord]: ...


@runtime_checkable
class Tokenizer(Protocol):
    """Must be the embedder's own tokenizer, or chunk sizes are measured wrongly."""

    def encode(self, text: str) -> Sequence[int]: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class ConceptStore(Protocol):
    """Backed by SKOS today; a graph DB later without touching callers.

    The same instance tags chunks at index time and queries at query time, so
    both ends share one normalisation path.
    """

    def match(self, text: str) -> list[ConceptMatch]: ...

    def expand(self, ids: Sequence[str], policy: ExpansionPolicy) -> list[WeightedConcept]: ...

    def labels(self, concept_id: str) -> tuple[str, ...]: ...


@runtime_checkable
class Embedder(Protocol):
    id: str
    dim: int

    def embed(self, texts: Sequence[str], kind: TextKind) -> list[Vector]: ...


@runtime_checkable
class SparseEncoder(Protocol):
    """`kind` matters: BM25 weights document terms by frequency and length but
    treats query terms as a set."""

    def encode(self, texts: Sequence[str], kind: TextKind) -> list[SparseVector]: ...


@runtime_checkable
class VectorStore(Protocol):
    capabilities: StoreCapabilities

    def upsert(self, points: Sequence[Point]) -> None: ...

    def search(self, req: SearchRequest) -> list[Candidate]: ...

    def delete_by(self, filter: Filter) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, cands: Sequence[Candidate], k: int) -> list[Candidate]: ...


@runtime_checkable
class Generator(Protocol):
    def generate(self, prompt: Prompt) -> Answer: ...
