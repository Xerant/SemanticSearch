"""Data contracts. Every pipeline stage is a function between these types.

    Element    one parsed unit of a document
    Document   document-level metadata + ordered Elements
    Chunk      chunk text + chunk-level metadata, referencing its DocumentMeta
    Point      Chunk + dense/sparse vectors
    Candidate  a retrieved chunk + per-signal scores
    Answer     generated text + citations + provenance

The doc/chunk metadata split is structural. Document-level fields (source, ACLs,
hashes, timestamps) exist only on DocumentMeta. A Chunk holds a reference to its
DocumentMeta and has no fields of its own for them, so chunking and tagging code
cannot write source-level metadata. All types are frozen; later stages derive new
values with `dataclasses.replace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Mapping

from semanticsearch.core.versioning import PIPELINE_VERSION

Vector = tuple[float, ...]


# --------------------------------------------------------------------- parsing


class ElementType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Element:
    type: ElementType
    text: str
    level: int = 0  # heading depth, 1 = top level; always 0 for non-headings
    page: int | None = None  # 1-based; None for formats without pages
    style: str | None = None  # source style name, e.g. docx "Heading 2"

    def __post_init__(self) -> None:
        if self.type is ElementType.HEADING and self.level < 1:
            raise ValueError(f"heading level must be >= 1, got {self.level}")
        if self.type is not ElementType.HEADING and self.level != 0:
            raise ValueError(f"non-heading {self.type.value} must have level 0, got {self.level}")
        if self.page is not None and self.page < 1:
            raise ValueError(f"page is 1-based, got {self.page}")


@dataclass(frozen=True, slots=True)
class DocumentMeta:
    """Document-level metadata. One row per document in the manifest store.

    These fields are also copied into each point's payload because the vector
    store needs `acl_groups` (and friends) for filtering. That copy is
    serialisation only: these values are written here and nowhere else.
    """

    document_id: str
    source: str
    mime: str
    sha256: str
    modified_ts: datetime
    acl_groups: tuple[str, ...] = ()
    pack_version: str | None = None


@dataclass(frozen=True, slots=True)
class Document:
    meta: DocumentMeta
    elements: tuple[Element, ...]

    @property
    def document_id(self) -> str:
        return self.meta.document_id


# -------------------------------------------------------------------- chunking


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    doc: DocumentMeta
    ordinal: int  # position within the document, used to expand to neighbouring chunks
    text: str  # never mutated after chunking; enrichment goes into the fields below
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    concept_ids: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()  # matched lexicon labels, fed to the sparse encoder only
    cluster_id: int | None = None
    pipeline_version: str = PIPELINE_VERSION

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError(f"ordinal must be >= 0, got {self.ordinal}")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError(f"page_end {self.page_end} < page_start {self.page_start}")

    @property
    def document_id(self) -> str:
        return self.doc.document_id


# ------------------------------------------------------------------- knowledge


class Relation(str, Enum):
    SELF = "self"  # the matched concept itself
    RELATED = "related"
    BROADER = "broader"
    NARROWER = "narrower"


@dataclass(frozen=True, slots=True)
class ConceptMatch:
    """A lexicon label found in a piece of text. `start`/`end` are character offsets."""

    concept_id: str
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class WeightedConcept:
    concept_id: str
    weight: float
    relation: Relation
    hops: int = 0


@dataclass(frozen=True, slots=True)
class ExpansionPolicy:
    """How strongly each SKOS relation contributes when expanding query concepts."""

    alt_label: float = 1.0
    related: float = 0.4
    broader: float = 0.15
    narrower: float = 0.15
    max_hops: int = 1


# ------------------------------------------------------------------- indexing


@dataclass(frozen=True, slots=True)
class SparseVector:
    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"indices ({len(self.indices)}) and values ({len(self.values)}) differ in length"
            )
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("sparse indices must be unique")


@dataclass(frozen=True, slots=True)
class Point:
    chunk: Chunk
    dense: Vector
    embedder_id: str  # the vector is only meaningful alongside the model that made it
    sparse: SparseVector | None = None


@dataclass(frozen=True, slots=True)
class StoreCapabilities:
    """What a vector store can do natively. The retriever plans around these."""

    native_sparse: bool
    named_vectors: bool
    payload_filters: bool
    array_payload_match: bool  # needed for concept_ids / acl_groups
    server_side_fusion: bool


# -------------------------------------------------------------------- filters


@dataclass(frozen=True, slots=True)
class MatchAny:
    """Payload field equals (or, for array fields, contains) any of `values`."""

    key: str
    values: tuple[str | int, ...]


@dataclass(frozen=True, slots=True)
class Range:
    key: str
    gte: float | datetime | None = None
    lte: float | datetime | None = None


Condition = MatchAny | Range


@dataclass(frozen=True, slots=True)
class Filter:
    must: tuple[Condition, ...] = ()
    must_not: tuple[Condition, ...] = ()


# ------------------------------------------------------------------ retrieval


@dataclass(frozen=True, slots=True)
class SearchRequest:
    k: int
    dense: Vector | None = None
    sparse: SparseVector | None = None
    filter: Filter | None = None

    def __post_init__(self) -> None:
        if self.dense is None and self.sparse is None:
            raise ValueError("SearchRequest needs a dense or sparse query vector")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")


@dataclass(frozen=True, slots=True)
class Candidate:
    """A retrieved chunk. `scores` holds one entry per signal ("dense", "sparse",
    "concept", "rerank"); `fused` is set by the FusionScorer."""

    chunk: Chunk
    scores: Mapping[str, float] = field(default_factory=dict)
    fused: float | None = None


# ----------------------------------------------------------------- generation


@dataclass(frozen=True, slots=True)
class Citation:
    document_id: str
    source: str
    heading_path: tuple[str, ...]
    page_start: int | None
    page_end: int | None

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Citation:
        return cls(
            document_id=chunk.document_id,
            source=chunk.doc.source,
            heading_path=chunk.heading_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
        )

    def label(self) -> str:
        """Render as `[DocID §Section p.Page]`, dropping parts that are unknown."""
        parts = [self.document_id]
        if self.heading_path:
            parts.append(f"§{self.heading_path[-1]}")
        if self.page_start is not None:
            if self.page_end is not None and self.page_end != self.page_start:
                parts.append(f"pp.{self.page_start}-{self.page_end}")
            else:
                parts.append(f"p.{self.page_start}")
        return "[" + " ".join(parts) + "]"


@dataclass(frozen=True, slots=True)
class Prompt:
    system: str
    question: str
    context: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    provenance: Mapping[str, str] = field(default_factory=dict)  # pack/embedder/strategy used
