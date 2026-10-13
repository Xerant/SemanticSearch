"""Schema for `pack.yaml`. Every section is optional, so a pack with only a name and
version is valid and the engine degrades to its baseline behaviour.

Unknown keys are rejected everywhere: a typo in a weight name would otherwise be
silently ignored and leave the default in place.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator, model_validator

from semanticsearch.core.types import ExpansionPolicy

_SEMVER = r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$"
_SLUG = r"^[a-z0-9][a-z0-9-]*$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LexiconMatch(_Strict):
    case_sensitive: bool = False
    min_label_length: PositiveInt = 3
    stopword_labels: tuple[str, ...] = ()  # labels too generic to match on


class LexiconConfig(_Strict):
    source: Path  # SKOS turtle, relative to the pack root
    scheme_uri: str | None = None
    match: LexiconMatch = LexiconMatch()


class SectioningConfig(_Strict):
    heading_regexes: tuple[str, ...] = ()
    # Empty -> Word's built-in "Title" and "Heading N" styles. Otherwise exactly these
    # styles are headings, at the level in the style name or else their list position.
    docx_styles: tuple[str, ...] = ()
    pdf_heuristic: Literal["font_size_delta", "none"] = "font_size_delta"
    boundary_level: PositiveInt = 1  # headings at or above this depth are hard chunk boundaries
    # A regex match longer than this, or ending in sentence punctuation, is a numbered
    # clause rather than a heading.
    max_heading_chars: PositiveInt = 120

    @field_validator("heading_regexes")
    @classmethod
    def _compiles(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for p in patterns:
            try:
                re.compile(p)
            except re.error as e:
                raise ValueError(f"invalid heading regex {p!r}: {e}") from e
        return patterns


class ChunkingConfig(_Strict):
    """Sizes are in the embedder's tokens. Overlap is whole sentences carried over from
    the previous chunk in the same section, up to `overlap_tokens`."""

    max_tokens: PositiveInt = 512
    overlap_tokens: int = Field(80, ge=0)
    abbreviations: tuple[str, ...] = ()  # extra "Abbr." forms that never end a sentence

    @model_validator(mode="after")
    def _overlap_fits(self) -> ChunkingConfig:
        if self.overlap_tokens * 2 > self.max_tokens:
            raise ValueError("chunking.overlap_tokens must be at most half of max_tokens")
        return self


class EmbedderConfig(_Strict):
    id: str
    dim: PositiveInt
    tokenizer: str | None = None  # defaults to the embedder's own tokenizer

    @property
    def tokenizer_id(self) -> str:
        return self.tokenizer or self.id


# Used when a pack names no embedder. Multilingual, 1024-d, 8192-token context, and
# fits an 11 GB GPU in fp16.
STOCK_EMBEDDER = EmbedderConfig(id="BAAI/bge-m3", dim=1024)


class ExpansionWeights(_Strict):
    alt_label: float = Field(1.0, ge=0)
    related: float = Field(0.4, ge=0)
    broader: float = Field(0.15, ge=0)
    narrower: float = Field(0.15, ge=0)
    max_hops: PositiveInt = 1

    def to_policy(self) -> ExpansionPolicy:
        return ExpansionPolicy(**self.model_dump())


class Weights(_Strict):
    alpha_dense: float = Field(0.55, ge=0)
    beta_sparse: float = Field(0.30, ge=0)
    gamma_concept: float = Field(0.15, ge=0)
    expansion: ExpansionWeights = ExpansionWeights()

    @model_validator(mode="after")
    def _not_all_zero(self) -> Weights:
        if self.alpha_dense + self.beta_sparse + self.gamma_concept == 0:
            raise ValueError("alpha_dense, beta_sparse and gamma_concept cannot all be 0")
        return self


class ClustersConfig(_Strict):
    model: Path | None = None
    enabled: bool = False  # off until the eval harness shows it doesn't cost recall
    top_n_centroids: PositiveInt = 2

    @model_validator(mode="after")
    def _model_when_enabled(self) -> ClustersConfig:
        if self.enabled and self.model is None:
            raise ValueError("clusters.enabled is true but clusters.model is not set")
        return self


class EvalConfig(_Strict):
    gold: Path | None = None


class PackManifest(_Strict):
    name: str = Field(pattern=_SLUG)
    version: str = Field(pattern=_SEMVER)
    lexicon: LexiconConfig | None = None
    sectioning: SectioningConfig = SectioningConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embedder: EmbedderConfig | None = None  # None -> STOCK_EMBEDDER
    weights: Weights = Weights()
    clusters: ClustersConfig = ClustersConfig()
    eval: EvalConfig = EvalConfig()
