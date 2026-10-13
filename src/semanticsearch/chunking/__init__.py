"""Chunking: Documents into Chunks that respect section boundaries, sized in the
embedder's tokens."""

from __future__ import annotations

from semanticsearch.chunking.chunker import StructureAwareChunker
from semanticsearch.chunking.sentences import DEFAULT_ABBREVIATIONS, SentenceSplitter
from semanticsearch.chunking.tokenizers import HFTokenizer
from semanticsearch.core.protocols import Tokenizer
from semanticsearch.packs.loader import DomainPack


def chunker_for_pack(pack: DomainPack, tokenizer: Tokenizer | None = None) -> StructureAwareChunker:
    """Build the chunker a pack describes. The tokenizer defaults to the pack's embedder's
    own (`embedder.tokenizer`, else the embedder ID)."""
    cfg = pack.manifest.chunking
    return StructureAwareChunker(
        tokenizer or HFTokenizer(pack.embedder.tokenizer_id),
        max_tokens=cfg.max_tokens,
        overlap_tokens=cfg.overlap_tokens,
        boundary_level=pack.manifest.sectioning.boundary_level,
        splitter=SentenceSplitter(cfg.abbreviations),
    )


__all__ = [
    "DEFAULT_ABBREVIATIONS",
    "HFTokenizer",
    "SentenceSplitter",
    "StructureAwareChunker",
    "chunker_for_pack",
]
