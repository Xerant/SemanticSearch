"""Tokenizer adapters. Chunk sizes must be counted in the embedder's own tokens: any
other tokenizer (tiktoken, a different model's) mis-sizes chunks, and the embedder then
silently truncates or under-fills them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from semanticsearch.core.errors import SemanticSearchError


class HFTokenizer:
    """A Hugging Face `tokenizers` tokenizer, counting tokens without special tokens.

    `source` is a `tokenizer.json` path, a model directory containing one, or a hub ID
    such as "BAAI/bge-m3". Air-gapped hosts use a path, or a pre-populated HF cache
    with HF_HUB_OFFLINE=1. The loaded tokenizer pickles, so it can go to worker processes.
    """

    def __init__(self, source: str | Path) -> None:
        from tokenizers import Tokenizer

        self.source = str(source)
        path = Path(source)
        if path.is_dir():
            path = path / "tokenizer.json"
        try:
            if path.is_file():
                self._tok = Tokenizer.from_file(str(path))
            else:
                self._tok = Tokenizer.from_pretrained(self.source)
        except Exception as e:
            raise SemanticSearchError(
                f"cannot load tokenizer {self.source!r}: {e}. On an offline host, point "
                "embedder.tokenizer at a local tokenizer.json or model directory."
            ) from e

    def encode(self, text: str) -> Sequence[int]:
        return self._tok.encode(text, add_special_tokens=False).ids
