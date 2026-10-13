"""Structure-aware chunking (principle #2): a chunk never spans a major section boundary.

1. Sections. Walk the elements keeping a heading stack. A heading at or above the
   pack's `boundary_level` closes the current section: a hard boundary. Levels are
   judged relative to the document (a lone title is skipped; see `_hard_boundary`).
2. Units. Each element becomes one or more units of at most `max_tokens`. Prose is split
   into sentences, a sentence still too long into word windows, and a single word
   longer than a chunk is cut by characters. A table that doesn't fit is split into
   row groups, each repeating the header row. Every unit keeps its element's page,
   which the review of the original spec flagged as lost when sections are
   concatenated before splitting.
3. Packing. Units fill chunks greedily. A sub-heading starts a new chunk if the
   current one is at least half full (a soft boundary), and a heading is never left
   dangling at the end of a chunk. A chunk that continues the same section starts
   with whole trailing sentences of the previous one, up to `overlap_tokens`.

Chunk text is assembled from the source elements and never edited afterwards:
sentences of one element join with a space, separate elements with a newline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from semanticsearch.chunking.sentences import SentenceSplitter
from semanticsearch.core.protocols import Tokenizer
from semanticsearch.core.types import Chunk, Document, DocumentMeta, Element, ElementType
from semanticsearch.core.versioning import chunk_id_for

SOFT_BREAK_FILL = 0.5  # a sub-heading starts a new chunk once the current one is this full

_PROSE = (ElementType.PARAGRAPH, ElementType.LIST_ITEM)


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    tokens: int
    element: int  # index of the source element
    path: tuple[str, ...]
    page: int | None
    heading: bool = False
    sentence: bool = False  # a whole prose sentence: eligible for overlap


class StructureAwareChunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        max_tokens: int = 512,
        overlap_tokens: int = 80,
        boundary_level: int = 1,
        splitter: SentenceSplitter | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        if not 0 <= overlap_tokens * 2 <= max_tokens:
            raise ValueError("overlap_tokens must be between 0 and max_tokens / 2")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.boundary_level = boundary_level
        self.splitter = splitter or SentenceSplitter()

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ------------------------------------------------------------------ public

    def chunk(self, doc: Document) -> list[Chunk]:
        groups = [g for section in self._sections(doc.elements) for g in self._pack(section)]
        return [_make_chunk(doc.meta, i, units) for i, units in enumerate(groups)]

    # ---------------------------------------------------------------- sections

    def _sections(self, elements: Sequence[Element]) -> Iterator[list[_Unit]]:
        boundary = _hard_boundary(elements, self.boundary_level)
        stack: list[tuple[int, str]] = []
        section: list[_Unit] = []
        for i, el in enumerate(elements):
            if el.type is ElementType.HEADING:
                if boundary(i, el) and section:
                    yield section
                    section = []
                while stack and stack[-1][0] >= el.level:
                    stack.pop()
                stack.append((el.level, el.text))
            path = tuple(text for _, text in stack)
            section.extend(self._units(el, i, path))
        if section:
            yield section

    # ------------------------------------------------------------------- units

    def _units(self, el: Element, i: int, path: tuple[str, ...]) -> list[_Unit]:
        def unit(text: str, tokens: int, **kw) -> _Unit:
            return _Unit(text, tokens, i, path, el.page, **kw)

        if el.type is ElementType.HEADING:
            n = self.count(el.text)
            if n <= self.max_tokens:
                return [unit(el.text, n, heading=True)]
            return [unit(t, k) for t, k in self._windows(el.text)]

        if el.type in _PROSE:
            out = []
            for s in self.splitter.split(el.text):
                n = self.count(s)
                if n <= self.max_tokens:
                    out.append(unit(s, n, sentence=True))
                else:
                    out.extend(unit(t, k) for t, k in self._windows(s))
            return out

        # TABLE and OTHER: line-structured; keep whole if it fits, else group lines.
        n = self.count(el.text)
        if n <= self.max_tokens:
            return [unit(el.text, n)]
        lines = el.text.split("\n")
        header = lines[0] if el.type is ElementType.TABLE and len(lines) > 1 else None
        return [unit(t, k) for t, k in self._line_groups(lines, header)]

    def _line_groups(self, lines: list[str], header: str | None) -> Iterator[tuple[str, int]]:
        """Greedy groups of whole lines. Table groups after the first repeat the header
        row, so a continuation still says what its columns are."""
        header_cost = self.count(header) if header else 0
        if header_cost * 2 > self.max_tokens:  # a header that big can't be repeated usefully
            header, header_cost = None, 0
        fresh = ([header], header_cost) if header else ([], 0)
        group: list[str] = []  # the first group starts with the header row itself
        total = 0
        for line in lines:
            n = self.count(line)
            has_rows = bool(group) and group != fresh[0]
            if n + header_cost > self.max_tokens:  # one line too long: word windows
                if has_rows:
                    yield "\n".join(group), total
                yield from self._windows(line)
                group, total = list(fresh[0]), fresh[1]
                continue
            if has_rows and total + n > self.max_tokens:
                yield "\n".join(group), total
                group, total = list(fresh[0]), fresh[1]
            group.append(line)
            total += n
        if group and group != fresh[0]:
            yield "\n".join(group), total

    def _windows(self, text: str) -> Iterator[tuple[str, int]]:
        """Greedy word windows, measured in tokens. Only a single word longer than a whole
        chunk is cut mid-word."""
        words: list[str] = []
        total = 0
        for word in text.split():
            n = self.count(word)
            if n > self.max_tokens:
                if words:
                    yield " ".join(words), total
                    words, total = [], 0
                yield from self._hard_split(word)
                continue
            if words and total + n > self.max_tokens:
                yield " ".join(words), total
                words, total = [], 0
            words.append(word)
            total += n
        if words:
            yield " ".join(words), total

    def _hard_split(self, word: str) -> Iterator[tuple[str, int]]:
        while word:
            lo, hi = 1, len(word)  # longest prefix within budget, by binary search
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.count(word[:mid]) <= self.max_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            piece, word = word[:lo], word[lo:]
            yield piece, self.count(piece)

    # ----------------------------------------------------------------- packing

    def _pack(self, units: list[_Unit]) -> list[list[_Unit]]:
        out: list[list[_Unit]] = []
        cur: list[_Unit] = []
        size = 0

        for u in units:
            soft_break = u.heading and size >= SOFT_BREAK_FILL * self.max_tokens
            if cur and (soft_break or size + u.tokens > self.max_tokens):
                # Never end a chunk on a heading: carry trailing headings forward.
                split = len(cur)
                while split and cur[split - 1].heading:
                    split -= 1
                body, carried = cur[:split], cur[split:]
                if body:
                    out.append(body)
                overlap = [] if u.heading or carried else self._overlap(body)
                cur = overlap + carried + [u]
                while overlap and _size(cur) > self.max_tokens:
                    overlap.pop(0)
                    cur = overlap + carried + [u]
                if _size(cur) > self.max_tokens:  # oversized heading run: path keeps them
                    cur = [u]
                size = _size(cur)
            else:
                cur.append(u)
                size += u.tokens

        if any(not u.heading for u in cur):
            while cur[-1].heading:
                cur.pop()
            out.append(cur)
        return out

    def _overlap(self, body: list[_Unit]) -> list[_Unit]:
        """Whole trailing sentences of `body`, up to the overlap budget."""
        picked: list[_Unit] = []
        total = 0
        for u in reversed(body):
            if not u.sentence or total + u.tokens > self.overlap_tokens:
                break
            picked.insert(0, u)
            total += u.tokens
        return picked if len(picked) < len(body) else []  # never repeat a whole chunk


def _hard_boundary(elements: Sequence[Element], boundary_level: int):
    """Which headings close a section, judged against the document's own levels.

    A first heading that is alone at the shallowest level is the document title
    (Markdown "# Title" then "## sections"; Word "Title" style) and is never a boundary.
    `boundary_level` then counts from the shallowest remaining level, so a document
    whose sections are H2s gets the same boundaries as one whose sections are H1s.
    """
    headings = [(i, el.level) for i, el in enumerate(elements) if el.type is ElementType.HEADING]
    if not headings:
        return lambda i, el: False
    top = min(level for _, level in headings)
    first_i, first_level = headings[0]
    title = first_i if first_level == top and sum(lv == top for _, lv in headings) == 1 else None
    rest = [level for i, level in headings if i != title]
    if not rest:
        return lambda i, el: False
    base = min(rest)
    return lambda i, el: i != title and el.level - base + 1 <= boundary_level


def _size(units: list[_Unit]) -> int:
    return sum(u.tokens for u in units)


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    first = paths[0]
    n = len(first)
    for p in paths[1:]:
        n = min(n, len(p))
        for k in range(n):
            if p[k] != first[k]:
                n = k
                break
    return first[:n]


def _make_chunk(meta: DocumentMeta, ordinal: int, units: list[_Unit]) -> Chunk:
    parts = [units[0].text]
    for prev, u in zip(units, units[1:]):
        same_sentence_run = prev.element == u.element and prev.sentence and u.sentence
        parts.append((" " if same_sentence_run else "\n") + u.text)
    pages = [u.page for u in units if u.page is not None]
    return Chunk(
        chunk_id=chunk_id_for(meta.document_id, ordinal),
        doc=meta,
        ordinal=ordinal,
        text="".join(parts),
        heading_path=_common_prefix([u.path for u in units]),
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
    )
