from datetime import datetime, timezone

import pytest

from semanticsearch.chunking import (
    HFTokenizer,
    SentenceSplitter,
    StructureAwareChunker,
    chunker_for_pack,
)
from semanticsearch.core import (
    Document,
    DocumentMeta,
    Element,
    ElementType,
    PackError,
    SemanticSearchError,
    Tokenizer,
    chunk_id_for,
    document_id_for,
)
from semanticsearch.packs import load_pack

H, P, L, T = ElementType.HEADING, ElementType.PARAGRAPH, ElementType.LIST_ITEM, ElementType.TABLE


class WordTokenizer:
    """One token per whitespace-separated word, so joined sizes are exact."""

    def encode(self, text):
        return [0] * len(text.split())


class CharTokenizer:
    def encode(self, text):
        return [0] * len(text.replace(" ", ""))


META = DocumentMeta(
    document_id=document_id_for("doc.pdf"),
    source="doc.pdf",
    mime="application/pdf",
    sha256="0" * 64,
    modified_ts=datetime(2025, 1, 1, tzinfo=timezone.utc),
)


def doc(*elements: Element) -> Document:
    return Document(meta=META, elements=elements)


def words(n: int, stem: str = "w") -> str:
    """`n` words forming one sentence."""
    return " ".join(f"{stem}{i}" for i in range(n - 1)) + f" {stem}end."


def chunker(max_tokens=20, overlap=0, **kw) -> StructureAwareChunker:
    return StructureAwareChunker(WordTokenizer(), max_tokens=max_tokens, overlap_tokens=overlap, **kw)


def n_tokens(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------- sentences


@pytest.mark.parametrize(
    "text,expected",
    [
        ("One. Two! Three? Four", ["One.", "Two!", "Three?", "Four"]),
        ("Price rose 3.5 per cent. Then fell.", ["Price rose 3.5 per cent.", "Then fell."]),
        ("See cl. 2.3.1 of the Rules. Next.", ["See cl. 2.3.1 of the Rules.", "Next."]),
        ("AGL Energy Ltd. Is a retailer.", ["AGL Energy Ltd. Is a retailer."]),
        ("Use e.g. Caps. Or i.e. Swaps.", ["Use e.g. Caps.", "Or i.e. Swaps."]),
        ("Signed by J. Smith today.", ["Signed by J. Smith today."]),
        ("Pick option c. Then go.", ["Pick option c.", "Then go."]),  # initials are capitals
        ("He said \"Stop.\" Then left.", ["He said \"Stop.\"", "Then left."]),
        ("Item No. 5 is due.", ["Item No. 5 is due."]),
        ("ends with lowercase. next", ["ends with lowercase. next"]),  # needs a capital to split
    ],
)
def test_sentence_splitter(text, expected):
    assert SentenceSplitter().split(text) == expected


def test_sentence_splitter_pack_abbreviations():
    assert len(SentenceSplitter().split("Per Sch. A of the Code.")) == 1  # single-letter "A."
    assert SentenceSplitter().split("See Reg. Two applies.") == ["See Reg.", "Two applies."]
    assert SentenceSplitter(["Reg."]).split("See Reg. Two applies.") == ["See Reg. Two applies."]


# ---------------------------------------------------------------- sections


def test_major_sections_never_share_a_chunk():
    chunks = chunker(max_tokens=100).chunk(
        doc(Element(H, "1 Intro", level=1), Element(P, "Short."), Element(H, "2 Risk", level=1), Element(P, "Also short."))
    )
    assert [c.text for c in chunks] == ["1 Intro\nShort.", "2 Risk\nAlso short."]
    assert [c.heading_path for c in chunks] == [("1 Intro",), ("2 Risk",)]


def test_lone_title_is_not_a_boundary_and_levels_are_document_relative():
    els = (
        Element(H, "Report", level=1),
        Element(P, "Preface."),
        Element(H, "A", level=2),
        Element(P, "Alpha."),
        Element(H, "B", level=2),
        Element(P, "Bravo."),
    )
    chunks = chunker(max_tokens=100).chunk(doc(*els))
    assert [c.text for c in chunks] == ["Report\nPreface.", "A\nAlpha.", "B\nBravo."]
    assert chunks[2].heading_path == ("Report", "B")

    # Same shape without a title, headings starting at H3: still split at each H3.
    no_title = [el for el in els[2:]]
    no_title = tuple(Element(H, e.text, level=3) if e.type is H else e for e in no_title)
    assert [c.text for c in chunker(max_tokens=100).chunk(doc(*no_title))] == ["A\nAlpha.", "B\nBravo."]


def test_subsections_merge_when_small_and_path_is_common_prefix():
    chunks = chunker(max_tokens=100).chunk(
        doc(
            Element(H, "2 Risk", level=1),
            Element(H, "2.1 Credit", level=2),
            Element(P, "Credit text."),
            Element(H, "2.2 Market", level=2),
            Element(P, "Market text."),
            Element(H, "3 Other", level=1),  # a second H1, so "2 Risk" is not a lone title
            Element(P, "Other text."),
        )
    )
    assert len(chunks) == 2
    assert chunks[0].heading_path == ("2 Risk",)
    assert "2.1 Credit" in chunks[0].text and "2.2 Market" in chunks[0].text


def test_subheading_is_a_soft_boundary_once_half_full():
    chunks = chunker(max_tokens=20).chunk(
        doc(
            Element(H, "Sec", level=1),
            Element(P, words(10)),
            Element(H, "Sub", level=2),
            Element(P, "Tail."),
        )
    )
    assert [c.text.split("\n")[0] for c in chunks] == ["Sec", "Sub"]
    assert chunks[1].heading_path == ("Sec", "Sub")


def test_heading_is_never_left_dangling():
    chunks = chunker(max_tokens=10).chunk(
        doc(Element(H, "S", level=1), Element(P, words(6)), Element(H, "Sub", level=2), Element(P, words(6, "x")))
    )
    assert all(not c.text.endswith(("S", "Sub")) for c in chunks)
    assert chunks[-1].text.startswith("Sub\n")


def test_heading_only_sections_produce_no_chunks():
    chunks = chunker().chunk(doc(Element(H, "A", level=1), Element(H, "B", level=1), Element(P, "Body.")))
    assert [c.text for c in chunks] == ["B\nBody."]


# ------------------------------------------------------- sizing and splitting


def test_every_chunk_fits_and_text_is_preserved():
    paras = [Element(P, " ".join(words(k, f"s{j}") for j in range(3))) for k in (3, 7, 11, 5)]
    c = chunker(max_tokens=15)
    chunks = c.chunk(doc(Element(H, "Sec", level=1), *paras))
    assert all(n_tokens(ch.text) <= 15 for ch in chunks)
    body = " ".join(ch.text for ch in chunks).replace("\n", " ").split()
    for p in paras:  # nothing lost: every word appears, in order
        assert all(w in body for w in p.text.split())


def test_long_sentence_falls_back_to_word_windows():
    chunks = chunker(max_tokens=10).chunk(doc(Element(P, words(25))))
    assert [n_tokens(c.text) for c in chunks] == [10, 10, 5]


def test_giant_word_is_hard_split():
    c = StructureAwareChunker(CharTokenizer(), max_tokens=10, overlap_tokens=0)
    chunks = c.chunk(doc(Element(P, "x" * 25)))
    assert [c.text for c in chunks] == ["x" * 10, "x" * 10, "x" * 5]


def test_big_table_splits_by_rows_and_repeats_header():
    rows = ["Region | Price"] + [f"R{i} | {i}" for i in range(9)]  # 3 tokens per line
    chunks = chunker(max_tokens=12).chunk(doc(Element(T, "\n".join(rows))))
    assert all(c.text.startswith("Region | Price\n") for c in chunks)
    assert all(n_tokens(c.text) <= 12 for c in chunks)
    seen = [line for c in chunks for line in c.text.split("\n")[1:]]
    assert seen == rows[1:]


# ------------------------------------------------------------------ overlap


def test_overlap_carries_whole_trailing_sentences_within_a_section():
    sentences = " ".join(f"S{i} a b c." for i in range(8))  # 4 tokens each
    chunks = chunker(max_tokens=12, overlap=5).chunk(doc(Element(P, sentences)))
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.text.rsplit("S", 1)[1]
        assert nxt.text.startswith("S" + last_sentence)  # exactly one 4-token sentence carried
    assert all(n_tokens(c.text) <= 12 for c in chunks)


def test_no_overlap_across_hard_boundaries_or_into_a_new_subsection():
    sents = lambda tag: " ".join(f"{tag}{i} a b c." for i in range(4))
    chunks = chunker(max_tokens=20, overlap=8).chunk(
        doc(
            Element(H, "A", level=1),
            Element(P, sents("a")),
            Element(H, "B", level=1),
            Element(P, sents("b")),
            Element(H, "B.1", level=2),
            Element(P, sents("c")),
        )
    )
    assert [c.text.split("\n")[0] for c in chunks] == ["A", "B", "B.1"]


# --------------------------------------------------------------- metadata


def test_pages_ids_and_document_reference():
    chunks = chunker(max_tokens=100).chunk(
        doc(Element(H, "S", level=1, page=2), Element(P, "On two.", page=2), Element(P, "On three.", page=3))
    )
    (c,) = chunks
    assert (c.page_start, c.page_end) == (2, 3)
    assert c.chunk_id == chunk_id_for(META.document_id, 0) and c.ordinal == 0
    assert c.doc is META


def test_formats_without_pages():
    (c,) = chunker().chunk(doc(Element(P, "No pages.")))
    assert c.page_start is None and c.page_end is None and c.heading_path == ()


def test_constructor_validation():
    with pytest.raises(ValueError):
        StructureAwareChunker(WordTokenizer(), max_tokens=10, overlap_tokens=6)


# ------------------------------------------------------------------- packs


def test_chunker_for_pack(tmp_path):
    (tmp_path / "pack.yaml").write_text(
        "name: x\nversion: 1.0.0\nsectioning: {boundary_level: 2}\n"
        "chunking: {max_tokens: 300, overlap_tokens: 40, abbreviations: [Reg.]}\n",
        encoding="utf-8",
    )
    c = chunker_for_pack(load_pack(tmp_path), tokenizer=WordTokenizer())
    assert (c.max_tokens, c.overlap_tokens, c.boundary_level) == (300, 40, 2)
    assert "reg." in c.splitter.abbreviations


def test_pack_rejects_overlap_over_half(tmp_path):
    (tmp_path / "pack.yaml").write_text(
        "name: x\nversion: 1.0.0\nchunking: {max_tokens: 100, overlap_tokens: 60}\n", encoding="utf-8"
    )
    with pytest.raises(PackError, match="overlap_tokens"):
        load_pack(tmp_path)


def test_stock_embedder_is_bge_m3(tmp_path):
    (tmp_path / "pack.yaml").write_text("name: x\nversion: 1.0.0\n", encoding="utf-8")
    pack = load_pack(tmp_path)
    assert (pack.embedder.id, pack.embedder.dim, pack.embedder.tokenizer_id) == ("BAAI/bge-m3", 1024, "BAAI/bge-m3")


@pytest.fixture(scope="module")
def bge_m3():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("HF_HUB_OFFLINE", "1")  # tests never download
        try:
            return HFTokenizer("BAAI/bge-m3")
        except SemanticSearchError:
            pytest.skip("bge-m3 tokenizer not in the local HF cache")


def test_real_bge_m3_sizes(bge_m3):
    assert isinstance(bge_m3, Tokenizer)
    text = " ".join(f"The cap contract {i} settles at $300/MWh in NSW1." for i in range(200))
    c = StructureAwareChunker(bge_m3, max_tokens=128, overlap_tokens=20)
    chunks = c.chunk(doc(Element(H, "Caps", level=1), Element(P, text)))
    assert len(chunks) > 10
    assert all(c.count(ch.text) <= 128 for ch in chunks)
