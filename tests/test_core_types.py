import dataclasses
import uuid
from datetime import datetime, timezone

import pytest

from semanticsearch.core import (
    Chunk,
    Citation,
    DocumentMeta,
    Element,
    ElementType,
    SearchRequest,
    SparseVector,
    chunk_id_for,
    document_id_for,
)

DOC_LEVEL_FIELDS = {"source", "mime", "sha256", "modified_ts", "acl_groups", "pack_version"}


@pytest.fixture
def meta() -> DocumentMeta:
    return DocumentMeta(
        document_id=document_id_for("contracts/hedge_Q1_2025.pdf"),
        source="contracts/hedge_Q1_2025.pdf",
        mime="application/pdf",
        sha256="0" * 64,
        modified_ts=datetime(2025, 9, 8, tzinfo=timezone.utc),
        acl_groups=("RiskTeam",),
    )


def test_chunk_has_no_document_level_fields(meta):
    chunk_fields = {f.name for f in dataclasses.fields(Chunk)}
    assert chunk_fields.isdisjoint(DOC_LEVEL_FIELDS)


def test_chunk_reads_document_fields_through_reference(meta):
    c = Chunk(chunk_id=chunk_id_for(meta.document_id, 0), doc=meta, ordinal=0, text="x")
    assert c.document_id == meta.document_id
    assert c.doc.acl_groups == ("RiskTeam",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "mutated"  # type: ignore[misc]


def test_enrichment_uses_replace_and_keeps_text(meta):
    c = Chunk(chunk_id="c", doc=meta, ordinal=0, text="Cap contract terms")
    tagged = dataclasses.replace(c, concept_ids=("aemo:CapContract",), terms=("cap contract",))
    assert tagged.text == c.text
    assert c.concept_ids == ()


def test_chunk_rejects_inverted_page_range(meta):
    with pytest.raises(ValueError):
        Chunk(chunk_id="c", doc=meta, ordinal=0, text="x", page_start=5, page_end=4)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(type=ElementType.HEADING, text="1 Intro", level=0),
        dict(type=ElementType.PARAGRAPH, text="body", level=2),
        dict(type=ElementType.PARAGRAPH, text="body", page=0),
    ],
)
def test_element_validation(kwargs):
    with pytest.raises(ValueError):
        Element(**kwargs)


def test_ids_are_stable_uuids():
    a = document_id_for("reports/q3.pdf")
    assert a == document_id_for("reports/q3.pdf")
    assert a != document_id_for("reports/q4.pdf")
    uuid.UUID(a)
    assert chunk_id_for(a, 3) == chunk_id_for(a, 3) != chunk_id_for(a, 4)
    uuid.UUID(chunk_id_for(a, 3))


def test_document_id_is_independent_of_host_path_style():
    assert document_id_for("reports/q3.pdf") == document_id_for("reports\\q3.pdf")
    assert document_id_for("reports/q3.pdf") == document_id_for("reports//q3.pdf")


def test_sparse_vector_validation():
    SparseVector(indices=(1, 5), values=(0.5, 0.2))
    with pytest.raises(ValueError):
        SparseVector(indices=(1, 5), values=(0.5,))
    with pytest.raises(ValueError):
        SparseVector(indices=(1, 1), values=(0.5, 0.2))


def test_search_request_needs_a_vector():
    with pytest.raises(ValueError):
        SearchRequest(k=10)


@pytest.mark.parametrize(
    "path,start,end,expected",
    [
        (("2 Risk", "2.3 Risk Adjustments"), 5, 6, "[D §2.3 Risk Adjustments pp.5-6]"),
        (("2 Risk",), 5, 5, "[D §2 Risk p.5]"),
        ((), None, None, "[D]"),
    ],
)
def test_citation_label(path, start, end, expected):
    assert Citation("D", "src", path, start, end).label() == expected
