import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from semanticsearch.core import (
    DocumentMeta,
    IngestStatus,
    ManifestRecord,
    ManifestStore,
    SemanticSearchError,
    document_id_for,
)
from semanticsearch.ingest import (
    DocumentStore,
    Ingestor,
    SectionRules,
    SqliteManifestStore,
    default_parsers,
    ingestor_for_pack,
)
from semanticsearch.packs import TEMPLATE_PACK_DIR, load_pack

RULES = SectionRules()


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    write(root, "a.md", "# Alpha\n\nAlpha body.\n")
    write(root, "sub/b.txt", "Bravo body.\n")
    return root


@pytest.fixture
def work(tmp_path):
    w = tmp_path / "work"
    stores = (SqliteManifestStore(w / "manifest.sqlite"), DocumentStore(w / "documents"))
    yield stores
    stores[0].close()


def make(corpus, work, **kw) -> Ingestor:
    manifest, documents = work
    return Ingestor(
        corpus, parsers=default_parsers(RULES), rules=RULES, manifest=manifest, documents=documents, **kw
    )


def ids(*sources: str) -> list[str]:
    return [document_id_for(s) for s in sources]


def bump_mtime(p: Path) -> None:
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))


def test_first_run_ingests_everything(corpus, work):
    manifest, documents = work
    write(corpus, "notes.doc", "legacy")
    write(corpus, ".hidden/x.md", "# hidden")
    write(corpus, "~$a.docx", "office lock file")

    r = make(corpus, work).run()

    assert sorted(r.to_index) == sorted(ids("a.md", "sub/b.txt"))
    assert r.unsupported == ["notes.doc"]
    assert r.stale == [] and r.failed == []

    rec = manifest.get(document_id_for("sub/b.txt"))
    assert rec.status is IngestStatus.OK
    assert rec.meta.source == "sub/b.txt" and rec.meta.mime == "text/plain"
    assert rec.n_elements == 1 and rec.meta.modified_ts.tzinfo is not None

    doc = documents.get(document_id_for("a.md"))
    assert doc.meta == manifest.get(doc.document_id).meta
    assert [e.text for e in doc.elements] == ["Alpha", "Alpha body."]


def test_second_run_skips_hashing_unchanged_files(corpus, work, monkeypatch):
    make(corpus, work).run()
    calls = []
    import semanticsearch.ingest.pipeline as pipeline

    monkeypatch.setattr(pipeline, "sha256_file", lambda p: calls.append(p) or "x")
    r = make(corpus, work).run()
    assert calls == []
    assert sorted(r.unchanged) == sorted(ids("a.md", "sub/b.txt"))
    assert r.to_index == [] and r.stale == []


def test_edit_reparses_and_marks_stale(corpus, work):
    make(corpus, work).run()
    p = write(corpus, "a.md", "# Alpha\n\nEdited.\n")
    bump_mtime(p)
    r = make(corpus, work).run()
    assert r.to_index == ids("a.md") == r.stale
    assert work[1].get(document_id_for("a.md")).elements[-1].text == "Edited."


def test_touch_without_edit_is_unchanged_but_recorded(corpus, work):
    make(corpus, work).run()
    p = corpus / "a.md"
    bump_mtime(p)
    r = make(corpus, work).run()
    assert r.to_index == [] and r.stale == []
    rec = work[0].get(document_id_for("a.md"))
    assert rec.meta.modified_ts == datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)


def test_duplicates_keep_the_existing_owner_and_promote_on_removal(corpus, work):
    manifest, documents = work
    make(corpus, work).run()
    shutil.copy(corpus / "sub/b.txt", corpus / "0-copy.txt")  # sorts before the original

    r = make(corpus, work).run()
    assert r.duplicates == [("0-copy.txt", "sub/b.txt")]
    dup = manifest.get(document_id_for("0-copy.txt"))
    assert dup.status is IngestStatus.DUPLICATE
    assert dup.duplicate_of == document_id_for("sub/b.txt")
    assert document_id_for("0-copy.txt") not in documents

    (corpus / "sub/b.txt").unlink()
    r = make(corpus, work).run()
    assert r.removed == ids("sub/b.txt") == r.stale
    assert r.to_index == ids("0-copy.txt")
    assert manifest.get(document_id_for("sub/b.txt")).status is IngestStatus.REMOVED
    assert document_id_for("sub/b.txt") not in documents


def test_identical_new_files_dedupe_in_path_order(tmp_path, work):
    root = tmp_path / "c"
    write(root, "z.txt", "same")
    write(root, "y.txt", "same")
    r = make(root, work).run()
    assert r.to_index == ids("y.txt")
    assert r.duplicates == [("z.txt", "y.txt")]


def test_empty_and_failed_documents(corpus, work):
    manifest, documents = work
    write(corpus, "blank.txt", "  \n\n ")
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4 truncated")

    r = make(corpus, work).run()
    assert r.empty == ["blank.txt"]
    assert [s for s, _ in r.failed] == ["broken.pdf"]
    assert len(r.to_index) == 2  # the good files still went through
    assert manifest.get(document_id_for("broken.pdf")).error
    assert manifest.get(document_id_for("blank.txt")).status is IngestStatus.EMPTY

    r = make(corpus, work).run()  # failures are retried; empties are not
    assert [s for s, _ in r.failed] == ["broken.pdf"]
    assert r.empty == [] and document_id_for("blank.txt") in r.unchanged


def test_pack_or_acl_change_forces_reparse(corpus, work):
    make(corpus, work, pack_version="p@1.0.0").run()
    r = make(corpus, work, pack_version="p@1.1.0").run()
    assert sorted(r.to_index) == sorted(ids("a.md", "sub/b.txt")) == sorted(r.stale)

    acl = lambda source: ("Risk",) if source.startswith("sub/") else ()
    r = make(corpus, work, pack_version="p@1.1.0", acl=acl).run()
    assert r.to_index == ids("sub/b.txt")
    assert work[0].get(document_id_for("sub/b.txt")).meta.acl_groups == ("Risk",)


def test_missing_normalised_document_is_rebuilt(corpus, work):
    make(corpus, work).run()
    work[1].delete(document_id_for("a.md"))
    assert make(corpus, work).run().to_index == ids("a.md")


def test_parallel_matches_serial(corpus, tmp_path):
    for i in range(4):
        write(corpus, f"many/{i}.txt", f"Document number {i}.")
    pack = load_pack(TEMPLATE_PACK_DIR)
    serial = ingestor_for_pack(corpus, pack, tmp_path / "w1", workers=1).run()
    parallel = ingestor_for_pack(corpus, pack, tmp_path / "w2", workers=2).run()
    assert sorted(serial.to_index) == sorted(parallel.to_index)
    assert len(parallel.to_index) == 6
    for doc_id in parallel.to_index:
        assert DocumentStore(tmp_path / "w1/documents").get(doc_id) == DocumentStore(
            tmp_path / "w2/documents"
        ).get(doc_id)


# ---------------------------------------------------------------------- stores


def test_manifest_store_round_trip(tmp_path):
    store = SqliteManifestStore(tmp_path / "m.sqlite")
    assert isinstance(store, ManifestStore)
    rec = ManifestRecord(
        meta=DocumentMeta(
            document_id=document_id_for("x.pdf"),
            source="x.pdf",
            mime="application/pdf",
            sha256="ab" * 32,
            modified_ts=datetime(2025, 1, 2, 3, 4, 5, 678901, tzinfo=timezone.utc),
            acl_groups=("A", "B"),
            pack_version="p@1.0.0",
        ),
        size_bytes=10,
        status=IngestStatus.FAILED,
        ingested_at=datetime(2025, 1, 3, tzinfo=timezone.utc),
        error="boom",
    )
    store.put(rec)
    assert store.get(rec.document_id) == rec
    assert list(store.records(IngestStatus.FAILED)) == [rec]
    assert list(store.records(IngestStatus.OK)) == []
    store.close()


def test_manifest_store_refuses_unknown_schema(tmp_path):
    path = tmp_path / "m.sqlite"
    SqliteManifestStore(path).close()
    con = sqlite3.connect(path)
    con.execute("PRAGMA user_version = 99")
    con.close()
    with pytest.raises(SemanticSearchError, match="schema v99"):
        SqliteManifestStore(path)


def test_corpus_root_must_exist(tmp_path, work):
    with pytest.raises(NotADirectoryError):
        make(tmp_path / "nope", work)
