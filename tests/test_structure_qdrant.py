import uuid
import json
from unittest.mock import Mock

import numpy as np
import pytest

from librarian.chunker import approx_tokens, chunk_documents
from librarian.corpus import Document, Section
from librarian.config import settings
from librarian.qdrant import QdrantStore


def test_long_paragraph_and_overlap_preserve_content_and_limit():
    text = " ".join(f"word{i}" for i in range(300))
    doc = Document("test.rst", "Test", "rst", "core", sections=[
        Section(["Test", "First"], text),
        Section(["Test", "Second"], "another separate section with enough words"),
    ])
    chunks = chunk_documents([doc], "structure", size=50, overlap=20)
    assert all(approx_tokens(c.text) <= 50 for c in chunks)
    assert set(text.split()) <= set(" ".join(c.text for c in chunks).split())
    assert {c.section for c in chunks} == {"Test > First", "Test > Second"}
    with pytest.raises(ValueError):
        chunk_documents([doc], "structure", size=50, overlap=50)


def test_qdrant_upload_and_filtered_query(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "https://example.qdrant.io")
    monkeypatch.setenv("CLUSTER_KEY", "test-secret")
    store = QdrantStore()
    assert store.session.headers["api-key"] == "test-secret"
    session = Mock()
    session.get.return_value.status_code = 404
    session.request.return_value.json.return_value = {"result": {"points": []}}
    store.session = session
    doc = Document("test.rst", "Test", "rst", "core", sections=[
        Section(["Test"], "some text with enough words for indexing")])
    chunks = chunk_documents([doc])
    store.upload(chunks, np.ones((1, 3)), "build1")
    upload = session.request.call_args.kwargs["json"]["points"][0]
    uuid.UUID(upload["id"])
    assert upload["payload"]["text"] == chunks[0].text
    assert upload["payload"]["generation"] == "build1"
    assert "test-secret" not in str(upload)
    store.search(np.ones(3), 5, "build1", [chunks[0].id])
    query = session.request.call_args.kwargs["json"]
    assert query["filter"]["must"][1]["match"]["any"] == [chunks[0].id]
    assert query["params"] == {"hnsw_ef": max(5, settings.qdrant_hnsw_ef), "exact": False}
    assert query["with_payload"] == ["id"]
    store.search(np.ones(3), 100, "build1", exact=True)
    assert session.request.call_args.kwargs["json"]["params"] == {
        "hnsw_ef": max(100, settings.qdrant_hnsw_ef), "exact": True,
    }
    assert store.search(np.ones(3), 5, "build1", []) == []


def test_hnsw_update_preserves_existing_collection(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_url", "https://example.qdrant.io")
    store = QdrantStore()
    store.request = Mock()
    store.configure_hnsw()
    args = store.request.call_args
    assert args.args == ("PATCH",)
    assert args.kwargs["json"]["hnsw_config"]["m"] > 0
    assert "vectors" not in args.kwargs["json"]
    monkeypatch.setattr(settings, "qdrant_hnsw_m", 0)
    with pytest.raises(ValueError):
        store.configure_hnsw()


def test_rebuild_only_embeds_changed_text(tmp_path, monkeypatch):
    from librarian import index

    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.delenv("CLUSTER_KEY", raising=False)
    first = Section(["Test", "First"], "unchanged paragraph with enough words for indexing")
    doc = Document("test.rst", "Test", "rst", "core", sections=[first])
    old_chunk = chunk_documents([doc])[0]
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "chunks.jsonl").write_text(json.dumps(old_chunk.to_dict()) + "\n")
    (previous / "meta.json").write_text(json.dumps({"embedder": "st:test"}))
    np.save(previous / "vectors.npy", np.array([[1., 0., 0.]]))
    doc.sections.append(Section(["Test", "Second"], "new paragraph requiring its own embedding vector"))
    monkeypatch.setattr(index, "load_corpus", lambda: [doc])
    embedder = Mock()
    embedder.encode_docs.return_value = np.array([[0., 1., 0.]])
    monkeypatch.setattr(index, "get_embedder", lambda *args: embedder)
    output = tmp_path / "output"
    index.build(str(output), embedder="st:test", reuse_index=str(previous))
    encoded = embedder.encode_docs.call_args.args[0]
    assert len(encoded) == 1 and "new paragraph" in encoded[0]
    np.testing.assert_array_equal(np.load(output / "vectors.npy"), [[1., 0., 0.], [0., 1., 0.]])
