from __future__ import annotations

import pytest

from app.rag.ingestion import (
    _infer_doc_type,
    chunk_document,
    ingest_domain,
    load_markdown_documents,
)
from app.rag.vectorstore import VectorStore


def test_load_markdown_documents_reads_all_md_files(sample_kb_dir):
    docs = load_markdown_documents(sample_kb_dir)
    filenames = {name for name, _ in docs}
    assert filenames == {"customer_segments.md", "brand_guidelines.md"}


def test_load_markdown_documents_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_markdown_documents("/nonexistent/path/xyz")


def test_infer_doc_type_maps_known_filenames():
    assert _infer_doc_type("customer_segments.md") == "segment"
    assert _infer_doc_type("brand_guidelines.md") == "brand"
    assert _infer_doc_type("campaign_guidelines.md") == "campaign"
    assert _infer_doc_type("previous_campaigns.md") == "campaign"
    assert _infer_doc_type("marketing_best_practices.md") == "best_practice"


def test_infer_doc_type_unknown_filename_falls_back_to_general():
    assert _infer_doc_type("something_else.md") == "general"


def test_chunk_document_produces_stable_ids_and_doc_type():
    text = "# Title\n\n## Section A\n" + ("word " * 300) + "\n\n## Section B\nmore text here."
    chunks = chunk_document("customer_segments.md", text)

    assert len(chunks) >= 1
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"segment::customer_segments.md::chunk_{i}"
        assert c.doc_type == "segment"
        assert c.doc_title == "customer_segments.md"
        assert c.text  # non-empty


def test_chunk_document_respects_chunk_size_roughly():
    # A long single paragraph should be split into multiple chunks.
    long_text = "This is one sentence about hiking gear. " * 100
    chunks = chunk_document("marketing_best_practices.md", long_text)
    assert len(chunks) > 1


def test_ingest_domain_stores_chunks_in_vectorstore(sample_kb_dir, isolated_chroma_dir):
    store = VectorStore(persist_dir=isolated_chroma_dir)
    n = ingest_domain(domain="test_marketing", kb_path=sample_kb_dir, store=store)

    assert n > 0
    assert store.count("test_marketing") == n


def test_ingest_domain_empty_kb_returns_zero(tmp_path, isolated_chroma_dir):
    empty_kb = tmp_path / "empty_kb"
    empty_kb.mkdir()
    store = VectorStore(persist_dir=isolated_chroma_dir)
    n = ingest_domain(domain="empty_domain", kb_path=str(empty_kb), store=store)
    assert n == 0
    assert store.count("empty_domain") == 0
