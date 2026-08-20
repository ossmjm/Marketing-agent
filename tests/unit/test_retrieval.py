from __future__ import annotations

from app.rag.ingestion import ingest_domain
from app.rag.retriever import Retriever
from app.rag.vectorstore import VectorStore


def _make_indexed_retriever(sample_kb_dir, isolated_chroma_dir, domain="test_marketing"):
    store = VectorStore(persist_dir=isolated_chroma_dir)
    ingest_domain(domain=domain, kb_path=sample_kb_dir, store=store)
    return Retriever(store=store), domain


def test_retrieve_returns_chunks_for_relevant_query(sample_kb_dir, isolated_chroma_dir):
    retriever, domain = _make_indexed_retriever(sample_kb_dir, isolated_chroma_dir)
    chunks = retriever.retrieve(domain=domain, query="Weekend Explorers hiking Instagram", top_k=5, similarity_threshold=0.0)
    assert len(chunks) > 0
    assert all(c.doc_id for c in chunks)


def test_retrieve_respects_top_k(sample_kb_dir, isolated_chroma_dir):
    retriever, domain = _make_indexed_retriever(sample_kb_dir, isolated_chroma_dir)
    chunks = retriever.retrieve(domain=domain, query="hiking", top_k=1, similarity_threshold=0.0)
    assert len(chunks) <= 1


def test_retrieve_applies_similarity_threshold(sample_kb_dir, isolated_chroma_dir):
    retriever, domain = _make_indexed_retriever(sample_kb_dir, isolated_chroma_dir)
    # An unreasonably high threshold should filter out everything.
    chunks = retriever.retrieve(domain=domain, query="hiking gear", top_k=5, similarity_threshold=0.999)
    assert chunks == []


def test_retrieve_metadata_filter_by_doc_type(sample_kb_dir, isolated_chroma_dir):
    retriever, domain = _make_indexed_retriever(sample_kb_dir, isolated_chroma_dir)
    chunks = retriever.retrieve(
        domain=domain, query="tone brand voice", doc_types=["brand"], top_k=5, similarity_threshold=0.0
    )
    assert all(c.doc_type == "brand" for c in chunks)


def test_retrieve_on_unindexed_domain_returns_empty(isolated_chroma_dir):
    store = VectorStore(persist_dir=isolated_chroma_dir)
    retriever = Retriever(store=store)
    chunks = retriever.retrieve(domain="never_indexed", query="anything", similarity_threshold=0.0)
    assert chunks == []


def test_is_indexed_reflects_store_state(sample_kb_dir, isolated_chroma_dir):
    store = VectorStore(persist_dir=isolated_chroma_dir)
    retriever = Retriever(store=store)
    assert retriever.is_indexed("test_marketing") is False

    ingest_domain(domain="test_marketing", kb_path=sample_kb_dir, store=store)
    assert retriever.is_indexed("test_marketing") is True
