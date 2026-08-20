"""
Retriever: the RAG-facing API used by the agent.

Responsibilities:
  - top-k semantic retrieval
  - similarity-threshold filtering (drops low-relevance matches instead
    of always returning top_k regardless of quality)
  - metadata filtering by doc_type (used by the planner to target
    specific KB categories, e.g. only "segment" docs for a segment
    analysis request)
  - converts raw vector-store matches into `RetrievedChunk` (the shape
    the rest of the agent understands), preserving score for source
    attribution and downstream confidence scoring

This module knows nothing about OpenAI or the agent's control flow --
it is a pure retrieval component, which keeps it independently
unit-testable (see tests/unit/test_retrieval.py).
"""

from __future__ import annotations

from app.agent.state import RetrievedChunk
from app.config import settings
from app.rag.vectorstore import VectorStore


class Retriever:
    def __init__(self, store: VectorStore | None = None):
        self._store = store or VectorStore()

    def retrieve(
        self,
        domain: str,
        query: str,
        doc_types: list[str] | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.retrieval_top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.retrieval_similarity_threshold
        )

        where = None
        if doc_types:
            where = (
                {"doc_type": doc_types[0]}
                if len(doc_types) == 1
                else {"doc_type": {"$in": doc_types}}
            )

        matches = self._store.query(domain=domain, query_text=query, top_k=top_k, where=where)

        chunks = [
            RetrievedChunk(
                doc_id=m.doc_id,
                doc_title=m.metadata.get("doc_title", "unknown"),
                doc_type=m.metadata.get("doc_type", "general"),
                text=m.text,
                score=m.score,
            )
            for m in matches
            if m.score >= threshold
        ]
        return chunks

    def is_indexed(self, domain: str) -> bool:
        return self._store.count(domain) > 0
