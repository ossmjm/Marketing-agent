"""
Thin wrapper around ChromaDB.

Trade-off (documented in README too): ChromaDB with a local persistent
client is the right choice for a take-home / small KB -- zero infra,
runs entirely on disk, no external service to stand up. It would NOT
be the right choice at production scale with concurrent writers,
multi-tenant filtering at high QPS, or the need for a managed
availability SLA -- at that point you'd move to a managed vector DB
(Pinecone, Weaviate Cloud, pgvector on managed Postgres). The
`VectorStore` interface below is intentionally narrow (add/query by
domain collection) so swapping the backend later only touches this
one file.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.rag.embeddings import EmbeddingFunction, get_embedding_function


@dataclass
class QueryMatch:
    doc_id: str
    text: str
    metadata: dict
    score: float  # cosine similarity, higher = more similar (0..1 roughly)


class VectorStore:
    def __init__(self, embedding_fn: EmbeddingFunction | None = None, persist_dir: str | None = None):
        self._client = chromadb.PersistentClient(
            path=persist_dir or settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = embedding_fn or get_embedding_function()

    def _collection(self, domain: str):
        return self._client.get_or_create_collection(
            name=f"domain_{domain}",
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self, domain: str) -> None:
        try:
            self._client.delete_collection(f"domain_{domain}")
        except Exception:
            pass

    def add_chunks(
        self,
        domain: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        embeddings = self._embedding_fn.embed_documents(texts)
        self._collection(domain).upsert(
            ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
        )

    def query(
        self,
        domain: str,
        query_text: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[QueryMatch]:
        collection = self._collection(domain)
        if collection.count() == 0:
            return []
        query_embedding = self._embedding_fn.embed_query(query_text)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            where=where,
        )
        matches: list[QueryMatch] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns cosine *distance*; convert to a 0..1 similarity score.
            similarity = max(0.0, 1.0 - dist / 2.0)
            matches.append(QueryMatch(doc_id=doc_id, text=text, metadata=meta, score=similarity))
        return matches

    def count(self, domain: str) -> int:
        return self._collection(domain).count()
