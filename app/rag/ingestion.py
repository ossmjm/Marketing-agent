"""
Ingestion pipeline: Documents -> Loading -> Chunking -> Embedding -> Vector DB.

Chunking choice
---------------
We use LangChain's `RecursiveCharacterTextSplitter` with markdown-aware
separators (headers first, then paragraphs, then sentences). This is a
pragmatic choice over a custom splitter: our knowledge base is
markdown with clear `##` section headers, so splitting on structural
boundaries first keeps each chunk topically coherent (e.g. one
customer segment stays in one chunk) before falling back to
character-count splitting for oversized sections.

`chunk_size` / `chunk_overlap` are configurable via Settings rather
than hardcoded, since the right size depends on document density --
documented in the README trade-offs section.

Each chunk gets metadata: doc_type (category, derived from filename),
source_file (original filename), and a stable chunk id
(`<doc_type>::<filename>::chunk_<n>`) used later for citation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.vectorstore import VectorStore


@dataclass
class IngestedChunk:
    chunk_id: str
    text: str
    doc_title: str
    doc_type: str


DOC_TYPE_FROM_FILENAME = {
    "brand_guidelines": "brand",
    "customer_segments": "segment",
    "campaign_guidelines": "campaign",
    "previous_campaigns": "campaign",
    "marketing_best_practices": "best_practice",
}


def _infer_doc_type(filename: str) -> str:
    stem = Path(filename).stem
    return DOC_TYPE_FROM_FILENAME.get(stem, "general")


def load_markdown_documents(kb_path: str) -> list[tuple[str, str]]:
    """Returns list of (filename, raw_text) for every .md file in kb_path."""
    p = Path(kb_path)
    if not p.exists():
        raise FileNotFoundError(f"Knowledge base path does not exist: {kb_path}")
    docs = []
    for md_file in sorted(p.glob("*.md")):
        docs.append((md_file.name, md_file.read_text(encoding="utf-8")))
    return docs


def chunk_document(filename: str, text: str) -> list[IngestedChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )
    raw_chunks = splitter.split_text(text)
    doc_type = _infer_doc_type(filename)
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk_id = f"{doc_type}::{filename}::chunk_{i}"
        chunks.append(
            IngestedChunk(
                chunk_id=chunk_id,
                text=chunk_text.strip(),
                doc_title=filename,
                doc_type=doc_type,
            )
        )
    return chunks


def ingest_domain(domain: str, kb_path: str, store: VectorStore | None = None, reset: bool = True) -> int:
    """
    Full ingestion run for one domain. Returns number of chunks stored.
    """
    store = store or VectorStore()
    if reset:
        store.reset(domain)

    documents = load_markdown_documents(kb_path)
    all_chunks: list[IngestedChunk] = []
    for filename, text in documents:
        all_chunks.extend(chunk_document(filename, text))

    if not all_chunks:
        return 0

    store.add_chunks(
        domain=domain,
        ids=[c.chunk_id for c in all_chunks],
        texts=[c.text for c in all_chunks],
        metadatas=[
            {"doc_title": c.doc_title, "doc_type": c.doc_type} for c in all_chunks
        ],
    )
    return len(all_chunks)
