"""
Embedding function used for both ingestion and query-time retrieval.

Two implementations:
  - OpenRouterEmbeddingFunction : real embeddings via OpenRouter / OpenAI-compatible API
  - LocalHashEmbeddingFunction : a deterministic, dependency-free
    fallback used when `USE_MOCK_LLM=true` (tests / offline dev). It
    is NOT semantically meaningful -- it exists purely so the RAG
    pipeline (chunking, storage, top-k, thresholding) can be exercised
    end-to-end in CI without network access or an API key.

Both implement the same tiny protocol so `vectorstore.py` doesn't care
which one it's using.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.config import settings


class EmbeddingFunction(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenRouterEmbeddingFunction:
    """
    Uses the OpenRouter OpenAI-compatible embeddings endpoint.
    Defaults to 'liquid/lfm-2.5-embedding-350m:free' if no embedding model is configured.
    """

    DEFAULT_MODEL = "liquid/lfm-2.5-embedding-350m:free"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str | None = None):
        from openai import OpenAI  # local import: keeps `openai` optional for mock mode

        # Fall back to main LLM API key / base URL if embedding-specific settings are unset
        api_key = settings.embedding_api_key or settings.llm_api_key
        base_url = settings.embedding_base_url or settings.llm_base_url or self.DEFAULT_BASE_URL

        if not api_key:
            raise ValueError(
                "API key is missing! Please set LLM_API_KEY or EMBEDDING_API_KEY in your environment/.env file."
            )

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self._model = model or getattr(settings, "embedding_model", None) or self.DEFAULT_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        results = self.embed_documents([text])
        if not results:
            return []
        return results[0]


class LocalHashEmbeddingFunction:
    """
    Deterministic bag-of-words hashing embedding for offline/mock mode.

    Not semantically meaningful in a deep-learning sense, but it does
    place lexically-similar texts closer together (shared tokens hash
    to the same buckets), which is enough to make retrieval tests
    behave sensibly without any external dependency.
    """

    DIM = 256

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        tokens = text.lower().split()
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embedding_function() -> EmbeddingFunction:
    if settings.use_mock_llm:
        return LocalHashEmbeddingFunction()
    return OpenRouterEmbeddingFunction()