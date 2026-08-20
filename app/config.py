"""
Central application configuration.

Everything that changes between environments (API keys, model names,
paths, thresholds) lives here and is read from environment variables
via `python-dotenv` + `os.environ`. Nothing else in the codebase should
call `os.environ` directly -- this is the single source of truth, which
makes the system easier to test (override `Settings` in tests) and
easier to reason about in an interview ("where do I change the model?").
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


@dataclass(frozen=True)
class Settings:
    # --- LLM provider ---
    # `openai_api_key` is a plain fallback/default source for both
    # `llm_api_key` and `embedding_api_key` below (kept for the common case
    # of "I just have one OpenAI key"). Set `llm_api_key`/`llm_base_url` and
    # `embedding_api_key`/`embedding_base_url` explicitly to route either or
    # both through a different provider (e.g. OpenRouter). OpenRouter now
    # serves both chat completions AND an OpenAI-compatible /embeddings
    # endpoint, so a single OpenRouter key + base_url can cover both.
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.2))
    llm_timeout_seconds: int = field(default_factory=lambda: _get_int("LLM_TIMEOUT_SECONDS", 30))
    # Some OpenRouter models/providers don't honor `response_format=json_object`.
    # When false, the client falls back to prompt-only JSON enforcement (the
    # prompts already instruct JSON-only output; validation is the real safety net).
    llm_use_json_mode: bool = field(default_factory=lambda: _get_bool("LLM_USE_JSON_MODE", True))

    # --- Embeddings / Vector store ---
    # Defaults to following the chat LLM provider (llm_api_key / llm_base_url)
    # so a single OpenRouter (or any other) config covers both chat and
    # embeddings. Set EMBEDDING_API_KEY / EMBEDDING_BASE_URL explicitly only
    # if you want embeddings on a *different* provider than chat.
    embedding_api_key: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_API_KEY",
            os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        )
    )
    embedding_base_url: str | None = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", os.getenv("LLM_BASE_URL")) or None
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", ".chroma")
    )

    # --- Retrieval tuning ---
    retrieval_top_k: int = field(default_factory=lambda: _get_int("RETRIEVAL_TOP_K", 4))
    retrieval_similarity_threshold: float = field(
        default_factory=lambda: _get_float("RETRIEVAL_SIMILARITY_THRESHOLD", 0.28)
    )
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))

    # --- Agent control-flow bounds (explicit, named, never "magic numbers") ---
    max_retrieval_retries: int = field(default_factory=lambda: _get_int("MAX_RETRIEVAL_RETRIES", 1))
    max_validation_repairs: int = field(default_factory=lambda: _get_int("MAX_VALIDATION_REPAIRS", 1))
    min_sources_required: int = field(default_factory=lambda: _get_int("MIN_SOURCES_REQUIRED", 1))

    # --- Misc ---
    use_mock_llm: bool = field(default_factory=lambda: _get_bool("USE_MOCK_LLM", False))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


settings = Settings()
