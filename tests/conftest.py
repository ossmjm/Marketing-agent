from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force mock/offline mode for the whole test session BEFORE app.config is imported
# anywhere, so no real network calls are ever attempted in tests.
os.environ["USE_MOCK_LLM"] = "true"
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

# The default similarity threshold (0.28) is tuned for real OpenAI embeddings.
# The offline LocalHashEmbeddingFunction used under USE_MOCK_LLM produces a
# different similarity distribution (bag-of-hashed-tokens, not a learned
# semantic space), so we relax the threshold for the test session. This is a
# test-environment concern only -- app/config.py's real default is unchanged.
os.environ["RETRIEVAL_SIMILARITY_THRESHOLD"] = "0.05"


@pytest.fixture()
def isolated_chroma_dir(tmp_path):
    """Give each test its own throwaway Chroma persistence directory path.

    `VectorStore(persist_dir=...)` accepts this directly -- we avoid relying
    on env-var reload timing since `app.config.settings` is a module-level
    singleton instantiated once at first import.
    """
    chroma_dir = tmp_path / ".chroma_test"
    return str(chroma_dir)


@pytest.fixture()
def sample_kb_dir(tmp_path):
    """A tiny synthetic knowledge base for ingestion/chunking tests."""
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "customer_segments.md").write_text(
        "# Customer Segments\n\n"
        "## Segment A: Weekend Explorers\n"
        "Weekend Explorers hike occasionally and value approachable, "
        "non-technical guidance. They respond well to Instagram content.\n\n"
        "## Segment B: Trail Veterans\n"
        "Trail Veterans are experienced backpackers who research heavily "
        "and value technical detail and durability.\n"
    )
    (kb / "brand_guidelines.md").write_text(
        "# Brand Guidelines\n\n"
        "## Tone\n"
        "Speak like a knowledgeable friend, not a luxury brand. Avoid "
        "fear-based urgency language.\n"
    )
    return str(kb)
