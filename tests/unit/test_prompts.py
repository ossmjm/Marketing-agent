from __future__ import annotations

import app.domains  # noqa: F401 -- registers marketing domain
from app.agent.domain_config import get_domain
from app.agent.prompts import (
    build_context_block,
    build_few_shot_block,
    build_generation_prompt,
    build_planner_prompt,
    build_validation_repair_prompt,
)
from app.agent.state import RetrievedChunk


def test_build_context_block_includes_doc_ids_for_citation():
    chunks = [
        RetrievedChunk(doc_id="segment::a.md::chunk_0", doc_title="a.md", doc_type="segment", text="hello", score=0.8),
        RetrievedChunk(doc_id="brand::b.md::chunk_1", doc_title="b.md", doc_type="brand", text="world", score=0.6),
    ]
    block = build_context_block(chunks)
    assert "segment::a.md::chunk_0" in block
    assert "brand::b.md::chunk_1" in block
    assert "hello" in block and "world" in block


def test_build_context_block_handles_empty_chunks():
    block = build_context_block([])
    assert "none retrieved" in block.lower()


def test_build_few_shot_block_renders_all_examples():
    domain = get_domain("marketing")
    block = build_few_shot_block(domain.few_shot_examples)
    for ex in domain.few_shot_examples:
        assert ex.title in block


def test_build_planner_prompt_lists_valid_intents_and_doc_types():
    domain = get_domain("marketing")
    system, user = build_planner_prompt(domain, "Create a campaign for Segment A")
    for intent in domain.intent_categories:
        assert intent in system
    for doc_type in domain.doc_type_labels:
        assert doc_type in system
    assert "Segment A" in user


def test_build_generation_prompt_includes_domain_instructions_and_context():
    domain = get_domain("marketing")
    chunks = [
        RetrievedChunk(doc_id="segment::x.md::chunk_0", doc_title="x.md", doc_type="segment", text="segment info", score=0.7)
    ]
    system, user = build_generation_prompt(domain, "Analyze Segment A", chunks, '{"example": true}')

    assert domain.system_prompt.strip()[:30] in system  # domain instructions present
    assert "segment info" in user
    assert "Analyze Segment A" in user


def test_build_generation_prompt_never_includes_raw_chain_of_thought_markers():
    domain = get_domain("marketing")
    system, user = build_generation_prompt(domain, "test", [], "{}")
    # We should never instruct the model to reveal step-by-step reasoning
    assert "think step by step" not in system.lower()
    assert "chain of thought" not in system.lower()


def test_build_validation_repair_prompt_includes_errors():
    system, user = build_validation_repair_prompt('{"bad": true}', ["field x is missing", "field y invalid"])
    assert "field x is missing" in user
    assert "field y invalid" in user
    assert '{"bad": true}' in user
