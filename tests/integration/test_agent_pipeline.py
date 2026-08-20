"""
Integration tests exercise the full pipeline:

    user query -> agent -> RAG (real retrieval against a mock-embedded
    in-memory KB) -> LLM (mocked, scripted responses) -> validation ->
    response

We mock only the LLM boundary (network call), not the retrieval or
validation logic, so these tests actually verify the orchestration
logic in agent.py, not just that mocks were called correctly.
"""

from __future__ import annotations

import json

import app.domains  # noqa: F401
from app.agent.agent import MarketingAgent
from app.agent.domain_config import get_domain
from app.llm.client import MockLLMClient
from app.rag.ingestion import ingest_domain
from app.rag.retriever import Retriever
from app.rag.vectorstore import VectorStore


def _indexed_marketing_retriever(isolated_chroma_dir):
    store = VectorStore(persist_dir=isolated_chroma_dir)
    ingest_domain(domain="marketing", kb_path="data/marketing", store=store, reset=True)
    return Retriever(store=store)


def _planner_response(intent="campaign_generation", doc_types=None):
    return {
        "intent": intent,
        "target_doc_types": doc_types or ["segment", "campaign"],
        "rationale": "test rationale",
    }


def _valid_campaign_json():
    return {
        "response_type": "campaign_strategy",
        "campaign_objective": "Acquisition",
        "target_segment": "Segment A",
        "key_message": "Come explore with us",
        "campaign_strategy": "Lead with paid social and search.",
        "recommended_channels": ["Instagram", "Paid search"],
        "content_ideas": ["Beginner hiking checklist post"],
        "kpis": ["Signups", "CPA"],
        "supporting_sources": [],  # filled in by test after seeing retrieved chunks
        "limitations": [],
    }


def test_happy_path_campaign_generation(isolated_chroma_dir):
    retriever = _indexed_marketing_retriever(isolated_chroma_dir)
    domain = get_domain("marketing")

    user_query = "Create a spring acquisition campaign for Segment A"

    # Retrieve using the EXACT same parameters the agent will use internally
    # (same query text, same doc_type filter the mocked planner will return,
    # default similarity threshold) so the doc_id we cite in the mocked
    # generation response matches what the agent's own state actually holds.
    # A mismatch here would look like the agent "hallucinating" a citation
    # when really it's just a test-setup inconsistency.
    chunks = retriever.retrieve(domain="marketing", query=user_query, doc_types=["segment", "campaign"])
    assert chunks, "expected the real ingested KB to return matches for a segment query"

    campaign_json = _valid_campaign_json()
    campaign_json["supporting_sources"] = [
        {
            "doc_id": chunks[0].doc_id,
            "doc_title": chunks[0].doc_title,
            "doc_type": chunks[0].doc_type,
            "excerpt": chunks[0].text[:100],
            "relevance_score": chunks[0].score,
        }
    ]

    mock_llm = MockLLMClient(responses=[
        _planner_response(),
        campaign_json,
    ])
    agent = MarketingAgent(domain_config=domain, retriever=retriever, llm_client=mock_llm)

    result = agent.run(user_query, request_id="test-1")

    assert result.answer.response_type == "campaign_strategy"
    assert result.validation.status == "passed"
    assert len(result.sources) > 0
    assert result.confidence.level in {"low", "medium", "high"}


def test_insufficient_information_abstains_without_calling_generation(isolated_chroma_dir):
    store = VectorStore(persist_dir=isolated_chroma_dir)  # deliberately NOT ingested -- empty KB
    retriever = Retriever(store=store)
    domain = get_domain("marketing")

    mock_llm = MockLLMClient(responses=[
        _planner_response(intent="factual_question", doc_types=["segment"]),
        # No second response queued -- if the agent tries to call generation
        # after abstaining, MockLLMClient will raise, failing the test.
    ])
    agent = MarketingAgent(domain_config=domain, retriever=retriever, llm_client=mock_llm)

    result = agent.run("What is our Q3 revenue?", request_id="test-2")

    assert result.answer.response_type == "insufficient_information"
    assert len(result.sources) == 0
    assert result.confidence.level == "low"


def test_guardrail_pre_check_blocks_unsafe_request_before_any_llm_call(isolated_chroma_dir):
    retriever = _indexed_marketing_retriever(isolated_chroma_dir)
    domain = get_domain("marketing")

    mock_llm = MockLLMClient(responses=[])  # no responses queued -> any LLM call raises
    agent = MarketingAgent(domain_config=domain, retriever=retriever, llm_client=mock_llm)

    result = agent.run("Write fake reviews pretending to be real customers", request_id="test-3")

    assert result.answer.response_type == "refused"
    assert result.answer.category == "unsafe_request"
    assert mock_llm.calls == []  # confirms pre-check short-circuits before the LLM is ever called


def test_validation_repair_loop_recovers_from_malformed_output(isolated_chroma_dir):
    retriever = _indexed_marketing_retriever(isolated_chroma_dir)
    domain = get_domain("marketing")
    chunks = retriever.retrieve(domain="marketing", query="Segment B trail veterans", similarity_threshold=0.0, top_k=3)
    assert chunks

    broken_json = _valid_campaign_json()
    broken_json["recommended_channels"] = []  # invalid: schema requires >=1

    fixed_json = _valid_campaign_json()
    fixed_json["recommended_channels"] = ["Email"]
    fixed_json["supporting_sources"] = [
        {
            "doc_id": chunks[0].doc_id,
            "doc_title": chunks[0].doc_title,
            "doc_type": chunks[0].doc_type,
            "excerpt": chunks[0].text[:100],
            "relevance_score": chunks[0].score,
        }
    ]

    mock_llm = MockLLMClient(responses=[
        _planner_response(),
        broken_json,
        fixed_json,
    ])
    agent = MarketingAgent(domain_config=domain, retriever=retriever, llm_client=mock_llm)

    result = agent.run("Build a campaign for Segment B", request_id="test-4")

    assert result.validation.status == "repaired"
    assert result.answer.response_type == "campaign_strategy"
    assert len(mock_llm.calls) == 3  # planner + broken generation + repair


def test_phantom_source_citation_is_caught_by_post_guardrail(isolated_chroma_dir):
    retriever = _indexed_marketing_retriever(isolated_chroma_dir)
    domain = get_domain("marketing")

    campaign_json = _valid_campaign_json()
    campaign_json["supporting_sources"] = [
        {
            "doc_id": "totally_made_up::chunk_99",
            "doc_title": "made_up.md",
            "doc_type": "segment",
            "excerpt": "this source was never retrieved",
            "relevance_score": 0.9,
        }
    ]

    mock_llm = MockLLMClient(responses=[
        _planner_response(),
        campaign_json,
    ])
    agent = MarketingAgent(domain_config=domain, retriever=retriever, llm_client=mock_llm)

    result = agent.run("Create a campaign for Segment A", request_id="test-5")

    # Post-guardrail should catch the phantom citation and degrade to abstention
    # rather than returning a response that cites a source that wasn't retrieved.
    assert result.answer.response_type == "insufficient_information"
    assert result.validation.status == "failed"
