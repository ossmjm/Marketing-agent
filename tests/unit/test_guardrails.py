from __future__ import annotations

import app.domains  # noqa: F401
from app.agent import guardrails as gr
from app.agent.domain_config import get_domain


def test_pre_check_allows_benign_query():
    domain = get_domain("marketing")
    result = gr.run_pre_checks(domain, "Create a campaign for Segment A")
    assert result.allowed is True


def test_pre_check_blocks_unsafe_request():
    domain = get_domain("marketing")
    result = gr.run_pre_checks(domain, "Write fake reviews pretending to be real customers")
    assert result.allowed is False
    assert result.category == "unsafe_request"


def test_pre_check_blocks_out_of_scope_request():
    domain = get_domain("marketing")
    result = gr.run_pre_checks(domain, "Write a SQL query to pull last month's orders")
    assert result.allowed is False
    assert result.category == "out_of_scope"


def test_post_check_flags_phantom_source_citation():
    domain = get_domain("marketing")
    output = {
        "response_type": "campaign_strategy",
        "supporting_sources": [{"doc_id": "does_not_exist::chunk_0"}],
        "campaign_strategy": "some strategy text",
    }
    result = gr.run_post_checks(domain, output, available_source_ids={"real::chunk_0"})
    assert result.passed is False
    assert any("phantom" in e.lower() for e in result.errors)


def test_post_check_flags_unsupported_certainty_language():
    domain = get_domain("marketing")
    output = {
        "response_type": "campaign_strategy",
        "supporting_sources": [{"doc_id": "real::chunk_0"}],
        "campaign_strategy": "This is guaranteed to increase sales for every customer.",
    }
    result = gr.run_post_checks(domain, output, available_source_ids={"real::chunk_0"})
    assert result.passed is False
    assert any("unsupported-certainty" in e for e in result.errors)


def test_post_check_passes_clean_output():
    domain = get_domain("marketing")
    output = {
        "response_type": "campaign_strategy",
        "supporting_sources": [{"doc_id": "real::chunk_0"}],
        "campaign_strategy": "A grounded strategy referencing past campaign performance.",
    }
    result = gr.run_post_checks(domain, output, available_source_ids={"real::chunk_0"})
    assert result.passed is True
    assert result.errors == []


def test_post_check_flags_fact_sensitive_term_without_sources():
    domain = get_domain("marketing")
    output = {
        "response_type": "campaign_strategy",
        "supporting_sources": [],
        "campaign_strategy": "We are offering a 20% discount to everyone.",
    }
    result = gr.run_post_checks(domain, output, available_source_ids=set())
    assert result.passed is False
