from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.outputs import (
    AbstentionResponse,
    CampaignStrategyResponse,
    ConfidenceReport,
    RefusalResponse,
    SegmentAnalysisResponse,
    SourceRef,
)


def _confidence():
    return ConfidenceReport(level="medium", score=0.5, factors=["test"])


def test_campaign_strategy_response_valid_payload():
    resp = CampaignStrategyResponse(
        campaign_objective="Acquisition",
        target_segment="Segment A",
        key_message="Come hike with us",
        campaign_strategy="Lead with paid social.",
        recommended_channels=["Instagram"],
        content_ideas=["Carousel post"],
        kpis=["Signups"],
        supporting_sources=[
            SourceRef(
                doc_id="segment::a.md::chunk_0",
                doc_title="a.md",
                doc_type="segment",
                excerpt="excerpt text",
                relevance_score=0.8,
            )
        ],
        confidence=_confidence(),
        limitations=[],
    )
    assert resp.response_type == "campaign_strategy"


def test_campaign_strategy_response_rejects_empty_channel_list():
    with pytest.raises(ValidationError):
        CampaignStrategyResponse(
            campaign_objective="Acquisition",
            target_segment="Segment A",
            key_message="msg",
            campaign_strategy="strategy",
            recommended_channels=[],
            content_ideas=["idea"],
            kpis=["kpi"],
            confidence=_confidence(),
        )


def test_campaign_strategy_response_rejects_blank_strings_in_lists():
    with pytest.raises(ValidationError):
        CampaignStrategyResponse(
            campaign_objective="Acquisition",
            target_segment="Segment A",
            key_message="msg",
            campaign_strategy="strategy",
            recommended_channels=["   "],
            content_ideas=["idea"],
            kpis=["kpi"],
            confidence=_confidence(),
        )


def test_segment_analysis_response_valid():
    resp = SegmentAnalysisResponse(
        segment_name="Segment B",
        segment_summary="Veterans",
        key_characteristics=["research-heavy"],
        recommended_approach="Use technical detail",
        confidence=_confidence(),
    )
    assert resp.response_type == "segment_analysis"


def test_confidence_report_score_bounds():
    with pytest.raises(ValidationError):
        ConfidenceReport(level="high", score=1.5, factors=[])
    with pytest.raises(ValidationError):
        ConfidenceReport(level="low", score=-0.1, factors=[])


def test_abstention_response_shape():
    resp = AbstentionResponse(
        reason="not enough info",
        what_was_searched=["segment.md"],
        what_would_help="more data",
        confidence=_confidence(),
    )
    assert resp.response_type == "insufficient_information"


def test_refusal_response_requires_valid_category():
    with pytest.raises(ValidationError):
        RefusalResponse(reason="nope", category="not_a_real_category")

    resp = RefusalResponse(reason="nope", category="out_of_scope")
    assert resp.response_type == "refused"
