"""
Structured output contracts.

Design note: discriminated union, not one "do everything" schema
-------------------------------------------------------------------
A first draft of this schema (see assignment spec) put "confidence"
and "limitations" fields on a single schema that always assumes a
successful campaign strategy was generated. That's awkward once you
take abstention seriously: "I don't have enough information" is not a
degraded CampaignStrategyResponse, it's a *different kind* of answer
with different fields (what was searched, what would help).

Modeling it as a `Literal["..."]`-discriminated union means:
  - the API/UI can render each case correctly without guessing,
  - Pydantic validates the *shape* is internally consistent,
  - it is impossible to accidentally return a "confident-looking"
    payload for a case where the agent actually abstained.

We also separate `SegmentAnalysisResponse` from
`CampaignStrategyResponse` because "analyze this segment" and "build
me a campaign" are genuinely different deliverables the assignment
asks for (capability #1 and #2), and cramming both into one schema
would force a lot of nullable fields.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class SourceRef(BaseModel):
    """A single retrieved document that supports (part of) the answer."""

    doc_id: str = Field(..., description="Stable identifier of the source chunk/document")
    doc_title: str = Field(..., description="Human-readable document name, e.g. 'brand_guidelines.md'")
    doc_type: str = Field(..., description="Category, e.g. 'brand', 'segment', 'campaign', 'best_practice'")
    excerpt: str = Field(..., description="Short supporting excerpt (<=280 chars) used for the claim")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score from the retriever")


class ConfidenceReport(BaseModel):
    """
    Heuristic confidence, NOT a calibrated probability.

    `score` is a deterministic function of retrieval quality, source
    count, and validation outcome (see app/agent/confidence.py). It is
    intentionally NOT produced by asking the LLM "how confident are
    you 0-100?" -- that number would be an uncalibrated guess dressed
    up as a statistic. We say so explicitly in `basis` so downstream
    consumers of this API never mistake it for a real probability.
    """

    level: Literal["low", "medium", "high"]
    score: float = Field(..., ge=0.0, le=1.0)
    basis: str = Field(
        default=(
            "Heuristic score derived from retrieval relevance, source count, "
            "and validation outcome. Not a calibrated statistical probability."
        )
    )
    factors: list[str] = Field(default_factory=list, description="Human-readable contributing factors")


# ---------------------------------------------------------------------------
# Success responses
# ---------------------------------------------------------------------------

class CampaignStrategyResponse(BaseModel):
    response_type: Literal["campaign_strategy"] = "campaign_strategy"

    campaign_objective: str
    target_segment: str
    key_message: str
    campaign_strategy: str = Field(..., description="2-4 sentence narrative strategy")
    recommended_channels: list[str] = Field(..., min_length=1)
    content_ideas: list[str] = Field(..., min_length=1)
    kpis: list[str] = Field(..., min_length=1)

    supporting_sources: list[SourceRef] = Field(default_factory=list)
    confidence: ConfidenceReport
    limitations: list[str] = Field(
        default_factory=list,
        description="Explicit caveats, e.g. 'no budget data available in KB'",
    )

    @field_validator("recommended_channels", "content_ideas", "kpis")
    @classmethod
    def _no_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [item.strip() for item in v if item and item.strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty item")
        return cleaned


class SegmentAnalysisResponse(BaseModel):
    response_type: Literal["segment_analysis"] = "segment_analysis"

    segment_name: str
    segment_summary: str
    key_characteristics: list[str] = Field(..., min_length=1)
    recommended_approach: str
    relevant_channels: list[str] = Field(default_factory=list)

    supporting_sources: list[SourceRef] = Field(default_factory=list)
    confidence: ConfidenceReport
    limitations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Non-success responses
# ---------------------------------------------------------------------------

class AbstentionResponse(BaseModel):
    """
    Returned when retrieval could not find sufficient grounding
    evidence for the request. This is a *first-class successful
    outcome* of the pipeline (the guardrail worked), not an error.
    """

    response_type: Literal["insufficient_information"] = "insufficient_information"
    reason: str
    what_was_searched: list[str] = Field(default_factory=list)
    what_would_help: str = ""
    confidence: ConfidenceReport


class RefusalResponse(BaseModel):
    """
    Returned when the guardrail layer rejects the request itself
    (out of scope, unsafe/manipulative request, or would require the
    agent to fabricate real-world facts not present in the KB).
    """

    response_type: Literal["refused"] = "refused"
    reason: str
    category: Literal["out_of_scope", "unsafe_request", "unsupported_claim_required"]


# ---------------------------------------------------------------------------
# Top-level envelope
# ---------------------------------------------------------------------------

AgentAnswer = Union[
    CampaignStrategyResponse,
    SegmentAnalysisResponse,
    AbstentionResponse,
    RefusalResponse,
]


class ValidationResult(BaseModel):
    status: Literal["passed", "repaired", "failed"]
    errors: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Top-level API response envelope for POST /chat."""

    request_id: str
    domain: str
    answer: AgentAnswer
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: ConfidenceReport
    validation: ValidationResult
    latency_ms: int
