"""
Heuristic confidence scoring.

Explicitly NOT "ask the LLM to rate itself 0-100". That approach
produces a number that looks like a probability but has no calibration
guarantee -- LLMs are known to be overconfident and inconsistent when
self-rating. Instead we compute a deterministic score from signals we
can actually measure:

  - retrieval_quality : mean similarity score of chunks actually used
  - source_coverage    : how many distinct sources support the answer
                          (bounded contribution, diminishing returns)
  - validation_penalty : whether the output needed a repair pass
  - retry_penalty      : whether retrieval needed a retry

This is still a heuristic -- we say so in `ConfidenceReport.basis` -- but
it is reproducible, testable, and explainable: given the same inputs
you always get the same score, and you can point at exactly which
factor moved the number.
"""

from __future__ import annotations

from app.agent.state import RetrievedChunk
from app.schemas.outputs import ConfidenceReport


def compute_confidence(
    chunks: list[RetrievedChunk],
    validation_status: str,
    retrieval_attempts: int,
) -> ConfidenceReport:
    factors: list[str] = []

    if not chunks:
        return ConfidenceReport(
            level="low",
            score=0.05,
            factors=["no supporting sources were retrieved"],
        )

    mean_score = sum(c.score for c in chunks) / len(chunks)
    retrieval_component = min(mean_score, 1.0) * 0.55
    factors.append(f"mean retrieval relevance = {mean_score:.2f}")

    n_sources = len({c.doc_id for c in chunks})
    source_component = min(n_sources / 3.0, 1.0) * 0.25
    factors.append(f"{n_sources} distinct supporting source(s)")

    validation_component = {
        "passed": 0.20,
        "repaired": 0.10,
        "failed": 0.0,
    }.get(validation_status, 0.0)
    factors.append(f"validation status = {validation_status}")

    retry_penalty = 0.05 if retrieval_attempts > 1 else 0.0
    if retry_penalty:
        factors.append("required a retrieval retry")

    raw_score = retrieval_component + source_component + validation_component - retry_penalty
    score = max(0.0, min(1.0, raw_score))

    if score >= 0.70:
        level = "high"
    elif score >= 0.40:
        level = "medium"
    else:
        level = "low"

    return ConfidenceReport(level=level, score=round(score, 2), factors=factors)
