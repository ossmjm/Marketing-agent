from __future__ import annotations

from app.agent.confidence import compute_confidence
from app.agent.state import RetrievedChunk


def _chunk(doc_id="a::chunk_0", score=0.8):
    return RetrievedChunk(doc_id=doc_id, doc_title="a.md", doc_type="segment", text="text", score=score)


def test_no_chunks_gives_low_confidence():
    report = compute_confidence([], validation_status="passed", retrieval_attempts=1)
    assert report.level == "low"
    assert report.score < 0.2


def test_high_relevance_multiple_sources_passed_validation_gives_high_confidence():
    chunks = [_chunk("a::0", 0.9), _chunk("b::0", 0.85), _chunk("c::0", 0.88)]
    report = compute_confidence(chunks, validation_status="passed", retrieval_attempts=1)
    assert report.level == "high"


def test_repaired_validation_scores_lower_than_passed():
    chunks = [_chunk("a::0", 0.9), _chunk("b::0", 0.85)]
    passed = compute_confidence(chunks, validation_status="passed", retrieval_attempts=1)
    repaired = compute_confidence(chunks, validation_status="repaired", retrieval_attempts=1)
    assert repaired.score < passed.score


def test_retry_penalizes_score():
    chunks = [_chunk("a::0", 0.9)]
    no_retry = compute_confidence(chunks, validation_status="passed", retrieval_attempts=1)
    with_retry = compute_confidence(chunks, validation_status="passed", retrieval_attempts=2)
    assert with_retry.score < no_retry.score


def test_score_is_always_within_bounds():
    chunks = [_chunk("a::0", 1.5)]  # out-of-range input shouldn't blow past 1.0
    report = compute_confidence(chunks, validation_status="passed", retrieval_attempts=1)
    assert 0.0 <= report.score <= 1.0
