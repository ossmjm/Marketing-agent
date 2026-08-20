"""
Evaluation harness.

What this measures, and how (mapped to the assignment's required
dimensions)
------------------------------------------------------------------
- Retrieval quality  : did we retrieve >= expected number of sources,
                        and from a plausible doc_type category?
                        (measured directly from AgentState/ChatResponse,
                        not judged by an LLM)
- Structure           : does the returned object satisfy its Pydantic
                        schema? (this is enforced automatically -- if
                        `agent.run()` returns a `ChatResponse` at all,
                        structure already passed; we still assert
                        `response_type` matches the expected category)
- Safety / abstention : for cases expecting refusal or abstention, did
                        the agent actually refuse/abstain rather than
                        answer?
- Groundedness        : approximated by checking that every
                        `supporting_sources[].doc_id` in the answer
                        actually appears in the chunks that were
                        retrieved for that request (this reuses the
                        same phantom-citation check as the runtime
                        guardrail, applied here as an eval metric)
- Relevance            : OPTIONAL, LLM-as-judge (see below) -- off by
                        default.

LLM-as-judge limitations (explicitly documented, per assignment
requirement)
------------------------------------------------------------------
An optional `judge_relevance()` function is provided that asks an LLM
whether a response addresses the user's query. This is included
because "does this actually answer the question" is hard to check
with pure heuristics. But it is NOT treated as ground truth:
  - it inherits the judge model's own biases and blind spots,
  - it is not independently validated against human judgment here,
  - it can be inconsistent across runs at temperature > 0,
  - a model may be more lenient toward outputs whose style matches its
    own preferences.
It is reported as a separate, clearly-labeled score, not blended into
the deterministic metrics above.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.agent.agent import MarketingAgent
from app.agent.domain_config import DomainConfig, get_domain


@dataclass
class CaseResult:
    id: str
    category: str
    passed: bool
    details: list[str] = field(default_factory=list)
    response_type: str | None = None
    n_sources: int = 0
    latency_ms: int = 0


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def by_category(self) -> dict[str, float]:
        cats: dict[str, list[bool]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.passed)
        return {c: sum(v) / len(v) for c, v in cats.items()}

    def to_dict(self) -> dict:
        return {
            "pass_rate": round(self.pass_rate, 3),
            "by_category": {k: round(v, 3) for k, v in self.by_category().items()},
            "cases": [
                {
                    "id": r.id,
                    "category": r.category,
                    "passed": r.passed,
                    "details": r.details,
                    "response_type": r.response_type,
                    "n_sources": r.n_sources,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
        }


def load_test_cases(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _check_groundedness(chat_response) -> tuple[bool, list[str]]:
    """Every cited source doc_id must correspond to an actually-retrieved chunk."""
    retrieved_ids = {s.doc_id for s in chat_response.sources}
    answer = chat_response.answer
    cited = getattr(answer, "supporting_sources", []) or []
    phantom = [s.doc_id for s in cited if s.doc_id not in retrieved_ids]
    if phantom:
        return False, [f"cited phantom source ids not in retrieved context: {phantom}"]
    return True, []


def run_case(agent: MarketingAgent, case: dict) -> CaseResult:
    details: list[str] = []
    passed = True

    chat_response = agent.run(case["query"], request_id=f"eval-{case['id']}")
    answer = chat_response.answer
    response_type = getattr(answer, "response_type", None)

    expected_type = case.get("expect_response_type")
    if expected_type and response_type != expected_type:
        passed = False
        details.append(f"expected response_type={expected_type!r}, got {response_type!r}")

    n_sources = len(chat_response.sources)
    min_sources = case.get("expect_min_sources", 0)
    if n_sources < min_sources:
        passed = False
        details.append(f"expected >= {min_sources} sources, got {n_sources}")

    expect_doc_types_any = case.get("expect_doc_types_any")
    if expect_doc_types_any:
        actual_types = {s.doc_type for s in chat_response.sources}
        if not (actual_types & set(expect_doc_types_any)):
            passed = False
            details.append(
                f"expected at least one source doc_type in {expect_doc_types_any}, got {sorted(actual_types)}"
            )

    grounded, ground_details = _check_groundedness(chat_response)
    if not grounded:
        passed = False
        details.extend(ground_details)

    if chat_response.validation.status == "failed" and expected_type not in (
        "insufficient_information",
        "refused",
    ):
        passed = False
        details.append(f"validation failed unexpectedly: {chat_response.validation.errors}")

    return CaseResult(
        id=case["id"],
        category=case["category"],
        passed=passed,
        details=details,
        response_type=response_type,
        n_sources=n_sources,
        latency_ms=chat_response.latency_ms,
    )


def run_evaluation(test_cases_path: str, domain_name: str = "marketing") -> EvalReport:
    import app.domains  # noqa: F401 -- ensure domain registration

    domain_config: DomainConfig = get_domain(domain_name)
    if domain_config is None:
        raise ValueError(f"Unknown domain: {domain_name}")

    agent = MarketingAgent(domain_config=domain_config)
    cases = load_test_cases(test_cases_path)
    results = [run_case(agent, c) for c in cases if c.get("domain", domain_name) == domain_name]
    return EvalReport(results=results)


def judge_relevance(agent_response_text: str, user_query: str, llm_client) -> dict:
    """
    OPTIONAL LLM-as-judge relevance score. Not used in the pass/fail
    metrics above -- see module docstring for why. Returns a dict with
    a 0-1 score and short justification, clearly separate from the
    deterministic EvalReport.
    """
    system = (
        "You are grading whether a response addresses a user's request. "
        "Respond as JSON: {\"relevant\": true|false, \"justification\": \"<one sentence>\"}. "
        "This is a rough relevance check, not a factual accuracy check."
    )
    user = f"User request: {user_query}\n\nResponse to grade: {agent_response_text}"
    return llm_client.complete_json(system, user)
