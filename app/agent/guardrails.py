"""
Guardrail mechanisms.

Split into PRE-checks (run on the raw user query, before spending any
retrieval/generation budget) and POST-checks (run on the generated
structured output, before it's returned to the user).

Everything here is domain-agnostic *mechanism*. The domain-specific
*content* (keyword lists, fact-sensitive terms) comes from
`DomainConfig.guardrails` (a `GuardrailConfig`). This is what makes the
guardrail layer reusable across Finance/Medical/Pharma/Hospitality: you
swap the keyword lists, not the logic.

Explicit limitation (documented, not hidden): keyword/heuristic
matching will not catch cleverly-obfuscated unsafe requests and may
occasionally over-trigger on benign phrasing. It is a first line of
defense, not a complete safety system. A production system would layer
a moderation-classifier model on top of this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.agent.domain_config import DomainConfig
from app.schemas.outputs import ConfidenceReport


@dataclass
class PreCheckResult:
    allowed: bool
    category: str | None = None  # "out_of_scope" | "unsafe_request"
    reason: str | None = None


def _matches_any(text: str, keywords: list[str]) -> str | None:
    lowered = text.lower()
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", lowered):
            return kw
    return None


def run_pre_checks(domain: DomainConfig, user_query: str) -> PreCheckResult:
    """Run before retrieval/generation. Cheap, fast, fails closed on unsafe content."""
    unsafe_hit = _matches_any(user_query, domain.guardrails.unsafe_request_keywords)
    if unsafe_hit:
        return PreCheckResult(
            allowed=False,
            category="unsafe_request",
            reason=(
                f"Request appears to ask for content matching a disallowed "
                f"pattern ('{unsafe_hit}'). This agent will not produce "
                f"manipulative, deceptive, or discriminatory marketing content."
            ),
        )

    out_of_scope_hit = _matches_any(user_query, domain.guardrails.out_of_scope_keywords)
    if out_of_scope_hit:
        return PreCheckResult(
            allowed=False,
            category="out_of_scope",
            reason=(
                f"This request ('{out_of_scope_hit}') falls outside the "
                f"{domain.display_name} agent's scope."
            ),
        )

    return PreCheckResult(allowed=True)


@dataclass
class PostCheckResult:
    passed: bool
    errors: list[str]


UNSUPPORTED_CLAIM_PHRASES = [
    "guaranteed", "guarantee", "will definitely", "proven to increase",
    "studies show", "research shows", "always results in", "100% of",
]


def run_post_checks(
    domain: DomainConfig,
    output_dict: dict,
    available_source_ids: set[str],
) -> PostCheckResult:
    """
    Run after generation, before returning to the user. Checks:
      1. every cited source id actually exists in what was retrieved
         (prevents citation hallucination)
      2. no absolute/unsupported-certainty language leaked into free-text fields
      3. domain-specific fact-sensitive terms are not asserted without
         at least one supporting source
    """
    errors: list[str] = []

    cited = {s.get("doc_id") for s in output_dict.get("supporting_sources", [])}
    phantom_sources = cited - available_source_ids
    if phantom_sources:
        errors.append(
            f"Output cites phantom source id(s) not present in retrieved context: {sorted(phantom_sources)}"
        )

    text_fields = _collect_text_fields(output_dict)
    full_text = " ".join(text_fields).lower()
    for phrase in UNSUPPORTED_CLAIM_PHRASES:
        if phrase in full_text:
            errors.append(f"Output contains unsupported-certainty language: '{phrase}'")

    if domain.guardrails.fact_sensitive_terms and output_dict.get("response_type") not in (
        "insufficient_information",
        "refused",
    ):
        for term in domain.guardrails.fact_sensitive_terms:
            if term.lower() in full_text and not output_dict.get("supporting_sources"):
                errors.append(
                    f"Output asserts fact-sensitive term '{term}' with no supporting sources cited."
                )
                break

    return PostCheckResult(passed=(len(errors) == 0), errors=errors)


def _collect_text_fields(d: dict) -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        if k in ("supporting_sources", "confidence"):
            continue
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            out.extend([x for x in v if isinstance(x, str)])
    return out


def build_low_confidence_report(reason_factors: list[str]) -> ConfidenceReport:
    return ConfidenceReport(level="low", score=0.15, factors=reason_factors)
