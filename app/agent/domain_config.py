"""
DomainConfig: the single extension point for adapting the agent to a
new industry (Finance, Medical, Pharma, Hospitality, ...).

This is the answer to the assignment's "domain adaptability" requirement.
`app/agent/agent.py` imports NOTHING marketing-specific. It only knows
about a `DomainConfig` object. To add a new domain you:

  1. Write new markdown docs under data/<domain>/
  2. Define Pydantic output schema(s) for that domain (or reuse existing)
  3. Write domain system-prompt text + few-shot examples
  4. Define a guardrail keyword/policy set
  5. Register a `DomainConfig` instance in app/domains/<domain>.py

No changes to agent.py, state.py, retriever.py, or the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class FewShotExample:
    """One worked example injected into the generation prompt."""

    title: str
    user_query: str
    retrieved_context_summary: str
    assistant_response_json: str  # pre-rendered JSON string matching the domain schema
    why_included: str  # documented for interview defensibility, not sent to the LLM


@dataclass(frozen=True)
class GuardrailConfig:
    """
    Domain-specific guardrail policy. The *mechanism* (how these are
    applied) lives in app/agent/guardrails.py and is domain-agnostic;
    only the *content* of these lists changes per domain.
    """

    # Requests that should be refused outright (pattern-matched, case-insensitive)
    unsafe_request_keywords: list[str] = field(default_factory=list)
    # Topics that are structurally out of scope for this domain's agent
    out_of_scope_keywords: list[str] = field(default_factory=list)
    # Claim types the agent must never assert without a retrieved source
    # (used by the post-generation unsupported-claim heuristic)
    fact_sensitive_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DomainConfig:
    name: str
    display_name: str
    description: str

    # RAG
    knowledge_base_path: str
    doc_type_labels: dict[str, str]  # maps filename prefix/category -> human label

    # Prompting
    system_prompt: str
    intent_categories: list[str]
    few_shot_examples: list[FewShotExample]

    # Output contracts: intent -> Pydantic model class (as a callable/type)
    output_schemas: dict[str, Any]

    # Guardrails
    guardrails: GuardrailConfig

    # Optional tools available in this domain: name -> callable
    tools: dict[str, Callable[..., Any]] = field(default_factory=dict)


_REGISTRY: dict[str, DomainConfig] = {}


def register_domain(config: DomainConfig) -> None:
    _REGISTRY[config.name] = config


def get_domain(name: str) -> Optional[DomainConfig]:
    return _REGISTRY.get(name)


def list_domains() -> list[str]:
    return sorted(_REGISTRY.keys())
