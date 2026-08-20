"""
AgentState: the single mutable record threaded through the pipeline.

Why this exists (for engineers new to agentic patterns)
---------------------------------------------------------
In a plain RAG script you'd just have local variables. Once you add
retries, guardrail checks, and repair loops, you need one place that
answers "what has happened so far in this request?" -- both so each
pipeline step can make decisions (e.g. "have we already retried
retrieval once?") and so we can log/inspect the whole execution trail
after the fact for observability.

Crucially, `AgentState` stores *facts about execution* (what was
retrieved, what validation said, how many retries happened) -- it does
NOT store hidden chain-of-thought text from the LLM. The `rationale`
field on each step is a short, user-safe, human-authored string (e.g.
"retrieved 3 segment docs, above threshold"), never a raw reasoning
dump from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Intent(str, Enum):
    CAMPAIGN_GENERATION = "campaign_generation"
    SEGMENT_ANALYSIS = "segment_analysis"
    FACTUAL_QUESTION = "factual_question"
    KPI_CALCULATION = "kpi_calculation"
    OUT_OF_SCOPE = "out_of_scope"
    UNSAFE = "unsafe"


@dataclass
class RetrievedChunk:
    doc_id: str
    doc_title: str
    doc_type: str
    text: str
    score: float


@dataclass
class ExecutionStep:
    """One entry in the audit trail -- shown to the user as 'execution info', never raw CoT."""

    name: str
    detail: str


@dataclass
class AgentState:
    request_id: str
    domain: str
    user_query: str

    # Planning
    intent: Optional[Intent] = None
    target_doc_types: list[str] = field(default_factory=list)
    planner_rationale: str = ""

    # Retrieval
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_attempts: int = 0
    retrieval_sufficient: bool = False

    # Tools
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    # Generation
    draft_response: Optional[dict[str, Any]] = None

    # Validation
    validation_status: str = "pending"  # pending | passed | repaired | failed
    validation_errors: list[str] = field(default_factory=list)
    validation_repairs: int = 0

    # Final
    final_response: Optional[dict[str, Any]] = None

    # Observability
    execution_trail: list[ExecutionStep] = field(default_factory=list)

    def log_step(self, name: str, detail: str) -> None:
        self.execution_trail.append(ExecutionStep(name=name, detail=detail))
