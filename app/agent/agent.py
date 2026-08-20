"""
Agent orchestrator: the domain-agnostic control flow.

This module implements the pipeline described in the design doc:

    pre-guardrail -> plan -> retrieve -> sufficiency check
         -> (retry once if insufficient) -> generate -> validate
         -> (repair once if invalid) -> post-guardrail -> confidence
         -> final response

Note: the cheap, keyword-based pre-guardrail check runs BEFORE the
planner's LLM call, not after -- there's no reason to spend an LLM
call classifying intent for a request we're about to refuse anyway.

Why this is "agentic" and not "just RAG"
------------------------------------------
A plain RAG script always does: embed query -> retrieve top_k -> stuff
into prompt -> answer. It cannot decide NOT to retrieve, cannot decide
WHICH categories of knowledge are relevant, and has no notion of
"I tried, retrieval wasn't good enough, do something else." This agent
makes three explicit decisions per request:
  1. What does the user want, and what KB categories are relevant?
     (the planner step)
  2. Is the retrieved evidence actually sufficient to answer, or should
     we retry / abstain? (the sufficiency check + bounded retry)
  3. Is the generated output structurally and factually acceptable, or
     should we attempt one repair before giving up? (validation loop)

Each decision point is bounded (see app/config.py:
MAX_RETRIEVAL_RETRIES, MAX_VALIDATION_REPAIRS) -- this is a controlled
workflow, not an open-ended autonomous loop. That bound is a
deliberate reliability/explainability choice: an unbounded agent loop
is harder to test, harder to put a cost ceiling on, and harder to
explain in an interview than "at most 2 retrieval attempts, at most 2
generation attempts."
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict

from pydantic import ValidationError

from app.agent import guardrails as gr
from app.agent.confidence import compute_confidence
from app.agent.domain_config import DomainConfig
from app.agent.prompts import (
    build_generation_prompt,
    build_planner_prompt,
    build_validation_repair_prompt,
)
from app.agent.state import AgentState, Intent, RetrievedChunk
from app.config import settings
from app.llm.client import LLMClient, LLMError, get_llm_client
from app.logging_config import get_logger
from app.rag.retriever import Retriever
from app.schemas.outputs import (
    AbstentionResponse,
    ChatResponse,
    ConfidenceReport,
    RefusalResponse,
    SourceRef,
    ValidationResult,
)

logger = get_logger("agent")

# Maps a planner-declared intent to the Pydantic schema used for generation/validation.
_SCHEMA_EXAMPLE_CACHE: dict[str, str] = {}


class AgentError(Exception):
    """Raised for unrecoverable pipeline failures (not user-facing abstentions)."""


class MarketingAgent:
    """
    Domain-agnostic orchestrator. Despite the class name (kept for
    clarity in this assignment, since Marketing is the implemented
    domain), nothing in this class is marketing-specific -- every
    marketing-specific value comes from the `DomainConfig` passed in.
    A hypothetical `FinanceAgent` would literally be this same class
    invoked with a different `DomainConfig`.
    """

    def __init__(
        self,
        domain_config: DomainConfig,
        retriever: Retriever | None = None,
        llm_client: LLMClient | None = None,
    ):
        self.domain = domain_config
        self.retriever = retriever or Retriever()
        self.llm = llm_client or get_llm_client()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(self, user_query: str, request_id: str | None = None) -> ChatResponse:
        request_id = request_id or str(uuid.uuid4())
        start = time.monotonic()
        state = AgentState(request_id=request_id, domain=self.domain.name, user_query=user_query)

        try:
            response_payload, confidence, validation = self._run_pipeline(state)
        except LLMError as exc:
            logger.error(
                "llm_provider_failure",
                extra={"request_id": request_id, "domain": self.domain.name, "error": str(exc)},
            )
            response_payload, confidence, validation = self._graceful_llm_failure(state, str(exc))
        except Exception as exc:  # noqa: BLE001 - top-level safety net, logged with context
            logger.error(
                "agent_pipeline_failure",
                extra={"request_id": request_id, "domain": self.domain.name, "error": str(exc)},
            )
            response_payload, confidence, validation = self._graceful_llm_failure(
                state, f"internal pipeline error: {exc}"
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        sources = [
            SourceRef(
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                doc_type=c.doc_type,
                excerpt=c.text[:280],
                relevance_score=round(c.score, 3),
            )
            for c in state.retrieved_chunks
        ]

        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "domain": self.domain.name,
                "intent": state.intent.value if state.intent else None,
                "retrieved_doc_ids": [c.doc_id for c in state.retrieved_chunks],
                "retrieval_count": len(state.retrieved_chunks),
                "retrieval_attempts": state.retrieval_attempts,
                "validation_status": validation.status,
                "tool_calls": [t.get("tool") for t in state.tool_calls],
                "latency_ms": latency_ms,
            },
        )

        return ChatResponse(
            request_id=request_id,
            domain=self.domain.name,
            answer=response_payload,
            sources=sources,
            confidence=confidence,
            validation=validation,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_pipeline(self, state: AgentState):
        # 1. PRE-GUARDRAIL: cheap, keyword-based, no LLM call involved -- reject
        # unsafe requests before spending any retrieval/generation budget on them.
        pre_check = gr.run_pre_checks(self.domain, state.user_query)
        if not pre_check.allowed:
            state.log_step("guardrail_pre_check", f"blocked: {pre_check.category}")
            refusal = RefusalResponse(reason=pre_check.reason, category=pre_check.category)
            confidence = ConfidenceReport(level="low", score=0.0, factors=["request blocked by guardrail"])
            validation = ValidationResult(status="passed", errors=[])
            return refusal, confidence, validation

        # 2. PLAN: classify intent + decide what to retrieve (first LLM call)
        self._plan(state)

        if state.intent == Intent.OUT_OF_SCOPE:
            state.log_step("planner", "classified as out_of_scope")
            refusal = RefusalResponse(
                reason=(
                    f"This request falls outside the {self.domain.display_name} agent's scope "
                    f"({state.planner_rationale})."
                ),
                category="out_of_scope",
            )
            confidence = ConfidenceReport(level="low", score=0.0, factors=["out of scope"])
            validation = ValidationResult(status="passed", errors=[])
            return refusal, confidence, validation

        # 3. RETRIEVE (with one bounded retry if insufficient)
        self._retrieve_with_retry(state)

        if not state.retrieval_sufficient:
            state.log_step("sufficiency_check", "insufficient evidence after retries; abstaining")
            confidence = compute_confidence(state.retrieved_chunks, "failed", state.retrieval_attempts)
            abstention = AbstentionResponse(
                reason=(
                    "The available knowledge base did not return sufficiently relevant "
                    "information to answer this request reliably."
                ),
                what_was_searched=list(self.domain.doc_type_labels.keys()),
                what_would_help="More specific knowledge-base content covering this topic.",
                confidence=confidence,
            )
            validation = ValidationResult(status="passed", errors=[])
            return abstention, confidence, validation

        # 4. GENERATE (with one bounded repair if validation fails)
        parsed, validation = self._generate_with_validation(state)

        # 5. POST-GUARDRAIL
        available_ids = {c.doc_id for c in state.retrieved_chunks}
        post_check = gr.run_post_checks(self.domain, parsed, available_ids)
        if not post_check.passed:
            validation.status = "failed"
            validation.errors.extend(post_check.errors)
            state.log_step("guardrail_post_check", f"failed: {post_check.errors}")
            confidence = compute_confidence(state.retrieved_chunks, "failed", state.retrieval_attempts)
            abstention = AbstentionResponse(
                reason=(
                    "The generated response did not pass post-generation safety checks "
                    "(e.g. citing sources not actually retrieved, or unsupported-certainty "
                    "language), so it is being withheld rather than returned as-is."
                ),
                what_was_searched=[c.doc_title for c in state.retrieved_chunks],
                what_would_help="Refine the request or expand the knowledge base.",
                confidence=confidence,
            )
            return abstention, confidence, validation

        # 6. CONFIDENCE
        confidence = compute_confidence(state.retrieved_chunks, validation.status, state.retrieval_attempts)
        parsed["confidence"] = confidence.model_dump()

        final_model = self._build_typed_response(state.intent, parsed)
        state.log_step("complete", f"validation={validation.status}, confidence={confidence.level}")
        return final_model, confidence, validation

    def _plan(self, state: AgentState) -> None:
        system, user = build_planner_prompt(self.domain, state.user_query)
        raw = self.llm.complete_json(system, user)

        intent_str = raw.get("intent", Intent.FACTUAL_QUESTION.value)
        try:
            state.intent = Intent(intent_str)
        except ValueError:
            state.intent = Intent.FACTUAL_QUESTION

        state.target_doc_types = [
            t for t in raw.get("target_doc_types", []) if t in self.domain.doc_type_labels
        ]
        state.planner_rationale = raw.get("rationale", "")
        state.log_step(
            "planner",
            f"intent={state.intent.value}, target_doc_types={state.target_doc_types}",
        )

    def _retrieve_with_retry(self, state: AgentState) -> None:
        max_attempts = settings.max_retrieval_retries + 1
        threshold = settings.retrieval_similarity_threshold

        for attempt in range(1, max_attempts + 1):
            state.retrieval_attempts = attempt
            chunks = self.retriever.retrieve(
                domain=self.domain.name,
                query=state.user_query,
                doc_types=state.target_doc_types or None,
                similarity_threshold=threshold,
            )
            state.log_step(
                "retrieval",
                f"attempt={attempt}, threshold={threshold:.2f}, returned={len(chunks)}",
            )
            if chunks:
                state.retrieved_chunks = chunks
                state.retrieval_sufficient = True
                return

            # Retry with relaxed threshold and broadened doc-type scope.
            threshold = max(0.0, threshold - 0.10)
            state.target_doc_types = []  # broaden: search all categories on retry

        state.retrieval_sufficient = False

    def _generate_with_validation(self, state: AgentState) -> tuple[dict, ValidationResult]:
        schema_cls = self._resolve_schema(state.intent)
        schema_example = self._schema_json_example(schema_cls)

        system, user = build_generation_prompt(
            self.domain, state.user_query, state.retrieved_chunks, schema_example
        )
        raw = self.llm.complete_json(system, user)
        state.draft_response = raw

        parsed, errors = self._validate(raw, schema_cls)
        if not errors:
            state.validation_status = "passed"
            return parsed, ValidationResult(status="passed", errors=[])

        # One bounded repair attempt
        if settings.max_validation_repairs > 0:
            state.validation_repairs += 1
            repair_system, repair_user = build_validation_repair_prompt(
                json.dumps(raw), errors
            )
            repaired_raw = self.llm.complete_json(repair_system, repair_user)
            parsed, repair_errors = self._validate(repaired_raw, schema_cls)
            if not repair_errors:
                state.validation_status = "repaired"
                state.log_step("validation", "repaired successfully after 1 retry")
                return parsed, ValidationResult(status="repaired", errors=errors)
            errors = repair_errors

        state.validation_status = "failed"
        state.log_step("validation", f"failed after repair attempt: {errors}")
        return raw, ValidationResult(status="failed", errors=errors)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_schema(self, intent: Intent):
        return self.domain.output_schemas.get(
            intent.value, next(iter(self.domain.output_schemas.values()))
        )

    def _schema_json_example(self, schema_cls) -> str:
        key = schema_cls.__name__
        if key not in _SCHEMA_EXAMPLE_CACHE:
            _SCHEMA_EXAMPLE_CACHE[key] = json.dumps(
                schema_cls.model_json_schema(), indent=None
            )
        return _SCHEMA_EXAMPLE_CACHE[key]

    def _validate(self, raw: dict, schema_cls) -> tuple[dict, list[str]]:
        try:
            # confidence is injected by the pipeline, not required from the LLM
            payload = dict(raw)
            payload.setdefault(
                "confidence", {"level": "medium", "score": 0.5, "factors": []}
            )
            model = schema_cls(**payload)
            return model.model_dump(), []
        except ValidationError as exc:
            errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            return raw, errors

    def _build_typed_response(self, intent: Intent, parsed: dict):
        schema_cls = self._resolve_schema(intent)
        return schema_cls(**parsed)

    def _graceful_llm_failure(self, state: AgentState, error_detail: str):
        confidence = ConfidenceReport(level="low", score=0.0, factors=["provider or pipeline failure"])
        validation = ValidationResult(status="failed", errors=[error_detail])
        abstention = AbstentionResponse(
            reason=(
                "The agent could not complete this request due to an internal or provider "
                "error, rather than a lack of knowledge. Please retry."
            ),
            what_was_searched=[c.doc_title for c in state.retrieved_chunks],
            what_would_help="Retrying the request, or checking service status.",
            confidence=confidence,
        )
        return abstention, confidence, validation
