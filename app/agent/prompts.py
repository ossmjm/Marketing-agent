"""
Prompt construction.

Design intent
-------------
Prompts are assembled from small, named, testable fragments rather
than one giant hardcoded f-string buried in agent.py. Each fragment
has ONE job:

  - CORE_SYSTEM_PREAMBLE        : behavior common to every domain
                                   (grounding discipline, no fabrication,
                                   no chain-of-thought leakage)
  - domain.system_prompt        : domain persona + domain-specific rules
                                   (comes from DomainConfig, not hardcoded here)
  - build_context_block()       : renders retrieved chunks with source ids
  - build_few_shot_block()      : renders the domain's worked examples
  - OUTPUT_FORMAT_INSTRUCTIONS  : how to shape the structured output
  - build_planner_prompt()      : the intent-classification call
  - build_generation_prompt()   : the main grounded-generation call
  - build_validation_repair_prompt(): re-prompt with validation errors

Keeping these separate means each piece can be unit-tested for its
own content (e.g. "does the context block include doc_id for every
chunk?") without mocking the whole pipeline, and a new domain can
override only what it needs to.
"""

from __future__ import annotations

from app.agent.domain_config import DomainConfig, FewShotExample
from app.agent.state import RetrievedChunk

# ---------------------------------------------------------------------------
# Domain-agnostic core instructions
# ---------------------------------------------------------------------------

CORE_SYSTEM_PREAMBLE = """\
You are an AI assistant operating inside a controlled, retrieval-grounded \
agent pipeline. You must follow these non-negotiable rules regardless of \
domain:

1. GROUNDING: Base every factual claim about the organization, its \
   customers, its brand, or its past performance ONLY on the retrieved \
   context provided to you. Never invent statistics, dates, names, or \
   outcomes that are not present in the retrieved context.
2. DISTINGUISH FACT FROM SUGGESTION: When you state something that comes \
   directly from retrieved context, it must be traceable to a source. When \
   you propose something new (a creative idea, a suggested channel, a new \
   angle), that is a generated suggestion, not a fact -- do not present it \
   with the same certainty as a sourced fact.
3. UNCERTAINTY: If the retrieved context does not adequately cover the \
   request, say so plainly rather than filling gaps with plausible-sounding \
   invented details.
4. NO HIDDEN REASONING IN OUTPUT: Respond only with the structured fields \
   requested. Do not include step-by-step internal reasoning, meta-commentary \
   about your process, or apologies -- only the final structured content.
5. OUTPUT FORMAT: Respond with a single JSON object that matches the schema \
   you are given. No markdown code fences, no prose outside the JSON.
"""

OUTPUT_FORMAT_INSTRUCTIONS = """\
Return ONLY a single valid JSON object. Do not wrap it in markdown code \
fences. Do not include any text before or after the JSON. Every field in \
the schema must be present. For `supporting_sources`, cite only doc_ids \
that actually appear in the "RETRIEVED CONTEXT" section below -- never \
invent a doc_id.
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks with explicit source ids for citation."""
    if not chunks:
        return "RETRIEVED CONTEXT: (none retrieved)"

    lines = ["RETRIEVED CONTEXT:"]
    for c in chunks:
        lines.append(
            f'--- source doc_id="{c.doc_id}" title="{c.doc_title}" '
            f'type="{c.doc_type}" relevance_score={c.score:.2f} ---'
        )
        lines.append(c.text.strip())
    return "\n".join(lines)


def build_few_shot_block(examples: list[FewShotExample]) -> str:
    if not examples:
        return ""
    parts = ["FEW-SHOT EXAMPLES (for behavior calibration only, not real data):"]
    for i, ex in enumerate(examples, start=1):
        parts.append(
            f"Example {i} - {ex.title}\n"
            f"User query: {ex.user_query}\n"
            f"Retrieved context summary: {ex.retrieved_context_summary}\n"
            f"Expected assistant JSON output:\n{ex.assistant_response_json}\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Planner prompt (intent classification + retrieval targeting)
# ---------------------------------------------------------------------------

def build_planner_prompt(domain: DomainConfig, user_query: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the planning LLM call."""
    system = (
        CORE_SYSTEM_PREAMBLE
        + "\nYou are now acting as the PLANNING step of the agent, for the "
        f"domain: {domain.display_name}.\n"
        "Your job is ONLY to classify the user's request and decide what "
        "knowledge-base categories are needed. Do not answer the request itself.\n"
        f"Valid intent categories: {', '.join(domain.intent_categories)}.\n"
        f"Valid knowledge-base categories to request: {', '.join(domain.doc_type_labels.keys())}.\n"
        "Respond as JSON: "
        '{"intent": "<one of the valid intents>", '
        '"target_doc_types": ["<zero or more valid kb categories>"], '
        '"rationale": "<one short sentence>"}'
    )
    user = f'User request: "{user_query}"'
    return system, user


# ---------------------------------------------------------------------------
# Generation prompt (grounded answer, structured output)
# ---------------------------------------------------------------------------

def build_generation_prompt(
    domain: DomainConfig,
    user_query: str,
    chunks: list[RetrievedChunk],
    schema_json_example: str,
) -> tuple[str, str]:
    system_parts = [
        CORE_SYSTEM_PREAMBLE,
        "\nDOMAIN INSTRUCTIONS:\n" + domain.system_prompt,
    ]
    few_shot = build_few_shot_block(domain.few_shot_examples)
    if few_shot:
        system_parts.append("\n" + few_shot)
    system_parts.append(
        "\nOUTPUT SCHEMA (produce a JSON object with exactly this shape):\n"
        + schema_json_example
    )
    system_parts.append("\n" + OUTPUT_FORMAT_INSTRUCTIONS)
    system = "\n".join(system_parts)

    user = (
        f'USER REQUEST:\n"{user_query}"\n\n'
        + build_context_block(chunks)
    )
    return system, user


# ---------------------------------------------------------------------------
# Validation-repair prompt
# ---------------------------------------------------------------------------

def build_validation_repair_prompt(
    previous_output_json: str, validation_errors: list[str]
) -> tuple[str, str]:
    system = (
        CORE_SYSTEM_PREAMBLE
        + "\nYour previous JSON output failed schema validation. Fix ONLY the "
        "listed problems and return a corrected, complete JSON object that "
        "matches the required schema. Do not change fields that were not "
        "flagged as problematic.\n" + OUTPUT_FORMAT_INSTRUCTIONS
    )
    user = (
        f"PREVIOUS OUTPUT:\n{previous_output_json}\n\n"
        f"VALIDATION ERRORS TO FIX:\n- " + "\n- ".join(validation_errors)
    )
    return system, user
