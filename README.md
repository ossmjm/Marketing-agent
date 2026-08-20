# Marketing Campaign Strategist — Agentic AI Assignment

A domain-agnostic, retrieval-grounded agent core, implemented and demonstrated
for a **Marketing** domain (Marketing Campaign Strategist for a synthetic demo
brand, "Northbrook Outfitters"). The core is built so that Finance, Medical,
Pharma, or Hospitality could be added as a **configuration**, not a rewrite.

---

## 1. Problem Statement

Marketing teams need a way to generate campaign ideas, analyze customer
segments, and build campaign strategies that are **actually grounded** in
their own brand guidelines, customer data, and campaign history — not
generic, plausible-sounding advice an LLM would produce from general
training data alone. A raw LLM call will happily invent statistics,
misstate brand rules, or contradict a company's own past campaign lessons.

This system solves that by combining:
- **Retrieval-Augmented Generation** against a real (synthetic, clearly
  labeled) marketing knowledge base, so recommendations are traceable to
  specific source documents.
- **Agentic control flow** that decides what to retrieve, whether retrieved
  evidence is sufficient, and whether to abstain — rather than always
  answering regardless of evidence quality.
- **Structured, validated output** so the result is directly usable by
  downstream tools (a campaign brief template, a dashboard, etc.), not free
  text that has to be re-parsed.
- **Guardrails** that reject unsafe/out-of-scope requests and catch
  unsupported claims before they reach the user.

---

## 2. Architecture

```
                                USER QUERY
                                    |
                                    v
                    +-------------------------------+
                    | 1. PRE-GUARDRAIL (keyword,      |
                    |    no LLM call -- cheap reject)  |
                    +---------------+-----------------+
                                    | allowed
                                    v
                    +-------------------------------+
                    | 2. PLANNER (LLM call #1)         |
                    |    - classify intent             |
                    |    - choose KB categories        |
                    +---------------+-----------------+
                                    v
                    +-------------------------------+
                    | 3. RETRIEVAL (RAG)               |
                    |    top-k + similarity threshold  |
                    |    + doc_type metadata filter    |
                    +---------------+-----------------+
                                    v
                    +-------------------------------+
                    | 4. SUFFICIENCY CHECK             |
                    |    enough relevant evidence?     |
                    +-------+-----------------+-------+
                       no   |                  | yes
                            v                  v
                  +------------------+  +-------------------------+
                  | retry (relaxed   |  | 5. GENERATE (LLM call #2)|
                  | threshold, wider |  |    grounded, structured  |
                  | doc scope) once, |  |    output via schema     |
                  | then ABSTAIN     |  +-----------+---------------+
                  +------------------+              v
                                       +-------------------------+
                                       | 6. VALIDATE (Pydantic)   |
                                       |    fail -> 1 repair retry|
                                       +-----------+---------------+
                                                   v
                                       +-------------------------+
                                       | 7. POST-GUARDRAIL        |
                                       |    phantom citations,    |
                                       |    unsupported-certainty |
                                       |    language, fact-terms  |
                                       +-----------+---------------+
                                                   v
                                       +-------------------------+
                                       | 8. CONFIDENCE (heuristic)|
                                       +-----------+---------------+
                                                   v
                                            FINAL RESPONSE
                                        (structured + sources +
                                         validation status)
```

### Why this deviates from a naive "diagram-as-given" implementation

The original brief's diagram groups "Query/Intent Understanding" and
"Agent Orchestrator" as separate boxes and shows a generic
LLM-reasoning-plus-tools block. In this implementation:

- **Planner and Orchestrator are one step** (`agent.py`'s `_plan()`), because
  in this domain intent classification and retrieval-targeting are the same
  decision — there's no separate "orchestration" logic beyond that decision
  plus the bounded retry/repair control flow.
- **No LangGraph / agent-executor framework.** The control flow has exactly
  two decision points (retrieval sufficiency, output validity), each with a
  bounded retry. That's a state machine simple enough to write and reason
  about directly in Python (`agent.py`), which is easier to defend in an
  interview than "here's a 6-node graph" for what is functionally two
  if/else branches with retry counters. LangGraph is the natural next step
  if branching complexity grows (e.g. multiple tools competing for
  selection, parallel multi-domain retrieval) — see [Scaling](#10-scaling).
- **LangChain is used only for `RecursiveCharacterTextSplitter`**, not its
  agent/executor abstractions — this keeps the control flow fully visible
  in `agent.py` instead of hidden inside a framework.

### Why this IS agentic, not just RAG

A plain RAG pipeline is `User → Retriever → LLM → Answer` — no decisions.
This system makes three explicit, inspectable decisions per request:
1. **What does the user want, and what KB categories matter?** (planner)
2. **Is retrieved evidence actually sufficient, or should we retry/abstain?**
   (sufficiency check + bounded retry — this is the difference between "we
   retrieved something" and "we retrieved something good enough")
3. **Is the generated output acceptable, or should we attempt one repair
   before giving up?** (validation loop)

Each decision is **bounded** (`MAX_RETRIEVAL_RETRIES=1`,
`MAX_VALIDATION_REPAIRS=1` in `app/config.py`) — a controlled workflow, not
an open-ended autonomous loop, per the assignment's explicit guidance to
prioritize reliability and explainability over artificial complexity.

---

## 3. Agent Workflow

`Query → Retrieval → Reasoning → Validation → Structured Response`, concretely:

1. **Query** hits `POST /chat` (or the Streamlit UI, or `MarketingAgent.run()`
   directly).
2. **Pre-guardrail** (`app/agent/guardrails.py::run_pre_checks`) — cheap
   keyword check, no LLM call. Runs *before* the planner so we never spend
   an LLM call classifying a request we're about to refuse.
3. **Planner LLM call** classifies intent (`campaign_generation`,
   `segment_analysis`, `factual_question`, `kpi_calculation`,
   `out_of_scope`, `unsafe`) and picks target KB categories
   (`brand`/`segment`/`campaign`/`best_practice`).
4. **Retrieval** (`app/rag/retriever.py`) — semantic top-k search, filtered
   by the planner's chosen categories, filtered again by similarity
   threshold. If empty, retry once with a relaxed threshold and broadened
   category scope; if still empty, **abstain** (return
   `AbstentionResponse`, not a guess).
5. **Generation LLM call** — grounded-generation prompt with the retrieved
   context injected, few-shot examples, and explicit schema instructions.
6. **Validation** — the LLM's JSON output is parsed into the domain's
   Pydantic schema. If invalid, one repair call re-prompts with the
   specific validation errors; if still invalid, degrade to abstention.
7. **Post-guardrail** — checks for phantom source citations (citing a
   `doc_id` that wasn't actually retrieved), unsupported-certainty language
   ("guaranteed", "studies show", etc.), and domain-specific fact-sensitive
   terms asserted without a source.
8. **Confidence** — computed deterministically from retrieval relevance,
   source count, and validation outcome (see [Confidence](#confidence--not-a-probability)).
9. **Response** returned as a validated `ChatResponse` with the structured
   answer, source list, confidence, and validation status.

---

## 4. RAG Design

### Chunking
`RecursiveCharacterTextSplitter` (LangChain) with markdown-aware separators
(`## ` headers first, then paragraphs, then sentences) — `app/rag/ingestion.py`.
Splitting on structural boundaries first keeps each chunk topically coherent
(e.g. one customer segment stays in one chunk) before falling back to
character-count splitting for oversized sections. `chunk_size=900`,
`chunk_overlap=150` by default, both configurable via `.env`.

### Embeddings
`text-embedding-3-small` via the OpenAI API by default
(`app/rag/embeddings.py::OpenAIEmbeddingFunction`). A deterministic,
dependency-free `LocalHashEmbeddingFunction` is used when `USE_MOCK_LLM=true`
— this exists **only** so tests and offline development can exercise the
full RAG pipeline (chunking → storage → top-k → thresholding) without
network access or an API key. It is not semantically meaningful and is
explicitly documented as such; do not treat mock-mode retrieval quality as
representative of production behavior.

### Vector database
ChromaDB, local persistent client (`app/rag/vectorstore.py`), one collection
per domain (`domain_marketing`, etc.). Zero infrastructure to run — see
[Trade-offs](#9-trade-offs) for why this wouldn't scale as-is.

### Retrieval
`app/rag/retriever.py` supports:
- **top-k** (`RETRIEVAL_TOP_K`, default 4)
- **similarity threshold** (`RETRIEVAL_SIMILARITY_THRESHOLD`, default 0.28)
  — chunks below this are dropped rather than always returning `top_k`
  regardless of quality
- **metadata filtering** by `doc_type` (`brand`/`segment`/`campaign`/`best_practice`),
  driven by the planner's output
- **source attribution** — every `RetrievedChunk` carries `doc_id`,
  `doc_title`, `doc_type`, and its similarity `score`, which flow straight
  into `supporting_sources` in the final response

### Source attribution end-to-end
The retrieved chunk's `doc_id` is injected into the generation prompt
(`RETRIEVED CONTEXT: --- source doc_id="..." ...`), the LLM is instructed to
only cite `doc_id`s that actually appear there, and the post-guardrail
verifies every cited `doc_id` really was retrieved (catching "phantom
citations" — see [Guardrails](#6-guardrails)).

---

## 5. Prompt Engineering

Prompts are assembled from small, named, composable fragments
(`app/agent/prompts.py`) rather than one hardcoded string:
- `CORE_SYSTEM_PREAMBLE` — domain-agnostic grounding discipline (fact vs.
  suggestion, no chain-of-thought leakage, JSON-only output)
- `DomainConfig.system_prompt` — the Marketing Campaign Strategist persona
  and Northbrook-specific rules (`app/domains/marketing.py`)
- `build_context_block()` — renders retrieved chunks with explicit source ids
- `build_few_shot_block()` — renders the domain's worked examples
- `OUTPUT_FORMAT_INSTRUCTIONS` — JSON-only, schema-matching output rules
- `build_planner_prompt()` / `build_generation_prompt()` /
  `build_validation_repair_prompt()` — compose the above per pipeline stage

This separation means each fragment is independently unit-tested
(`tests/unit/test_prompts.py`) without mocking the whole pipeline, and a new
domain only needs to supply its own `system_prompt` and few-shot examples —
the core preamble and output-format rules are shared.

### Few-shot examples (5, one per required behavior — see `app/domains/marketing.py`)

| # | Behavior | Why this example specifically |
|---|---|---|
| 1 | Normal campaign generation | Establishes the baseline expected shape: concrete channels/content tied to retrieved evidence, with sources cited. |
| 2 | Segment analysis | Shows the *other* schema (`SegmentAnalysisResponse`), demonstrating capability #2 is genuinely distinct from #1, not a subset. |
| 3 | Insufficient knowledge → abstain | Directly demonstrates hallucination avoidance: recognizing an out-of-KB request (financial data) and abstaining instead of guessing a number. |
| 4 | Unsupported / out-of-authority request | Tests whether the agent applies a KB-derived policy rule (discount/delivery claims need Operations sign-off) rather than a hardcoded refusal — guardrails driven by retrieved policy, not just keyword matching. |
| 5 | Conflicting information | Exercises the case where general best-practice guidance and brand-specific campaign history point in different directions; the agent must notice the conflict, state a resolution principle (prefer brand-specific evidence), and surface the tension in `limitations` rather than silently picking one. |

Five examples, not more — enough to calibrate each distinct behavior without
bloating every prompt call with redundant demonstrations.

---

## 6. Guardrails

Guardrails are split into **mechanism** (domain-agnostic, `app/agent/guardrails.py`)
and **content** (domain-specific keyword/term lists, `DomainConfig.guardrails`
— a `GuardrailConfig`). This split is what makes guardrails reusable across
domains: adding Finance means writing new keyword lists, not new logic.

**Pre-checks** (before any LLM call):
- `unsafe_request_keywords` — e.g. "fake reviews", "deceptive", "astroturf" → refused outright
- `out_of_scope_keywords` — e.g. "SQL query", "medical advice" → refused as out of scope

**Post-checks** (after generation, before returning to the user):
- **Phantom citation detection** — every `supporting_sources[].doc_id` in the
  output must correspond to a chunk that was *actually retrieved*; if not,
  the response is withheld and the agent abstains instead of returning it.
  This is the concrete mechanism preventing citation hallucination.
- **Unsupported-certainty language** — phrases like "guaranteed",
  "studies show", "100% of" are flagged regardless of source.
- **Fact-sensitive terms without sources** — domain-configured terms
  (`discount`, `delivery date`, `carbon offset`, `revenue`, ...) trigger a
  failure if asserted with zero supporting sources.

**Hallucination handling principle:** *if available knowledge doesn't
sufficiently support a claim, don't present it as fact.* Implemented via:
retrieval relevance threshold → sufficiency check → bounded retry →
abstention if still insufficient; post-generation phantom-citation and
unsupported-certainty checks; and explicit grounded-generation instructions
in the system prompt distinguishing "sourced fact" from "generated
suggestion."

**Documented limitation:** keyword/heuristic matching will not catch
cleverly-obfuscated unsafe requests and may occasionally over-trigger on
benign phrasing that happens to contain a flagged term. This is a first line
of defense, not a complete safety system — a production system would layer
a moderation-classifier model on top (see [Scaling](#10-scaling)).

### Confidence — not a probability

Per the assignment's explicit requirement, confidence is **not** "ask the
LLM to rate itself 0–100." `app/agent/confidence.py::compute_confidence()`
is a deterministic function of:
- mean retrieval relevance score of chunks actually used (55% weight)
- number of distinct supporting sources, with diminishing returns (25%)
- validation outcome — passed / repaired / failed (20%)
- a small penalty if retrieval needed a retry

Every `ConfidenceReport` carries a `basis` field stating explicitly:
*"Heuristic score derived from retrieval relevance, source count, and
validation outcome. Not a calibrated statistical probability."* This is
reproducible and testable (`tests/unit/test_confidence.py`) — same inputs
always produce the same score, and you can point at exactly which factor
moved the number — but it is still a heuristic, not a calibrated
probability, and the code and docs say so.

---

## 7. Evaluation

`evaluation/test_cases.json` — 11 cases covering every required category:
factual retrieval, campaign generation, segment analysis, insufficient
information, hallucination resistance, source attribution, structured
output validity, guardrail behavior (×3: unsafe, out-of-scope, unsupported
claim), and conflicting information.

`app/evaluation/evaluator.py::run_evaluation()` runs each case through the
**real pipeline** (not a separate judge-only path) and scores, per case:

| Dimension | How it's measured |
|---|---|
| Retrieval quality | Actual retrieved-source count and `doc_type` vs. `expect_min_sources` / `expect_doc_types_any` |
| Structure | Enforced automatically — the response is a validated Pydantic model or the pipeline degrades to abstention; `response_type` is also checked against expectation |
| Safety / abstention | For cases expecting `insufficient_information` or `refused`, did the agent actually abstain/refuse? |
| Groundedness | Every `supporting_sources[].doc_id` in the answer must appear in the sources the *same request* actually retrieved (reuses the runtime phantom-citation check as an eval metric) |
| Relevance | Optional, LLM-as-judge (`judge_relevance()`) — off by default, reported separately, never blended into the deterministic pass/fail metrics |

Run it with a live OpenAI key and an ingested KB:
```bash
python scripts/ingest.py --domain marketing
python scripts/run_eval.py
```

**LLM-as-judge limitations (explicitly not treated as ground truth):** the
optional relevance judge inherits the judge model's own biases and blind
spots, is not independently validated against human judgment in this
project, can be inconsistent across runs, and may be more lenient toward
outputs whose style matches its own preferences. It is reported as a
separate, clearly-labeled score.

**A note on the offline/mock test path:** `tests/integration/test_evaluator.py`
verifies the evaluation harness's own scoring/aggregation logic (pass/fail
conditions, groundedness check, category breakdown) using the same mocked-LLM
approach as the agent pipeline tests. It does **not** run the full 11-case
`test_cases.json` suite against the mock embedding space, because that
dataset is designed to be judged with real semantic retrieval — e.g.
distinguishing "Q3 revenue" as genuinely unrelated to the marketing KB is a
semantic judgment the offline hash-based mock embedding doesn't reliably
reproduce. `scripts/run_eval.py` against a live, ingested agent is the
intended way to run the real suite.

---

## 8. Domain Adaptability

The entire domain-adaptation story is `app/agent/domain_config.py`'s
`DomainConfig` dataclass. `app/agent/agent.py` (`MarketingAgent` — the name
is a historical artifact of this being the implemented domain; the class
contains **zero** marketing-specific logic) only ever touches a
`DomainConfig` object — never a hardcoded prompt string, hardcoded schema,
or hardcoded keyword list.

To add **Finance** (or Medical, Pharma, Hospitality):
1. Write `data/finance/*.md` — new knowledge base documents.
2. Define Pydantic schema(s) in `app/schemas/outputs.py` if Finance needs a
   different output shape (or reuse `CampaignStrategyResponse`-style models
   if applicable).
3. Write `app/domains/finance.py`: a `FINANCE_SYSTEM_PROMPT`, a handful of
   few-shot examples calibrating Finance-specific behaviors, a
   `GuardrailConfig` with Finance-appropriate unsafe/out-of-scope/fact-sensitive
   term lists, and a `DomainConfig(...)` instance, ending with
   `register_domain(FINANCE_DOMAIN)`.
4. Add `from app.domains import finance` to `app/domains/__init__.py`.

**No changes to** `agent.py`, `state.py`, `prompts.py`'s core fragments,
`guardrails.py`'s mechanism functions, `confidence.py`, `retriever.py`,
`vectorstore.py`, or the API layer. The API's `POST /chat` already accepts a
`domain` field and looks the config up by name (`app/api/routes.py`).

---

## 9. Trade-offs

| Decision | What we chose | What we gave up / when to reconsider |
|---|---|---|
| Vector DB | ChromaDB, local persistent client | No concurrent-writer support, no managed HA/SLA, filtering doesn't scale past a small corpus. Move to Pinecone/Weaviate/pgvector at production scale or multi-tenancy. |
| Control flow | Hand-written bounded state machine in `agent.py` | Less flexible than LangGraph for genuinely branching, multi-tool, cyclic behavior. Right call here (one clear decision point + one retry each), would need revisiting if tool selection or multi-domain parallel retrieval were added. |
| Retrieval threshold | Fixed default (0.28), configurable | A static threshold is a blunt instrument — too high loses recall on paraphrased queries, too low lets weak matches through. A production system might use adaptive/reranked thresholds instead. |
| Confidence | Deterministic heuristic, not LLM self-rating | Reproducible and explainable, but still not a statistically calibrated probability — documented in every `ConfidenceReport.basis`. |
| LLM-as-judge (relevance) | Optional, off by default, never blended into pass/fail | Judge-model bias and run-to-run inconsistency mean it should inform, not decide. |
| Two LLM calls per request (planner + generator) | Better separation of concerns, intent-aware retrieval | Costs latency/tokens vs. a single-shot RAG call. Worth it here because it enables targeted retrieval and clean pre-guardrail short-circuiting. |
| Guardrails | Keyword/heuristic, domain-configured | Fast, explainable, zero extra latency — but won't catch obfuscated unsafe requests and can false-positive on benign phrasing. First line of defense, not a complete safety system. |
| Synthetic KB | Fully fictional brand ("Northbrook Outfitters"), clearly labeled in every document header | Realistic enough for meaningful retrieval, but not real-world data — appropriate for a demo, would need a real KB pipeline (ingestion from CMS/wiki/CRM) for production. |
| Mock embedding for tests | Deterministic hash-based bag-of-words, not semantically meaningful | Lets the full pipeline run in CI with zero network calls, but retrieval *quality* observed in mock mode isn't representative — see the note in [Evaluation](#7-evaluation). |

---

## 10. Scaling

Deliberately **not** implemented here (per the assignment's own guidance not
to add production infra just to mention it), but this is how the system
would evolve:

- **Managed vector DB** (Pinecone / Weaviate Cloud / pgvector) once corpus
  size, concurrent writers, or multi-tenant filtering exceed what a local
  Chroma instance handles well.
- **Caching** — cache embeddings for repeated/similar queries; cache planner
  classifications for common request patterns.
- **Async processing** — the two LLM calls per request (planner, generator)
  are currently sequential; retrieval and any independent tool calls could
  run concurrently via `asyncio`.
- **Observability** — the structured JSON logging here (`app/logging_config.py`)
  is a starting point; production would pipe this into a real aggregator
  (Datadog/CloudWatch) with request tracing across the planner → retrieval →
  generation → validation stages, plus dashboards on abstention rate,
  validation-repair rate, and per-category confidence distribution.
- **Model routing** — cheaper/faster models for the planner classification
  step (a small, low-stakes decision) vs. a stronger model for generation.
- **Authentication & rate limiting** — the current API has neither; would
  need API keys/OAuth and per-key rate limits before any real exposure.
- **Evaluation pipelines** — `scripts/run_eval.py` run manually here would
  become a CI-gated check (block deploys on regression), with a larger,
  versioned test-case set and periodic human review of a sample of live
  responses to catch drift the heuristics miss.
- **LangGraph migration** — if tool selection grows beyond one deterministic
  KPI calculator, or if domains need to compose (e.g. a request spanning
  Marketing + Finance), a real graph-based orchestrator would be justified
  over the current hand-written state machine.

---

## Running It

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
```

### Ingest the knowledge base
```bash
python scripts/ingest.py --domain marketing
```

### Run the API
```bash
uvicorn app.main:app --reload
# POST http://localhost:8000/chat  {"query": "...", "domain": "marketing"}
# GET  http://localhost:8000/health
```

### Run the demo UI
```bash
streamlit run streamlit_app.py
```

### Run the evaluation suite
```bash
python scripts/run_eval.py
```

### Run tests (no API key required — uses a mocked LLM boundary and a
deterministic offline embedding)
```bash
USE_MOCK_LLM=true pytest
```
(`tests/conftest.py` sets this automatically for the whole test session.)

---

## Project Structure

```
app/
  agent/         # domain-agnostic orchestrator, state, prompts, guardrails, confidence
  api/           # FastAPI routes (thin -- HTTP concerns only)
  domains/       # DomainConfig instances (marketing.py implemented; add finance.py etc.)
  evaluation/    # evaluation harness
  llm/           # LLM client abstraction (real + mock)
  rag/           # ingestion, embeddings, vector store, retriever
  schemas/       # Pydantic output contracts
  tools/         # kpi_calculator (the one justified tool)
  config.py      # single source of truth for env-driven settings
  logging_config.py
  main.py        # FastAPI app entrypoint
data/marketing/  # synthetic, clearly-labeled knowledge base
evaluation/      # test_cases.json
tests/
  unit/          # ingestion, chunking, retrieval, prompts, schemas, guardrails, confidence, tool
  integration/   # full pipeline with mocked LLM boundary; evaluator harness tests
scripts/
  ingest.py      # CLI: build the vector store from data/<domain>/
  run_eval.py    # CLI: run evaluation/test_cases.json
streamlit_app.py # minimal demo UI
```
