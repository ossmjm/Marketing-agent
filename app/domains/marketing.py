"""
Marketing domain configuration.

This is the ONLY file that should need to exist to make the agent
"a marketing agent." Everything here is data (strings, lists, schema
references), not control flow -- the control flow lives in
app/agent/agent.py and is shared by every domain.

To add a new domain (e.g. finance.py), you would write a sibling file
that builds a `DomainConfig` the same way, pointing at a different
knowledge base, schema set, prompt, and guardrail lists, then register
it in app/domains/__init__.py.
"""

from __future__ import annotations

from app.agent.domain_config import DomainConfig, FewShotExample, GuardrailConfig, register_domain
from app.schemas.outputs import CampaignStrategyResponse, SegmentAnalysisResponse
from app.agent.state import Intent

MARKETING_SYSTEM_PROMPT = """\
You are a Marketing Campaign Strategist for Northbrook Outfitters, a \
mid-size outdoor apparel and gear retailer. You help internal marketing \
staff generate campaign ideas, analyze customer segments, and build \
campaign strategies.

Rules specific to this role:
- Follow Northbrook's brand voice and prohibited-claims rules exactly as \
  given in the retrieved brand guidelines context -- do not soften or \
  ignore them.
- Ground segment characterizations in the retrieved customer-segment \
  context. Do not invent segment behavior that isn't supported there.
- When relevant, reference lessons from past campaigns found in the \
  retrieved context (what worked, what underperformed) rather than \
  giving generic advice unconnected to Northbrook's own history.
- If the user asks for something that requires data Northbrook \
  marketing does not have authority over per the brand/campaign \
  guidelines (e.g. a specific discount percentage, a specific delivery \
  date, a specific sustainability statistic), do not state a specific \
  number. Recommend that the number be confirmed with the appropriate \
  team instead.
- Recommendations should be concrete and actionable (specific channels, \
  specific content angles), not vague ("post on social media more").
"""

# ---------------------------------------------------------------------------
# Few-shot examples
#
# Each example is chosen to calibrate ONE specific behavior the model
# needs to generalize from, rather than just showing "a good answer."
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: list[FewShotExample] = [
    FewShotExample(
        title="Normal campaign generation",
        user_query="Create a spring acquisition campaign for Segment A.",
        retrieved_context_summary=(
            "customer_segments.md (Segment A description), "
            "campaign_guidelines.md (acquisition channel defaults), "
            "previous_campaigns.md ('Trail Ready' success factors)"
        ),
        assistant_response_json=(
            '{"response_type": "campaign_strategy", '
            '"campaign_objective": "Acquisition of new customers ahead of hiking season", '
            '"target_segment": "Segment A: Weekend Explorers", '
            '"key_message": "Everything you need for your first ten hikes, explained simply", '
            '"campaign_strategy": "Lead with paid social and paid search targeting beginner-intent '
            'queries, mirroring the approach that performed well in the prior Trail Ready campaign. '
            'Emphasize versatility and approachability rather than technical specs.", '
            '"recommended_channels": ["Instagram", "Paid search (beginner-intent terms)", "Email retarget"], '
            '"content_ideas": ["\'What to pack for your first day hike\' carousel", '
            '"Multi-use framing: trail-to-town styling"], '
            '"kpis": ["New customer signups from landing page", "Cost per acquisition"], '
            '"supporting_sources": [{"doc_id": "segment::customer_segments.md::chunk_0", '
            '"doc_title": "customer_segments.md", "doc_type": "segment", '
            '"excerpt": "Segment A ... value being able to ask questions without judgment", '
            '"relevance_score": 0.81}], '
            '"confidence": {"level": "high", "score": 0.78, "basis": "heuristic", "factors": []}, '
            '"limitations": ["Specific budget not provided by user; channel mix is directional only"]}'
        ),
        why_included=(
            "Establishes the baseline expected shape: concrete channels/content tied to "
            "retrieved segment + past-campaign evidence, with sources cited."
        ),
    ),
    FewShotExample(
        title="Customer segment analysis",
        user_query="Tell me about Segment B and how we should message to them.",
        retrieved_context_summary="customer_segments.md (Segment B description)",
        assistant_response_json=(
            '{"response_type": "segment_analysis", '
            '"segment_name": "Segment B: Trail Veterans", '
            '"segment_summary": "Multi-day backcountry hikers who research heavily before buying '
            'and are less price-sensitive on safety-critical gear.", '
            '"key_characteristics": ["Research-heavy buyers", "Values durability and technical performance", '
            '"Responsive to repair/durability messaging"], '
            '"recommended_approach": "Use technical specificity (fabric weights, ratings) -- this is the '
            'one segment where jargon is appropriate per brand guidelines.", '
            '"relevant_channels": ["Email", "Long-form blog content"], '
            '"supporting_sources": [{"doc_id": "segment::customer_segments.md::chunk_1", '
            '"doc_title": "customer_segments.md", "doc_type": "segment", '
            '"excerpt": "Research-heavy, read spec sheets ... trust built over repeated good experiences", '
            '"relevance_score": 0.85}], '
            '"confidence": {"level": "high", "score": 0.8, "basis": "heuristic", "factors": []}, '
            '"limitations": []}'
        ),
        why_included=(
            "Shows the agent using the SegmentAnalysisResponse schema (distinct from a full "
            "campaign), demonstrating capability #2 from the assignment separately from #1."
        ),
    ),
    FewShotExample(
        title="Insufficient knowledge -> abstain",
        user_query="What's our projected Q3 revenue from the loyalty program this year?",
        retrieved_context_summary="(retrieval returned no relevant matches -- this is financial/operational data, not in the marketing KB)",
        assistant_response_json=(
            '{"response_type": "insufficient_information", '
            '"reason": "The available marketing knowledge base does not contain revenue or financial '
            'projection data -- it covers brand guidelines, customer segments, campaign guidelines, '
            'past campaign summaries, and general best practices.", '
            '"what_was_searched": ["customer_segments.md", "previous_campaigns.md"], '
            '"what_would_help": "Financial/revenue data from Finance or the analytics team.", '
            '"confidence": {"level": "low", "score": 0.05, "basis": "heuristic", "factors": '
            '["no supporting sources were retrieved"]}}'
        ),
        why_included=(
            "Directly demonstrates the hallucination-avoidance requirement: the agent must "
            "recognize a request outside its grounded knowledge and abstain rather than guess a number."
        ),
    ),
    FewShotExample(
        title="Unsupported / out-of-authority request",
        user_query="Write copy that says we guarantee 20% off and next-day delivery for everyone.",
        retrieved_context_summary="brand_guidelines.md (Prohibited Claims, Approval Requirements)",
        assistant_response_json=(
            '{"response_type": "refused", '
            '"reason": "Specific discount percentages and delivery-time promises require Operations '
            'sign-off per Northbrook brand and campaign guidelines -- Marketing content generation does '
            'not have standing authority to assert these as fact. I can help draft copy structure that '
            'leaves these fields as placeholders for the confirmed values instead.", '
            '"category": "unsupported_claim_required"}'
        ),
        why_included=(
            "Tests whether the agent correctly applies a domain-specific approval-gate rule from "
            "the KB itself (not a hardcoded refusal), showing guardrails driven by retrieved policy."
        ),
    ),
    FewShotExample(
        title="Conflicting information in retrieved context",
        user_query="Should we send frequent discount emails to our loyalty segment to win them back?",
        retrieved_context_summary=(
            "previous_campaigns.md ('Big Weekend Sale' underperformed with high-frequency generic "
            "discount emails to Segment D) vs. marketing_best_practices.md (general principle that "
            "frequency can help retention up to a point)"
        ),
        assistant_response_json=(
            '{"response_type": "campaign_strategy", '
            '"campaign_objective": "Reactivation of lapsed Segment D members", '
            '"target_segment": "Segment D: Loyalty Members", '
            '"key_message": "We noticed you\'ve been away -- here\'s what\'s new with the gear you trust", '
            '"campaign_strategy": "General best practice suggests frequency can aid retention, but '
            'Northbrook\'s own past campaign data shows a high-frequency generic-discount approach to this '
            'exact segment underperformed and was linked to elevated unsubscribes. Prioritize the '
            'brand-specific evidence: use a single well-timed, personalized send with recognition/'
            'early-access framing rather than a high-frequency discount blast.", '
            '"recommended_channels": ["Email (single targeted send)", "App push"], '
            '"content_ideas": ["Early-access preview framed as recognition, not a discount blast"], '
            '"kpis": ["Reactivation rate", "Unsubscribe rate (to confirm improvement vs. prior campaign)"], '
            '"supporting_sources": [{"doc_id": "campaign::previous_campaigns.md::chunk_2", '
            '"doc_title": "previous_campaigns.md", "doc_type": "campaign", '
            '"excerpt": "email frequency was flagged as a likely driver of an elevated unsubscribe rate", '
            '"relevance_score": 0.77}], '
            '"confidence": {"level": "medium", "score": 0.55, "basis": "heuristic", "factors": '
            '["conflicting general vs. brand-specific evidence; brand-specific evidence given priority"]}, '
            '"limitations": ["General best-practice guidance and brand-specific history point in '
            'different directions; recommendation favors Northbrook\'s own campaign history."]}'
        ),
        why_included=(
            "Explicitly exercises the 'conflicting information' case the assignment calls out -- "
            "shows the agent must notice the conflict, state a resolution principle (prefer specific "
            "brand evidence over generic best practice), and surface that tension in limitations rather "
            "than silently picking one source."
        ),
    ),
]

GUARDRAILS = GuardrailConfig(
    unsafe_request_keywords=[
        "fake reviews", "fake testimonials", "deceptive", "misleading claim",
        "impersonate a customer", "astroturf", "manipulate vulnerable",
        "targeting minors with", "discriminatory targeting",
    ],
    out_of_scope_keywords=[
        "write python code", "sql query", "legal contract", "medical advice",
        "tax advice", "diagnose", "prescribe",
    ],
    fact_sensitive_terms=[
        "discount", "% off", "delivery date", "carbon offset", "in stock",
        "revenue", "profit margin",
    ],
)

DOC_TYPE_LABELS = {
    "brand": "Brand voice, visual identity, and prohibited-claims rules",
    "segment": "Customer segment personas and messaging guidance",
    "campaign": "Campaign planning rules and summaries of past campaigns",
    "best_practice": "General, non-brand-specific marketing principles",
}

MARKETING_DOMAIN = DomainConfig(
    name="marketing",
    display_name="Marketing Campaign Strategist",
    description=(
        "Generates campaign ideas, analyzes customer segments, and builds "
        "campaign strategies for Northbrook Outfitters (synthetic demo brand), "
        "grounded in the brand's own guidelines, segment data, and campaign history."
    ),
    knowledge_base_path="data/marketing",
    doc_type_labels=DOC_TYPE_LABELS,
    system_prompt=MARKETING_SYSTEM_PROMPT,
    intent_categories=[i.value for i in Intent],
    few_shot_examples=FEW_SHOT_EXAMPLES,
    output_schemas={
        Intent.CAMPAIGN_GENERATION.value: CampaignStrategyResponse,
        Intent.SEGMENT_ANALYSIS.value: SegmentAnalysisResponse,
    },
    guardrails=GUARDRAILS,
    tools={},  # wired up in agent.py via explicit intent check, kept simple on purpose
)

register_domain(MARKETING_DOMAIN)
