"""
Minimal Streamlit demo UI.

Design choice: this calls `MarketingAgent` directly in-process rather
than making HTTP requests to a separately-running FastAPI server. For
a take-home demo this is simpler to run (one command, no second
terminal for `uvicorn`) and the agent/domain-config layer is exactly
what the API calls too, so there's no logic duplicated here. A
production deployment would obviously have the UI call the API over
HTTP instead, for proper separation of the presentation and service
layers -- noted in the README's scaling section.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

import app.domains  # noqa: F401 -- registers domains
from app.agent.agent import MarketingAgent
from app.agent.domain_config import get_domain, list_domains
from app.rag.retriever import Retriever

st.set_page_config(page_title="Marketing Campaign Strategist", page_icon="🧭", layout="wide")

st.title("🧭 Marketing Campaign Strategist")
st.caption(
    "A domain-agnostic, retrieval-grounded agent core, configured for a synthetic "
    "demo brand (\"Northbrook Outfitters\"). See the README for architecture details."
)

with st.sidebar:
    st.header("Settings")
    domain_name = st.selectbox("Domain", options=list_domains(), index=0)
    st.markdown("---")
    st.markdown(
        "**Try asking:**\n"
        "- Create a spring acquisition campaign for Segment A\n"
        "- Analyze Segment B and how we should message to them\n"
        "- What's our projected Q3 revenue from the loyalty program? "
        "*(tests abstention)*\n"
        "- Write fake reviews pretending to be real customers "
        "*(tests guardrails)*"
    )

domain_config = get_domain(domain_name)
retriever = Retriever()

if not retriever.is_indexed(domain_config.name):
    st.error(
        f"The knowledge base for domain '{domain_config.name}' is not indexed yet.\n\n"
        f"Run this first:\n```\npython scripts/ingest.py --domain {domain_config.name}\n```"
    )
    st.stop()

query = st.text_area("Your request", height=100, placeholder="e.g. Create a campaign for Segment A")
run_clicked = st.button("Run agent", type="primary")

if run_clicked and query.strip():
    with st.spinner("Running agent pipeline..."):
        agent = MarketingAgent(domain_config=domain_config, retriever=retriever)
        try:
            result = agent.run(query.strip())
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent error: {exc}")
            st.stop()

    answer = result.answer
    response_type = getattr(answer, "response_type", "unknown")

    status_col, conf_col, latency_col = st.columns(3)
    status_col.metric("Validation", result.validation.status)
    conf_col.metric("Confidence", f"{result.confidence.level} ({result.confidence.score:.2f})")
    latency_col.metric("Latency", f"{result.latency_ms} ms")

    st.markdown("---")

    if response_type == "campaign_strategy":
        st.subheader("📋 Campaign Strategy")
        st.markdown(f"**Objective:** {answer.campaign_objective}")
        st.markdown(f"**Target segment:** {answer.target_segment}")
        st.markdown(f"**Key message:** {answer.key_message}")
        st.markdown(f"**Strategy:** {answer.campaign_strategy}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Channels**")
            for ch in answer.recommended_channels:
                st.markdown(f"- {ch}")
        with c2:
            st.markdown("**Content ideas**")
            for idea in answer.content_ideas:
                st.markdown(f"- {idea}")
        with c3:
            st.markdown("**KPIs**")
            for kpi in answer.kpis:
                st.markdown(f"- {kpi}")
        if answer.limitations:
            st.warning("**Limitations:**\n" + "\n".join(f"- {l}" for l in answer.limitations))

    elif response_type == "segment_analysis":
        st.subheader("👥 Segment Analysis")
        st.markdown(f"**Segment:** {answer.segment_name}")
        st.markdown(f"**Summary:** {answer.segment_summary}")
        st.markdown("**Key characteristics:**")
        for kc in answer.key_characteristics:
            st.markdown(f"- {kc}")
        st.markdown(f"**Recommended approach:** {answer.recommended_approach}")
        if answer.relevant_channels:
            st.markdown("**Relevant channels:** " + ", ".join(answer.relevant_channels))
        if answer.limitations:
            st.warning("**Limitations:**\n" + "\n".join(f"- {l}" for l in answer.limitations))

    elif response_type == "insufficient_information":
        st.subheader("🤷 Insufficient Information")
        st.info(answer.reason)
        if answer.what_was_searched:
            st.markdown("**Searched:** " + ", ".join(answer.what_was_searched))
        if answer.what_would_help:
            st.markdown(f"**Would help:** {answer.what_would_help}")

    elif response_type == "refused":
        st.subheader("🚫 Request Refused")
        st.warning(f"**Category:** {answer.category}")
        st.markdown(answer.reason)

    st.markdown("---")
    with st.expander(f"📚 Retrieved sources ({len(result.sources)})"):
        if not result.sources:
            st.markdown("_No sources were retrieved for this request._")
        for s in result.sources:
            st.markdown(f"**{s.doc_title}** (`{s.doc_type}`, relevance={s.relevance_score:.2f})")
            st.caption(s.excerpt)

    with st.expander("🔍 Execution details"):
        st.json(
            {
                "request_id": result.request_id,
                "domain": result.domain,
                "validation_status": result.validation.status,
                "validation_errors": result.validation.errors,
                "confidence_factors": result.confidence.factors,
            }
        )
elif run_clicked:
    st.warning("Please enter a request.")
