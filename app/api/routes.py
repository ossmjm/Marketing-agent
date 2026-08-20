"""
API layer.

Kept intentionally thin: request validation in, `MarketingAgent.run()`
call, response out. All the interesting behavior lives in app/agent
and app/rag -- the API layer's job is HTTP concerns (status codes,
error envelopes) and nothing else.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.agent import MarketingAgent
from app.agent.domain_config import get_domain, list_domains
from app.llm.client import LLMError
from app.logging_config import get_logger
from app.rag.retriever import Retriever
from app.schemas.outputs import ChatResponse

router = APIRouter()
logger = get_logger("api")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    domain: str = Field(default="marketing")


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "domains_available": list_domains()}


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    domain_config = get_domain(request.domain)
    if domain_config is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{request.domain}'. Available: {list_domains()}",
        )

    retriever = Retriever()
    if not retriever.is_indexed(domain_config.name):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Knowledge base for domain '{domain_config.name}' is not indexed yet. "
                f"Run `python scripts/ingest.py --domain {domain_config.name}` first."
            ),
        )

    request_id = str(uuid.uuid4())
    agent = MarketingAgent(domain_config=domain_config, retriever=retriever)

    try:
        return agent.run(request.query, request_id=request_id)
    except LLMError as exc:
        logger.error("chat_llm_error", extra={"request_id": request_id, "error": str(exc)})
        raise HTTPException(status_code=502, detail="Upstream LLM provider error.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("chat_unhandled_error", extra={"request_id": request_id, "error": str(exc)})
        raise HTTPException(status_code=500, detail="Internal agent error.") from exc
