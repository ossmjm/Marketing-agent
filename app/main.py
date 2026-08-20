"""
FastAPI application entrypoint.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI

import app.domains  # noqa: F401  -- side effect: registers all domains
from app.api.routes import router
from app.logging_config import configure_logging

configure_logging()

app = FastAPI(
    title="Agentic Marketing Campaign Strategist",
    description=(
        "A domain-agnostic, retrieval-grounded agent core with a Marketing "
        "domain configuration. See /docs for the API contract."
    ),
    version="1.0.0",
)

app.include_router(router)
