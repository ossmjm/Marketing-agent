"""
Observability: structured logging.

Design intent
-------------
We log *execution metadata* (request id, domain, retrieved doc ids,
retrieval count, validation result, latency, errors) because that is
what lets an engineer debug or audit the system in production.

We deliberately never log:
  - API keys / secrets
  - Full prompts sent to the LLM (may contain PII from user queries)
  - The model's private reasoning / chain-of-thought

Each log line is a single JSON object so it can be piped into any log
aggregator (CloudWatch, Datadog, ELK) without a custom parser.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from app.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Structured extras attached via logger.info(msg, extra={...})
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "levelname", "levelno", "pathname", "filename",
                       "module", "exc_info", "exc_text", "stack_info", "lineno",
                       "funcName", "created", "msecs", "relativeCreated", "thread",
                       "threadName", "processName", "process", "name"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    # Avoid duplicate handlers on reload
    root.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
