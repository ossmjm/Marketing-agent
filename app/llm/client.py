"""
LLM client abstraction.

`LLMClient` is a tiny protocol (`complete_json`) so that:
  - `agent.py` never imports the `openai` package directly,
  - tests can inject a `MockLLMClient` with scripted responses and run
    the entire pipeline (planner -> retrieval -> generation ->
    validation) with zero network calls,
  - swapping providers (Azure OpenAI, Anthropic, etc.) means writing
    one new class here, not touching agent logic.

We use the Chat Completions API with `response_format={"type":
"json_object"}` to reduce (not eliminate) malformed-JSON risk; the
agent's validation layer is the actual safety net, not this flag.
"""

from __future__ import annotations

import json
from typing import Protocol

from app.config import settings


class LLMError(Exception):
    """Raised on provider failures (timeout, auth, rate limit, etc.)."""


class LLMClient(Protocol):
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict: ...


class OpenAIChatClient:
    def __init__(self, model: str | None = None):
        from openai import APIError, APITimeoutError, AuthenticationError, OpenAI

        self._OpenAI = OpenAI
        self._errors = (APIError, APITimeoutError, AuthenticationError)
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,  # None -> OpenAI SDK default (api.openai.com)
            timeout=settings.llm_timeout_seconds,
        )
        self._model = model or settings.llm_model

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        kwargs: dict = dict(
            model=self._model,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if settings.llm_use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except self._errors as exc:
            raise LLMError(f"LLM provider call failed: {exc}") from exc

        content = resp.choices[0].message.content
        content = _strip_markdown_fences(content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned non-JSON content: {exc}") from exc


def _strip_markdown_fences(content: str) -> str:
    """
    Some non-OpenAI models (routed via OpenRouter etc.) wrap JSON in
    ```json ... ``` fences even when told not to, especially when
    `response_format=json_object` isn't strictly enforced by the
    provider/model. Strip fences defensively before parsing; this is a
    tolerance shim, not a replacement for the prompt-level instruction
    or the Pydantic validation layer that catches anything worse.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


class MockLLMClient:
    """
    Deterministic, scriptable client for tests and offline demos.

    Usage: register a callable per "call kind" (planner/generation/repair)
    inferred from a marker in the prompt, or just push a queue of
    canned responses consumed in order. We use the queue approach here
    because it's simplest to reason about in test code.
    """

    def __init__(self, responses: list[dict] | None = None):
        self._queue: list[dict] = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    def push(self, response: dict) -> None:
        self._queue.append(response)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if not self._queue:
            raise LLMError("MockLLMClient: no more scripted responses available")
        return self._queue.pop(0)


def get_llm_client() -> LLMClient:
    if settings.use_mock_llm:
        return MockLLMClient()
    return OpenAIChatClient()
