"""LLM provider interface and implementations for claim extraction and polarity classification."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by the fact-checker.

    Any object with a matching ``complete`` signature works via structural subtyping.
    """

    async def complete(self, system: str, user: str) -> str: ...


class AnthropicProvider:
    """Wraps an ``anthropic.Anthropic`` or ``anthropic.AsyncAnthropic`` client."""

    def __init__(self, client: Any, model: str = "claude-haiku-4-5-20251001") -> None:
        self._client = client
        self._model = model
        create_fn = getattr(getattr(client, "messages", None), "create", None)
        self._is_async = create_fn is not None and asyncio.iscoroutinefunction(create_fn)

    async def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._is_async:
            response = await self._client.messages.create(**kwargs)
        else:
            response = await asyncio.to_thread(self._client.messages.create, **kwargs)
        return response.content[0].text


class OpenAIProvider:
    """Wraps an ``openai.OpenAI`` or ``openai.AsyncOpenAI`` client."""

    def __init__(self, client: Any, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model
        completions = getattr(getattr(client, "chat", None), "completions", None)
        create_fn = getattr(completions, "create", None)
        self._is_async = create_fn is not None and asyncio.iscoroutinefunction(create_fn)

    async def complete(self, system: str, user: str) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self._is_async:
            response = await self._client.chat.completions.create(**kwargs)
        else:
            response = await asyncio.to_thread(self._client.chat.completions.create, **kwargs)
        return response.choices[0].message.content


class GeminiProvider:
    """Wraps a ``google.genai.Client`` instance."""

    def __init__(self, client: Any, model: str = "gemini-2.0-flash") -> None:
        self._client = client
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        prompt = f"{system}\n\n{user}"
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
        )
        return response.text


def detect_provider(client: Any, model: str | None = None) -> LLMProvider:
    """Auto-detect the LLM provider from a client object.

    Supports Anthropic, OpenAI, and Google GenAI clients.  Objects that already
    implement :class:`LLMProvider` are returned as-is.
    """
    if isinstance(client, LLMProvider):
        return client

    module = getattr(type(client), "__module__", "") or ""

    if "anthropic" in module:
        return AnthropicProvider(client, model or "claude-haiku-4-5-20251001")
    if "openai" in module:
        return OpenAIProvider(client, model or "gpt-4o-mini")
    if "google" in module:
        return GeminiProvider(client, model or "gemini-2.0-flash")

    raise ValueError(
        f"Cannot auto-detect LLM provider from {type(client).__name__}. "
        "Pass an Anthropic, OpenAI, or Google GenAI client, "
        "or an object implementing the LLMProvider protocol."
    )


def extract_json(text: str) -> Any:
    """Extract JSON from an LLM response, handling markdown code blocks."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    return json.loads(text)
