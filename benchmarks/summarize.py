"""Retrieval-directed span summariser for the hierarchical experiment (H2).

See ``hierarchical-test-plan.md`` §6.3. The directed-summary arm represents a
section or document not by its raw text but by an LLM summary written to surface
the content a searcher would query for -- the summary denoises a long, mixed span
and points it at retrieval. This module produces those summaries; the caller
embeds them with the same encoder as every other arm.

Design mirrors ``CachedEncoder``:

* **Cache hard.** Summaries cost LLM tokens and are non-deterministic, so every
  summary is cached to disk keyed on (model, directive, system-prompt, text). A
  prompt edit changes the key and invalidates cleanly. Re-runs are free. Cache
  lives under ``benchmarks/.cache/`` (gitignored).
* **No new dependency.** The ``openai`` SDK is not installed; we call
  ``/v1/chat/completions`` directly over ``httpx`` (already a dependency), with
  the same Bearer auth and error unwrapping as ``OpenAIEmbeddings``.

The directive is itself a knob (plan §6.3): ``retrieval`` (default) asks for the
claims/entities/findings behind the text. Add variants to ``DIRECTIVES`` to test
whether a more pointed directive helps.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from benchmarks.config import CACHE_DIR

logger = logging.getLogger("benchmarks.summarize")

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# directive key -> system prompt. The prompt is part of the cache key, so editing
# one here re-summarises only that directive on the next run.
DIRECTIVES: Dict[str, str] = {
    "retrieval": (
        "You are a search-index summariser. Given a passage, write a concise summary "
        "(2-4 sentences) capturing the specific claims, entities, findings, and facts a "
        "user would search for to find this text. Prefer concrete nouns and specifics over "
        "generalities; omit filler and meta-commentary. Output only the summary."
    ),
}


class CachedSummarizer:
    """Summarise spans with a directive, caching every result to disk.

    Parameters
    ----------
    model
        OpenAI chat model, e.g. ``gpt-5.4-nano``.
    directive
        Key into :data:`DIRECTIVES` selecting the system prompt.
    api_key, timeout, max_retries, max_concurrent
        Standard knobs; ``api_key`` falls back to ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str,
        *,
        directive: str = "retrieval",
        api_key: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
        max_concurrent: int = 8,
    ) -> None:
        if directive not in DIRECTIVES:
            raise ValueError(f"Unknown directive {directive!r}; known: {sorted(DIRECTIVES)}")
        self.model = model
        self.directive = directive
        self.system_prompt = DIRECTIVES[directive]
        # No key required at construction: a fully-cached set of summaries (every
        # span seen on a prior run) needs no live API, which lets the offline
        # analysis harness re-use cached summaries for free. The key is demanded
        # in _run only when there is a genuine cache miss.
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.cache_dir = CACHE_DIR / "hier_summary" / f"{model}__{directive}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_called = 0
        self.n_cached = 0

    def _path(self, text: str) -> Path:
        key = f"{self.model}\x00{self.directive}\x00{self.system_prompt}\x00{text}"
        return self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.txt"

    async def _summarize_one(self, client, sem: asyncio.Semaphore, text: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # Minimal body: a nano/reasoning model may reject temperature or max_tokens,
        # so send only what every chat model accepts.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
        }
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                async with sem:
                    resp = await client.post(_OPENAI_CHAT_URL, headers=headers, json=payload, timeout=self.timeout)
                if not resp.is_success:
                    detail = ""
                    try:
                        body = resp.json()
                        detail = body.get("error", {}).get("message", "") if isinstance(body, dict) else ""
                    except (ValueError, KeyError, TypeError):
                        detail = resp.text[:200]
                    raise RuntimeError(f"OpenAI chat error {resp.status_code}: {detail}")
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    raise RuntimeError("OpenAI chat returned empty content")
                return content.strip()
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                last_exc = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
        raise RuntimeError(f"Summarisation failed after {self.max_retries} attempts: {last_exc}") from last_exc

    async def _run(self, texts: Sequence[str]) -> List[str]:
        import httpx

        results: List[Optional[str]] = [None] * len(texts)
        miss_idx: List[int] = []
        for i, text in enumerate(texts):
            path = self._path(text)
            if path.exists():
                results[i] = path.read_text(encoding="utf-8")
                self.n_cached += 1
            else:
                miss_idx.append(i)

        if miss_idx:
            if not self.api_key:
                raise ValueError(
                    f"OPENAI_API_KEY is required: {len(miss_idx)} summaries are not cached. "
                    "Run with the same params as the cached run, or set the key."
                )
            sem = asyncio.Semaphore(self.max_concurrent)
            async with httpx.AsyncClient() as client:
                summaries = await asyncio.gather(*(self._summarize_one(client, sem, texts[i]) for i in miss_idx))
            for i, summary in zip(miss_idx, summaries, strict=True):
                self._path(texts[i]).write_text(summary, encoding="utf-8")
                results[i] = summary
                self.n_called += 1

        return [r for r in results if r is not None]

    def summarize_all(self, texts: Sequence[str]) -> List[str]:
        """Summarise every text (cached), returning summaries in input order."""
        if not texts:
            return []
        return asyncio.run(self._run(texts))
