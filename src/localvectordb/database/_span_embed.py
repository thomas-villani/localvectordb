"""Raw-span level-vector embedding (window-mean-pooled).

Represents a section by the embedding of its **own text** rather than the
centroid (mean) of its chunk vectors.

WHEN THIS WINS, AND WHEN IT LOSES
---------------------------------
This module used to cite findings F2/F3 -- "raw-span beats the centroid on real
long documents" -- as blanket justification. **That is only true for short
sections.** Later work established that a raw-span vector and a centroid are one
operator at two granularities, and that the coarser one degrades as the span
grows: measured on long sections (median >8k tokens), raw-span lost **0.25-0.36
nDCG** to the centroid on every encoder tested, while on short sections (~190
tokens) it still won. The span length, not the representation, is the axis.

Callers should therefore not read ``section_vector_strategy="rawspan"`` as the
generally-better arm; it is the better arm for sections that fit comfortably
inside the encoder's context.

A span longer than that context is embedded in **windows and mean-pooled** --
never truncated -- so a long section is represented in full. That promise was
not kept until 2026-08-04: the window was sized from the provider's reported
context, and local providers reported none, so a fixed 24,000-char window was
handed to models with far smaller contexts and silently truncated by them (7.3%
of a 24.5k-char section survived on ``all-MiniLM-L6-v2``). Windows are now sized
from the provider's real context; see ``EmbeddingProvider.max_input_tokens``.

``_WINDOW_CHARS_PER_TOKEN`` (3.0) is deliberately below ``_CHARS_PER_TOKEN``
(3.5). The harness needed token-exact windows to locate the pooling threshold
precisely, but ``src/`` needs only "never truncate", and erring small is both
safe and -- since finer granularity is what helps on long spans -- the harmless
direction to err in.

Vectors are returned **raw** (not unit-normalised); the caller's
``_add_vectors_to_section_index`` unit-normalises at the write.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import numpy as np

# One embeddings request is bounded by a token budget and an input count; a
# single input over the model's context window is truncated (silently, by most
# providers) or a hard error (hosted). We window over-long spans instead, so the
# window must be sized to the *encoder's* context, not a fixed guess: a fixed
# 24k-char (~8k-token) window overflows a 2k-context model ~3.4x, and each window
# is then truncated -- defeating the whole point of windowing. The window is
# derived from the provider's context (``num_ctx`` / ``max_input_tokens``) at a
# conservative ~3 chars/token so a window reliably fits, falling back to the 8k
# assumption only when the context is unknown.
_DEFAULT_WINDOW_CHARS = 24_000
_WINDOW_CHARS_PER_TOKEN = 3.0
_MAX_TOKENS_PER_REQUEST = 200_000
_MAX_INPUTS_PER_REQUEST = 2000
_CHARS_PER_TOKEN = 3.5


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _provider_context_tokens(provider: object) -> Optional[int]:
    """The encoder's usable context window in tokens, if the provider exposes one.

    Prefers an explicit ``num_ctx`` (e.g. an Ollama provider told to load a larger
    window) over the generic ``max_input_tokens`` per-text cap. Returns None when
    neither is a positive int, so the caller falls back to the fixed default.
    """
    for attr in ("num_ctx", "max_input_tokens"):
        value = getattr(provider, attr, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _window_chars_for(provider: object) -> int:
    """Char budget per embedding window, sized to the provider's context window."""
    tokens = _provider_context_tokens(provider)
    if tokens is None:
        return _DEFAULT_WINDOW_CHARS
    return max(1, int(tokens * _WINDOW_CHARS_PER_TOKEN))


def _windows(text: str, window_chars: int) -> List[str]:
    """Split ``text`` into <= window-sized pieces, breaking on whitespace when possible."""
    if len(text) <= window_chars:
        return [text]
    out: List[str] = []
    i = 0
    while i < len(text):
        end = min(i + window_chars, len(text))
        if end < len(text):
            cut = text.rfind(" ", i + window_chars // 2, end)
            if cut > i:
                end = cut
        out.append(text[i:end])
        i = end
    return out


def _batch_by_budget(texts: Sequence[str]) -> Iterator[List[str]]:
    """Split ``texts`` into request-sized batches under the token and input caps.

    Order is preserved and every text appears exactly once, so concatenating the
    per-batch embeddings reproduces a single-call result.
    """
    batch: List[str] = []
    tokens = 0
    for text in texts:
        cost = _estimate_tokens(text)
        if batch and (len(batch) >= _MAX_INPUTS_PER_REQUEST or tokens + cost > _MAX_TOKENS_PER_REQUEST):
            yield batch
            batch, tokens = [], 0
        batch.append(text)
        tokens += cost
    if batch:
        yield batch


def _embed_raw(provider: object, texts: List[str], dim: int) -> np.ndarray:
    """Embed each text (<= one window) via the provider, batched; return raw ``(n, dim)``."""
    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    parts = [np.asarray(provider.embed_sync(batch), dtype=np.float32) for batch in _batch_by_budget(texts)]  # type: ignore[attr-defined]
    return np.vstack(parts).astype(np.float32)


def embed_spans_pooled(provider: object, texts: Sequence[str], dim: int) -> np.ndarray:
    """Embed each span's text into an ``(n, dim)`` raw (un-normalised) array.

    A span over the encoder window is embedded in windows and mean-pooled, so a
    long section/document is represented in full rather than truncated. An empty
    span becomes a zero row (an empty-section-equivalent, left untouched by the
    downstream unit-normalisation).
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, dim), dtype=np.float32)

    window_chars = _window_chars_for(provider)
    windows_per_text: List[Optional[List[str]]] = []
    flat: List[str] = []
    for t in texts:
        if not t:
            windows_per_text.append(None)  # zero-row marker; no windows to embed
            continue
        ws = _windows(t, window_chars)
        windows_per_text.append(ws)
        flat.extend(ws)

    flat_raw = _embed_raw(provider, flat, dim)
    width = flat_raw.shape[1] if flat_raw.shape[0] else dim
    out = np.zeros((n, width), dtype=np.float32)
    cursor = 0
    for i, win_list in enumerate(windows_per_text):
        if win_list is None:
            continue
        m = len(win_list)
        out[i] = flat_raw[cursor : cursor + m].mean(axis=0)
        cursor += m
    return out
