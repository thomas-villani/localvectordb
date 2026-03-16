"""Inline citation annotation for fact-checked text."""

from __future__ import annotations

import difflib
from typing import Optional

from .result import ClaimResult


def _find_best_match(text: str, sentence: str, threshold: float = 0.6) -> Optional[tuple[int, int]]:
    """Locate *sentence* in *text* via exact then fuzzy matching.

    Returns ``(start, end)`` character offsets or ``None``.
    """
    # Exact match
    idx = text.find(sentence)
    if idx != -1:
        return idx, idx + len(sentence)

    # Fuzzy sliding-window match
    sent_len = len(sentence)
    best_ratio = 0.0
    best_span: Optional[tuple[int, int]] = None

    lo = max(1, int(sent_len * 0.7))
    hi = int(sent_len * 1.3) + 1
    for window_size in range(lo, hi):
        for start in range(0, len(text) - window_size + 1):
            candidate = text[start : start + window_size]
            ratio = difflib.SequenceMatcher(None, sentence, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, start + window_size)

    if best_ratio >= threshold and best_span is not None:
        return best_span
    return None


def annotate_response(
    response_text: str,
    claim_results: list[ClaimResult],
    similarity_threshold: float = 0.6,
) -> Optional[str]:
    """Insert ``[N]`` citation markers into *response_text* and append footnotes.

    Only annotates grounded claims that have both an ``original_sentence`` and a
    ``source_id``.  Returns ``None`` if no annotations are possible.
    """
    citations: list[tuple[int, int, str, str]] = []  # start, end, source_id, excerpt
    footnote_num = 0

    for cr in claim_results:
        if not (cr.grounded and cr.original_sentence and cr.source_id):
            continue
        span = _find_best_match(response_text, cr.original_sentence, similarity_threshold)
        if span is None:
            continue
        footnote_num += 1
        citations.append((span[0], span[1], cr.source_id, cr.source_excerpt or cr.claim))

    if not citations:
        return None

    # Sort by start position descending so insertions don't shift earlier offsets
    citations.sort(key=lambda c: c[0], reverse=True)

    annotated = response_text
    footnotes: list[str] = []

    for i, (_start, end, source_id, excerpt) in enumerate(reversed(citations), 1):
        marker = f" [{i}]"
        annotated = annotated[:end] + marker + annotated[end:]
        footnotes.append(f'[{i}] {source_id} -- "{excerpt}"')

    annotated += "\n\n---\n" + "\n".join(footnotes)
    return annotated
