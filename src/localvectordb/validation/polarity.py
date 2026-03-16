"""Polarity classification between claims and source chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .llm import LLMProvider, extract_json
from .result import Polarity

POLARITY_SYSTEM = """\
Classify the relationship between a claim and each source text chunk.

Return a JSON array with one entry per chunk:
- "index": The chunk index (0-based)
- "polarity": "supports", "contradicts", or "unrelated"
- "confidence": 0.0 to 1.0
- "excerpt": A short relevant quote from the source (null if unrelated)

IMPORTANT: Absence of information is NOT contradiction. Only classify as \
"contradicts" if the source makes a statement that directly conflicts with \
the claim.

Return valid JSON only, no other text."""


@dataclass
class PolarityResult:
    """Classification of a single claim-chunk pair."""

    polarity: Polarity
    confidence: float
    excerpt: Optional[str] = None


def _fallback(n: int) -> list[PolarityResult]:
    return [PolarityResult(polarity=Polarity.UNRELATED, confidence=0.0) for _ in range(n)]


async def classify_polarity(
    llm: LLMProvider,
    claim: str,
    chunks: list[dict],
) -> list[PolarityResult]:
    """Classify polarity between *claim* and each chunk.

    Each entry in *chunks* must have a ``"content"`` key.
    Returns one :class:`PolarityResult` per chunk, in the same order.
    """
    if not chunks:
        return []

    chunks_text = "\n\n".join(f"[Chunk {i}]:\n{c['content'][:3000]}" for i, c in enumerate(chunks))

    try:
        response = await llm.complete(
            system=POLARITY_SYSTEM,
            user=f"Claim: {claim}\n\nSource chunks:\n{chunks_text}",
        )
        results_data = extract_json(response)
        if not isinstance(results_data, list):
            return _fallback(len(chunks))

        result_map: dict[int, PolarityResult] = {}
        for item in results_data:
            idx = item.get("index", -1)
            polarity_str = str(item.get("polarity", "unrelated")).lower()
            try:
                polarity = Polarity(polarity_str)
            except ValueError:
                polarity = Polarity.UNRELATED
            result_map[idx] = PolarityResult(
                polarity=polarity,
                confidence=float(item.get("confidence", 0.0)),
                excerpt=item.get("excerpt"),
            )

        default = PolarityResult(polarity=Polarity.UNRELATED, confidence=0.0)
        return [result_map.get(i, default) for i in range(len(chunks))]
    except Exception:
        return _fallback(len(chunks))
