"""Claim extraction from LLM-generated text."""

from __future__ import annotations

from .llm import LLMProvider, extract_json

CLAIM_EXTRACTION_SYSTEM = """\
Extract all factual claims from the following text.

Return a JSON array of objects, each with:
- "claim": A specific, verifiable factual claim
- "sentence": The exact sentence from the text that contains this claim

Only extract verifiable factual claims. Skip opinions, questions, hedged \
statements, and subjective assessments.

Return valid JSON only, no other text."""


async def extract_claims(llm: LLMProvider, text: str) -> list[dict]:
    """Extract factual claims from *text* using the LLM.

    Returns a list of ``{"claim": str, "sentence": str}`` dicts.
    Returns an empty list on failure.
    """
    try:
        response = await llm.complete(system=CLAIM_EXTRACTION_SYSTEM, user=text)
        claims = extract_json(response)
        if not isinstance(claims, list):
            return []
        return [
            c
            for c in claims
            if isinstance(c, dict) and "claim" in c and "sentence" in c
        ]
    except Exception:
        return []
