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


class ClaimExtractionError(Exception):
    """Raised when claim extraction could not be completed.

    This is deliberately distinct from "the text contained no factual claims"
    (a *successful* extraction that returns an empty list). A provider outage or
    an unparseable response must never be read as "nothing to verify" -- see
    :class:`~localvectordb.validation.checker.FactChecker`, which maps this to a
    could-not-verify result (``overall_score=0.0``) rather than a perfect score.
    """


async def extract_claims(llm: LLMProvider, text: str) -> list[dict]:
    """Extract factual claims from *text* using the LLM.

    Returns a list of ``{"claim": str, "sentence": str}`` dicts. An **empty
    list** means extraction succeeded and the text held no verifiable claims.

    Raises
    ------
    ClaimExtractionError
        If the provider call fails or its response cannot be parsed as a JSON
        array of claims. Callers must treat this as "verification could not be
        performed", never as "no claims found".
    """
    try:
        response = await llm.complete(system=CLAIM_EXTRACTION_SYSTEM, user=text)
    except Exception as e:
        raise ClaimExtractionError(f"LLM claim-extraction call failed: {e}") from e

    try:
        claims = extract_json(response)
    except Exception as e:
        raise ClaimExtractionError(f"Could not parse claim-extraction response as JSON: {e}") from e

    if not isinstance(claims, list):
        raise ClaimExtractionError(f"Claim-extraction response was not a JSON array (got {type(claims).__name__})")
    return [c for c in claims if isinstance(c, dict) and "claim" in c and "sentence" in c]
