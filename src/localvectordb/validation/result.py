"""Data models for fact-checking results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Polarity(str, Enum):
    """Relationship between a claim and a source chunk."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNRELATED = "unrelated"


@dataclass
class ClaimResult:
    """Result of checking a single factual claim against sources."""

    claim: str
    grounded: bool
    confidence: float
    source_id: Optional[str] = None
    source_excerpt: Optional[str] = None
    contradiction: bool = False
    polarity: Optional[Polarity] = None
    similarity: Optional[float] = None
    original_sentence: Optional[str] = None
    database_name: Optional[str] = None


@dataclass
class FactCheckResult:
    """Aggregate result of fact-checking a text against one or more databases.

    ``error`` is set only when the check could not be performed at all (e.g. the
    LLM provider failed during claim extraction). It is distinct from a genuine
    zero score: callers should treat a non-``None`` ``error`` as "unverified",
    never as "verified" or "refuted".
    """

    claims: list[ClaimResult] = field(default_factory=list)
    overall_score: float = 0.0
    has_contradictions: bool = False
    citation_text: str = ""
    annotated_text: Optional[str] = None
    error: Optional[str] = None
