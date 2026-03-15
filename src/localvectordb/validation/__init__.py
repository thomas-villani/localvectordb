"""Fact-checking / validation module for LocalVectorDB.

Provides "reverse RAG" -- verify LLM-generated text against documents stored
in one or more LocalVectorDB instances.

Quick start::

    from localvectordb.validation import FactChecker

    checker = FactChecker(databases=[db], llm=anthropic_client)
    result = checker.check("The policy allows 10 days PTO per year.")

    print(result.overall_score)
    print(result.annotated_text)
"""

from .checker import FactChecker
from .llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMProvider,
    OpenAIProvider,
)
from .result import ClaimResult, FactCheckResult, Polarity

__all__ = [
    "FactChecker",
    "FactCheckResult",
    "ClaimResult",
    "Polarity",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
]
