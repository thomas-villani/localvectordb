"""Tests for the localvectordb.validation module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from localvectordb.validation import (
    ClaimResult,
    FactChecker,
    FactCheckResult,
    Polarity,
)
from localvectordb.validation.annotator import _find_best_match, annotate_response
from localvectordb.validation.claims import extract_claims
from localvectordb.validation.llm import (
    AnthropicProvider,
    GeminiProvider,
    OpenAIProvider,
    detect_provider,
    extract_json,
)
from localvectordb.validation.polarity import classify_polarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockLLM:
    """Simple mock implementing the LLMProvider protocol."""

    def __init__(self, response: str = "[]"):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.response


@dataclass
class FakeQueryResult:
    """Minimal stand-in for localvectordb.core.QueryResult."""

    id: str
    score: float
    content: str
    document_id: Optional[str] = None
    metadata: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def make_mock_db(name: str = "testdb", query_results: list | None = None):
    """Create a mock LocalVectorDB with canned query results."""
    db = MagicMock()
    db.name = name
    db.query.return_value = query_results or []
    return db


# ---------------------------------------------------------------------------
# Tests: result.py
# ---------------------------------------------------------------------------


class TestResultModels:
    def test_polarity_values(self):
        assert Polarity.SUPPORTS == "supports"
        assert Polarity.CONTRADICTS == "contradicts"
        assert Polarity.UNRELATED == "unrelated"

    def test_claim_result_defaults(self):
        cr = ClaimResult(claim="test", grounded=False, confidence=0.0)
        assert cr.source_id is None
        assert cr.contradiction is False
        assert cr.polarity is None
        assert cr.database_name is None

    def test_fact_check_result_defaults(self):
        r = FactCheckResult()
        assert r.claims == []
        assert r.overall_score == 0.0
        assert r.has_contradictions is False
        assert r.annotated_text is None


# ---------------------------------------------------------------------------
# Tests: llm.py
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('[{"a": 1}]') == [{"a": 1}]

    def test_code_block_json(self):
        text = '```json\n[{"a": 1}]\n```'
        assert extract_json(text) == [{"a": 1}]

    def test_code_block_no_lang(self):
        text = "```\n{}\n```"
        assert extract_json(text) == {}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("not json at all")


class TestDetectProvider:
    def test_already_implements_protocol(self):
        llm = MockLLM()
        assert detect_provider(llm) is llm

    def test_anthropic_detection(self):
        client = MagicMock()
        type(client).__module__ = "anthropic._client"
        provider = detect_provider(client, "my-model")
        assert isinstance(provider, AnthropicProvider)

    def test_openai_detection(self):
        client = MagicMock()
        type(client).__module__ = "openai._client"
        provider = detect_provider(client)
        assert isinstance(provider, OpenAIProvider)

    def test_gemini_detection(self):
        client = MagicMock()
        type(client).__module__ = "google.genai._client"
        provider = detect_provider(client)
        assert isinstance(provider, GeminiProvider)

    def test_unknown_client_raises(self):
        client = MagicMock()
        type(client).__module__ = "unknown_lib"
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_provider(client)


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_sync_client(self):
        client = MagicMock()
        type(client).__module__ = "anthropic._client"
        # messages.create is not a coroutine → sync path
        response = MagicMock()
        response.content = [MagicMock(text="hello")]
        client.messages.create.return_value = response

        provider = AnthropicProvider(client, model="test-model")
        result = await provider.complete("sys", "usr")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_async_client(self):
        client = MagicMock()
        type(client).__module__ = "anthropic._client"
        response = MagicMock()
        response.content = [MagicMock(text="async hello")]
        client.messages.create = AsyncMock(return_value=response)

        provider = AnthropicProvider(client, model="test-model")
        result = await provider.complete("sys", "usr")
        assert result == "async hello"


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_sync_client(self):
        client = MagicMock()
        type(client).__module__ = "openai._client"
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="oi hello"))]
        client.chat.completions.create.return_value = response

        provider = OpenAIProvider(client, model="test-model")
        result = await provider.complete("sys", "usr")
        assert result == "oi hello"


class TestGeminiProvider:
    @pytest.mark.asyncio
    async def test_complete(self):
        client = MagicMock()
        type(client).__module__ = "google.genai._client"
        response = MagicMock()
        response.text = "gemini hello"
        client.models.generate_content.return_value = response

        provider = GeminiProvider(client, model="test-model")
        result = await provider.complete("sys", "usr")
        assert result == "gemini hello"


# ---------------------------------------------------------------------------
# Tests: claims.py
# ---------------------------------------------------------------------------


class TestExtractClaims:
    @pytest.mark.asyncio
    async def test_valid_extraction(self):
        claims = [
            {"claim": "X is true", "sentence": "X is true and important."},
            {"claim": "Y equals 5", "sentence": "Y equals 5."},
        ]
        llm = MockLLM(json.dumps(claims))
        result = await extract_claims(llm, "some text")
        assert len(result) == 2
        assert result[0]["claim"] == "X is true"

    @pytest.mark.asyncio
    async def test_filters_invalid_entries(self):
        claims = [
            {"claim": "ok", "sentence": "ok."},
            {"claim_only": "missing sentence"},
            "not a dict",
        ]
        llm = MockLLM(json.dumps(claims))
        result = await extract_claims(llm, "text")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_returns_empty_on_failure(self):
        llm = MockLLM("not valid json")
        result = await extract_claims(llm, "text")
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_non_list(self):
        llm = MockLLM('{"not": "a list"}')
        result = await extract_claims(llm, "text")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: polarity.py
# ---------------------------------------------------------------------------


class TestClassifyPolarity:
    @pytest.mark.asyncio
    async def test_supports(self):
        response = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.95, "excerpt": "evidence"},
            ]
        )
        llm = MockLLM(response)
        chunks = [{"content": "chunk text"}]
        results = await classify_polarity(llm, "claim", chunks)
        assert len(results) == 1
        assert results[0].polarity == Polarity.SUPPORTS
        assert results[0].confidence == 0.95

    @pytest.mark.asyncio
    async def test_contradicts(self):
        response = json.dumps(
            [
                {"index": 0, "polarity": "contradicts", "confidence": 0.9, "excerpt": "nope"},
            ]
        )
        llm = MockLLM(response)
        results = await classify_polarity(llm, "claim", [{"content": "c"}])
        assert results[0].polarity == Polarity.CONTRADICTS

    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        llm = MockLLM("[]")
        results = await classify_polarity(llm, "claim", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_fallback_on_bad_json(self):
        llm = MockLLM("not json")
        results = await classify_polarity(llm, "claim", [{"content": "c"}, {"content": "d"}])
        assert len(results) == 2
        assert all(r.polarity == Polarity.UNRELATED for r in results)
        assert all(r.confidence == 0.0 for r in results)

    @pytest.mark.asyncio
    async def test_missing_index_uses_default(self):
        response = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.8, "excerpt": "ok"},
                # index 1 is missing from response
            ]
        )
        llm = MockLLM(response)
        results = await classify_polarity(llm, "claim", [{"content": "a"}, {"content": "b"}])
        assert results[0].polarity == Polarity.SUPPORTS
        assert results[1].polarity == Polarity.UNRELATED

    @pytest.mark.asyncio
    async def test_invalid_polarity_string(self):
        response = json.dumps(
            [
                {"index": 0, "polarity": "maybe", "confidence": 0.5, "excerpt": None},
            ]
        )
        llm = MockLLM(response)
        results = await classify_polarity(llm, "claim", [{"content": "c"}])
        assert results[0].polarity == Polarity.UNRELATED


# ---------------------------------------------------------------------------
# Tests: annotator.py
# ---------------------------------------------------------------------------


class TestFindBestMatch:
    def test_exact_match(self):
        text = "Hello world. This is a test."
        result = _find_best_match(text, "This is a test.")
        assert result == (13, 28)

    def test_fuzzy_match(self):
        text = "Hello world. This is a test."
        result = _find_best_match(text, "This is a tset.", threshold=0.7)
        assert result is not None
        assert text[result[0] : result[1]] == "This is a test."

    def test_no_match(self):
        text = "Hello world."
        result = _find_best_match(text, "Completely different sentence.", threshold=0.9)
        assert result is None


class TestAnnotateResponse:
    def test_basic_annotation(self):
        text = "The policy allows 10 days PTO. Sick leave is separate."
        claims = [
            ClaimResult(
                claim="10 days PTO",
                grounded=True,
                confidence=0.9,
                source_id="policies/pto.md",
                source_excerpt="Employees receive 10 days",
                original_sentence="The policy allows 10 days PTO.",
            ),
        ]
        result = annotate_response(text, claims)
        assert result is not None
        assert "[1]" in result
        assert "policies/pto.md" in result

    def test_no_grounded_claims(self):
        claims = [
            ClaimResult(claim="x", grounded=False, confidence=0.0, original_sentence="x."),
        ]
        result = annotate_response("x.", claims)
        assert result is None

    def test_no_source_id(self):
        claims = [
            ClaimResult(
                claim="x",
                grounded=True,
                confidence=0.9,
                source_id=None,
                original_sentence="x.",
            ),
        ]
        result = annotate_response("x.", claims)
        assert result is None

    def test_multiple_citations(self):
        text = "Fact A is true. Fact B is also true."
        claims = [
            ClaimResult(
                claim="A",
                grounded=True,
                confidence=0.9,
                source_id="docA",
                source_excerpt="A excerpt",
                original_sentence="Fact A is true.",
            ),
            ClaimResult(
                claim="B",
                grounded=True,
                confidence=0.8,
                source_id="docB",
                source_excerpt="B excerpt",
                original_sentence="Fact B is also true.",
            ),
        ]
        result = annotate_response(text, claims)
        assert result is not None
        assert "[1]" in result
        assert "[2]" in result
        assert "docA" in result
        assert "docB" in result


# ---------------------------------------------------------------------------
# Tests: checker.py
# ---------------------------------------------------------------------------


class TestFactChecker:
    def _make_checker(self, llm_responses: list[str] | None = None, query_results=None):
        """Create a FactChecker with mocked LLM and DB."""
        responses = list(llm_responses or ["[]"])
        call_idx = {"i": 0}

        class SequentialLLM:
            async def complete(self, system: str, user: str) -> str:
                idx = min(call_idx["i"], len(responses) - 1)
                call_idx["i"] += 1
                return responses[idx]

        db = make_mock_db("testdb", query_results or [])
        return FactChecker(
            databases=[db],
            llm=SequentialLLM(),
            similarity_threshold=0.1,
            min_grounding_score=0.7,
        )

    @pytest.mark.asyncio
    async def test_no_claims_returns_perfect_score(self):
        checker = self._make_checker(llm_responses=["[]"])
        result = await checker.check_async("Some text with no facts.")
        assert result.overall_score == 1.0
        assert len(result.claims) == 0
        assert result.citation_text == "No factual claims detected."

    @pytest.mark.asyncio
    async def test_claim_with_no_sources(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        checker = self._make_checker(llm_responses=[claims])
        result = await checker.check_async("X is 5.")
        assert result.overall_score == 0.0
        assert len(result.claims) == 1
        assert result.claims[0].grounded is False

    @pytest.mark.asyncio
    async def test_supported_claim(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.95, "excerpt": "X equals 5"},
            ]
        )
        qr = [FakeQueryResult(id="chunk_0", score=0.9, content="X equals 5", document_id="doc1")]
        checker = self._make_checker(
            llm_responses=[claims, polarity],
            query_results=qr,
        )
        result = await checker.check_async("X is 5.")
        assert result.overall_score == 0.95
        assert result.claims[0].grounded is True
        assert result.claims[0].polarity == Polarity.SUPPORTS

    @pytest.mark.asyncio
    async def test_contradicted_claim(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "contradicts", "confidence": 0.9, "excerpt": "X is 3"},
            ]
        )
        qr = [FakeQueryResult(id="chunk_0", score=0.8, content="X is 3", document_id="doc1")]
        # First polarity response (scoped or expanded) returns contradiction,
        # second call on expanded search also returns contradiction
        checker = self._make_checker(
            llm_responses=[claims, polarity, polarity],
            query_results=qr,
        )
        result = await checker.check_async("X is 5.")
        assert result.overall_score == 0.0
        assert result.has_contradictions is True
        assert result.claims[0].contradiction is True

    @pytest.mark.asyncio
    async def test_scoped_search_with_support(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.9, "excerpt": "X=5"},
            ]
        )
        qr = [FakeQueryResult(id="chunk_0", score=0.85, content="X=5", document_id="src_doc")]
        checker = self._make_checker(
            llm_responses=[claims, polarity],
            query_results=qr,
        )
        result = await checker.check_async("X is 5.", sources=["src_doc"])
        assert result.claims[0].grounded is True
        assert result.claims[0].source_id == "src_doc"

    @pytest.mark.asyncio
    async def test_scoped_search_falls_through(self):
        """When scoped search finds no matching docs, expanded search runs."""
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.85, "excerpt": "X=5"},
            ]
        )
        # doc_id is "other_doc" which doesn't match source "src_doc"
        qr = [FakeQueryResult(id="chunk_0", score=0.8, content="X=5", document_id="other_doc")]
        checker = self._make_checker(
            llm_responses=[claims, polarity],
            query_results=qr,
        )
        result = await checker.check_async("X is 5.", sources=["src_doc"])
        # Should still find via expanded search
        assert result.claims[0].grounded is True
        assert result.claims[0].source_id == "other_doc"

    @pytest.mark.asyncio
    async def test_multiple_claims(self):
        claims = json.dumps(
            [
                {"claim": "A is 1", "sentence": "A is 1."},
                {"claim": "B is 2", "sentence": "B is 2."},
            ]
        )
        pol1 = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.9, "excerpt": "A=1"},
            ]
        )
        pol2 = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.8, "excerpt": "B=2"},
            ]
        )
        qr = [FakeQueryResult(id="c0", score=0.85, content="A=1 B=2", document_id="doc1")]
        checker = self._make_checker(
            llm_responses=[claims, pol1, pol2],
            query_results=qr,
        )
        result = await checker.check_async("A is 1. B is 2.")
        assert len(result.claims) == 2
        assert result.overall_score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_citation_text_format(self):
        claims = json.dumps([{"claim": "X", "sentence": "X."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.9, "excerpt": "X evidence"},
            ]
        )
        qr = [FakeQueryResult(id="c0", score=0.8, content="X evidence", document_id="doc1")]
        checker = self._make_checker(
            llm_responses=[claims, polarity],
            query_results=qr,
        )
        result = await checker.check_async("X.")
        assert "Sources consulted:" in result.citation_text
        assert "doc1" in result.citation_text

    @pytest.mark.asyncio
    async def test_contradiction_in_citation_text(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "contradicts", "confidence": 0.9, "excerpt": "X is 3"},
            ]
        )
        qr = [FakeQueryResult(id="c0", score=0.8, content="X is 3", document_id="doc1")]
        checker = self._make_checker(
            llm_responses=[claims, polarity, polarity],
            query_results=qr,
        )
        result = await checker.check_async("X is 5.")
        assert "Contradictions detected:" in result.citation_text

    def test_sync_check(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.9, "excerpt": "X=5"},
            ]
        )
        qr = [FakeQueryResult(id="c0", score=0.8, content="X=5", document_id="doc1")]
        checker = self._make_checker(
            llm_responses=[claims, polarity],
            query_results=qr,
        )
        result = checker.check("X is 5.")
        assert result.claims[0].grounded is True

    @pytest.mark.asyncio
    async def test_single_db_accepted(self):
        """FactChecker accepts a single DB (not wrapped in a list)."""
        db = make_mock_db("solo")
        llm = MockLLM("[]")
        checker = FactChecker(databases=db, llm=llm)
        result = await checker.check_async("text")
        assert result.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_multi_db_search(self):
        claims = json.dumps([{"claim": "X is 5", "sentence": "X is 5."}])
        polarity = json.dumps(
            [
                {"index": 0, "polarity": "supports", "confidence": 0.9, "excerpt": "X=5"},
            ]
        )

        responses = [claims, polarity]
        call_idx = {"i": 0}

        class SeqLLM:
            async def complete(self, system, user):
                idx = min(call_idx["i"], len(responses) - 1)
                call_idx["i"] += 1
                return responses[idx]

        db1 = make_mock_db("db1", [])
        db2 = make_mock_db(
            "db2",
            [FakeQueryResult(id="c0", score=0.9, content="X=5", document_id="doc_db2")],
        )

        checker = FactChecker(databases=[db1, db2], llm=SeqLLM(), similarity_threshold=0.1)
        result = await checker.check_async("X is 5.")
        assert result.claims[0].database_name == "db2"


class TestFactCheckerFormatCitations:
    def test_ungrounded_claims_section(self):
        claims = [
            ClaimResult(claim="mystery claim", grounded=False, confidence=0.0),
        ]
        text = FactChecker._format_citations(claims)
        assert "Ungrounded claims:" in text
        assert "mystery claim" in text

    def test_no_sources(self):
        text = FactChecker._format_citations([])
        assert text == "No sources found."


# ---------------------------------------------------------------------------
# Tests: integration (import from top-level)
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    def test_import_from_localvectordb(self):
        from localvectordb import ClaimResult, FactChecker, FactCheckResult, Polarity

        assert FactChecker is not None
        assert FactCheckResult is not None
        assert ClaimResult is not None
        assert Polarity is not None
