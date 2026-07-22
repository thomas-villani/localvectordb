"""Tests for token/word/character context sizing (``context_unit`` / ``context_truncate``).

The ``return_type='context'`` and ``'enriched'`` surfaces historically sized the
returned context purely by chunk count (``context_window``). These tests cover the
budget modes added for v0.1.0, where ``context_window`` is interpreted as a
token/word/character budget:

- the low-level measurement/truncation/validation helpers,
- greedy whole-chunk selection staying within the budget (including the
  inter-chunk separator), sync and async, for both context and enriched,
- the ``context_truncate`` hard cap (and the single-chunk-exceeds-budget case it
  exists for),
- backwards-compatible ``context_unit='chunks'`` behaviour, and
- the server request-model validation of ``context_window`` vs ``context_unit``.
"""

import tempfile

import pytest
import tiktoken

from localvectordb.database import LocalVectorDB
from localvectordb.database._search import (
    _CONTEXT_SEPARATOR,
    _measure_text,
    _truncate_text_to_budget,
    _validate_context_unit,
)

_ENC = tiktoken.get_encoding("cl100k_base")


def _count(text: str, unit: str) -> int:
    if unit == "tokens":
        return len(_ENC.encode(text))
    if unit == "words":
        return len(text.split())
    return len(text)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
class TestMeasureText:
    def test_tokens_prefers_stored_count(self):
        # When a stored token count is supplied it is used verbatim.
        assert _measure_text("anything at all here", tokens=7, unit="tokens") == 7

    def test_tokens_falls_back_to_encoding(self):
        text = "hello world foo bar"
        assert _measure_text(text, tokens=0, unit="tokens") == len(_ENC.encode(text))
        assert _measure_text(text, tokens=None, unit="tokens") == len(_ENC.encode(text))

    def test_words_and_characters(self):
        assert _measure_text("one two three", tokens=None, unit="words") == 3
        assert _measure_text("abcde", tokens=None, unit="characters") == 5


class TestTruncateToBudget:
    def test_characters_backs_off_to_word_boundary(self):
        out = _truncate_text_to_budget("alpha beta gamma delta", 12, "characters")
        assert len(out) <= 12
        # Should not end in the middle of a word.
        assert not out.endswith("gam")
        assert out == "alpha beta"

    def test_words_slices_at_word_boundary_preserving_spacing(self):
        text = "one two three four five"
        out = _truncate_text_to_budget(text, 3, "words")
        assert out == "one two three"

    def test_tokens_are_exact(self):
        text = "The quick brown fox jumps over the lazy dog repeatedly and often."
        out = _truncate_text_to_budget(text, 4, "tokens")
        assert len(_ENC.encode(out)) <= 4

    def test_no_truncation_when_within_budget(self):
        assert _truncate_text_to_budget("short", 100, "characters") == "short"

    def test_zero_budget_returns_empty(self):
        assert _truncate_text_to_budget("anything", 0, "tokens") == ""


class TestValidateContextUnit:
    @pytest.mark.parametrize("unit", ["chunks", "tokens", "words", "characters"])
    def test_valid_units(self, unit):
        _validate_context_unit(unit)  # no raise

    @pytest.mark.parametrize("bad", ["token", "chars", "", "CHUNKS", "sentences"])
    def test_invalid_units_raise(self, bad):
        with pytest.raises(ValueError, match="Invalid context_unit"):
            _validate_context_unit(bad)


# ---------------------------------------------------------------------------
# Integration: budget-based context/enrichment against a real database
# ---------------------------------------------------------------------------
@pytest.fixture
def sized_db():
    """A single multi-chunk document with small, even chunks."""
    tmp = tempfile.mkdtemp()
    db = LocalVectorDB(
        name="ctxsize",
        base_path=tmp,
        embedding_provider="mock",
        embedding_model="mock-model",
        embedding_config={"dimension": 32},
        chunk_size=12,  # small chunks (~12 tokens) so budgets span a few
        chunk_overlap=0,
    )
    doc = " ".join(f"sentence{i} alpha beta gamma delta epsilon." for i in range(80))
    db.upsert(documents=[doc], metadata=[{}])
    yield db
    db.close()


@pytest.mark.database
class TestContextBudgetSync:
    @pytest.mark.parametrize("unit,budget", [("words", 40), ("characters", 200), ("tokens", 60)])
    def test_context_budget_is_respected(self, sized_db, unit, budget):
        results = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=budget,
            context_unit=unit,
        )
        assert results and results[0].type == "context"
        # Whole-chunk greedy never exceeds the budget once it spans >1 chunk.
        assert _count(results[0].content, unit) <= budget
        assert results[0].metadata["_context_unit"] == unit
        # Should have pulled in more than the single matched chunk for a big budget.
        assert results[0].metadata["_context_chunk_count"] >= 1

    def test_bigger_budget_returns_more_context(self, sized_db):
        small = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=20,
            context_unit="words",
        )
        big = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=120,
            context_unit="words",
        )
        assert big[0].metadata["_context_chunk_count"] >= small[0].metadata["_context_chunk_count"]
        assert len(big[0].content) >= len(small[0].content)

    def test_single_chunk_exceeds_budget_keeps_matched_chunk(self, sized_db):
        # A budget smaller than one chunk: whole-chunk mode still returns the
        # matched chunk (so it can overshoot), truncate mode enforces the cap.
        whole = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=3,
            context_unit="tokens",
        )
        assert whole and whole[0].content  # matched chunk retained
        assert whole[0].metadata["_context_chunk_count"] == 1

        truncated = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=3,
            context_unit="tokens",
            context_truncate=True,
        )
        assert _count(truncated[0].content, "tokens") <= 3
        assert truncated[0].metadata["_context_truncated"] is True

    @pytest.mark.parametrize("unit,budget", [("words", 25), ("characters", 120), ("tokens", 35)])
    def test_truncate_enforces_exact_cap(self, sized_db, unit, budget):
        results = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=budget,
            context_unit=unit,
            context_truncate=True,
        )
        assert _count(results[0].content, unit) <= budget

    def test_separator_counts_toward_budget(self, sized_db):
        # Regression: the inter-chunk separator must be charged against the budget
        # so multi-chunk assembly never exceeds a word budget.
        results = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=45,
            context_unit="words",
        )
        content = results[0].content
        if results[0].metadata["_context_chunk_count"] > 1:
            assert _CONTEXT_SEPARATOR in content
        assert len(content.split()) <= 45


@pytest.mark.database
class TestEnrichedBudgetSync:
    def test_enriched_budget_marks_method_and_respects_budget(self, sized_db):
        results = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="enriched",
            k=1,
            context_window=60,
            context_unit="tokens",
        )
        assert results and results[0].type == "enriched"
        assert results[0].metadata["_enrichment_method"] == "budget"
        assert results[0].metadata["_context_unit"] == "tokens"
        assert _count(results[0].content, "tokens") <= 60

    def test_enriched_truncate(self, sized_db):
        # A budget smaller than a single chunk forces the hard cut (greedy would
        # otherwise keep the whole matched chunk).
        results = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="enriched",
            k=1,
            context_window=3,
            context_unit="words",
            context_truncate=True,
        )
        assert _count(results[0].content, "words") <= 3
        assert results[0].metadata.get("_context_truncated") is True


@pytest.mark.database
class TestContextBudgetAsync:
    async def test_async_context_budget(self, sized_db):
        results = await sized_db.query_async(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=40,
            context_unit="words",
        )
        assert results and results[0].type == "context"
        assert _count(results[0].content, "words") <= 40

    async def test_async_enriched_budget_truncate(self, sized_db):
        # Budget below one chunk forces the hard cut regardless of chunk selection.
        results = await sized_db.query_async(
            "alpha beta",
            search_type="hybrid",
            return_type="enriched",
            k=1,
            context_window=4,
            context_unit="tokens",
            context_truncate=True,
        )
        assert results and results[0].type == "enriched"
        assert _count(results[0].content, "tokens") <= 4
        assert results[0].metadata.get("_context_truncated") is True


@pytest.mark.database
class TestChunkModeUnchanged:
    def test_default_is_chunk_mode(self, sized_db):
        # context_unit defaults to 'chunks'; window counts neighbours as before.
        results = sized_db.query(
            "alpha beta",
            search_type="vector",
            return_type="context",
            k=1,
            context_window=1,
        )
        assert results[0].type == "context"
        assert results[0].metadata["_context_unit"] == "chunks"
        # window=1 -> at most matched + 1 before + 1 after
        assert results[0].metadata["_context_chunk_count"] <= 3

    def test_invalid_unit_rejected_at_query(self, sized_db):
        with pytest.raises(ValueError, match="Invalid context_unit"):
            sized_db.query("alpha", context_unit="sentences")


# ---------------------------------------------------------------------------
# Server request-model validation
# ---------------------------------------------------------------------------
class TestServerModelValidation:
    def test_chunk_window_not_capped(self):
        # M2 (relax server to match local): local query() imposes no context_window
        # ceiling, so the server must not either -- a value that succeeds locally must
        # not 422 remotely. The old <=20-for-chunks cap was removed.
        from localvectordb_server.routers._models import QueryBody

        body = QueryBody(query="x", context_unit="chunks", context_window=50)
        assert body.context_window == 50

    def test_budget_window_allows_large_values(self):
        from localvectordb_server.routers._models import QueryBody

        body = QueryBody(query="x", context_unit="tokens", context_window=5000, context_truncate=True)
        assert body.context_window == 5000
        assert body.context_unit == "tokens"
        assert body.context_truncate is True

    def test_defaults(self):
        from localvectordb_server.routers._models import QueryBody

        body = QueryBody(query="x")
        assert body.context_unit == "chunks"
        assert body.context_truncate is False
        assert body.rerank_k is None

    def test_rerank_k_accepted_and_serialized(self):
        from localvectordb_server.routers._models import QueryBody

        body = QueryBody(query="x", rerank_k=50)
        assert body.rerank_k == 50
        assert body.model_dump()["rerank_k"] == 50

    def test_rerank_k_rejects_non_positive(self):
        from pydantic import ValidationError

        from localvectordb_server.routers._models import QueryBody

        with pytest.raises(ValidationError):
            QueryBody(query="x", rerank_k=0)
