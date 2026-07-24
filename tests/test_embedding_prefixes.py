"""Tests for asymmetric retrieval prefixes.

Asymmetric embedding models are trained with a different instruction on each side
(EmbeddingGemma, the new default, is unusually sensitive to this). These tests pin
three things that are easy to break silently:

* the prefix a model resolves to from its name,
* that ingest embeds with the document prefix and search with the query prefix,
* that a database reproduces the prefixes its vectors were built with on reopen.

A prefix bug does not raise -- it just ranks worse -- so every assertion here is
on the exact text handed to the provider rather than on results.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, List

import pytest

from localvectordb.database import LocalVectorDB
from localvectordb.embeddings import (
    EmbeddingRegistry,
    JinaEmbeddings,
    MockEmbeddings,
    ModelPrefixes,
    resolve_model_prefixes,
)

GEMMA_DOC = "title: none | text: "
GEMMA_QUERY = "task: search result | query: "


class SpyEmbeddings(MockEmbeddings):
    """Mock provider that records the (task, texts) of every batch it embeds."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.seen: List[tuple] = []

    @property
    def provider_name(self) -> str:
        # Must not inherit "mock", or a database saves that name and reopening
        # rebuilds a plain MockEmbeddings with no `seen` to assert against.
        return "spy"

    async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        self.seen.append((kwargs.get("task"), list(texts)))
        return await super()._embed_single_batch(texts, **kwargs)

    def texts_for(self, task: str) -> List[str]:
        return [t for seen_task, texts in self.seen if seen_task == task for t in texts]


@pytest.fixture
def spy_provider():
    """Register the spy under its own provider name for the duration of a test."""
    EmbeddingRegistry.register("spy", SpyEmbeddings)
    yield
    EmbeddingRegistry._providers.pop("spy", None)


@pytest.mark.unit
@pytest.mark.embedding
class TestModelPrefixResolution:
    """Prefixes are looked up from a normalised model name."""

    @pytest.mark.parametrize(
        "model,expected",
        [
            ("embeddinggemma", ModelPrefixes(GEMMA_DOC, GEMMA_QUERY)),
            ("nomic-embed-text", ModelPrefixes("search_document: ", "search_query: ")),
            ("multilingual-e5-large", ModelPrefixes("passage: ", "query: ")),
            (
                "snowflake-arctic-embed2",
                ModelPrefixes("", "Represent this sentence for searching relevant passages: "),
            ),
        ],
    )
    def test_known_models(self, model, expected):
        assert resolve_model_prefixes(model) == expected

    @pytest.mark.parametrize(
        "model",
        [
            "embeddinggemma:300m",
            "embeddinggemma:latest",
            "hf.co/google/EmbeddingGemma-300M",
            "hf.co/google/EmbeddingGemma-300M:Q8_0",
        ],
    )
    def test_tags_and_registry_paths_normalize(self, model):
        """A version tag or registry path must not defeat the lookup."""
        assert resolve_model_prefixes(model) == ModelPrefixes(GEMMA_DOC, GEMMA_QUERY)

    @pytest.mark.parametrize("model", ["bge-m3", "text-embedding-3-small", "gte-large", "some-private-encoder"])
    def test_unknown_and_symmetric_models_get_no_prefix(self, model):
        """An unrecognised model is assumed symmetric rather than guessed at.

        bge-m3 is the trap: it is symmetric, but sits next to bge-large-en which
        is not, so a bare "bge-" pattern would silently corrupt it.
        """
        assert resolve_model_prefixes(model) == ModelPrefixes("", "")


@pytest.mark.unit
@pytest.mark.embedding
class TestProviderPrefixConfiguration:
    def test_auto_detected_from_model_name(self):
        provider = MockEmbeddings("embeddinggemma", auto_prefix=True)
        assert (provider.document_prefix, provider.query_prefix) == (GEMMA_DOC, GEMMA_QUERY)

    def test_explicit_prefixes_win(self):
        provider = MockEmbeddings("embeddinggemma", document_prefix="D: ", query_prefix="Q: ")
        assert (provider.document_prefix, provider.query_prefix) == ("D: ", "Q: ")

    def test_empty_string_forces_no_prefix(self):
        """'' is a real value (opt out), distinct from None (auto-detect)."""
        provider = MockEmbeddings("embeddinggemma", auto_prefix=True, document_prefix="", query_prefix="")
        assert provider.uses_prefixes is False

    def test_half_specified_does_not_autofill_the_other_side(self):
        """Mixing a caller's prefix with a detected one would pair mismatched instructions."""
        provider = MockEmbeddings("embeddinggemma", auto_prefix=True, query_prefix="Q: ")
        assert provider.document_prefix == ""
        assert provider.query_prefix == "Q: "

    def test_auto_prefix_disabled(self):
        provider = MockEmbeddings("embeddinggemma", auto_prefix=False)
        assert provider.uses_prefixes is False

    def test_mock_does_not_auto_detect_by_default(self):
        """MockEmbeddings seeds vectors from a hash of the text.

        Auto-detecting would make the query vector for "foo" unrelated to the
        document vector for "foo" purely because the model name matched, turning a
        symmetric test double asymmetric.
        """
        assert MockEmbeddings("embeddinggemma").uses_prefixes is False

    def test_apply_prefix(self):
        provider = MockEmbeddings("embeddinggemma", auto_prefix=True)
        assert provider.apply_prefix(["hi"], "document") == [GEMMA_DOC + "hi"]
        assert provider.apply_prefix(["hi"], "query") == [GEMMA_QUERY + "hi"]

    def test_apply_prefix_is_identity_without_a_prefix(self):
        provider = MockEmbeddings("plain-model")
        texts = ["a", "b"]
        assert provider.apply_prefix(texts, "query") is texts

    def test_unknown_task_rejected(self):
        with pytest.raises(ValueError, match="Unknown embedding task"):
            MockEmbeddings("m").prefix_for("passage")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.embedding
class TestEmbedTaskPlumbing:
    def test_default_task_is_document(self):
        provider = SpyEmbeddings("embeddinggemma", auto_prefix=True)
        provider.embed_sync(["hello"])
        assert provider.texts_for("document") == [GEMMA_DOC + "hello"]

    def test_embed_query_applies_query_prefix_and_returns_1d(self):
        provider = SpyEmbeddings("embeddinggemma", auto_prefix=True)
        vector = provider.embed_query("hello")
        assert vector.ndim == 1
        assert provider.texts_for("query") == [GEMMA_QUERY + "hello"]

    def test_embed_query_async_matches_sync(self):
        provider = SpyEmbeddings("embeddinggemma", auto_prefix=True)
        vector = asyncio.run(provider.embed_query_async("hello"))
        assert vector.ndim == 1
        assert provider.texts_for("query") == [GEMMA_QUERY + "hello"]

    def test_task_survives_multiple_batches(self):
        """Regression: the batch loop's local shadowed the `task` parameter.

        The closure read the enclosing `task`, so rebinding that name in the loop
        handed every provider a coroutine object instead of "query" -- which no
        provider errors on, it just silently stops distinguishing the two sides.
        """
        provider = SpyEmbeddings("embeddinggemma", auto_prefix=True)
        provider.embed_sync([f"t{i}" for i in range(5)], batch_size=2, task="query")

        assert [task for task, _ in provider.seen] == ["query"] * 3
        assert sorted(provider.texts_for("query")) == sorted(GEMMA_QUERY + f"t{i}" for i in range(5))

    def test_prefix_applied_once_across_retries(self):
        """The prefix is attached outside the retry loop, so a retry cannot double it."""
        attempts = []

        class FlakyEmbeddings(SpyEmbeddings):
            async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
                attempts.append(list(texts))
                if len(attempts) == 1:
                    raise ConnectionError("transient")
                return await super()._embed_single_batch(texts, **kwargs)

        provider = FlakyEmbeddings("embeddinggemma", auto_prefix=True, retry_delay=0.0)
        provider.embed_sync(["hello"], task="query")

        assert attempts == [[GEMMA_QUERY + "hello"]] * 2


@pytest.mark.unit
@pytest.mark.embedding
class TestNativeTaskProviders:
    """Providers whose API takes a task parameter instead of a text prefix."""

    def test_jina_defaults_both_sides_to_task(self):
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="k", task="text-matching")
        assert provider.document_task == "text-matching"
        assert provider.query_task == "text-matching"

    def test_jina_per_side_tasks(self):
        provider = JinaEmbeddings(
            "jina-embeddings-v4",
            api_key="k",
            document_task="retrieval.passage",
            query_task="retrieval.query",
        )
        assert provider.document_task == "retrieval.passage"
        assert provider.query_task == "retrieval.query"

    def test_jina_rejects_unsupported_per_side_task(self):
        with pytest.raises(ValueError, match="query_task"):
            JinaEmbeddings("jina-embeddings-v4", api_key="k", query_task="not-a-task")

    def test_google_defaults_both_sides_to_task_type(self):
        from localvectordb.embeddings import GoogleEmbeddings

        provider = GoogleEmbeddings("gemini-embedding-001", api_key="k", requested_dimensions=8)
        assert provider.document_task_type == provider.task_type == "SEMANTIC_SIMILARITY"
        assert provider.query_task_type == "SEMANTIC_SIMILARITY"

    def test_google_per_side_task_types(self):
        from localvectordb.embeddings import GoogleEmbeddings

        provider = GoogleEmbeddings(
            "gemini-embedding-001",
            api_key="k",
            requested_dimensions=8,
            document_task_type="retrieval_document",
            query_task_type="retrieval_query",
        )
        assert provider.document_task_type == "RETRIEVAL_DOCUMENT"
        assert provider.query_task_type == "RETRIEVAL_QUERY"


@pytest.mark.integration
@pytest.mark.embedding
class TestDatabasePrefixes:
    """The database must use, persist, and reproduce its prefixes."""

    def _db(self, tmp_path, **embedding_config):
        return LocalVectorDB(
            name="prefixdb",
            base_path=str(tmp_path),
            embedding_provider="spy",
            embedding_model="embeddinggemma",
            embedding_config=embedding_config or None,
        )

    def test_ingest_uses_document_prefix_and_search_uses_query_prefix(self, tmp_path, spy_provider):
        db = self._db(tmp_path, auto_prefix=True)
        try:
            db.upsert("The capital of France is Paris.")
            provider = db.embedding_provider
            assert provider.texts_for("document") == [GEMMA_DOC + "The capital of France is Paris."]

            provider.seen.clear()
            db.query("capital of France", k=1)
            assert provider.texts_for("query") == [GEMMA_QUERY + "capital of France"]
            assert provider.texts_for("document") == []
        finally:
            db.close()

    def test_async_query_uses_query_prefix(self, tmp_path, spy_provider):
        db = self._db(tmp_path, auto_prefix=True)
        try:
            db.upsert("The capital of France is Paris.")
            db.embedding_provider.seen.clear()
            asyncio.run(db.query_async("capital of France", k=1))
            assert db.embedding_provider.texts_for("query") == [GEMMA_QUERY + "capital of France"]
        finally:
            db.close()

    def test_prefixes_are_persisted(self, tmp_path, spy_provider):
        db = self._db(tmp_path, auto_prefix=True)
        db.close()

        conn = sqlite3.connect(str(Path(tmp_path) / "prefixdb.sqlite"))
        try:
            saved = dict(conn.execute("SELECT key, value FROM config WHERE key LIKE '%prefix%'").fetchall())
        finally:
            conn.close()

        assert saved["embedding_document_prefix"] == GEMMA_DOC
        assert saved["embedding_query_prefix"] == GEMMA_QUERY

    def test_reopen_restores_saved_prefixes(self, tmp_path, spy_provider):
        self._db(tmp_path, auto_prefix=True).close()

        reopened = self._db(tmp_path)
        try:
            assert reopened.embedding_provider.document_prefix == GEMMA_DOC
            assert reopened.embedding_provider.query_prefix == GEMMA_QUERY
        finally:
            reopened.close()

    def test_database_predating_prefixes_stays_unprefixed(self, tmp_path, spy_provider):
        """The back-compat guarantee: an existing index must not silently change space.

        A database built before prefixes existed has vectors with no instruction in
        them. Auto-detecting on reopen would embed queries in a different space than
        the stored chunks and degrade every search, with nothing raised.
        """
        db = self._db(tmp_path, auto_prefix=True, document_prefix="", query_prefix="")
        db.upsert("The capital of France is Paris.")
        db.close()

        # Simulate the pre-upgrade on-disk state: no prefix keys at all.
        conn = sqlite3.connect(str(Path(tmp_path) / "prefixdb.sqlite"))
        try:
            conn.execute("DELETE FROM config WHERE key LIKE '%prefix%'")
            conn.commit()
        finally:
            conn.close()

        reopened = self._db(tmp_path)
        try:
            assert reopened.embedding_provider.uses_prefixes is False
            reopened.query("capital of France", k=1)
            assert reopened.embedding_provider.texts_for("query") == ["capital of France"]
        finally:
            reopened.close()

    def test_explicit_override_beats_saved_prefix(self, tmp_path, spy_provider):
        """An escape hatch for re-ingesting under different prefixes."""
        self._db(tmp_path, auto_prefix=True).close()

        reopened = self._db(tmp_path, document_prefix="D: ", query_prefix="Q: ")
        try:
            assert reopened.embedding_provider.document_prefix == "D: "
            assert reopened.embedding_provider.query_prefix == "Q: "
        finally:
            reopened.close()
