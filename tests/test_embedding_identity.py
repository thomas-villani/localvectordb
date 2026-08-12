"""The embedding identity a database is built with must survive reopening.

A database's stored vectors are only meaningful under the provider, model and
dimension that produced them, and there is no in-place migration. So the saved
identity always wins on reopen -- but it used to win *silently*, discarding a
caller-supplied provider or model without a word. These tests pin the three
outcomes: silent when nothing was overridden, warned when something was, and
raised when the mismatch is one the FAISS index cannot survive.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pytest

from localvectordb.database import LocalVectorDB
from localvectordb.embeddings import EmbeddingRegistry, MockEmbeddings
from localvectordb.exceptions import ConfigurationError


class AlphaEmbeddings(MockEmbeddings):
    """Stands in for an HTTP provider: carries a base_url like the real ones."""

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.base_url = kwargs.pop("base_url", None)
        super().__init__(model, **kwargs)

    @property
    def provider_name(self) -> str:
        return "alpha"


class BetaEmbeddings(MockEmbeddings):
    @property
    def provider_name(self) -> str:
        return "beta"


@pytest.fixture
def providers():
    """Two interchangeable named providers, so divergence is testable offline."""
    EmbeddingRegistry.register("alpha", AlphaEmbeddings)
    EmbeddingRegistry.register("beta", BetaEmbeddings)
    yield
    EmbeddingRegistry._providers.pop("alpha", None)
    EmbeddingRegistry._providers.pop("beta", None)


def _db(tmp_path, **kwargs) -> LocalVectorDB:
    kwargs.setdefault("embedding_provider", "alpha")
    kwargs.setdefault("embedding_model", "model-a")
    return LocalVectorDB(name="identitydb", base_path=str(tmp_path), **kwargs)


@pytest.mark.unit
@pytest.mark.embedding
class TestSavedIdentityResolution:
    """Unit-level: the resolver, independent of a real database."""

    def test_empty_config_takes_the_request(self):
        assert LocalVectorDB._resolve_saved_embedding_identity("openai", "text-embedding-3-small", {}) == (
            "openai",
            "text-embedding-3-small",
        )

    def test_empty_config_and_no_request_falls_back_to_defaults(self):
        assert LocalVectorDB._resolve_saved_embedding_identity(None, None, {}) == ("ollama", "embeddinggemma")

    def test_partial_saved_config_is_not_trusted(self):
        """A provider with no model recorded is not an identity; take the request."""
        saved: Dict[str, str] = {"embedding_provider": "ollama"}
        assert LocalVectorDB._resolve_saved_embedding_identity("openai", "text-embedding-3-small", saved) == (
            "openai",
            "text-embedding-3-small",
        )

    def test_saved_identity_wins_over_an_override(self):
        saved = {"embedding_provider": "alpha", "embedding_model": "model-a"}
        assert LocalVectorDB._resolve_saved_embedding_identity("beta", "model-b", saved) == ("alpha", "model-a")


@pytest.mark.unit
@pytest.mark.embedding
class TestReopenWarnsBeforeDropping:
    def test_overriding_the_model_warns(self, tmp_path, providers, caplog):
        _db(tmp_path).close()

        with caplog.at_level("WARNING"):
            reopened = _db(tmp_path, embedding_model="model-b")
        try:
            assert reopened.embedding_provider.model == "model-a"
            assert "model-b" in caplog.text
            assert "re-ingest" in caplog.text.lower()
        finally:
            reopened.close()

    def test_overriding_the_provider_warns(self, tmp_path, providers, caplog):
        _db(tmp_path).close()

        with caplog.at_level("WARNING"):
            reopened = _db(tmp_path, embedding_provider="beta")
        try:
            assert reopened.embedding_provider.provider_name == "alpha"
            assert "beta" in caplog.text
        finally:
            reopened.close()

    def test_reopening_without_naming_anything_is_silent(self, tmp_path, providers, caplog):
        """The regression guard for the fix itself.

        Constructor defaults used to be concrete strings, so every reopen looked
        like an override of ollama/embeddinggemma. Warning on that would fire on
        every open of every non-Ollama database and train people to ignore it.
        """
        _db(tmp_path).close()

        with caplog.at_level("WARNING"):
            reopened = LocalVectorDB(name="identitydb", base_path=str(tmp_path))
        try:
            assert reopened.embedding_provider.provider_name == "alpha"
            assert reopened.embedding_provider.model == "model-a"
            assert "Ignoring embedding" not in caplog.text
        finally:
            reopened.close()

    def test_restating_the_saved_identity_is_silent(self, tmp_path, providers, caplog):
        """Naming the same values the database already has is not an override."""
        _db(tmp_path).close()

        with caplog.at_level("WARNING"):
            reopened = _db(tmp_path)
        try:
            assert "Ignoring embedding" not in caplog.text
        finally:
            reopened.close()


@pytest.mark.unit
@pytest.mark.embedding
class TestDimensionMismatchRaises:
    def test_narrower_provider_is_refused(self, tmp_path, providers):
        """The index physically cannot hold the vectors, so this one raises."""
        _db(tmp_path, embedding_config={"dimension": 384}).close()

        with pytest.raises(ConfigurationError, match="384"):
            _db(tmp_path, embedding_config={"dimension": 256})

    def test_matching_dimension_opens(self, tmp_path, providers):
        _db(tmp_path, embedding_config={"dimension": 384}).close()

        reopened = _db(tmp_path, embedding_config={"dimension": 384})
        try:
            assert reopened.embedding_dimension == 384
        finally:
            reopened.close()


@pytest.mark.unit
@pytest.mark.embedding
class TestSavedEndpoint:
    """Under an OpenAI-compatible provider the model name no longer pins the server."""

    def _saved(self, tmp_path) -> Dict[str, str]:
        conn = sqlite3.connect(str(Path(tmp_path) / "identitydb.sqlite"))
        try:
            return dict(conn.execute("SELECT key, value FROM config").fetchall())
        finally:
            conn.close()

    def test_endpoint_is_persisted(self, tmp_path, providers):
        _db(tmp_path, embedding_config={"base_url": "http://localhost:8080/v1"}).close()
        assert self._saved(tmp_path)["embedding_base_url"] == "http://localhost:8080/v1"

    def test_provider_without_an_endpoint_saves_empty(self, tmp_path, providers):
        _db(tmp_path).close()
        assert self._saved(tmp_path)["embedding_base_url"] == ""

    def test_serving_the_same_model_from_a_different_host_warns(self, tmp_path, providers, caplog):
        """The case nothing else in the saved config can detect."""
        _db(tmp_path, embedding_config={"base_url": "http://localhost:8080/v1"}).close()

        with caplog.at_level("WARNING"):
            reopened = _db(tmp_path, embedding_config={"base_url": "http://localhost:1234/v1"})
        try:
            assert "localhost:8080" in caplog.text
            assert "localhost:1234" in caplog.text
            # Warned, not overridden: the server may legitimately have moved.
            assert reopened.embedding_provider.base_url == "http://localhost:1234/v1"
        finally:
            reopened.close()

    def test_same_endpoint_is_silent(self, tmp_path, providers, caplog):
        _db(tmp_path, embedding_config={"base_url": "http://localhost:8080/v1"}).close()

        with caplog.at_level("WARNING"):
            reopened = _db(tmp_path, embedding_config={"base_url": "http://localhost:8080/v1"})
        try:
            assert "were embedded via" not in caplog.text
        finally:
            reopened.close()

    def test_database_predating_the_key_is_not_warned_at(self, tmp_path, providers, caplog):
        """An unrecorded endpoint is unknown, not divergent."""
        _db(tmp_path, embedding_config={"base_url": "http://localhost:8080/v1"}).close()

        conn = sqlite3.connect(str(Path(tmp_path) / "identitydb.sqlite"))
        try:
            conn.execute("DELETE FROM config WHERE key = 'embedding_base_url'")
            conn.commit()
        finally:
            conn.close()

        with caplog.at_level("WARNING"):
            reopened = _db(tmp_path, embedding_config={"base_url": "http://elsewhere:9999/v1"})
        try:
            assert "were embedded via" not in caplog.text
        finally:
            reopened.close()


@pytest.mark.unit
@pytest.mark.embedding
class TestOpenDoesNotTouchTheDefaultProvider:
    def test_reopening_never_constructs_the_default_provider(self, tmp_path, providers, monkeypatch):
        """Opening a non-Ollama database must not require a running Ollama.

        The provider used to be built eagerly from the constructor defaults and
        then thrown away once the saved config was read -- but OllamaEmbeddings
        .validate_model() reaches the network, so the throwaway could raise
        OllamaNotFoundError and take the open down with it.
        """
        _db(tmp_path).close()

        requested: List[str] = []
        real = EmbeddingRegistry.create_provider

        def _record(provider: str, model: str, **kwargs: Any):
            requested.append(provider)
            return real(provider, model, **kwargs)

        monkeypatch.setattr(EmbeddingRegistry, "create_provider", staticmethod(_record))

        reopened = LocalVectorDB(name="identitydb", base_path=str(tmp_path))
        try:
            assert requested == ["alpha"], f"expected one provider build, got {requested}"
        finally:
            reopened.close()
