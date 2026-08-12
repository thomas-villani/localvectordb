"""The one adapter for every server speaking OpenAI's /v1/embeddings format.

llama.cpp, LM Studio, vLLM, text-embeddings-inference and LocalAI differ only in
their address, so they share a provider. What they do *not* share with hosted
OpenAI is the assumptions: no API key, an open-ended model set, and a family of
setup mistakes (server not started with embedding support, a missing '/v1') whose
raw HTTP errors say nothing useful. Most of these tests pin the diagnostics.
"""

from typing import Any, List
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
import pytest

from localvectordb.embeddings import EmbeddingRegistry, OpenAICompatibleEmbeddings
from localvectordb.exceptions import EmbeddingError, ProviderHTTPError

LLAMA_CPP = "http://localhost:8080/v1"


def _provider(**kwargs: Any) -> OpenAICompatibleEmbeddings:
    kwargs.setdefault("base_url", LLAMA_CPP)
    kwargs.setdefault("dimension", 3)
    return OpenAICompatibleEmbeddings("nomic-embed-text", **kwargs)


def _response(status: int, json_body: Any = None, text: str = "") -> httpx.Response:
    request = httpx.Request("POST", f"{LLAMA_CPP}/embeddings")
    if json_body is not None:
        return httpx.Response(status, json=json_body, request=request)
    return httpx.Response(status, text=text, request=request)


@pytest.mark.unit
@pytest.mark.embedding
class TestConstruction:
    def test_base_url_is_required(self):
        """Guessing an endpoint would silently embed against the wrong server."""
        with pytest.raises(ValueError, match="base_url is required"):
            OpenAICompatibleEmbeddings("nomic-embed-text")

    def test_missing_base_url_names_the_common_servers(self):
        with pytest.raises(ValueError) as exc:
            OpenAICompatibleEmbeddings("nomic-embed-text")
        message = str(exc.value)
        assert "llama.cpp" in message and "LM Studio" in message and "vLLM" in message

    def test_no_api_key_is_fine(self):
        """The defining difference from OpenAIEmbeddings, which raises here."""
        assert _provider().api_key is None

    def test_api_key_read_from_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-local")
        assert _provider().api_key == "sk-local"

    def test_endpoint_appends_embeddings_once(self):
        assert _provider(base_url="http://localhost:8080/v1/")._endpoint == f"{LLAMA_CPP}/embeddings"

    def test_disagreeing_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="disagree"):
            OpenAICompatibleEmbeddings("nomic-embed-text", base_url=LLAMA_CPP, dimension=768, requested_dimensions=256)

    def test_any_model_name_validates(self):
        """The server decides what it serves; a bad name surfaces on first use."""
        provider = OpenAICompatibleEmbeddings("whatever-gguf-Q4_K_M", base_url=LLAMA_CPP, dimension=3)
        assert provider.validate_model() is True

    def test_registered_under_openai_compatible(self):
        assert EmbeddingRegistry.get("openai_compatible") is OpenAICompatibleEmbeddings


@pytest.mark.unit
@pytest.mark.embedding
class TestHeaders:
    def test_no_authorization_header_without_a_key(self):
        """Some local servers reject a malformed bearer, so send none at all."""
        assert "Authorization" not in _provider()._headers()

    def test_authorization_header_with_a_key(self):
        assert _provider(api_key="sk-local")._headers()["Authorization"] == "Bearer sk-local"


@pytest.mark.unit
@pytest.mark.embedding
class TestDimensionResolution:
    def test_declared_dimension_makes_no_network_call(self):
        with patch("httpx.Client") as client_class:
            assert _provider(dimension=768).get_dimension() == 768
        client_class.assert_not_called()

    def test_requested_dimensions_wins_without_a_probe(self):
        provider = OpenAICompatibleEmbeddings("m", base_url=LLAMA_CPP, requested_dimensions=256)
        with patch("httpx.Client") as client_class:
            assert provider.get_dimension() == 256
        client_class.assert_not_called()

    def test_unknown_dimension_is_probed(self):
        provider = OpenAICompatibleEmbeddings("m", base_url=LLAMA_CPP)
        client = Mock()
        client.post = Mock(return_value=_response(200, {"data": [{"embedding": [0.0] * 768}]}))

        with patch("httpx.Client") as client_class:
            client_class.return_value.__enter__ = Mock(return_value=client)
            client_class.return_value.__exit__ = Mock(return_value=None)
            assert provider.get_dimension() == 768

        assert client.post.call_args.args[0] == f"{LLAMA_CPP}/embeddings"

    def test_probe_result_is_cached(self):
        provider = OpenAICompatibleEmbeddings("m", base_url=LLAMA_CPP)
        client = Mock()
        client.post = Mock(return_value=_response(200, {"data": [{"embedding": [0.0] * 768}]}))

        with patch("httpx.Client") as client_class:
            client_class.return_value.__enter__ = Mock(return_value=client)
            client_class.return_value.__exit__ = Mock(return_value=None)
            provider.get_dimension()
            provider.get_dimension()

        assert client.post.call_count == 1

    def test_probe_against_a_non_embedding_model_explains_itself(self):
        provider = OpenAICompatibleEmbeddings("m", base_url=LLAMA_CPP)
        client = Mock()
        client.post = Mock(return_value=_response(200, {"data": []}))

        with patch("httpx.Client") as client_class:
            client_class.return_value.__enter__ = Mock(return_value=client)
            client_class.return_value.__exit__ = Mock(return_value=None)
            with pytest.raises(EmbeddingError, match="embedding model"):
                provider.get_dimension()


@pytest.mark.unit
@pytest.mark.embedding
class TestErrorDiagnostics:
    """The actual value of the adapter: the failures are setup mistakes."""

    def test_404_points_at_the_v1_suffix(self):
        with pytest.raises(ProviderHTTPError, match="/v1") as exc:
            _provider()._raise_for_response(_response(404, {"error": "not found"}))
        assert exc.value.status_code == 404

    def test_400_points_at_embedding_support(self):
        with pytest.raises(ProviderHTTPError, match="embedding") as exc:
            _provider()._raise_for_response(_response(400, {"error": {"message": "model does not support embeddings"}}))
        assert "model does not support embeddings" in str(exc.value)
        assert exc.value.status_code == 400

    def test_422_is_treated_like_400(self):
        """vLLM validates with FastAPI and answers 422 where llama.cpp says 400."""
        with pytest.raises(ProviderHTTPError, match="embedding"):
            _provider()._raise_for_response(_response(422, {"detail": "bad input"}))

    def test_non_json_error_body_still_reports(self):
        with pytest.raises(ProviderHTTPError, match="502"):
            _provider()._raise_for_response(_response(502, text="upstream gone"))

    def test_connection_failure_names_the_likely_causes(self):
        error = _provider()._connection_error(httpx.ConnectError("refused"))
        message = str(error)
        assert "--embedding" in message
        assert "/v1" in message
        assert LLAMA_CPP in message


@pytest.mark.unit
@pytest.mark.embedding
class TestEmbedding:
    def _patched_client(self, response: httpx.Response):
        client = Mock()
        client.post = AsyncMock(return_value=response)
        patcher = patch("httpx.AsyncClient")
        client_class = patcher.start()
        client_class.return_value.__aenter__ = AsyncMock(return_value=client)
        client_class.return_value.__aexit__ = AsyncMock(return_value=None)
        return patcher, client

    @pytest.mark.asyncio
    async def test_posts_to_the_configured_endpoint(self):
        patcher, client = self._patched_client(_response(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]}))
        try:
            result = await _provider().embed_batch(["hello"])
        finally:
            patcher.stop()

        assert result.shape == (1, 3)
        assert client.post.call_args.args[0] == f"{LLAMA_CPP}/embeddings"
        assert client.post.call_args.kwargs["json"]["model"] == "nomic-embed-text"

    @pytest.mark.asyncio
    async def test_requested_dimensions_reaches_the_payload(self):
        patcher, client = self._patched_client(_response(200, {"data": [{"embedding": [0.1, 0.2]}]}))
        try:
            provider = OpenAICompatibleEmbeddings("m", base_url=LLAMA_CPP, requested_dimensions=2)
            await provider.embed_batch(["hello"])
        finally:
            patcher.stop()

        assert client.post.call_args.kwargs["json"]["dimensions"] == 2

    @pytest.mark.asyncio
    async def test_normalize_produces_unit_vectors(self):
        patcher, _ = self._patched_client(_response(200, {"data": [{"embedding": [3.0, 4.0, 0.0]}]}))
        try:
            result = await _provider(normalize=True).embed_batch(["hello"])
        finally:
            patcher.stop()

        assert np.isclose(np.linalg.norm(result[0]), 1.0)

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self):
        provider = _provider()
        assert await provider._embed_single_batch([]) == []

    @pytest.mark.asyncio
    async def test_batch_size_is_configurable(self):
        assert _provider().max_batch_size == 64
        assert _provider(max_batch_size=8).max_batch_size == 8


@pytest.mark.unit
@pytest.mark.embedding
class TestProviderIdentity:
    def test_provider_name_is_recorded_as_openai_compatible(self):
        """Databases save this string; changing it orphans existing databases."""
        assert _provider().provider_name == "openai_compatible"

    def test_base_url_is_exposed_for_persistence(self):
        """_save_config reads base_url off the provider to record provenance."""
        assert _provider().base_url == LLAMA_CPP


@pytest.mark.unit
@pytest.mark.embedding
class TestServerVariants:
    """Each runtime's default address must work through the same adapter."""

    @pytest.mark.parametrize(
        "base_url",
        [
            "http://localhost:8080/v1",  # llama.cpp / text-embeddings-inference
            "http://localhost:1234/v1",  # LM Studio
            "http://localhost:8000/v1",  # vLLM
            "http://127.0.0.1:11434/v1",  # Ollama's OpenAI shim
        ],
    )
    def test_endpoint_for_each_server(self, base_url: str):
        provider = OpenAICompatibleEmbeddings("m", base_url=base_url, dimension=4)
        assert provider._endpoint == f"{base_url}/embeddings"


@pytest.mark.unit
@pytest.mark.embedding
class TestListedForDiscovery:
    def test_appears_in_available_providers(self):
        names: List[str] = list(EmbeddingRegistry._providers.keys())
        assert "openai_compatible" in names
