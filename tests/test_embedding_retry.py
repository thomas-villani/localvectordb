"""Retry classification for provider HTTP failures.

Regression cover for a bug that killed a real bulk ingest: the OpenAI provider
rendered any JSON error body into a bare ``RuntimeError``, discarding the status
code. ``EmbeddingProvider._should_retry`` classifies on ``httpx`` types and a
status code, so a **429 matched nothing and was never retried** -- despite
``max_retries=3`` and a docstring promising "automatic retry handling". The
error message itself said "Please try again in 174ms".
"""

from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from localvectordb.embeddings import OpenAIEmbeddings
from localvectordb.exceptions import EmbeddingError, ProviderHTTPError


def _provider(**kw):
    return OpenAIEmbeddings("text-embedding-3-small", api_key="test-key", **kw)


class TestShouldRetryClassification:
    def test_rate_limit_with_status_is_retryable(self):
        p = _provider()
        err = ProviderHTTPError("OpenAI error: Rate limit reached", status_code=429)
        assert p._should_retry(err, attempt=0) is True

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_server_errors_are_retryable(self, status):
        p = _provider()
        assert p._should_retry(ProviderHTTPError("boom", status_code=status), attempt=0) is True

    @pytest.mark.parametrize("status", [400, 401, 404, 422])
    def test_client_errors_are_not_retryable(self, status):
        """A bad key or malformed request must fail fast, not burn the retry budget."""
        p = _provider()
        assert p._should_retry(ProviderHTTPError("nope", status_code=status), attempt=0) is False

    def test_status_free_runtime_error_is_still_not_retryable(self):
        """The old behaviour for genuinely unclassifiable errors is unchanged."""
        p = _provider()
        assert p._should_retry(RuntimeError("OpenAI error: something"), attempt=0) is False

    def test_last_attempt_never_retries(self):
        p = _provider(max_retries=2)
        err = ProviderHTTPError("OpenAI error: Rate limit reached", status_code=429)
        assert p._should_retry(err, attempt=2) is False

    def test_httpx_status_error_path_still_works(self):
        p = _provider()
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        response = httpx.Response(429, request=request)
        assert p._should_retry(httpx.HTTPStatusError("x", request=request, response=response), attempt=0) is True


class TestProviderRaisesClassifiableError:
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_openai_429_carries_its_status_code(self, mock_client_class):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "Rate limit reached for text-embedding-3-small"}}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(ProviderHTTPError) as excinfo:
            await _provider()._embed_single_batch(["hello"])
        assert excinfo.value.status_code == 429
        assert "Rate limit reached" in str(excinfo.value)

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_error_is_still_a_runtime_error(self, mock_client_class):
        """Back-compat: callers catching RuntimeError/EmbeddingError keep working."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        with pytest.raises(RuntimeError, match="OpenAI error: Invalid API key"):
            await _provider()._embed_single_batch(["hello"])
        assert issubclass(ProviderHTTPError, EmbeddingError)
        assert issubclass(ProviderHTTPError, RuntimeError)


class TestRetryActuallyHappens:
    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_retries_a_rate_limit_then_succeeds(self, mock_client_class):
        """The end-to-end behaviour the bug denied: a 429 followed by a success."""
        limited = Mock()
        limited.is_success = False
        limited.status_code = 429
        limited.json.return_value = {"error": {"message": "Rate limit reached"}}
        limited.raise_for_status = Mock()

        ok = Mock()
        ok.is_success = True
        ok.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}

        mock_client = Mock()
        mock_client.post = AsyncMock(side_effect=[limited, ok])
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = _provider(max_retries=3, retry_delay=0.0)
        result = await provider.embed_batch(["hello"])
        assert mock_client.post.await_count == 2  # retried rather than surfacing the 429
        assert result.shape[0] == 1
