"""
Tests for localvectordb.embeddings module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import numpy as np
import pytest

from localvectordb.embeddings import (
    EmbeddingProvider,
    EmbeddingRegistry,
    GoogleEmbeddings,
    HuggingFaceInferenceEmbeddings,
    HuggingFaceLocalEmbeddings,
    JinaEmbeddings,
    MockEmbeddings,
    OllamaEmbeddings,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    create_embedding_provider,
    embed_texts,
    embed_texts_sync,
    list_providers,
)
from localvectordb.exceptions import EmbeddingError, OllamaNotFoundError


@pytest.mark.unit
@pytest.mark.embedding
class TestEmbeddingProvider:
    """Test abstract EmbeddingProvider class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            EmbeddingProvider("test-model")

    def test_abstract_methods_defined(self):
        """Test that abstract methods are defined."""
        abstract_methods = EmbeddingProvider.__abstractmethods__
        expected_methods = {"get_dimension", "validate_model", "provider_name", "max_batch_size", "_embed_single_batch"}
        assert abstract_methods == expected_methods


@pytest.mark.unit
@pytest.mark.embedding
class TestMockEmbeddings:
    """Test MockEmbeddings provider."""

    def test_create_mock_provider(self):
        """Test creating mock embedding provider."""
        provider = MockEmbeddings("test-model", dimension=256)
        assert provider.model == "test-model"
        assert provider.provider_name == "mock"
        assert provider.max_batch_size == 1000
        assert provider.get_dimension() == 256

    def test_validate_model(self):
        """Test model validation always returns True."""
        provider = MockEmbeddings("test-model")
        assert provider.validate_model() is True

    @pytest.mark.asyncio
    async def test_embed_batch_async(self):
        """Test async embedding generation."""
        provider = MockEmbeddings("test-model", dimension=384)
        texts = ["hello world", "test text"]

        embeddings = await provider.embed_batch(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

        # Test deterministic behavior
        embeddings2 = await provider.embed_batch(texts)
        np.testing.assert_array_equal(embeddings, embeddings2)

    def test_embed_sync(self):
        """Test synchronous embedding generation."""
        provider = MockEmbeddings("test-model", dimension=384)
        texts = ["hello world", "test text"]

        embeddings = provider.embed_sync(texts)

        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """Test embedding empty list."""
        provider = MockEmbeddings("test-model", dimension=384)

        embeddings = await provider.embed_batch([])

        assert embeddings.shape == (0, 384)

    def test_deterministic_embeddings(self):
        """Test that embeddings are deterministic based on text."""
        provider = MockEmbeddings("test-model", dimension=384)

        # Same text should produce same embedding
        emb1 = provider.embed_sync(["hello"])
        emb2 = provider.embed_sync(["hello"])
        np.testing.assert_array_equal(emb1, emb2)

        # Different text should produce different embeddings
        emb3 = provider.embed_sync(["world"])
        assert not np.array_equal(emb1, emb3)


@pytest.mark.unit
@pytest.mark.embedding
@pytest.mark.network
class TestOllamaEmbeddings:
    """Test OllamaEmbeddings provider."""

    def test_create_ollama_provider(self):
        """Test creating Ollama embedding provider."""
        provider = OllamaEmbeddings("nomic-embed-text")
        assert provider.model == "nomic-embed-text"
        assert provider.base_url == "http://localhost:11434"
        assert provider.provider_name == "ollama"
        assert provider.max_batch_size == 64

    def test_create_with_custom_url(self):
        """Test creating with custom base URL."""
        provider = OllamaEmbeddings("test-model", base_url="http://custom:8080/")
        assert provider.base_url == "http://custom:8080"

    @patch("httpx.Client")
    def test_validate_model_success(self, mock_client_class):
        """Test successful model validation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "llama2:7b"}]}
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        provider = OllamaEmbeddings("nomic-embed-text")
        result = provider.validate_model()

        assert result is True

    @patch("httpx.Client")
    def test_validate_model_not_found(self, mock_client_class):
        """Test model validation when model not found."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "llama2:7b"}]}
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        provider = OllamaEmbeddings("missing-model")
        result = provider.validate_model()

        assert result is False

    @patch("httpx.Client")
    def test_validate_model_connection_error(self, mock_client_class):
        """Test model validation with connection error."""
        mock_client = Mock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")
        mock_client_class.return_value.__enter__.return_value = mock_client

        provider = OllamaEmbeddings("test-model")
        with pytest.raises(OllamaNotFoundError):
            provider.validate_model()

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OllamaEmbeddings("test-model")
        provider._dimension = 3  # Set dimension to avoid test call

        texts = ["hello", "world"]
        embeddings = await provider.embed_batch(texts)

        assert embeddings.shape == (2, 3)
        np.testing.assert_array_equal(embeddings[0], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(embeddings[1], np.asarray([0.4, 0.5, 0.6], dtype=np.float32))

        mock_client.post.assert_called_once_with(
            "http://localhost:11434/api/embed",
            json={"model": "test-model", "input": texts, "truncate": True},
            timeout=300.0,
        )

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_error_response(self, mock_client_class):
        """Test embedding with error response."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Model not found"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response  # AsyncMock(return_value=mock_response)

        # Fix: Properly mock the async context manager
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = async_context_manager

        provider = OllamaEmbeddings("test-model")
        provider._dimension = 384

        with pytest.raises(RuntimeError, match="Ollama error: Model not found"):
            await provider.embed_batch(["test"])

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_connection_error(self, mock_client_class):
        """Test embedding with connection error."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Model not found"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection failed")
        mock_client.post.return_value = mock_response  # AsyncMock(return_value=mock_response)

        # Fix: Properly mock the async context manager
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = async_context_manager

        provider = OllamaEmbeddings("test-model")
        provider._dimension = 384

        with pytest.raises(EmbeddingError):
            await provider.embed_batch(["test"])

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_batching(self, mock_client_class):
        """Test that large inputs are batched correctly."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OllamaEmbeddings("test-model")
        provider._dimension = 3

        # Create more texts than batch size
        texts = [f"text {i}" for i in range(70)]  # More than max_batch_size of 64

        await provider.embed_batch(texts, batch_size=64)

        # Should make multiple API calls
        assert mock_client.post.call_count == 2

    @patch.object(OllamaEmbeddings, "_get_model_dimension_sync")
    def test_get_dimension_calls_embed_batch(self, mock_get_dimension):
        """Test get_dimension makes test call to determine dimension."""
        mock_get_dimension.return_value = 4

        provider = OllamaEmbeddings("test-model")
        dimension = provider.get_dimension()

        assert dimension == 4
        assert provider._dimension == 4

        # Should call sync method
        mock_get_dimension.assert_called_once()


@pytest.mark.unit
@pytest.mark.embedding
@pytest.mark.network
class TestOpenAIEmbeddings:
    """Test OpenAIEmbeddings provider."""

    def test_create_openai_provider(self):
        """Test creating OpenAI embedding provider."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")
        assert provider.model == "text-embedding-ada-002"
        assert provider.api_key == "test-key"
        assert provider.provider_name == "openai"
        assert provider.max_batch_size == 1000

    def test_create_without_api_key(self):
        """Test creating without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIEmbeddings("text-embedding-ada-002")

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    def test_api_key_from_environment(self):
        """Test getting API key from environment."""
        provider = OpenAIEmbeddings("text-embedding-ada-002")
        assert provider.api_key == "env-key"

    def test_validate_known_model(self):
        """Test validation of known models."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test")
        assert provider.validate_model() is True

        provider = OpenAIEmbeddings("text-embedding-3-small", api_key="test")
        assert provider.validate_model() is True

        with pytest.raises(ValueError):
            OpenAIEmbeddings("unknown-model", api_key="test")

    def test_get_dimension_known_model(self):
        """Test getting dimension for known models."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test")
        assert provider.get_dimension() == 1536

        provider = OpenAIEmbeddings("text-embedding-3-large", api_key="test")
        assert provider.get_dimension() == 3072

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful OpenAI embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")
        provider._dimension = 3

        texts = ["hello", "world"]
        embeddings = await provider.embed_batch(texts)

        assert embeddings.shape == (2, 3)
        np.testing.assert_array_equal(embeddings[0], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(embeddings[1], np.asarray([0.4, 0.5, 0.6], dtype=np.float32))

        # Check API call
        mock_client.post.assert_called_once_with(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
            json={"model": "text-embedding-ada-002", "input": texts},
            timeout=provider.timeout,
        )

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_openai_error(self, mock_client_class):
        """Test OpenAI API error handling when response contains error JSON."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")

        with pytest.raises(RuntimeError, match="OpenAI error: Invalid API key"):
            await provider.embed_batch(["test"])

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_openai_error_token_limit(self, mock_client_class):
        """Test OpenAI API error handling preserves token limit error message."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.json.return_value = {
            "error": {
                "message": "This model's maximum context length is 8192 tokens, however you requested 12431 tokens",
                "type": "invalid_request_error",
                "code": "context_length_exceeded",
            }
        }
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")

        with pytest.raises(RuntimeError, match="maximum context length is 8192 tokens"):
            await provider.embed_batch(["test"])

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_openai_error_fallback_to_http_error(self, mock_client_class):
        """Test OpenAI API error falls back to HTTP error when JSON parsing fails."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.is_success = False
        mock_response.status_code = 400  # Must be an int for _should_retry check
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock(
            side_effect=httpx.HTTPStatusError("Bad Request", request=Mock(), response=mock_response)
        )
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")

        # 400 errors are not retried, so it should raise EmbeddingError
        from localvectordb.exceptions import EmbeddingError

        with pytest.raises(EmbeddingError, match="Bad Request"):
            await provider.embed_batch(["test"])

    def test_max_input_tokens_property(self):
        """Test that OpenAI provider has max_input_tokens property."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")
        assert provider.max_input_tokens == 8191

        provider2 = OpenAIEmbeddings("text-embedding-3-small", api_key="test-key")
        assert provider2.max_input_tokens == 8191

    def test_truncate_to_token_limit(self):
        """Test text truncation to fit within token limit."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")

        # Create a simple text
        text = "Hello world, this is a test of truncation."

        # Truncate to a very small limit
        truncated = provider._truncate_to_token_limit(text, 5)

        # Should be shorter than original
        tokenizer = provider._get_tokenizer()
        assert len(tokenizer.encode(truncated)) <= 5

    def test_validate_and_truncate_texts(self):
        """Test that oversized texts are truncated with warning."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")

        # Create a very long text that exceeds max_input_tokens
        # Generate text with ~10000 tokens (each word is roughly 1 token)
        long_text = " ".join(["word"] * 10000)
        short_text = "short text"

        texts = [short_text, long_text]

        validated = provider._validate_and_truncate_texts(texts)

        # Short text should be unchanged
        assert validated[0] == short_text

        # Long text should be truncated
        tokenizer = provider._get_tokenizer()
        assert len(tokenizer.encode(validated[1])) <= provider.max_input_tokens


@pytest.mark.unit
@pytest.mark.embedding
@pytest.mark.network
class TestJinaEmbeddings:
    """Test JinaEmbeddings provider."""

    def test_create_jina_provider(self):
        """Test creating JinaAI embedding provider."""
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test-key", task="text-matching")
        assert provider.model == "jina-embeddings-v4"
        assert provider.api_key == "test-key"
        assert provider.provider_name == "jina"
        assert provider.max_batch_size == 512

    def test_create_without_api_key(self):
        """Test creating without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key is required"):
                JinaEmbeddings("jina-embeddings-v4", task="text-matching")

    @patch.dict("os.environ", {"JINA_API_KEY": "env-key"})
    def test_api_key_from_environment(self):
        """Test getting API key from environment."""
        provider = JinaEmbeddings("jina-embeddings-v4", task="text-matching")
        assert provider.api_key == "env-key"

    def test_api_key_from_env_var_reference(self):
        """Test getting API key from custom environment variable."""
        with patch.dict("os.environ", {"CUSTOM_JINA_KEY": "custom-key"}):
            provider = JinaEmbeddings("jina-embeddings-v4", api_key="$CUSTOM_JINA_KEY")
            assert provider.api_key == "custom-key"

    def test_known_model_dimensions(self):
        """Test getting dimension for known models."""
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test", task="text-matching")
        assert provider.get_dimension() == 2048

        provider = JinaEmbeddings("jina-embeddings-v3", api_key="test")
        assert provider.get_dimension() == 1024

        provider = JinaEmbeddings("jina-code-embeddings-1.5b", api_key="test")
        assert provider.get_dimension() == 1536

    def test_requested_dimensions_override(self):
        """Test that requested_dimensions overrides known dimensions."""
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test", requested_dimensions=512)
        assert provider.get_dimension() == 512

    def test_validate_model_task_validation(self):
        """Test task validation for different models."""
        # Valid task for v4
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test", task="retrieval.query")
        assert provider.task == "retrieval.query"

        # Valid task for code model
        provider = JinaEmbeddings("jina-code-embeddings-1.5b", api_key="test", task="code2code.query")
        assert provider.task == "code2code.query"

        # Invalid task should raise error
        with pytest.raises(ValueError, match="`task` must be one of"):
            JinaEmbeddings("jina-embeddings-v4", api_key="test", task="invalid_task")

    def test_validate_model_always_returns_true(self):
        """Test model validation (currently simplified to return True)."""
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test")
        assert provider.validate_model() is True

    @patch("httpx.Client")
    def test_get_dimension_api_probe(self, mock_client_class):
        """Test dimension detection via API probe for unknown models."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]}
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        # Unknown model should probe API
        provider = JinaEmbeddings("unknown-model", api_key="test-key")
        provider._dimension = None  # Force dimension detection
        dimension = provider._get_model_dimension_api()

        assert dimension == 5
        mock_client.post.assert_called_once()

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}, {"embedding": [0.4, 0.5, 0.6]}]}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test-key")
        provider._dimension = 3  # Set dimension to avoid test call

        texts = ["hello", "world"]
        embeddings = await provider.embed_batch(texts)

        assert embeddings.shape == (2, 3)
        np.testing.assert_array_equal(embeddings[0], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(embeddings[1], np.asarray([0.4, 0.5, 0.6], dtype=np.float32))

        # Check API call
        expected_payload = {
            "model": "jina-embeddings-v4",
            "input": texts,
            "task": "text-matching",
        }
        mock_client.post.assert_called_once_with(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=expected_payload,
            timeout=provider.timeout,
        )

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_with_task_and_dimensions(self, mock_client_class):
        """Test embedding with task and dimensions configuration."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaEmbeddings(
            "jina-embeddings-v4",
            api_key="test-key",
            task="retrieval.passage",
            requested_dimensions=512,
            truncate=True,
            late_chunking=True,
        )
        provider._dimension = 2

        texts = ["test"]
        await provider.embed_batch(texts)

        # Check that advanced options are included in payload
        expected_payload = {
            "model": "jina-embeddings-v4",
            "input": texts,
            "task": "retrieval.passage",
            "dimensions": 512,
            "truncate": True,
            "late_chunking": True,
        }
        mock_client.post.assert_called_once_with(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            json=expected_payload,
            timeout=provider.timeout,
        )

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_error_response(self, mock_client_class):
        """Test embedding with error response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test-key")
        provider._dimension = 384

        with pytest.raises(RuntimeError, match="Jina API error: Invalid API key"):
            await provider.embed_batch(["test"])

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, mock_client_class):
        """Test embedding empty list returns single empty embedding."""
        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test-key")
        provider._dimension = 384

        embeddings = await provider.embed_batch([])
        assert len(embeddings) == 0


@pytest.mark.unit
@pytest.mark.embedding
@pytest.mark.network
class TestGoogleEmbeddings:
    """Test GoogleEmbeddings provider."""

    def test_create_google_provider(self):
        """Test creating Google AI embedding provider."""
        provider = GoogleEmbeddings(api_key="test-key")
        assert provider.model == "gemini-embedding-001"
        assert provider.api_key == "test-key"
        assert provider.provider_name == "google"
        assert provider.max_batch_size == 200

    def test_create_without_api_key(self):
        """Test creating without API key raises error."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google AI.*API key is required"):
                GoogleEmbeddings()

    @patch.dict("os.environ", {"GEMINI_API_KEY": "env-key"})
    def test_api_key_from_gemini_env(self):
        """Test getting API key from GEMINI_API_KEY environment."""
        provider = GoogleEmbeddings()
        assert provider.api_key == "env-key"

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "google-key"})
    def test_api_key_from_google_env(self):
        """Test getting API key from GOOGLE_API_KEY environment."""
        provider = GoogleEmbeddings()
        assert provider.api_key == "google-key"

    def test_custom_configuration(self):
        """Test custom configuration options."""
        provider = GoogleEmbeddings(
            model="gemini-embedding-001",
            api_key="test-key",
            task_type="retrieval_document",
            requested_dimensions=1536,
            normalize=False,
            base_url="https://custom.googleapis.com",
        )
        assert provider.model == "gemini-embedding-001"
        assert provider.task_type == "RETRIEVAL_DOCUMENT"
        assert provider.requested_dimensions == 1536
        assert provider.normalize is False
        assert provider.base_url == "https://custom.googleapis.com"

    def test_requested_dimensions_sets_dimension(self):
        """Test that requested_dimensions immediately sets _dimension."""
        provider = GoogleEmbeddings(api_key="test", requested_dimensions=1024)
        assert provider._dimension == 1024
        assert provider.get_dimension() == 1024

    @patch("httpx.Client")
    def test_validate_model_success(self, mock_client_class):
        """Test successful model validation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        provider = GoogleEmbeddings(api_key="test-key")
        result = provider.validate_model()

        assert result is True
        mock_client.get.assert_called_once_with(
            f"{provider.base_url}/models/{provider.model}",
            headers={"x-goog-api-key": "test-key"},
            timeout=provider.timeout,
        )

    @patch("httpx.Client")
    def test_validate_model_not_found(self, mock_client_class):
        """Test model validation when model not found."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        provider = GoogleEmbeddings(api_key="test-key")
        result = provider.validate_model()

        assert result is False

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = GoogleEmbeddings(api_key="test-key", requested_dimensions=3, normalize=False)

        texts = ["hello", "world"]
        embeddings = await provider.embed_batch(texts)

        assert embeddings.shape == (2, 3)
        np.testing.assert_array_equal(embeddings[0], np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
        np.testing.assert_array_equal(embeddings[1], np.asarray([0.4, 0.5, 0.6], dtype=np.float32))

        # Check API call structure
        expected_contents = [{"parts": [{"text": "hello"}]}, {"parts": [{"text": "world"}]}]
        expected_payload = {
            "contents": expected_contents,
            "embedding_config": {"task_type": "SEMANTIC_SIMILARITY", "output_dimensionality": 3},
        }
        mock_client.post.assert_called_once_with(
            f"{provider.base_url}/models/{provider.model}:embedContent",
            headers={"x-goog-api-key": "test-key", "Content-Type": "application/json"},
            json=expected_payload,
            timeout=provider.timeout,
        )

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_with_normalization(self, mock_client_class):
        """Test embedding with vector normalization."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [{"values": [3.0, 4.0]}]}  # Magnitude = 5.0
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = GoogleEmbeddings(api_key="test-key", normalize=True, requested_dimensions=2)

        texts = ["test"]
        embeddings = await provider.embed_batch(texts)

        # Should be normalized to unit vector
        expected = np.array([[0.6, 0.8]], dtype=np.float32)  # [3/5, 4/5]
        np.testing.assert_array_almost_equal(embeddings, expected, decimal=6)

    @patch("httpx.AsyncClient")
    @pytest.mark.asyncio
    async def test_embed_batch_error_response(self, mock_client_class):
        """Test embedding with malformed response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = GoogleEmbeddings(api_key="test-key", requested_dimensions=3)

        with pytest.raises(RuntimeError, match="Unexpected response from Google AI embeddings API"):
            await provider.embed_batch(["test"])


@pytest.mark.unit
@pytest.mark.embedding
class TestEmbeddingRegistry:
    """Test EmbeddingRegistry class."""

    def test_register_provider(self):
        """Test registering a new provider."""

        class CustomProvider(EmbeddingProvider):
            @property
            def provider_name(self):
                return "custom"

            @property
            def max_batch_size(self):
                return 100

            async def embed_batch(self, texts, batch_size=None):
                return np.array([])

            def get_dimension(self):
                return 384

            def validate_model(self):
                return True

        EmbeddingRegistry.register("custom", CustomProvider)

        assert "custom" in EmbeddingRegistry._providers
        assert EmbeddingRegistry._providers["custom"] == CustomProvider

    def test_get_provider(self):
        """Test getting a registered provider."""
        provider_class = EmbeddingRegistry.get("mock")
        assert provider_class == MockEmbeddings

    def test_get_unknown_provider(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown embedding provider: unknown"):
            EmbeddingRegistry.get("unknown")

    def test_create_provider(self):
        """Test creating provider instance."""
        provider = EmbeddingRegistry.create_provider("mock", "test-model", dimension=256)
        assert isinstance(provider, MockEmbeddings)
        assert provider.model == "test-model"
        assert provider.get_dimension() == 256

    def test_list_providers(self):
        """Test listing available providers."""
        providers = EmbeddingRegistry.list()
        assert "mock" in providers
        assert "ollama" in providers
        assert "openai" in providers
        assert "jina" in providers
        assert "google" in providers
        assert isinstance(providers, list)

    def test_builtin_providers_registered(self):
        """Test that built-in providers are auto-registered."""
        assert "ollama" in EmbeddingRegistry._providers
        assert "openai" in EmbeddingRegistry._providers
        assert "mock" in EmbeddingRegistry._providers
        assert "jina" in EmbeddingRegistry._providers
        assert "google" in EmbeddingRegistry._providers

    @patch("importlib.metadata.entry_points")
    def test_discover_plugins_importlib_metadata(self, mock_entry_points):
        """Test plugin discovery with importlib.metadata."""
        # Mock entry points
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = MockEmbeddings
        mock_entry_points.return_value = [mock_ep]

        # Reset discovery state
        EmbeddingRegistry._plugins_discovered = False
        EmbeddingRegistry._discover_plugins()

        assert "test_plugin" in EmbeddingRegistry._providers

    def test_refresh_plugins(self):
        """Test refreshing plugin discovery."""
        # Set as discovered
        EmbeddingRegistry._plugins_discovered = True

        # Refresh should reset state
        EmbeddingRegistry.refresh_plugins()
        assert EmbeddingRegistry._plugins_discovered is True  # Will be set again after discovery


@pytest.mark.unit
@pytest.mark.embedding
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_embedding_provider(self):
        """Test create_embedding_provider function."""
        provider = create_embedding_provider("mock", "test-model", dimension=256)
        assert isinstance(provider, MockEmbeddings)
        assert provider.model == "test-model"
        assert provider.get_dimension() == 256

    def test_list_providers_function(self):
        """Test list_providers function."""
        providers = list_providers()
        assert "mock" in providers
        assert "ollama" in providers
        assert "openai" in providers
        assert "jina" in providers
        assert "google" in providers

    @pytest.mark.asyncio
    async def test_embed_texts_async(self):
        """Test async embed_texts function."""
        texts = ["hello", "world"]
        embeddings = await embed_texts(texts, "mock", "test-model", dimension=384)
        assert embeddings.shape == (2, 384)

    def test_embed_texts_sync(self):
        """Test synchronous embed_texts_sync function."""
        texts = ["hello", "world"]
        embeddings = embed_texts_sync(texts, "mock", "test-model", dimension=384)
        assert embeddings.shape == (2, 384)

    @pytest.mark.asyncio
    async def test_embed_texts_with_batch_size(self):
        """Test embedding with custom batch size."""
        texts = ["hello", "world", "test"]
        embeddings = await embed_texts(texts, "mock", "test-model", batch_size=2, dimension=384)
        assert embeddings.shape == (3, 384)


@pytest.mark.unit
@pytest.mark.embedding
class TestEmbeddingSyncWrapper:
    """Test synchronous wrapper for async embedding methods."""

    def test_embed_sync_no_event_loop(self):
        """Test sync wrapper when no event loop exists."""
        provider = MockEmbeddings("test-model", dimension=384)

        # Ensure no event loop exists
        try:
            asyncio.get_event_loop().close()
        except Exception:
            pass

        embeddings = provider.embed_sync(["hello", "world"])
        assert embeddings.shape == (2, 384)

    def test_embed_sync_with_running_loop(self):
        """Test sync wrapper with running event loop raises RuntimeError."""
        provider = MockEmbeddings("test-model", dimension=384)

        # Create a running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def test_within_loop():
            # This should raise a RuntimeError - either because it detects the async context
            # or because asyncio.run() cannot be called from a running loop
            with pytest.raises(RuntimeError):
                provider.embed_sync(["hello"])

        try:
            loop.run_until_complete(test_within_loop())
        finally:
            loop.close()

    @patch("asyncio.run")
    def test_embed_sync_with_stopped_loop(self, mock_asyncio_run):
        """Test sync wrapper with no running event loop uses asyncio.run."""
        provider = MockEmbeddings("test-model", dimension=384)
        expected_result = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_asyncio_run.return_value = expected_result

        result = provider.embed_sync(["hello", "world"])

        assert mock_asyncio_run.called
        assert np.array_equal(result, expected_result)


# =========================================================================
# New embedding provider tests
# =========================================================================


@pytest.mark.unit
@pytest.mark.embedding
class TestSentenceTransformerEmbeddings:
    """Test SentenceTransformerEmbeddings provider."""

    def test_init(self):
        provider = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        assert provider.model == "all-MiniLM-L6-v2"
        assert provider.provider_name == "sentence_transformers"
        assert provider.max_batch_size == 256

    def test_init_with_requested_dimensions(self):
        provider = SentenceTransformerEmbeddings("all-MiniLM-L6-v2", requested_dimensions=128)
        assert provider.get_dimension() == 128

    def test_load_model_import_error(self):
        provider = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            assert provider.validate_model() is False

    @pytest.mark.asyncio
    async def test_embed_with_mock_model(self):
        provider = SentenceTransformerEmbeddings("test-model")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        provider._model = mock_model

        embeddings = await provider._embed_single_batch(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    @pytest.mark.asyncio
    async def test_embed_with_dimension_truncation(self):
        provider = SentenceTransformerEmbeddings("test-model", requested_dimensions=128, normalize=True)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 384).astype(np.float32)
        provider._model = mock_model

        embeddings = await provider._embed_single_batch(["hello", "world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 128
        # Check normalization
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            assert abs(norm - 1.0) < 1e-5


@pytest.mark.unit
@pytest.mark.embedding
class TestHuggingFaceInferenceEmbeddings:
    """Test HuggingFaceInferenceEmbeddings provider."""

    def test_init(self):
        provider = HuggingFaceInferenceEmbeddings("BAAI/bge-small-en-v1.5")
        assert provider.model == "BAAI/bge-small-en-v1.5"
        assert provider.provider_name == "huggingface"
        assert provider.max_batch_size == 128

    def test_env_var_api_key(self):
        with patch.dict("os.environ", {"HF_TOKEN": "test-hf-token"}):
            provider = HuggingFaceInferenceEmbeddings("test-model")
            assert provider.api_key == "test-hf-token"

    def test_dollar_env_var(self):
        with patch.dict("os.environ", {"CUSTOM_KEY": "custom-val"}):
            provider = HuggingFaceInferenceEmbeddings("test-model", api_key="$CUSTOM_KEY")
            assert provider.api_key == "custom-val"

    def test_explicit_base_url(self):
        provider = HuggingFaceInferenceEmbeddings("test-model", base_url="http://custom:8080")
        assert provider._explicit_base_url is True
        assert provider.base_url == "http://custom:8080"

    def test_requested_dimensions(self):
        provider = HuggingFaceInferenceEmbeddings("test-model", requested_dimensions=256)
        assert provider.get_dimension() == 256


@pytest.mark.unit
@pytest.mark.embedding
class TestHuggingFaceLocalEmbeddings:
    """Test HuggingFaceLocalEmbeddings provider."""

    def test_init(self):
        provider = HuggingFaceLocalEmbeddings("BAAI/bge-small-en-v1.5")
        assert provider.model == "BAAI/bge-small-en-v1.5"
        assert provider.provider_name == "huggingface_local"
        assert provider.max_batch_size == 128
        assert provider.pooling_strategy == "mean"

    def test_init_with_pooling(self):
        provider = HuggingFaceLocalEmbeddings("test-model", pooling_strategy="cls")
        assert provider.pooling_strategy == "cls"

    def test_init_with_requested_dimensions(self):
        provider = HuggingFaceLocalEmbeddings("test-model", requested_dimensions=128)
        assert provider.get_dimension() == 128

    def test_load_model_import_error(self):
        provider = HuggingFaceLocalEmbeddings("test-model")
        with patch.dict("sys.modules", {"transformers": None}):
            assert provider.validate_model() is False


# =========================================================================
# Matryoshka (MRL) dimension tests
# =========================================================================


@pytest.mark.unit
@pytest.mark.embedding
class TestOpenAIMatryoshka:
    """Test OpenAI Matryoshka dimension support."""

    def test_requested_dimensions_v3_small(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddings("text-embedding-3-small", requested_dimensions=256)
            assert provider.get_dimension() == 256

    def test_requested_dimensions_v3_large(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddings("text-embedding-3-large", requested_dimensions=1024)
            assert provider.get_dimension() == 1024

    def test_requested_dimensions_ada_raises(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with pytest.raises(ValueError, match="does not support the 'dimensions' parameter"):
                OpenAIEmbeddings("text-embedding-ada-002", requested_dimensions=256)

    def test_default_dimension_unchanged(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddings("text-embedding-3-small")
            assert provider.get_dimension() == 1536

    def test_payload_includes_dimensions(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddings("text-embedding-3-small", requested_dimensions=256)
            payload = provider._build_openai_payload(["test"])
            assert payload["dimensions"] == 256

    def test_payload_no_dimensions_when_not_set(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddings("text-embedding-3-small")
            payload = provider._build_openai_payload(["test"])
            assert "dimensions" not in payload

    def test_normalize_postprocessing(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIEmbeddings("text-embedding-3-small", normalize=True)
            embeddings = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            result = provider._postprocess_embeddings(embeddings)
            for vec in result:
                norm = np.linalg.norm(vec)
                assert abs(norm - 1.0) < 1e-5


@pytest.mark.unit
@pytest.mark.embedding
class TestOllamaMatryoshka:
    """Test Ollama Matryoshka dimension support."""

    def test_init_with_requested_dimensions(self):
        provider = OllamaEmbeddings("nomic-embed-text", requested_dimensions=256)
        assert provider.requested_dimensions == 256

    def test_get_dimension_with_requested(self):
        provider = OllamaEmbeddings("nomic-embed-text", requested_dimensions=256)
        provider._dimension = 768  # Simulate native dimension already cached
        assert provider.get_dimension() == 256

    def test_truncate_and_normalize(self):
        provider = OllamaEmbeddings("nomic-embed-text", requested_dimensions=3, normalize=True)
        embeddings = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        result = provider._truncate_and_normalize(embeddings)
        assert len(result[0]) == 3
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5

    def test_no_truncation_when_not_set(self):
        provider = OllamaEmbeddings("nomic-embed-text")
        embeddings = [[1.0, 2.0, 3.0, 4.0, 5.0]]
        result = provider._truncate_and_normalize(embeddings)
        assert len(result[0]) == 5


# =========================================================================
# Registry tests for new providers
# =========================================================================


@pytest.mark.unit
@pytest.mark.embedding
class TestNewProviderRegistry:
    """Test that new providers are properly registered."""

    def test_sentence_transformers_registered(self):
        assert "sentence_transformers" in EmbeddingRegistry.list()
        cls = EmbeddingRegistry.get("sentence_transformers")
        assert cls is SentenceTransformerEmbeddings

    def test_huggingface_registered(self):
        assert "huggingface" in EmbeddingRegistry.list()
        cls = EmbeddingRegistry.get("huggingface")
        assert cls is HuggingFaceInferenceEmbeddings

    def test_huggingface_local_registered(self):
        assert "huggingface_local" in EmbeddingRegistry.list()
        cls = EmbeddingRegistry.get("huggingface_local")
        assert cls is HuggingFaceLocalEmbeddings
