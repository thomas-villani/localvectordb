"""
Tests for localvectordb.embeddings module.
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import httpx

from localvectordb.embeddings import (
    EmbeddingProvider, OllamaEmbeddings, OpenAIEmbeddings, MockEmbeddings,
    JinaEmbeddings, GoogleEmbeddings, EmbeddingRegistry, create_embedding_provider, list_providers,
    embed_texts, embed_texts_sync
)
from localvectordb.exceptions import OllamaNotFoundError, EmbeddingError


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
        expected_methods = {
            'get_dimension', 'validate_model',
            'provider_name', 'max_batch_size', '_embed_single_batch'
        }
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

    @patch('httpx.Client')
    def test_validate_model_success(self, mock_client_class):
        """Test successful model validation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "llama2:7b"}
            ]
        }
        mock_client_class.return_value.__enter__.return_value = mock_client
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response

        provider = OllamaEmbeddings("nomic-embed-text")
        result = provider.validate_model()

        assert result is True

    @patch('httpx.Client')
    def test_validate_model_not_found(self, mock_client_class):
        """Test model validation when model not found."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [{"name": "llama2:7b"}]
        }
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        provider = OllamaEmbeddings("missing-model")
        result = provider.validate_model()

        assert result is False

    @patch('httpx.Client')
    def test_validate_model_connection_error(self, mock_client_class):
        """Test model validation with connection error."""
        mock_client = Mock()
        mock_client.get.side_effect = httpx.RequestError("Connection failed")
        mock_client_class.return_value.__enter__.return_value = mock_client

        provider = OllamaEmbeddings("test-model")
        with pytest.raises(OllamaNotFoundError):
            result = provider.validate_model()


    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "embeddings": [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]
        }
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
            json={
                "model": "test-model",
                "input": texts,
                "truncate": True
            },
            timeout=300.0
        )

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_error_response(self, mock_client_class):
        """Test embedding with error response."""
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Model not found"}
        mock_response.raise_for_status = Mock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response # AsyncMock(return_value=mock_response)

        # Fix: Properly mock the async context manager
        async_context_manager = AsyncMock()
        async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        async_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_client_class.return_value = async_context_manager

        provider = OllamaEmbeddings("test-model")
        provider._dimension = 384

        with pytest.raises(RuntimeError, match="Ollama error: Model not found"):
            await provider.embed_batch(["test"])

    @patch('httpx.AsyncClient')
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

    @patch('httpx.AsyncClient')
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

    @patch.object(OllamaEmbeddings, '_embed_single_batch')
    def test_get_dimension_calls_embed_batch(self, mock_embed):
        """Test get_dimension makes test call to determine dimension."""
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        provider = OllamaEmbeddings("test-model")
        dimension = provider.get_dimension()

        assert dimension == 4
        assert provider._dimension == 4

        # Should use asyncio.run
        mock_embed.assert_called_once()


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
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key is required"):
                OpenAIEmbeddings("text-embedding-ada-002")

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'})
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
            provider = OpenAIEmbeddings("unknown-model", api_key="test")

    def test_get_dimension_known_model(self):
        """Test getting dimension for known models."""
        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test")
        assert provider.get_dimension() == 1536

        provider = OpenAIEmbeddings("text-embedding-3-large", api_key="test")
        assert provider.get_dimension() == 3072

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful OpenAI embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
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
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-ada-002",
                "input": texts
            },
            timeout=provider.timeout
        )

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_openai_error(self, mock_client_class):
        """Test OpenAI API error handling."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = OpenAIEmbeddings("text-embedding-ada-002", api_key="test-key")

        with pytest.raises(RuntimeError, match="OpenAI error: Invalid API key"):
            await provider.embed_batch(["test"])


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
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Jina API key is required"):
                JinaEmbeddings("jina-embeddings-v4", task="text-matching")

    @patch.dict('os.environ', {'JINA_API_KEY': 'env-key'})
    def test_api_key_from_environment(self):
        """Test getting API key from environment."""
        provider = JinaEmbeddings("jina-embeddings-v4", task="text-matching")
        assert provider.api_key == "env-key"

    def test_api_key_from_env_var_reference(self):
        """Test getting API key from custom environment variable."""
        with patch.dict('os.environ', {'CUSTOM_JINA_KEY': 'custom-key'}):
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

    @patch('httpx.Client')
    def test_get_dimension_api_probe(self, mock_client_class):
        """Test dimension detection via API probe for unknown models."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}]
        }
        mock_response.raise_for_status = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        # Unknown model should probe API
        provider = JinaEmbeddings("unknown-model", api_key="test-key")
        provider._dimension = None  # Force dimension detection
        dimension = provider._get_model_dimension_api()

        assert dimension == 5
        mock_client.post.assert_called_once()

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]}
            ]
        }
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
                "Accept": "application/json"
            },
            json=expected_payload,
            timeout=provider.timeout
        )

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_with_task_and_dimensions(self, mock_client_class):
        """Test embedding with task and dimensions configuration."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2]}]
        }
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
            late_chunking=True
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
            "late_chunking": True
        }
        mock_client.post.assert_called_once_with(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json=expected_payload,
            timeout=provider.timeout
        )

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_error_response(self, mock_client_class):
        """Test embedding with error response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_response.raise_for_status = Mock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = AsyncMock(return_value=None)

        provider = JinaEmbeddings("jina-embeddings-v4", api_key="test-key")
        provider._dimension = 384

        with pytest.raises(RuntimeError, match="Jina API error: Invalid API key"):
            await provider.embed_batch(["test"])

    @patch('httpx.AsyncClient')
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
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Google AI.*API key is required"):
                GoogleEmbeddings()

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'env-key'})
    def test_api_key_from_gemini_env(self):
        """Test getting API key from GEMINI_API_KEY environment."""
        provider = GoogleEmbeddings()
        assert provider.api_key == "env-key"

    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'google-key'})
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
            base_url="https://custom.googleapis.com"
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

    @patch('httpx.Client')
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
            timeout=provider.timeout
        )

    @patch('httpx.Client')
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

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_success(self, mock_client_class):
        """Test successful embedding generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "embeddings": [
                {"values": [0.1, 0.2, 0.3]},
                {"values": [0.4, 0.5, 0.6]}
            ]
        }
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
            "embedding_config": {
                "task_type": "SEMANTIC_SIMILARITY",
                "output_dimensionality": 3
            }
        }
        mock_client.post.assert_called_once_with(
            f"{provider.base_url}/models/{provider.model}:embedContent",
            headers={
                "x-goog-api-key": "test-key",
                "Content-Type": "application/json"
            },
            json=expected_payload,
            timeout=provider.timeout
        )

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_embed_batch_with_normalization(self, mock_client_class):
        """Test embedding with vector normalization."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "embeddings": [
                {"values": [3.0, 4.0]}  # Magnitude = 5.0
            ]
        }
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

    @patch('httpx.AsyncClient')
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

    @patch('importlib.metadata.entry_points')
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
        embeddings = await embed_texts(
            texts, "mock", "test-model", dimension=384
        )
        assert embeddings.shape == (2, 384)

    def test_embed_texts_sync(self):
        """Test synchronous embed_texts_sync function."""
        texts = ["hello", "world"]
        embeddings = embed_texts_sync(
            texts, "mock", "test-model", dimension=384
        )
        assert embeddings.shape == (2, 384)

    @pytest.mark.asyncio
    async def test_embed_texts_with_batch_size(self):
        """Test embedding with custom batch size."""
        texts = ["hello", "world", "test"]
        embeddings = await embed_texts(
            texts, "mock", "test-model", batch_size=2, dimension=384
        )
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
        except:
            pass

        embeddings = provider.embed_sync(["hello", "world"])
        assert embeddings.shape == (2, 384)

    @patch('asyncio.get_event_loop')
    def test_embed_sync_with_running_loop(self, mock_get_loop):
        """Test sync wrapper with running event loop."""
        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        mock_get_loop.return_value = mock_loop

        provider = MockEmbeddings("test-model", dimension=384)

        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_future = Mock()
            mock_future.result.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_executor.submit.return_value = mock_future
            mock_executor_class.return_value.__enter__.return_value = mock_executor

            result = provider.embed_sync(["hello"])

            assert mock_executor.submit.called
            assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))

    @patch('asyncio.get_event_loop')
    def test_embed_sync_with_stopped_loop(self, mock_get_loop):
        """Test sync wrapper with stopped event loop."""
        mock_loop = Mock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_loop.return_value = mock_loop

        provider = MockEmbeddings("test-model", dimension=384)
        result = provider.embed_sync(["hello"])

        assert mock_loop.run_until_complete.called
        assert np.array_equal(result, np.array([[0.1, 0.2, 0.3]]))