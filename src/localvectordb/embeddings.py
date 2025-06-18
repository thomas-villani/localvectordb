# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/embeddings.py
"""
Plugin-based embedding providers for LocalVectorDB v1.0

This module provides a flexible embedding system with support for multiple providers
through a registry pattern.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Type

import httpx
import numpy as np

from localvectordb.exceptions import OllamaNotFoundError, EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model: str, timeout=90, max_retries=3, retry_delay=1.0, **kwargs):
        self.model = model
        self.config = kwargs

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._dimension = None


    async def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Generate embeddings with automatic retry handling."""

        for attempt in range(self.max_retries + 1):
            try:
                return await self._embed_batch_impl(texts, batch_size)
            except Exception as e:
                if not self._should_retry(e, attempt):
                    raise EmbeddingError(f"Error retrieving embeddings: {str(e)}")

                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Embedding failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

        # Should never reach here, but safety net
        raise RuntimeError(f"All {self.max_retries + 1} embedding attempts failed")

    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an error should trigger a retry."""
        # Don't retry on last attempt
        if attempt >= self.max_retries:
            return False

        # Retry on network/timeout errors
        if isinstance(error, (httpx.RequestError, httpx.TimeoutException)):
            return True

        # Retry on HTTP errors that indicate temporary issues
        if isinstance(error, httpx.HTTPStatusError):
            # Retry on 429 (rate limit) and 5xx (server errors)
            return error.response.status_code == 429 or 500 <= error.response.status_code < 600

        # Retry on general connection/timeout issues
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True

        return False

    @abstractmethod
    async def _embed_batch_impl(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Actual embedding implementation - child classes implement this."""
        pass

    async def embed_async(self, texts: List[str], batch_size: Optional[int] = None):
        """Generate embeddings for a list of texts."""
        return await self.embed_batch(texts, batch_size)

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the embedding dimension for this model"""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Check if the model is available/valid"""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name"""
        pass

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Maximum batch size for this provider"""
        pass

    def embed_sync(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Synchronous wrapper for embed_batch"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.embed_batch(texts, batch_size))
                    return future.result()
            else:
                return loop.run_until_complete(self.embed_batch(texts, batch_size))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.embed_batch(texts, batch_size))


class HTTPEmbeddingProvider(EmbeddingProvider, ABC):
    """Embedding Providers which utilize HTTP requests to get embeddings.

    Subclasses need to implement `_embed_single_batch(self, texts: list[str], client: httpx.AsyncClient)`
    which provides an async httpx client to use to make the http request.

    """

    async def _embed_batch_impl(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        batch_size = batch_size or self.max_batch_size
        if batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        final_embeddings = np.empty(
            (len(texts), self.get_dimension()),
            dtype=np.float32
        )
        async with httpx.AsyncClient() as client:
            current_idx = 0
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                embeddings = await self._embed_single_batch(batch, client)
                current_batch_size = len(embeddings)
                final_embeddings[current_idx:current_idx + current_batch_size] = embeddings
                current_idx += current_batch_size

                del embeddings  # free the memory!

        return final_embeddings

    @abstractmethod
    async def _embed_single_batch(self, texts, client: httpx.AsyncClient) -> List[List[float]]:
        """Embed a batch using asynchronous httpx client."""
        pass


class OllamaEmbeddings(HTTPEmbeddingProvider):
    """Ollama embedding provider.

    Parameters
    ----------
    model : str
        The OpenAI model to use for embeddding
    base_url : str
        The base url for the ollama server (default for Ollama install is http://localhost:11434)
        Alternatively, you can set the `OLLAMA_HOST` environment variable.
    timeout : int, default = 90
        Timeout in sceonds for the http request
    max_retries : int, default = 3
        How many times to retry on a failed request.
    retry_delay : float, default = 1.0
        How long to delay after a failed request (the backoff is exponential)
    """

    _model_info_cache = {}

    def __init__(self, model: str, base_url: str = None, timeout=300, max_retries=3, retry_delay=1.0, **kwargs):
        super().__init__(model, timeout, max_retries, retry_delay, **kwargs)
        base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = base_url.rstrip('/')
        self._validated = False

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def max_batch_size(self) -> int:
        return 64  # Ollama's typical batch size

    def _get_model_info(self, force=False):
        if not self._model_info_cache or not self._model_info_cache.get(self.base_url) or force:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags", timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])

                self._model_info_cache[self.base_url] = models

        return self._model_info_cache.get(self.base_url, {})

    def validate_model(self) -> bool:
        """Check if the model is available in Ollama"""
        if self._validated:
            return True

        def _check_it(_models):
            for model_info in _models:
                if model_info["name"].startswith(self.model):
                    return True
            return False

        try:
            models = self._get_model_info()
            if _check_it(models):
                self._validated = True
                return True
            models = self._get_model_info(force=True)
            if _check_it(models):
                self._validated = True
                return True
            return False
        except (httpx.ConnectError, TimeoutError, ConnectionError, httpx.RequestError) as e:
            logger.error(f"Could not connect to Ollama service: {str(e)}")
            raise OllamaNotFoundError(f"Could not connect to Ollama service at: {self.base_url}")

    def _get_model_dimension_api(self) -> int:
        # Because we may be calling get_dimension from the sync or async function, we need to handle it properly.
        client = httpx.AsyncClient()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to run in a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._embed_single_batch(["test"], client=client))
                    test_embedding = future.result()

            else:
                test_embedding = loop.run_until_complete(self._embed_single_batch(["test"], client=client))

        except RuntimeError:
            # No event loop, create one
            test_embedding = asyncio.run(self._embed_single_batch(["test"], client=client))

        return len(test_embedding[0])

    def get_dimension(self) -> int:
        """Get embedding dimension by making a test call"""
        if self._dimension is None:
            self._dimension = self._get_model_dimension_api()
        return self._dimension

    async def _embed_single_batch(self, texts, client: httpx.AsyncClient):
        """Gets the embeddings for a single batch, called from '_embed_batch_impl' with a single batch of texts."""
        response = await client.post(
            f"{self.base_url}/api/embed",
            json={
                "model": self.model,
                "input": texts,
                "truncate": True
            },
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error']}")

        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError("No embeddings returned from Ollama")

        return embeddings


class OpenAIEmbeddings(HTTPEmbeddingProvider):
    """OpenAI embedding provider.

    Parameters
    ----------
    model : str
        The OpenAI model to use for embeddding
    api_key : str, optional
        Optionally provide the api key as a str. If not provided, tries to use "OPENAI_API_KEY" environment variable.
        You can specify a custom environment variable to use by prefixing with a "$", for example using:
        apikey="$CUSTOM_ENV_VAR" would try to load the api key from the `CUSTOM_ENV_VAR` environment variable.
    timeout : int, default = 90
        Timeout in sceonds for the http request
    max_retries : int, default = 3
        How many times to retry on a failed request.
    retry_delay : float, default = 1.0
        How long to delay after a failed request (the backoff is exponential)
    """

    def __init__(self, model: str, api_key: Optional[str] = None, timeout=90, max_retries=3, retry_delay=1.0, **kwargs):
        super().__init__(model, timeout, max_retries, retry_delay, **kwargs)

        if api_key is not None and api_key.startswith("$"):
            api_key = os.getenv(api_key[1:])

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self._validated = False

        # Model dimension mapping
        self._model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        if model not in self._model_dimensions:
            raise ValueError("Currently the only supported OpenAI embedding models are: "
                             f"{', '.join(self._model_dimensions.keys())}")

        self._dimension = self._model_dimensions[model]

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def max_batch_size(self) -> int:
        return 1000  # OpenAI's batch size limit

    def validate_model(self) -> bool:
        """Check if the model exists"""
        if self._validated:
            return True

        self._validated = self.model in self._model_dimensions.keys()
        return self._validated

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is not None:
            return self._dimension

        # Try to get from known mappings first
        if self.model in self._model_dimensions:
            self._dimension = self._model_dimensions[self.model]
            return self._dimension
        else:
            # Should never get here.
            raise ValueError("Unknown model.")

    async def _embed_single_batch(self, texts, client: httpx.AsyncClient):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json={
                "model": self.model,
                "input": texts
            },
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()

        if "error" in data:
            raise RuntimeError(f"OpenAI error: {data['error']['message']}")

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    # async def _embed_batch_impl_OLD(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
    #     """Generate embeddings using OpenAI"""
    #     if not texts:
    #         return np.array([]).reshape(0, self.get_dimension())
    #
    #     batch_size = batch_size or self.max_batch_size
    #     all_embeddings = []
    #
    #     headers = {
    #         "Authorization": f"Bearer {self.api_key}",
    #         "Content-Type": "application/json"
    #     }
    #
    #     async with httpx.AsyncClient() as client:
    #         for i in range(0, len(texts), batch_size):
    #             batch = texts[i:i + batch_size]
    #
    #             try:
    #                 response = await client.post(
    #                     "https://api.openai.com/v1/embeddings",
    #                     headers=headers,
    #                     json={
    #                         "model": self.model,
    #                         "input": batch
    #                     },
    #                     timeout=self.timeout
    #                 )
    #                 response.raise_for_status()
    #
    #                 data = response.json()
    #
    #                 if "error" in data:
    #                     raise RuntimeError(f"OpenAI error: {data['error']['message']}")
    #
    #                 embeddings = [item["embedding"] for item in data["data"]]
    #                 all_embeddings.extend(embeddings)
    #
    #             except httpx.RequestError as e:
    #                 raise RuntimeError(f"Failed to connect to OpenAI: {e}")
    #             except Exception as e:
    #                 raise RuntimeError(f"Error generating embeddings: {e}")
    #
    #     return np.array(all_embeddings, dtype=np.float32)


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def __init__(self, model: str, dimension: int = 384, timeout=90, max_retries=3, retry_delay=1.0, **kwargs):
        super().__init__(model, timeout, max_retries, retry_delay, **kwargs)
        self._dimension = dimension
        self.number_of_calls = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def max_batch_size(self) -> int:
        return 1000

    def validate_model(self) -> bool:
        return True

    def get_dimension(self) -> int:
        return self._dimension

    async def _embed_batch_impl(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate random embeddings for testing"""
        if not texts:
            return np.array([]).reshape(0, self._dimension)

        # Generate deterministic "embeddings" based on text hash
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            np.random.seed(hash(text) % (2**31))
            embedding = np.random.normal(0, 1, self._dimension)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        self.number_of_calls += 1
        return np.array(embeddings, dtype=np.float32)


class EmbeddingRegistry:
    """Registry for embedding providers with plugin discovery"""

    _providers: Dict[str, Type[EmbeddingProvider]] = {}
    _plugins_discovered = False

    @classmethod
    def register(cls, name: str, provider_class: Type[EmbeddingProvider]):
        """Register a new embedding provider"""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def _discover_plugins(cls):
        """Discover embedding provider plugins using entry points"""
        if cls._plugins_discovered:
            return

        # Python 3.10+ importlib.metadata
        from importlib.metadata import entry_points


        provider_eps = entry_points(group='localvectordb.embedding_providers')
        for ep in provider_eps:
            try:
                provider_class = ep.load()
                cls.register(ep.name, provider_class)
                logger.info(f"Discovered embedding provider plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding provider plugin {ep.name}: {e}")

        cls._plugins_discovered = True

    @classmethod
    def get(cls, name: str) -> Type[EmbeddingProvider]:
        """Get an embedding provider by name"""
        cls._discover_plugins()

        name = name.lower()
        if name not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Unknown embedding provider: {name}. "
                f"Available providers: {available}"
            )
        return cls._providers[name]

    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        model: str,
        **kwargs
    ) -> EmbeddingProvider:
        """Create an embedding provider instance"""
        provider_class = cls.get(provider_name)
        return provider_class(model, **kwargs)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered providers"""
        cls._discover_plugins()
        return list(cls._providers.keys())

    @classmethod
    def refresh_plugins(cls):
        """Force re-discovery of plugins (useful for testing)"""
        cls._plugins_discovered = False
        cls._discover_plugins()


# Auto-register built-in providers
EmbeddingRegistry.register("ollama", OllamaEmbeddings)
EmbeddingRegistry.register("openai", OpenAIEmbeddings)
EmbeddingRegistry.register("mock", MockEmbeddings)


# Convenience functions
def create_embedding_provider(
    provider: str,
    model: str,
    **kwargs
) -> EmbeddingProvider:
    """Create an embedding provider instance"""
    return EmbeddingRegistry.create_provider(provider, model, **kwargs)


def list_providers() -> List[str]:
    """List available embedding providers"""
    return EmbeddingRegistry.list()


async def embed_texts(
    texts: List[str],
    provider: str,
    model: str,
    batch_size: Optional[int] = None,
    **provider_kwargs
) -> np.ndarray:
    """Convenience function to embed texts"""
    embedding_provider = create_embedding_provider(provider, model, **provider_kwargs)
    return await embedding_provider.embed_batch(texts, batch_size)


def embed_texts_sync(
    texts: List[str],
    provider: str,
    model: str,
    batch_size: Optional[int] = None,
    **provider_kwargs
) -> np.ndarray:
    """Synchronous convenience function to embed texts"""
    embedding_provider = create_embedding_provider(provider, model, **provider_kwargs)
    return embedding_provider.embed_sync(texts, batch_size)