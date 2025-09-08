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
from typing import Any, Callable, Dict, List, Optional, Type

import httpx
import numpy as np

from localvectordb.exceptions import EmbeddingError, OllamaNotFoundError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self,
                 model: str,
                 timeout: int = 90,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_concurrent_requests: int = 5,
                 **kwargs: Any) -> None:
        self.model = model
        self.config = kwargs

        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_concurrent_requests = max_concurrent_requests
        self._dimension: Optional[int] = None

    @property
    def async_supported(self) -> bool:
        return True

    async def embed_batch(self,
                          texts: List[str],
                          batch_size: Optional[int] = None,
                          progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Generate embeddings with automatic retry handling."""

        for attempt in range(self.max_retries + 1):
            try:
                return await self._embed_batch_impl(texts, batch_size, progress_callback)
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

    async def _embed_batch_impl(
            self,
            texts: List[str],
            batch_size: Optional[int] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Generate embeddings with progress tracking

        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        batch_size : Optional[int]
            Size of each batch
        progress_callback : Optional[callable]
            Callback function called with (completed_batches, total_batches)

        Returns
        -------
        np.ndarray
            Array of embeddings
        """
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        batch_size = batch_size or self.max_batch_size
        if batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        total_batches = (len(texts) + batch_size - 1) // batch_size
        completed_batches = 0

        # Pre-allocate final embeddings array
        final_embeddings = np.empty(
            (len(texts), self.get_dimension()),
            dtype=np.float32
        )

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_batch_with_progress(
                batch_texts: List[str],
                _start_index: int,
                _batch_num: int
        ) -> tuple[int, List[List[float]]]:
            """Process batch and update progress"""
            nonlocal completed_batches

            async with semaphore:
                try:
                    _embeddings = await self._embed_single_batch(batch_texts)

                    # Update progress
                    completed_batches += 1
                    if progress_callback is not None:
                        progress_callback(completed_batches, total_batches)

                    return _start_index, _embeddings

                except Exception as e:
                    logger.error(f"Error processing batch {_batch_num}: {e}")
                    raise

        # Create tasks for all batches
        tasks = []

        for batch_num, i in enumerate(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            task = process_batch_with_progress(batch, i, batch_num)
            tasks.append(task)

        # Execute all batches concurrently
        batch_results = await asyncio.gather(*tasks)

        # Assemble final results
        for start_index, embeddings in batch_results:
            batch_size_actual = len(embeddings)
            final_embeddings[start_index:start_index + batch_size_actual] = embeddings

        return final_embeddings

    @abstractmethod
    async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Embed a single batch"""
        pass

    async def embed_async(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
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

    async def _embed_batch_impl(
            self,
            texts: List[str],
            batch_size: Optional[int] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> np.ndarray:
        """
        Generate embeddings with progress tracking

        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        batch_size : Optional[int]
            Size of each batch
        progress_callback : Optional[Callable[[int, int], None]]
            Callback function called with (completed_batches, total_batches)

        Returns
        -------
        np.ndarray
            Array of embeddings
        """
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        batch_size = batch_size or self.max_batch_size
        if batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        total_batches = (len(texts) + batch_size - 1) // batch_size
        completed_batches = 0

        # Pre-allocate final embeddings array
        final_embeddings = np.empty(
            (len(texts), self.get_dimension()),
            dtype=np.float32
        )

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def process_batch_with_progress(
                batch_texts: List[str],
                _client: httpx.AsyncClient,
                _start_index: int,
                _batch_num: int
        ) -> tuple[int, List[List[float]]]:
            """Process batch and update progress"""
            nonlocal completed_batches

            async with semaphore:
                try:
                    _embeddings = await self._embed_single_batch(batch_texts, _client)

                    # Update progress
                    completed_batches += 1
                    if progress_callback is not None:
                        progress_callback(completed_batches, total_batches)

                    return _start_index, _embeddings

                except Exception as e:
                    logger.error(f"Error processing batch {_batch_num}: {e}")
                    raise

        async with httpx.AsyncClient() as client:
            # Create tasks for all batches
            tasks = []

            for batch_num, i in enumerate(range(0, len(texts), batch_size)):
                batch = texts[i:i + batch_size]
                task = process_batch_with_progress(batch, client, i, batch_num)
                tasks.append(task)

            # Execute all batches concurrently
            batch_results = await asyncio.gather(*tasks)

            # Assemble final results
            for start_index, embeddings in batch_results:
                batch_size_actual = len(embeddings)
                final_embeddings[start_index:start_index + batch_size_actual] = embeddings

        return final_embeddings


    @abstractmethod
    async def _embed_single_batch(self, texts: List[str], client: Optional[httpx.AsyncClient] = None, **kwargs: Any) -> List[List[float]]:
        """Embed a batch using asynchronous httpx client.

        The async httpx client is passed in as the `client` kwarg."""
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
    max_concurrent_requests : int, default = 3
        How many requests to make concurrently to the ollama server.
    """

    _model_info_cache: Dict[str, List[Dict]] = {}

    def __init__(self, model: str, base_url: Optional[str] = None, timeout: int = 300, max_retries: int = 3, retry_delay: float = 1.0,
                 max_concurrent_requests: int = 3) -> None:
        super().__init__(model, timeout, max_retries, retry_delay, max_concurrent_requests=max_concurrent_requests)
        effective_base_url = base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.base_url = (effective_base_url or "http://localhost:11434").rstrip('/')
        self._validated = False

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def max_batch_size(self) -> int:
        return 64  # Ollama's typical batch size

    def _get_model_info(self, force: bool = False) -> List[Dict]:
        if not self._model_info_cache or not self._model_info_cache.get(self.base_url) or force:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags", timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])

                self._model_info_cache[self.base_url] = models

        return self._model_info_cache.get(self.base_url, [])

    def validate_model(self) -> bool:
        """Check if the model is available in Ollama"""
        if self._validated:
            return True

        def _check_it(_models: List[Dict]) -> bool:
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
            dimension = self._get_model_dimension_api()
            self._dimension = dimension
        return self._dimension

    async def _embed_single_batch(self, texts: List[str], client: Optional[httpx.AsyncClient] = None, **kwargs: Any) -> List[List[float]]:
        """Gets the embeddings for a single batch, called from '_embed_batch_impl' with a single batch of texts."""
        if client is None:
            client = httpx.AsyncClient()

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

        return embeddings  # type: ignore[no-any-return]


class OpenAIEmbeddings(HTTPEmbeddingProvider):
    """OpenAI embedding provider.

    Parameters
    ----------
    model : str
        The OpenAI model to use for embedding
    api_key : str, optional
        Optionally provide the api key as a str. If not provided, tries to use "OPENAI_API_KEY" environment variable.
        You can specify a custom environment variable to use by prefixing with a "$", for example using:
        apikey="$CUSTOM_ENV_VAR" would try to load the api key from the `CUSTOM_ENV_VAR` environment variable.
    timeout : int, default = 90
        Timeout in seconds for the http request
    max_retries : int, default = 3
        How many times to retry on a failed request.
    retry_delay : float, default = 1.0
        How long to delay after a failed request (the backoff is exponential)
    max_concurrent_requests : int, default = 5
        How many requests to make concurrently to the OpenAI server.
    """

    def __init__(self, model: str, api_key: Optional[str] = None, timeout: int = 90, max_retries: int = 3, retry_delay: float = 1.0,
                 max_concurrent_requests: int = 5) -> None:
        super().__init__(model, timeout, max_retries, retry_delay, max_concurrent_requests=max_concurrent_requests)

        if api_key is not None and api_key.startswith("$") and api_key[1:].isupper():
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

    async def _embed_single_batch(self, texts: List[str], client: Optional[httpx.AsyncClient] = None, **kwargs: Any) -> List[List[float]]:
        if client is None:
            client = httpx.AsyncClient()

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


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def __init__(self, model: str, dimension: int = 384, timeout: int = 90, max_retries: int = 3, retry_delay: float = 1.0, **kwargs: Any) -> None:
        super().__init__(model, timeout, max_retries, retry_delay, **kwargs)
        self._dimension: int = dimension
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

    async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        if not texts:
            return [[]]

        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            np.random.seed(hash(text) % (2 ** 31))
            embedding_array = np.random.normal(0, 1, self._dimension)
            # Normalize
            embedding_array = embedding_array / np.linalg.norm(embedding_array)
            embeddings.append(embedding_array.tolist())

        self.number_of_calls += 1

        return embeddings



class EmbeddingRegistry:
    """Registry for embedding providers with plugin discovery"""

    _providers: Dict[str, Type[EmbeddingProvider]] = {}
    _plugins_discovered = False

    @classmethod
    def register(cls, name: str, provider_class: Type[EmbeddingProvider]) -> None:
        """Register a new embedding provider"""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def _discover_plugins(cls) -> None:
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
        **kwargs: Any
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
    def refresh_plugins(cls) -> None:
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
    **kwargs: Any
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
    **provider_kwargs: Any
) -> np.ndarray:
    """Convenience function to embed texts"""
    embedding_provider = create_embedding_provider(provider, model, **provider_kwargs)
    return await embedding_provider.embed_batch(texts, batch_size)


def embed_texts_sync(
    texts: List[str],
    provider: str,
    model: str,
    batch_size: Optional[int] = None,
    **provider_kwargs: Any
) -> np.ndarray:
    """Synchronous convenience function to embed texts"""
    embedding_provider = create_embedding_provider(provider, model, **provider_kwargs)
    return embedding_provider.embed_sync(texts, batch_size)
