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
import hashlib
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Type

import httpx
import numpy as np

from localvectordb.exceptions import EmbeddingError, OllamaNotFoundError

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(
            self,
            model: str,
            timeout: int = 90,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrent_requests: int = 5,
            **kwargs: Any
            ) -> None:
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

    async def embed_batch(
            self,
            texts: List[str],
            batch_size: Optional[int] = None,
            progress_callback: Optional[Callable] = None
            ) -> np.ndarray:
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

        # Sort results by start index to ensure correct ordering
        batch_results.sort(key=lambda x: x[0])

        # Build final array from sorted results - memory efficient approach
        embeddings_list = []
        for _, embeddings in batch_results:
            embeddings_list.extend(embeddings)

        # Convert to numpy array only at the end
        final_embeddings = np.array(embeddings_list, dtype=np.float32)

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
        """Synchronous wrapper for embed_batch with proper event loop handling."""
        try:
            # Try to get the current event loop without creating one
            asyncio.get_running_loop()
            # If we reach here, we're in an async context - delegate to sync implementation
            # Rather than creating complex threading, recommend using embed_batch directly
            raise RuntimeError(
                "embed_sync() cannot be called from within an async context. "
                "Use 'await provider.embed_batch(texts, batch_size)' instead."
            )
        except RuntimeError:
            # No running event loop - safe to create one
            return asyncio.run(self.embed_batch(texts, batch_size))


class HTTPEmbeddingProvider(EmbeddingProvider, ABC):
    """Embedding Providers which utilize HTTP requests to get embeddings.

    Subclasses need to implement `_embed_single_batch(self, texts: list[str], client: httpx.AsyncClient)`
    which provides an async httpx client to use to make the http request.

    """

    def __init__(self,
                 model: str,
                 base_url: Optional[str] = None,
                 timeout: int = 90,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_concurrent_requests: int = 5,
                 **kwargs: Any):
        self.base_url = base_url
        super().__init__(model,
                         timeout=timeout,
                         max_retries=max_retries,
                         retry_delay=retry_delay,
                         max_concurrent_requests=max_concurrent_requests,
                         **kwargs)

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

            # Sort results by start index to ensure correct ordering
            batch_results.sort(key=lambda x: x[0])

            # Build final array from sorted results - memory efficient approach
            embeddings_list = []
            for _, embeddings in batch_results:
                embeddings_list.extend(embeddings)

            # Convert to numpy array only at the end
            final_embeddings = np.array(embeddings_list, dtype=np.float32)

        return final_embeddings

    @abstractmethod
    async def _embed_single_batch(self, texts: List[str], client: Optional[httpx.AsyncClient] = None, **kwargs: Any) -> \
    List[List[float]]:
        """Embed a batch using asynchronous httpx client.

        The async httpx client is passed in as the `client` kwarg."""
        pass

# NOTE: OLLAMA_HOST is meant to tell ollama how to serve (e.g. 127.0.0.1 vs. 0.0.0.0) so we should choose a different var.
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
        Timeout in seconds for the http request
    max_retries : int, default = 3
        How many times to retry on a failed request.
    retry_delay : float, default = 1.0
        How long to delay after a failed request (the backoff is exponential)
    max_concurrent_requests : int, default = 3
        How many requests to make concurrently to the ollama server.
    """

    _model_info_cache: Dict[str, List[Dict]] = {}

    def __init__(
            self, model: str, base_url: Optional[str] = None, timeout: int = 300, max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrent_requests: int = 3
            ) -> None:
        super().__init__(model, base_url=base_url, timeout=timeout, max_retries=max_retries,
                         retry_delay=retry_delay, max_concurrent_requests=max_concurrent_requests)
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

    def _get_model_dimension_sync(self) -> int:
        """
        Get model dimension using synchronous HTTP client to avoid event loop issues.

        Returns
        -------
        int
            Embedding dimension for the model
        """
        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": ["dimension_test"],
                        "truncate": True
                    }
                )
                response.raise_for_status()
                data = response.json()

                if 'embeddings' in data and data['embeddings']:
                    return len(data['embeddings'][0])
                else:
                    raise ValueError("No embeddings returned from Ollama API")

            except Exception as e:
                logger.error(f"Failed to get dimension from Ollama API: {e}")
                # Fallback to a common default for Ollama models
                return 4096

    def get_dimension(self) -> int:
        """Get embedding dimension by making a test call"""
        if self._dimension is None:
            dimension = self._get_model_dimension_sync()
            self._dimension = dimension
        return self._dimension

    async def _embed_single_batch(self, texts: List[str], client: Optional[httpx.AsyncClient] = None, **kwargs: Any) -> \
    List[List[float]]:
        """Gets the embeddings for a single batch, called from '_embed_batch_impl' with a single batch of texts."""
        if client is None:
            # Use context manager to ensure proper AsyncClient cleanup
            async with httpx.AsyncClient() as client:
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
        else:
            # Use provided client
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

    def __init__(
            self, model: str, api_key: Optional[str] = None, timeout: int = 90, max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrent_requests: int = 5
            ) -> None:
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

    async def _embed_single_batch(self, texts: List[str], client: Optional[httpx.AsyncClient] = None, **kwargs: Any) -> \
    List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if client is None:
            # Use context manager to ensure proper AsyncClient cleanup
            async with httpx.AsyncClient() as client:
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
        else:
            # Use provided client
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


class GoogleEmbeddings(HTTPEmbeddingProvider):
    """Google AI (Gemini) embedding provider using the Generative Language API.

    Parameters
    ----------
    model : str, default "gemini-embedding-001"
        The Google AI embedding model, e.g.:
        - "gemini-embedding-001" (stable)
        - "gemini-embedding-exp-03-07" (experimental)
    api_key : str, optional
        API key string or an env var reference (e.g., "$GEMINI_API_KEY").
        If not provided, tries env vars (in order): GEMINI_API_KEY, GOOGLE_API_KEY.
    task_type : Literal, optional
        One of:
        {"semantic_similarity", "classification", "clustering", "retrieval_document", "retrieval_query",
        "code_retrieval_query", "question_answering", "fact_verification"}
        See: https://ai.google.dev/gemini-api/docs/embeddings#supported-task-types
    requested_dimensions : int, optional
        MRL-controlled output size (128–3072). Defaults to 3072 if not set by API.
        If provided, get_dimension() returns this value without a test call.
    normalize : bool, default False
        If True, L2-normalize returned vectors (recommended for non-3072 outputs).
        For 3072 output, vectors are already normalized by the API.
    base_url : str, optional
        Override the base API URL. Defaults to the public Google endpoint.
    timeout : int, default 90
        Request timeout in seconds.
    max_retries : int, default 3
        Retry attempts on transient errors.
    retry_delay : float, default 1.0
        Base delay between retries (exponential backoff).
    max_concurrent_requests : int, default 5
        Concurrency for batch processing.
    """

    def __init__(
            self,
            model: str = "gemini-embedding-001",
            api_key: Optional[str] = None,
            task_type: Literal["semantic_similarity", "classification", "clustering", "retrieval_document",
            "retrieval_query", "code_retrieval_query", "question_answering",
            "fact_verification"] = "semantic_similarity",
            requested_dimensions: Optional[int] = None,
            normalize: bool = True,
            base_url: Optional[str] = None,
            timeout: int = 90,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrent_requests: int = 5,
            **kwargs: Any,
    ) -> None:
        super().__init__(model, timeout, max_retries, retry_delay, max_concurrent_requests=max_concurrent_requests,
                         **kwargs)

        # Resolve API key (param or env)
        if api_key is not None and api_key.startswith("$") and api_key[1:].isupper():
            api_key = os.getenv(api_key[1:])
        self.api_key = (
                api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Google AI (Gemini) API key is required. Set api_key or one of: GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_GENAI_API_KEY")

        # API base URL (v1beta as in public docs)
        self.base_url = (base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")

        # Optional config
        self.task_type = str(task_type).upper()
        self.requested_dimensions = requested_dimensions
        self.normalize = normalize

        # If caller fixed the output dimensionality, we can set it immediately.
        if self.requested_dimensions:
            self._dimension = int(self.requested_dimensions)

        self._validated = False

        # Conservative default if we must assume without a test call.
        # The public docs state default is 3072 for gemini-embedding-001.
        self._default_dimensions_by_model = {
            "gemini-embedding-001": 3072,
        }

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def max_batch_size(self) -> int:
        # Google API supports multiple Contents in one call; keep a safe batch size.
        return 200

    def validate_model(self) -> bool:
        """Validate the model by querying the models endpoint."""
        if self._validated:
            return True

        url = f"{self.base_url}/models/{self.model}"
        headers = {"x-goog-api-key": self.api_key}
        try:
            with httpx.Client() as client:
                r = client.get(url, headers=headers, timeout=self.timeout)
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                self._validated = True
                return True
        except Exception as e:
            # If validation fails due to networking, treat as not validated but don't crash.
            logger.warning(f"Could not validate Google AI model '{self.model}': {e}")
            return False

    def _get_model_dimension_sync(self) -> int:
        """
        Determine embedding dimension via synchronous API call.

        Returns
        -------
        int
            Embedding dimension for the model
        """
        # If caller provided requested_dimensions, use it.
        if self.requested_dimensions:
            return int(self.requested_dimensions)

        # Otherwise attempt a lightweight embed to discover dimension using sync client
        with httpx.Client(timeout=self.timeout) as client:
            try:
                url = f"{self.base_url}/models/{self.model}:embedContent"
                headers = {
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                }

                # Build minimal request payload
                payload = {
                    "content": {"parts": [{"text": "dimension_probe"}]}
                }

                if self.requested_dimensions:
                    payload["embedding_config"] = {"output_dimensionality": self.requested_dimensions}

                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                if "embedding" in data and "values" in data["embedding"]:
                    return len(data["embedding"]["values"])
                else:
                    raise ValueError("No embedding values returned from Google API")

            except Exception as e:
                logger.error(f"Failed to get dimension from Google API: {e}")
                # Fallback to known defaults if probe fails
                return self._default_dimensions_by_model.get(self.model, 3072)

    def get_dimension(self) -> int:
        """Return embedding dimension, using API probe if needed."""
        if self._dimension is None:
            self._dimension = self._get_model_dimension_sync()
        return self._dimension

    async def _embed_single_batch(
            self,
            texts: List[str],
            client: Optional[httpx.AsyncClient] = None,
            **kwargs: Any
    ) -> List[List[float]]:
        url = f"{self.base_url}/models/{self.model}:embedContent"
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        # Build 'contents' as a list of Content objects: [{parts: [{text: "..."}]}, ...]
        contents = [{"parts": [{"text": t}]} for t in texts]

        payload: Dict[str, Any] = {
            "contents": contents
        }

        emb_cfg: Dict[str, Any] = {}
        if self.task_type:
            emb_cfg["task_type"] = self.task_type
        if self.requested_dimensions:
            emb_cfg["output_dimensionality"] = int(self.requested_dimensions)
        if emb_cfg:
            payload["embedding_config"] = emb_cfg

        if client is None:
            # Use context manager to ensure proper AsyncClient cleanup
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                return self._process_google_response(data)
        else:
            # Use provided client
            response = await client.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return self._process_google_response(data)

    def _process_google_response(self, data: Dict[str, Any]) -> List[List[float]]:
        """Process Google AI API response and extract embeddings."""
        # Possible shapes from API:
        # - {"embeddings": [ {"values": [...]}, {"values": [...]} ]}
        # - Rare: Single embedding forms (handle defensively)
        embeddings_raw = None
        if "embeddings" in data and isinstance(data["embeddings"], list):
            embeddings_raw = data["embeddings"]
        elif "embedding" in data and isinstance(data["embedding"], dict):
            embeddings_raw = [data["embedding"]]
        else:
            raise RuntimeError(f"Unexpected response from Google AI embeddings API: {data}")

        vectors: List[List[float]] = []
        for e in embeddings_raw:
            values = e.get("values")
            if values is None:
                # Some responses may nest differently; try 'embedding' field if present
                inner = e.get("embedding")
                if inner and isinstance(inner, dict):
                    values = inner.get("values")
            if values is None:
                raise RuntimeError("Malformed embedding response: missing 'values'")

            vec = [float(x) for x in values]

            # Optional normalization (recommended for non-3072 outputs)
            if self.normalize and vec:
                arr = np.asarray(vec, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                vec = arr.tolist()

            vectors.append(vec)

        # If no fixed dimension yet, set it from the first vector length.
        if self._dimension is None and vectors and len(vectors[0]) > 0:
            self._dimension = len(vectors[0])

        return vectors


class JinaEmbeddings(HTTPEmbeddingProvider):
    """Jina AI embedding provider.

    Parameters
    ----------
    model : str
        The Jina model to use for embedding. Examples:
          - "jina-embeddings-v4" (multimodal/multilingual, 2048 dims)
          - "jina-embeddings-v3" (1024 dims)
          - "jina-clip-v2" (1024 dims)
          - "jina-code-embeddings-0.5b"
          - "jina-code-embeddings-1.5b"
    api_key : str, optional
        Optionally provide the API key as a str. If it starts with "$" and the rest is uppercase,
        the key will be read from that environment variable. Otherwise, defaults to JINA_API_KEY env var.
        Get your Jina AI API key for free: https://jina.ai/?sui=apikey
    timeout : int, default = 90
        HTTP timeout (seconds)
    max_retries : int, default = 3
        Automatic retry attempts
    retry_delay : float, default = 1.0
        Base delay for exponential backoff
    max_concurrent_requests : int, default = 5
        Concurrent requests to Jina API

    Additional keyword arguments are passed through to the Jina Embeddings API request body, for example:
      - embedding_type: str, default "float" (other options: "base64", "binary", "ubinary")
      - task: str, e.g., for v4: "retrieval.query" | "retrieval.passage" | "text-matching" | "code.query" | "code.passage"
              for code models: "nl2code.query" | "nl2code.passage" | "code2code.query" | "code2code.passage" | "code2nl.query" | "code2nl.passage" | "code2completion.query" | "code2completion.passage" | "qa.query" | "qa.passage"
      - dimensions: int, to truncate output embeddings to this size
      - truncate: bool
      - late_chunking: bool (v4)
      - return_multivector: bool (v4; not supported by this provider, will raise if True)
      - normalized: bool (v3)

    Behavior
    --------
    - By default, embeddings are returned as float vectors.
    - If you provide `dimensions`, this provider will both:
        1) tell the API to output that dimension; and
        2) use that value to pre-allocate output arrays.
    - If no `dimensions` is provided, well-known models use known sizes (v4=2048, v3=1024, clip-v2=1024).
      Otherwise, dimension is determined via a one-off probe request.
    """

    _MODEL_DIMENSIONS = {
        "jina-embeddings-v4": 2048,
        "jina-embeddings-v3": 1024,
        "jina-code-embeddings-1.5b": 1536,
        "jina-code-embeddings-0.5b": 896,
    }
    _MODEL_TASKS = {
        "jina-embeddings-v3": ["text-matching", "retrieval.passage", "separation", "text-matching", None],
        "jina-embeddings-v4": ["text-matching", "retrieval.query", "retrieval.passage", "code.query", "code.passage"],
        "jina-code-embeddings-1.5b": [
            "nl2code.query",
            "nl2code.passage",
            "code2code.query",
            "code2code.passage",
            "code2nl.query",
            "code2nl.passage",
            "code2completion.query",
            "code2completion.passage",
            "qa.query",
            "qa.passage",
        ],
        "jina-code-embeddings-0.5b": [
            "nl2code.query",
            "nl2code.passage",
            "code2code.query",
            "code2code.passage",
            "code2nl.query",
            "code2nl.passage",
            "code2completion.query",
            "code2completion.passage",
            "qa.query",
            "qa.passage",
        ]
    }

    def __init__(
            self,
            model: str,
            api_key: Optional[str] = None,
            task: Optional[str] = "auto",
            truncate: bool = False,
            late_chunking: bool = False,
            requested_dimensions: Optional[int] = None,
            timeout: int = 90,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrent_requests: int = 5,
            **kwargs: Any
    ) -> None:
        super().__init__(model, timeout, max_retries, retry_delay, max_concurrent_requests=max_concurrent_requests,
                         **kwargs)

        # Resolve API key from env if formatted as "$ENVVAR"
        if api_key is not None and api_key.startswith("$") and api_key[1:].isupper():
            api_key = os.getenv(api_key[1:])

        # Default to JINA_API_KEY env var
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key is required. Please set JINA_API_KEY environment variable. "
                "Get your Jina AI API key for free: https://jina.ai/?sui=apikey"
            )

        # Requested truncation dimension (if provided by user)
        self.requested_dimensions: Optional[int] = requested_dimensions
        self.truncate: bool = truncate
        self.task: Optional[str] = task

        if model not in self._MODEL_TASKS:
            self.task = None
        else:
            allowed_tasks = self._MODEL_TASKS.get(model, [])
            if self.task == "auto":
                if allowed_tasks:
                    self.task = allowed_tasks[0]
                else:
                    self.task = None
            elif self.task not in allowed_tasks:
                raise ValueError(f"`task` must be one of {allowed_tasks} for Jina AI model {model}")

        self.late_chunking: bool = late_chunking

        # If user specified a target dimension, honor it. Else use known mapping if available.
        if self.requested_dimensions is not None:
            self._dimension = self.requested_dimensions
        elif model in self._MODEL_DIMENSIONS:
            self._dimension = self._MODEL_DIMENSIONS[model]
        else:
            self._dimension = None  # will probe

    @property
    def provider_name(self) -> str:
        return "jina"

    @property
    def max_batch_size(self) -> int:
        # Reasonable default; Jina API supports batching, but no explicit hard limit documented.
        # Keep moderate to limit payload sizes and latency.
        return 512

    def validate_model(self) -> bool:
        """Try a lightweight probe to confirm the model is usable."""
        return True

    def get_dimension(self) -> int:
        """Return embedding dimension. Uses known sizes, user-requested dimensions, or probes via API."""
        if self._dimension is not None:
            return self._dimension

        # No requested dimension and unknown model: probe the API without truncation to get true size
        self._dimension = self._get_model_dimension_api()
        return self._dimension

    def _get_model_dimension_api(self) -> int:
        """Probe Jina embeddings API to determine embedding size."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",  # required by Jina API
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": ["dimension probe"],
        }
        if self.task:
            payload["task"] = self.task

        # Do not include "dimensions" on probe, we want the native dimension if unknown
        with httpx.Client() as client:
            resp = client.post(
                "https://api.jina.ai/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                msg = data["error"]["message"] if isinstance(data["error"], dict) and "message" in data["error"] else \
                    data["error"]
                raise RuntimeError(f"Jina API error during dimension probe: {msg}")

            items = data.get("data", [])
            if not items:
                raise RuntimeError("No data returned from Jina API during dimension probe")
            emb = items[0].get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Unexpected embedding format during dimension probe (expected float list)")
            return len(emb)

    async def _embed_single_batch(
            self,
            texts: List[str],
            client: Optional[httpx.AsyncClient] = None,
            **kwargs: Any
    ) -> List[List[float]]:
        """Embed a single batch using Jina API."""
        if not texts:
            return [[]]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",  # required by Jina API
        }

        # Build payload; only include keys that are set/non-None
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.task:
            payload["task"] = self.task
        if self.truncate:
            payload["truncate"] = True

        if self.requested_dimensions is not None:
            payload["dimensions"] = self.requested_dimensions

        if self.late_chunking:
            payload["late_chunking"] = True

        if client is None:
            # Use context manager to ensure proper AsyncClient cleanup
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                return self._process_jina_response(data)
        else:
            # Use provided client
            response = await client.post(
                "https://api.jina.ai/v1/embeddings",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return self._process_jina_response(data)

    def _process_jina_response(self, data: Dict[str, Any]) -> List[List[float]]:
        """Process Jina API response and extract embeddings."""
        if "error" in data:
            # Jina returns {"error": {"message": "..."}}
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Jina API error: {msg}")

        items = data.get("data", [])
        if not items:
            raise RuntimeError("No embeddings returned from Jina API")

        # Expect float vectors
        embeddings: List[List[float]] = []
        for item in items:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Unexpected embedding format from Jina API (expected float list).")
            embeddings.append(emb)

        return embeddings


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def __init__(
            self, model: str, dimension: int = 384, timeout: int = 90, max_retries: int = 3, retry_delay: float = 1.0,
            **kwargs: Any
            ) -> None:
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
            seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
            np.random.seed(seed)
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
EmbeddingRegistry.register("google", GoogleEmbeddings)
EmbeddingRegistry.register("jina", JinaEmbeddings)


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
