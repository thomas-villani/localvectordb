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

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        pass

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


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama embedding provider"""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url.rstrip('/')
        self._dimension = None
        self._validated = False

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def max_batch_size(self) -> int:
        return 64  # Ollama's typical batch size

    def validate_model(self) -> bool:
        """Check if the model is available in Ollama"""
        if self._validated:
            return True

        # TODO: cache the models somewhere.
        # TODO: better error handling here, need to raise an error if service is not available
        try:
            with httpx.Client() as client:
                response = client.get(f"{self.base_url}/api/tags", timeout=10.0)
                response.raise_for_status()

                data = response.json()
                models = data.get("models", [])

                for model_info in models:
                    if model_info["name"].startswith(self.model):
                        self._validated = True
                        return True

                return False

        except Exception:
            return False

    def get_dimension(self) -> int:
        """Get embedding dimension by making a test call"""
        if self._dimension is not None:
            return self._dimension

        try:
            # Make a test embedding call
            test_embedding = asyncio.run(self.embed_batch(["test"], batch_size=1))
            self._dimension = test_embedding.shape[1]
            return self._dimension
        except Exception as e:
            raise RuntimeError(f"Failed to determine embedding dimension: {e}")

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings using Ollama"""
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        batch_size = batch_size or self.max_batch_size
        all_embeddings = []

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                try:
                    response = await client.post(
                        f"{self.base_url}/api/embed",
                        json={
                            "model": self.model,
                            "input": batch,
                            "truncate": True
                        },
                        timeout=300.0
                    )
                    response.raise_for_status()

                    data = response.json()
                    if "error" in data:
                        raise RuntimeError(f"Ollama error: {data['error']}")

                    embeddings = data.get("embeddings", [])
                    if not embeddings:
                        raise RuntimeError("No embeddings returned from Ollama")

                    all_embeddings.extend(embeddings)

                except httpx.RequestError as e:
                    raise RuntimeError(f"Failed to connect to Ollama: {e}")
                except Exception as e:
                    raise RuntimeError(f"Error generating embeddings: {e}")

        return np.array(all_embeddings, dtype=np.float32)


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider"""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        self._dimension = None
        self._validated = False

        # Model dimension mapping
        self._model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

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

        # For now, just check if it's a known model
        known_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]

        self._validated = self.model in known_models
        return self._validated

    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is not None:
            return self._dimension

        # Try to get from known mappings first
        if self.model in self._model_dimensions:
            self._dimension = self._model_dimensions[self.model]
            return self._dimension

        # Otherwise make a test call
        try:
            test_embedding = asyncio.run(self.embed_batch(["test"], batch_size=1))
            self._dimension = test_embedding.shape[1]
            return self._dimension
        except Exception as e:
            raise RuntimeError(f"Failed to determine embedding dimension: {e}")

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Generate embeddings using OpenAI"""
        if not texts:
            return np.array([]).reshape(0, self.get_dimension())

        batch_size = batch_size or self.max_batch_size
        all_embeddings = []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with httpx.AsyncClient() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                try:
                    response = await client.post(
                        "https://api.openai.com/v1/embeddings",
                        headers=headers,
                        json={
                            "model": self.model,
                            "input": batch
                        },
                        timeout=300.0
                    )
                    response.raise_for_status()

                    data = response.json()

                    if "error" in data:
                        raise RuntimeError(f"OpenAI error: {data['error']['message']}")

                    embeddings = [item["embedding"] for item in data["data"]]
                    all_embeddings.extend(embeddings)

                except httpx.RequestError as e:
                    raise RuntimeError(f"Failed to connect to OpenAI: {e}")
                except Exception as e:
                    raise RuntimeError(f"Error generating embeddings: {e}")

        return np.array(all_embeddings, dtype=np.float32)


class MockEmbeddings(EmbeddingProvider):
    """Mock embedding provider for testing"""

    def __init__(self, model: str, dimension: int = 384, **kwargs):
        super().__init__(model, **kwargs)
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

    async def embed_batch(
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

        try:
            # Python 3.10+ importlib.metadata
            from importlib.metadata import entry_points

            # Look for entry points in the 'localvectordb.embedding_providers' group
            eps = entry_points()
            if hasattr(eps, 'select'):
                # Python 3.10+ API
                provider_eps = eps.select(group='localvectordb.embedding_providers')
            else:
                # Python 3.8-3.9 API
                provider_eps = eps.get('localvectordb.embedding_providers', [])

            for ep in provider_eps:
                try:
                    provider_class = ep.load()
                    cls.register(ep.name, provider_class)
                    logger.info(f"Discovered embedding provider plugin: {ep.name}")
                except Exception as e:
                    logger.warning(f"Failed to load embedding provider plugin {ep.name}: {e}")

        except ImportError:
            # Fallback for older Python versions
            try:
                import pkg_resources
                for ep in pkg_resources.iter_entry_points('localvectordb.embedding_providers'):
                    try:
                        provider_class = ep.load()
                        cls.register(ep.name, provider_class)
                        logger.info(f"Discovered embedding provider plugin: {ep.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load embedding provider plugin {ep.name}: {e}")
            except ImportError:
                logger.warning("Entry point discovery not available (importlib.metadata and pkg_resources not found)")

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