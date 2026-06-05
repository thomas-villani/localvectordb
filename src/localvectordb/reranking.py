# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Reranking providers for LocalVectorDB.

This module provides cross-encoder and API-based reranking to improve search result
quality by re-scoring candidates with more powerful models.
"""

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import numpy as np

from localvectordb.core import QueryResult
from localvectordb.exceptions import RerankerError
from localvectordb.utils import resolve_env_ref

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for reranking providers."""

    def __init__(self, model: str, *, timeout: int = 90, max_retries: int = 3, **kwargs: Any) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.config = kwargs

    @abstractmethod
    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        """Rerank search results synchronously.

        Parameters
        ----------
        query : str
            The original query text.
        results : List[QueryResult]
            Search results to rerank.
        top_k : int, optional
            Maximum number of results to return. If None, returns all.

        Returns
        -------
        List[QueryResult]
            Reranked results with updated scores. Original scores are preserved
            in result.metadata["original_score"].
        """
        pass

    async def rerank_async(
        self, query: str, results: List[QueryResult], top_k: Optional[int] = None
    ) -> List[QueryResult]:
        """Rerank search results asynchronously. Default delegates to sync."""
        return self.rerank(query, results, top_k)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the reranker provider name."""
        pass

    @abstractmethod
    def validate_model(self) -> bool:
        """Check if the model is available/valid."""
        pass


class JinaReranker(Reranker):
    """Jina AI reranker using the Jina Reranker API.

    Parameters
    ----------
    model : str
        The Jina reranker model. Default: "jina-reranker-v2-base-multilingual"
    api_key : str, optional
        API key. Falls back to JINA_API_KEY env var.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retry attempts.
    """

    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        *,
        api_key: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, max_retries=max_retries, **kwargs)

        api_key = resolve_env_ref(api_key, what="api_key")

        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key is required. Set JINA_API_KEY environment variable. "
                "Get your key at: https://jina.ai/?sui=apikey"
            )

    @property
    def provider_name(self) -> str:
        return "jina"

    def validate_model(self) -> bool:
        return True

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        if not results:
            return results

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        documents = [r.content or "" for r in results]
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            payload["top_n"] = top_k

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                reranked = []
                for item in data.get("results", []):
                    idx = item["index"]
                    score = float(item["relevance_score"])
                    result = results[idx]
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["original_score"] = result.score
                    result.score = score
                    reranked.append(result)

                reranked.sort(key=lambda x: x.score, reverse=True)
                if top_k is not None:
                    reranked = reranked[:top_k]
                return reranked

            except Exception as e:
                if attempt >= self.max_retries:
                    raise RerankerError(f"Jina reranking failed: {e}") from e
                logger.warning(f"Jina rerank attempt {attempt + 1} failed: {e}")

        raise RerankerError("All Jina reranking attempts failed")

    async def rerank_async(
        self, query: str, results: List[QueryResult], top_k: Optional[int] = None
    ) -> List[QueryResult]:
        if not results:
            return results

        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        documents = [r.content or "" for r in results]
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }
        if top_k is not None:
            payload["top_n"] = top_k

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        "https://api.jina.ai/v1/rerank",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()

                reranked = []
                for item in data.get("results", []):
                    idx = item["index"]
                    score = float(item["relevance_score"])
                    result = results[idx]
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["original_score"] = result.score
                    result.score = score
                    reranked.append(result)

                reranked.sort(key=lambda x: x.score, reverse=True)
                if top_k is not None:
                    reranked = reranked[:top_k]
                return reranked

            except Exception as e:
                if attempt >= self.max_retries:
                    raise RerankerError(f"Jina reranking failed: {e}") from e
                logger.warning(f"Jina rerank attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1.0 * (2**attempt))

        raise RerankerError("All Jina reranking attempts failed")


class SentenceTransformersReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers.

    Parameters
    ----------
    model : str
        The cross-encoder model name. Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device : str, optional
        Device for inference (cpu/cuda/mps). Default: auto-detect.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, max_retries=max_retries, **kwargs)
        self.device = device
        self._cross_encoder = None

    def _load_model(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersReranker. "
                "Install it with: pip install sentence-transformers"
            ) from e
        kwargs = {}
        if self.device is not None:
            kwargs["device"] = self.device
        self._cross_encoder = CrossEncoder(self.model, **kwargs)
        return self._cross_encoder

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"

    def validate_model(self) -> bool:
        try:
            self._load_model()
            return True
        except Exception:
            return False

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        if not results:
            return results

        cross_encoder = self._load_model()
        pairs = [[query, r.content or ""] for r in results]
        scores = cross_encoder.predict(pairs)

        # Normalize scores to 0-1 range using sigmoid
        scores_array = np.array(scores, dtype=np.float64)
        normalized = 1.0 / (1.0 + np.exp(-scores_array))

        for i, result in enumerate(results):
            if result.metadata is None:
                result.metadata = {}
            result.metadata["original_score"] = result.score
            result.score = float(normalized[i])

        ranked = sorted(results, key=lambda x: x.score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


class HuggingFaceReranker(Reranker):
    """HuggingFace Inference API reranker.

    Parameters
    ----------
    model : str
        HuggingFace model ID. Default: "BAAI/bge-reranker-v2-m3"
    api_key : str, optional
        API key. Falls back to HF_TOKEN / HUGGINGFACE_TOKEN env vars.
    base_url : str, optional
        API base URL. Default: https://api-inference.huggingface.co
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, max_retries=max_retries, **kwargs)

        api_key = resolve_env_ref(api_key, what="api_key")

        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.base_url = (base_url or "https://api-inference.huggingface.co").rstrip("/")

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def validate_model(self) -> bool:
        return True

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        if not results:
            return results

        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        documents = [r.content or "" for r in results]
        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": documents,
            }
        }

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/models/{self.model}",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    scores = response.json()

                if not isinstance(scores, list):
                    raise RerankerError(f"Unexpected response format: {type(scores)}")

                # Normalize scores to 0-1
                scores_array = np.array(scores, dtype=np.float64)
                min_s, max_s = scores_array.min(), scores_array.max()
                if max_s > min_s:
                    normalized = (scores_array - min_s) / (max_s - min_s)
                else:
                    normalized = np.ones_like(scores_array)

                for i, result in enumerate(results):
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["original_score"] = result.score
                    result.score = float(normalized[i])

                ranked = sorted(results, key=lambda x: x.score, reverse=True)
                if top_k is not None:
                    ranked = ranked[:top_k]
                return ranked

            except RerankerError:
                raise
            except Exception as e:
                if attempt >= self.max_retries:
                    raise RerankerError(f"HuggingFace reranking failed: {e}") from e
                logger.warning(f"HuggingFace rerank attempt {attempt + 1} failed: {e}")

        raise RerankerError("All HuggingFace reranking attempts failed")


class MockReranker(Reranker):
    """Mock reranker for testing. Uses word-overlap scoring."""

    def __init__(self, model: str = "mock-reranker", *, timeout: int = 90, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(model, timeout=timeout, max_retries=max_retries, **kwargs)

    @property
    def provider_name(self) -> str:
        return "mock"

    def validate_model(self) -> bool:
        return True

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        if not results:
            return results

        query_words = set(query.lower().split())

        for result in results:
            content_words = set((result.content or "").lower().split())
            if query_words:
                overlap = len(query_words & content_words) / len(query_words)
            else:
                overlap = 0.0

            if result.metadata is None:
                result.metadata = {}
            result.metadata["original_score"] = result.score
            result.score = overlap

        ranked = sorted(results, key=lambda x: x.score, reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked


class RerankerRegistry:
    """Registry for reranker providers with plugin discovery."""

    _providers: Dict[str, Type[Reranker]] = {}
    _plugins_discovered = False

    @classmethod
    def register(cls, name: str, provider_class: Type[Reranker]) -> None:
        """Register a new reranker provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def _discover_plugins(cls) -> None:
        """Discover reranker provider plugins using entry points."""
        if cls._plugins_discovered:
            return

        from importlib.metadata import entry_points

        provider_eps = entry_points(group="localvectordb.reranker_providers")
        for ep in provider_eps:
            try:
                provider_class = ep.load()
                cls.register(ep.name, provider_class)
                logger.info(f"Discovered reranker provider plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load reranker provider plugin {ep.name}: {e}")

        cls._plugins_discovered = True

    @classmethod
    def get(cls, name: str) -> Type[Reranker]:
        """Get a reranker provider by name."""
        cls._discover_plugins()

        name = name.lower()
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown reranker provider: {name}. " f"Available providers: {available}")
        return cls._providers[name]

    @classmethod
    def create_reranker(cls, provider_name: str, model: Optional[str] = None, **kwargs: Any) -> Reranker:
        """Create a reranker instance."""
        provider_class = cls.get(provider_name)
        if model is not None:
            return provider_class(model, **kwargs)
        return provider_class(**kwargs)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered reranker providers."""
        cls._discover_plugins()
        return list(cls._providers.keys())

    @classmethod
    def refresh_plugins(cls) -> None:
        """Force re-discovery of plugins (useful for testing)."""
        cls._plugins_discovered = False
        cls._discover_plugins()


# Auto-register built-in rerankers
RerankerRegistry.register("jina", JinaReranker)
RerankerRegistry.register("sentence_transformers", SentenceTransformersReranker)
RerankerRegistry.register("huggingface", HuggingFaceReranker)
RerankerRegistry.register("mock", MockReranker)


# Convenience functions
def create_reranker(provider: str, model: Optional[str] = None, **kwargs: Any) -> Reranker:
    """Create a reranker instance."""
    return RerankerRegistry.create_reranker(provider, model, **kwargs)


def list_rerankers() -> List[str]:
    """List available reranker providers."""
    return RerankerRegistry.list()
