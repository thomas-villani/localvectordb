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
            Reranked results with updated scores, best first.

        Notes
        -----
        Every provider writes two metadata keys and leaves ``result.score`` holding
        an **absolute** ``[0, 1]`` relevance score -- one that means the same thing
        regardless of the other candidates in the batch, so it survives
        ``score_threshold`` filtering and cross-query comparison:

        * ``metadata["original_score"]`` -- the pre-rerank search score.
        * ``metadata["rerank_raw_score"]`` -- the reranker model's raw output before
          the provider's ``[0, 1]`` mapping.

        The mapping is provider-specific because the raw scores are:

        * **Jina** -- the API returns a native ``[0, 1]`` relevance score; used as-is.
        * **SentenceTransformers / HuggingFace** -- a cross-encoder logit, squashed
          with a logistic sigmoid. This is deliberately *not* a per-batch min-max:
          min-max is pool-relative (top always 1.0, bottom always 0.0) and would
          break ``score_threshold`` exactly as the pre-T1.1 hybrid fusion did.
        * **OpenAI-compatible / OpenRouter** -- a self-hosted or routed server may
          return either, so the mapping is decided per score: values already in
          ``[0, 1]`` pass through and anything else is squashed. Override with
          ``score_transform`` when you know which your server produces.
        * **Mock** -- word-overlap fraction, already in ``[0, 1]``.
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


class HTTPRerankerBase(Reranker):
    """Shared machinery for the ``/rerank`` wire format Jina popularised.

    The same request shape -- ``{model, query, documents, top_n}`` answered with
    per-document ``{index, score}`` pairs -- is served by Jina, vLLM,
    text-embeddings-inference, Infinity and OpenRouter. Only the address, the
    credential and the exact spelling of the response differ, so subclasses
    override those and inherit the request, retry and scoring behaviour.

    Response shapes accepted, because servers disagree on all three axes:

    * ``{"results": [...]}`` (Jina, vLLM, OpenRouter), ``{"data": [...]}``, or a
      bare top-level list (text-embeddings-inference).
    * ``relevance_score`` or ``score`` for the value.
    * Scores already in ``[0, 1]``, or raw cross-encoder logits.

    That last one matters more than it looks. The module contract is that
    ``result.score`` is an *absolute* ``[0, 1]`` relevance, because
    ``score_threshold`` filtering and cross-query comparison depend on it. A
    server returning logits would silently break both, so ``score_transform``
    defaults to squashing anything that falls outside ``[0, 1]``.
    """

    #: Endpoint root used when the caller gives no base_url. None means required.
    DEFAULT_BASE_URL: Optional[str] = None
    #: Environment variable consulted when no api_key is passed.
    API_KEY_ENV: Optional[str] = None
    #: Whether construction fails without a credential.
    REQUIRES_API_KEY: bool = False
    #: Human-readable name used in log and error messages.
    DISPLAY_NAME: str = "Reranker"
    #: Message raised when a required key is absent.
    MISSING_KEY_MESSAGE: str = "An API key is required."

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 90,
        max_retries: int = 3,
        score_transform: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, max_retries=max_retries, **kwargs)

        api_key = resolve_env_ref(api_key, what="api_key")
        if not api_key and self.API_KEY_ENV:
            api_key = os.getenv(self.API_KEY_ENV)
        self.api_key = api_key
        if self.REQUIRES_API_KEY and not self.api_key:
            raise ValueError(self.MISSING_KEY_MESSAGE)

        resolved_base_url = base_url or self.DEFAULT_BASE_URL
        if not resolved_base_url:
            raise ValueError(
                f"base_url is required for the {self.provider_name} reranker -- there is no default "
                f"endpoint to fall back to. Point it at your server's rerank root, e.g. "
                f"http://localhost:8000/v1 (vLLM) or http://localhost:8080/v1 "
                f"(text-embeddings-inference)."
            )
        self.base_url = resolved_base_url.rstrip("/")

        if score_transform not in ("auto", "none", "sigmoid"):
            raise ValueError(f"Unknown score_transform: {score_transform!r}. Use 'auto', 'none' or 'sigmoid'.")
        self.score_transform = score_transform

    def validate_model(self) -> bool:
        # The endpoint decides what it serves; a bad name surfaces on first use.
        return True

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/rerank"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # Local servers usually have no credential, and some reject a malformed
        # Authorization header outright, so omit it entirely when absent.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(self, query: str, results: List[QueryResult], top_k: Optional[int]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [r.content or "" for r in results],
        }
        if top_k is not None:
            payload["top_n"] = top_k
        return payload

    def _to_absolute_score(self, raw: float) -> float:
        """Map a server's raw score into the absolute [0, 1] the module promises."""
        if self.score_transform == "none":
            return raw
        if self.score_transform == "auto" and 0.0 <= raw <= 1.0:
            return raw
        return float(1.0 / (1.0 + np.exp(-raw)))

    def _parse(self, data: Any, results: List[QueryResult], top_k: Optional[int]) -> List[QueryResult]:
        items: List[Dict[str, Any]]
        if isinstance(data, dict):
            # `or` rather than a get() default: a server that sends an explicit
            # null for the envelope should be treated as empty, not iterated.
            items = data.get("results") or data.get("data") or []
        elif isinstance(data, list):
            items = data
        else:
            raise RerankerError(f"{self.DISPLAY_NAME} returned an unreadable response of type {type(data).__name__}")

        reranked = []
        for item in items:
            idx = item["index"]
            if not 0 <= idx < len(results):
                raise RerankerError(f"{self.DISPLAY_NAME} returned index {idx} for a batch of {len(results)} documents")
            if "relevance_score" in item:
                raw = float(item["relevance_score"])
            elif "score" in item:
                raw = float(item["score"])
            else:
                raise RerankerError(
                    f"{self.DISPLAY_NAME} returned a result with no relevance_score or score field: {item!r}"
                )

            result = results[idx]
            if result.metadata is None:
                result.metadata = {}
            result.metadata["original_score"] = result.score
            result.metadata["rerank_raw_score"] = raw
            result.score = self._to_absolute_score(raw)
            reranked.append(result)

        reranked.sort(key=lambda x: x.score, reverse=True)
        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked

    def rerank(self, query: str, results: List[QueryResult], top_k: Optional[int] = None) -> List[QueryResult]:
        if not results:
            return results

        import httpx

        headers = self._headers()
        payload = self._payload(query, results, top_k)

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return self._parse(data, results, top_k)
            except Exception as e:
                if attempt >= self.max_retries:
                    raise RerankerError(f"{self.DISPLAY_NAME} reranking failed: {e}") from e
                logger.warning(f"{self.DISPLAY_NAME} rerank attempt {attempt + 1} failed: {e}")

        raise RerankerError(f"All {self.DISPLAY_NAME} reranking attempts failed")

    async def rerank_async(
        self, query: str, results: List[QueryResult], top_k: Optional[int] = None
    ) -> List[QueryResult]:
        if not results:
            return results

        import httpx

        headers = self._headers()
        payload = self._payload(query, results, top_k)

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                return self._parse(data, results, top_k)
            except Exception as e:
                if attempt >= self.max_retries:
                    raise RerankerError(f"{self.DISPLAY_NAME} reranking failed: {e}") from e
                logger.warning(f"{self.DISPLAY_NAME} rerank attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1.0 * (2**attempt))

        raise RerankerError(f"All {self.DISPLAY_NAME} reranking attempts failed")


class JinaReranker(HTTPRerankerBase):
    """Jina AI reranker using the Jina Reranker API.

    Parameters
    ----------
    model : str
        The Jina reranker model. Default: "jina-reranker-v2-base-multilingual"
    api_key : str, optional
        API key. Falls back to JINA_API_KEY env var.
    base_url : str, optional
        Override the API root. Defaults to ``https://api.jina.ai/v1``. Useful for
        a proxy; to reach a self-hosted reranker prefer the ``openai_compatible``
        provider, which does not require a credential.
    timeout : int
        Request timeout in seconds.
    max_retries : int
        Number of retry attempts.
    """

    DEFAULT_BASE_URL = "https://api.jina.ai/v1"
    API_KEY_ENV = "JINA_API_KEY"
    REQUIRES_API_KEY = True
    DISPLAY_NAME = "Jina"
    MISSING_KEY_MESSAGE = (
        "Jina API key is required. Set JINA_API_KEY environment variable. "
        "Get your key at: https://jina.ai/?sui=apikey"
    )

    def __init__(self, model: str = "jina-reranker-v2-base-multilingual", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)

    @property
    def provider_name(self) -> str:
        return "jina"


class OpenAICompatibleReranker(HTTPRerankerBase):
    """Any self-hosted server exposing a Jina/Cohere-shaped ``/rerank`` endpoint.

    The reranking counterpart to
    :class:`~localvectordb.embeddings.OpenAICompatibleEmbeddings`, so a fully
    local stack does not have to fall back to a hosted API for its second stage.

    =========================  ====================================
    Server                     Typical ``base_url``
    =========================  ====================================
    vLLM                       ``http://localhost:8000/v1``
    text-embeddings-inference  ``http://localhost:8080``
    Infinity                   ``http://localhost:7997``
    =========================  ====================================

    Parameters
    ----------
    model : str
        Model name as the server reports it.
    base_url : str
        Required. The rerank root; ``/rerank`` is appended. Note that
        text-embeddings-inference serves it at the root rather than under
        ``/v1``.
    api_key : str, optional
        Sent as a bearer token when present. Optional -- most local servers need
        none. Falls back to ``RERANKER_API_KEY``.
    score_transform : {"auto", "none", "sigmoid"}, default "auto"
        How to map raw scores into the absolute ``[0, 1]`` range the rest of the
        library assumes. ``"auto"`` passes through values already in range and
        squashes anything else, which is what makes a logit-returning
        cross-encoder safe to use with ``score_threshold``.
    """

    API_KEY_ENV = "RERANKER_API_KEY"
    DISPLAY_NAME = "OpenAI-compatible"

    @property
    def provider_name(self) -> str:
        return "openai_compatible"


class OpenRouterReranker(HTTPRerankerBase):
    """OpenRouter reranker, routing to the rerank models it hosts.

    Parameters
    ----------
    model : str
        OpenRouter model slug for a rerank-capable model.
    api_key : str, optional
        Falls back to ``OPENROUTER_API_KEY``. Get a key at
        https://openrouter.ai/keys
    base_url : str, optional
        Defaults to ``https://openrouter.ai/api/v1``.
    """

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    API_KEY_ENV = "OPENROUTER_API_KEY"
    REQUIRES_API_KEY = True
    DISPLAY_NAME = "OpenRouter"
    MISSING_KEY_MESSAGE = (
        "OpenRouter API key is required. Set the OPENROUTER_API_KEY environment "
        "variable or pass api_key=. Get a key at https://openrouter.ai/keys"
    )

    @property
    def provider_name(self) -> str:
        return "openrouter"


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

        # ``CrossEncoder.predict`` applies the model's default activation, which
        # is a Sigmoid for the num_labels=1 rerankers used here. Applying our own
        # sigmoid on top of that compresses every score into ~[0.5, 0.73] and
        # breaks the documented absolute [0, 1] scale. Request the raw logits
        # instead (the activation kwarg was renamed across the supported
        # sentence-transformers range, so probe for it) and map them exactly
        # once. Falls back to whatever ``predict`` returns if neither name exists.
        import inspect

        identity = lambda logits: logits  # noqa: E731 - keep raw logits
        predict_params = inspect.signature(cross_encoder.predict).parameters
        if "activation_fn" in predict_params:
            scores = cross_encoder.predict(pairs, activation_fn=identity)
        elif "activation_fct" in predict_params:
            scores = cross_encoder.predict(pairs, activation_fct=identity)
        else:
            scores = cross_encoder.predict(pairs)

        # Absolute [0, 1] mapping of the cross-encoder logit via a logistic sigmoid.
        scores_array = np.array(scores, dtype=np.float64)
        normalized = 1.0 / (1.0 + np.exp(-scores_array))

        for i, result in enumerate(results):
            if result.metadata is None:
                result.metadata = {}
            result.metadata["original_score"] = result.score
            result.metadata["rerank_raw_score"] = float(scores_array[i])
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

                # Absolute [0, 1] mapping via a logistic sigmoid, matching the local
                # cross-encoder path. NOT per-batch min-max: that is pool-relative
                # (top always 1.0, bottom always 0.0), which breaks score_threshold
                # and cross-query comparison -- the same defect T1.1 removed from
                # hybrid fusion. See the Reranker.rerank docstring.
                scores_array = np.array(scores, dtype=np.float64)
                normalized = 1.0 / (1.0 + np.exp(-scores_array))

                for i, result in enumerate(results):
                    if result.metadata is None:
                        result.metadata = {}
                    result.metadata["original_score"] = result.score
                    result.metadata["rerank_raw_score"] = float(scores_array[i])
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
            result.metadata["rerank_raw_score"] = overlap
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
RerankerRegistry.register("openai_compatible", OpenAICompatibleReranker)
RerankerRegistry.register("openrouter", OpenRouterReranker)
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
