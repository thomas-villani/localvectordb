"""PrecomputedEmbeddings provider for injecting SIFT vectors into LocalVectorDB."""

import re
from typing import Any, List, Optional

import numpy as np

from localvectordb.embeddings import EmbeddingProvider


class PrecomputedEmbeddings(EmbeddingProvider):
    """Embedding provider that returns precomputed vectors.

    Maps document text of the form ``"sift_vec_{i}"`` to ``vectors[i]``.
    Query vectors are registered explicitly via :meth:`register_query`.
    """

    def __init__(
        self,
        model: str = "precomputed",
        *,
        vectors: Optional[np.ndarray] = None,
        dimension: int = 128,
        timeout: int = 90,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, timeout=timeout, max_retries=max_retries, retry_delay=retry_delay, **kwargs)
        self._vectors = vectors  # shape (N, D)
        self._dimension = dimension
        self._query_map: dict[str, np.ndarray] = {}
        self._vec_pattern = re.compile(r"^sift_vec_(\d+)$")

    @property
    def provider_name(self) -> str:
        return "precomputed"

    @property
    def max_batch_size(self) -> int:
        return 10_000

    def validate_model(self) -> bool:
        return True

    def get_dimension(self) -> int:
        return self._dimension

    def set_vectors(self, vectors: np.ndarray) -> None:
        """Set the precomputed vector table."""
        self._vectors = vectors
        self._dimension = vectors.shape[1]

    def register_query(self, text: str, vector: np.ndarray) -> None:
        """Register a query text → vector mapping."""
        self._query_map[text] = vector

    def clear_queries(self) -> None:
        """Clear all registered query mappings."""
        self._query_map.clear()

    def _resolve(self, text: str) -> List[float]:
        """Resolve a text string to a vector."""
        # Check query map first
        if text in self._query_map:
            return self._query_map[text].tolist()
        # Try sift_vec_{i} pattern
        m = self._vec_pattern.match(text)
        if m and self._vectors is not None:
            idx = int(m.group(1))
            return self._vectors[idx].tolist()
        # Fallback: return zeros (shouldn't happen in correct usage)
        return [0.0] * self._dimension

    async def _embed_single_batch(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        return [self._resolve(t) for t in texts]
