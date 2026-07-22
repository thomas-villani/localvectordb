"""
QueryCursor for streaming query results with lazy SQLite hydration.

A QueryCursor holds cached FAISS/FTS search results (lightweight ID+score pairs)
and lazily loads content and metadata from SQLite in batches as the consumer iterates.
This avoids the cost of re-querying for paginated access.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Literal, Optional

from localvectordb.core import ChunkPosition, DocumentScoringMethod, QueryResult
from localvectordb.exceptions import CursorExpiredError

if TYPE_CHECKING:
    from localvectordb.database._search import SearchMixin

logger = logging.getLogger(__name__)


@dataclass
class CursorCandidate:
    """A raw candidate from FAISS/FTS before SQLite hydration."""

    score: float
    source: Literal["vector", "keyword", "hybrid"]
    faiss_id: Optional[int] = None
    chunk_rowid: Optional[int] = None
    document_id: Optional[str] = None
    # Raw BM25 for keyword candidates. Hybrid fusion normalizes this, not `score`:
    # `_fts_rank_to_similarity` saturates to ~1.0 for any decent match.
    raw_rank: Optional[float] = None


@dataclass
class CursorConfig:
    """Configuration captured at cursor creation time."""

    search_type: Literal["vector", "keyword", "hybrid"]
    return_type: Literal["documents", "chunks", "sections", "context", "enriched"]
    search_level: Literal["chunks", "sections", "documents", "fused"]
    score_threshold: float
    filters: Optional[Dict[str, Any]]
    vector_weight: float
    context_window: int
    semantic_dedup_threshold: Optional[float]
    document_scoring_method: DocumentScoringMethod
    document_scoring_options: Optional[dict]
    total_k: int
    context_unit: str = "chunks"
    context_truncate: bool = False


@dataclass
class DocumentCandidate:
    """Pre-aggregated document-level candidate for document return type."""

    document_id: str
    score: float
    chunk_scores: List[float] = field(default_factory=list)


class QueryCursor:
    """
    Holds cached search results with lazy SQLite metadata/content loading.

    Lifecycle:
    1. Created by ``SearchMixin.query_cursor()`` which runs FAISS/FTS search
    2. Consumer calls ``fetch_batch()`` or iterates via ``stream()`` / ``__aiter__()``
    3. Each batch triggers a SQLite query for only that batch's metadata
    4. Cursor tracks position and expires after configurable TTL
    5. Must be closed explicitly or via context manager

    Parameters
    ----------
    db : LocalVectorDBBase
        Reference to the database for lazy SQLite loading.
    candidates : list of CursorCandidate
        Scored candidates from FAISS/FTS, sorted by score descending.
    config : CursorConfig
        Query configuration captured at cursor creation time.
    ttl_seconds : float
        Time-to-live in seconds (default 300 = 5 minutes).
    default_batch_size : int
        Default number of results per batch (default 50).
    """

    def __init__(
        self,
        db: "SearchMixin",
        candidates: List[CursorCandidate],
        config: CursorConfig,
        *,
        ttl_seconds: float = 300.0,
        default_batch_size: int = 50,
    ):
        self._db = db
        self._config = config
        self._created_at: float = time.monotonic()
        self._last_access: float = self._created_at
        self._ttl_seconds = ttl_seconds
        self._default_batch_size = default_batch_size
        self._closed = False
        self._exhausted = False
        self._lock = threading.Lock()
        self._position: int = 0
        # Results emitted so far. The candidate pool is deliberately over-fetched
        # (k*2..k*4) to give filtering/dedup headroom, but the cursor must still
        # return no more than the caller's requested k (``config.total_k``).
        self._emitted: int = 0
        self._candidates: List[CursorCandidate] = []
        self._doc_candidates: List[DocumentCandidate] = []

        # For document return type, pre-aggregate candidates into document scores
        if config.return_type == "documents":
            self._doc_candidates = self._aggregate_to_documents(candidates, config)
        else:
            self._candidates = candidates

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _check_alive(self) -> None:
        """Raise if cursor is closed or expired."""
        if self._closed:
            raise CursorExpiredError("Cursor has been closed")
        elapsed = time.monotonic() - self._last_access
        if elapsed > self._ttl_seconds:
            self._closed = True
            raise CursorExpiredError(f"Cursor expired after {self._ttl_seconds}s of inactivity")

    def close(self) -> None:
        """Close the cursor and release the database reference."""
        self._closed = True
        self._db = None  # type: ignore[assignment]

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "QueryCursor":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "QueryCursor":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def total_candidates(self) -> int:
        """Total number of scored candidates (or documents for document return type)."""
        if self._config.return_type == "documents":
            return len(self._doc_candidates)
        return len(self._candidates)

    @property
    def remaining(self) -> int:
        """Number of results still available (bounded by the requested ``k``)."""
        total = self.total_candidates
        unfetched = max(0, total - self._position)
        allowance = max(0, self._config.total_k - self._emitted)
        return min(unfetched, allowance)

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    # ------------------------------------------------------------------
    # Sync fetch
    # ------------------------------------------------------------------

    def fetch_batch(self, batch_size: Optional[int] = None) -> List[QueryResult]:
        """
        Fetch the next batch of results, hydrating from SQLite lazily.

        Parameters
        ----------
        batch_size : int, optional
            Number of results to fetch. Defaults to ``default_batch_size``.

        Returns
        -------
        list of QueryResult
            The next batch. Empty list when exhausted.

        Raises
        ------
        CursorExpiredError
            If the cursor has been closed or its TTL has elapsed.
        """
        with self._lock:
            self._check_alive()
            batch_size = batch_size or self._default_batch_size

            allowance = self._config.total_k - self._emitted
            if allowance <= 0:
                self._exhausted = True
                return []

            if self._config.return_type == "documents":
                results = self._fetch_document_batch(batch_size)
            else:
                results = self._fetch_chunk_batch(batch_size)
            return self._apply_k_cap(results, allowance)

    def _apply_k_cap(self, results: List[QueryResult], allowance: int) -> List[QueryResult]:
        """Truncate a hydrated batch so cumulative output never exceeds ``total_k``."""
        if len(results) > allowance:
            results = results[:allowance]
            self._exhausted = True
        self._emitted += len(results)
        return results

    def fetch_all(self) -> List[QueryResult]:
        """Fetch all remaining results at once."""
        all_results: List[QueryResult] = []
        while True:
            batch = self.fetch_batch()
            if not batch:
                break
            all_results.extend(batch)
        return all_results

    def stream(self, batch_size: Optional[int] = None) -> Iterator[List[QueryResult]]:
        """Yield batches of results as a sync generator."""
        batch_size = batch_size or self._default_batch_size
        while True:
            batch = self.fetch_batch(batch_size)
            if not batch:
                return
            yield batch

    def stream_individual(self, batch_size: Optional[int] = None) -> Iterator[QueryResult]:
        """Yield individual QueryResult objects."""
        for batch in self.stream(batch_size):
            yield from batch

    # ------------------------------------------------------------------
    # Async fetch
    # ------------------------------------------------------------------

    async def fetch_batch_async(self, batch_size: Optional[int] = None) -> List[QueryResult]:
        """
        Async version of fetch_batch. Hydrates from SQLite using the async pool.

        Parameters
        ----------
        batch_size : int, optional
            Number of results to fetch. Defaults to ``default_batch_size``.

        Returns
        -------
        list of QueryResult
        """
        with self._lock:
            self._check_alive()
            batch_size = batch_size or self._default_batch_size

            allowance = self._config.total_k - self._emitted
            if allowance <= 0:
                self._exhausted = True
                return []

            if self._config.return_type == "documents":
                results = await self._fetch_document_batch_async(batch_size)
            else:
                results = await self._fetch_chunk_batch_async(batch_size)
            return self._apply_k_cap(results, allowance)

    async def fetch_all_async(self) -> List[QueryResult]:
        """Fetch all remaining results asynchronously."""
        all_results: List[QueryResult] = []
        while True:
            batch = await self.fetch_batch_async()
            if not batch:
                break
            all_results.extend(batch)
        return all_results

    async def stream_async(self, batch_size: Optional[int] = None) -> AsyncIterator[List[QueryResult]]:
        """Yield batches of results as an async generator."""
        batch_size = batch_size or self._default_batch_size
        while True:
            batch = await self.fetch_batch_async(batch_size)
            if not batch:
                return
            yield batch

    async def stream_individual_async(self, batch_size: Optional[int] = None) -> AsyncIterator[QueryResult]:
        """Yield individual QueryResult objects asynchronously."""
        async for batch in self.stream_async(batch_size):
            for result in batch:
                yield result

    # ------------------------------------------------------------------
    # Internal: chunk-level hydration (sync)
    # ------------------------------------------------------------------

    def _fetch_chunk_batch(self, batch_size: int) -> List[QueryResult]:
        """Hydrate a batch of chunk candidates from SQLite."""
        if self._position >= len(self._candidates):
            self._exhausted = True
            return []

        end = min(self._position + batch_size, len(self._candidates))
        batch_candidates = self._candidates[self._position : end]
        self._position = end
        self._last_access = time.monotonic()

        results = self._hydrate_chunks_sync(batch_candidates)

        # Apply metadata filters per-batch
        if self._config.filters:
            from localvectordb._filters import matches_metadata_filter

            results = [r for r in results if matches_metadata_filter(r.metadata, self._config.filters)]

        # Apply context/enrichment post-processing per-batch
        if self._config.return_type == "context":
            results = self._db._add_context_window(
                results, self._config.context_window, self._config.context_unit, self._config.context_truncate
            )
        elif self._config.return_type == "enriched":
            results = self._db._enrich_with_intra_doc_context(
                results, self._config.context_window, self._config.context_unit, self._config.context_truncate
            )

        if not results and self._position < len(self._candidates):
            # Filters removed everything in this batch; fetch more
            return self._fetch_chunk_batch(batch_size)

        return results

    def _hydrate_chunks_sync(self, candidates: List[CursorCandidate]) -> List[QueryResult]:
        """Load content + metadata from SQLite for a batch of chunk candidates."""
        faiss_ids = [c.faiss_id for c in candidates if c.faiss_id is not None]
        chunk_rowids = [c.chunk_rowid for c in candidates if c.chunk_rowid is not None]

        if not faiss_ids and not chunk_rowids:
            return []

        with self._db.connection_pool.get_connection() as conn:
            if faiss_ids:
                placeholders = ",".join(["?"] * len(faiss_ids))
                cursor = conn.execute(
                    f"""
                    SELECT c.*, d.id as doc_id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.faiss_id IN ({placeholders})
                    """,
                    faiss_ids,
                )
            else:
                placeholders = ",".join(["?"] * len(chunk_rowids))
                cursor = conn.execute(
                    f"""
                    SELECT c.*, d.id as doc_id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id IN ({placeholders})
                    """,
                    chunk_rowids,
                )

            rows_by_key: Dict[int, Any] = {}
            doc_ids_to_fetch: set[str] = set()
            for row in cursor.fetchall():
                key = row["faiss_id"] if faiss_ids else row["id"]
                rows_by_key[key] = row
                doc_ids_to_fetch.add(row["doc_id"])

            doc_metadata_batch = self._db._get_documents_metadata_batch(conn, list(doc_ids_to_fetch))

        # Build QueryResults maintaining candidate order
        results: List[QueryResult] = []
        for candidate in candidates:
            key = candidate.faiss_id if candidate.faiss_id is not None else candidate.chunk_rowid
            if key is None:
                continue
            row = rows_by_key.get(key)
            if not row:
                continue

            position = ChunkPosition(
                start=row["start_pos"],
                end=row["end_pos"],
                line=row["start_line"],
                column=row["start_col"],
                end_line=row["end_line"],
                end_column=row["end_col"],
            )
            doc_metadata = doc_metadata_batch.get(row["doc_id"], {})
            result = QueryResult(
                id=f"{row['document_id']}:{row['chunk_index']}",
                score=candidate.score,
                type="chunk",
                content=row["content"],
                metadata=doc_metadata,
                document_id=row["doc_id"],
                position=position,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Internal: chunk-level hydration (async)
    # ------------------------------------------------------------------

    async def _fetch_chunk_batch_async(self, batch_size: int) -> List[QueryResult]:
        """Hydrate a batch of chunk candidates from SQLite asynchronously."""
        if self._position >= len(self._candidates):
            self._exhausted = True
            return []

        end = min(self._position + batch_size, len(self._candidates))
        batch_candidates = self._candidates[self._position : end]
        self._position = end
        self._last_access = time.monotonic()

        results = await self._hydrate_chunks_async(batch_candidates)

        if self._config.filters:
            from localvectordb._filters import matches_metadata_filter

            results = [r for r in results if matches_metadata_filter(r.metadata, self._config.filters)]

        cw = self._config.context_window
        cu = self._config.context_unit
        ct = self._config.context_truncate
        if self._config.return_type == "context":
            if hasattr(self._db, "_add_context_window_async"):
                results = await self._db._add_context_window_async(results, cw, cu, ct)
            else:
                results = self._db._add_context_window(results, cw, cu, ct)
        elif self._config.return_type == "enriched":
            if hasattr(self._db, "_enrich_with_intra_doc_context_async"):
                results = await self._db._enrich_with_intra_doc_context_async(results, cw, cu, ct)
            else:
                results = self._db._enrich_with_intra_doc_context(results, cw, cu, ct)

        if not results and self._position < len(self._candidates):
            return await self._fetch_chunk_batch_async(batch_size)

        return results

    async def _hydrate_chunks_async(self, candidates: List[CursorCandidate]) -> List[QueryResult]:
        """Load content + metadata from SQLite asynchronously."""
        faiss_ids = [c.faiss_id for c in candidates if c.faiss_id is not None]
        chunk_rowids = [c.chunk_rowid for c in candidates if c.chunk_rowid is not None]

        if not faiss_ids and not chunk_rowids:
            return []

        self._db._ensure_async_pool()
        assert self._db.async_connection_pool is not None

        async with self._db.async_connection_pool.get_connection_context() as conn:
            if faiss_ids:
                placeholders = ",".join(["?"] * len(faiss_ids))
                cursor = await conn.execute(
                    f"""
                    SELECT c.*, d.id as doc_id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.faiss_id IN ({placeholders})
                    """,
                    faiss_ids,
                )
            else:
                placeholders = ",".join(["?"] * len(chunk_rowids))
                cursor = await conn.execute(
                    f"""
                    SELECT c.*, d.id as doc_id
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE c.id IN ({placeholders})
                    """,
                    chunk_rowids,
                )

            rows_by_key: Dict[int, Any] = {}
            doc_ids_to_fetch: set[str] = set()
            async for row in cursor:
                key = row["faiss_id"] if faiss_ids else row["id"]
                rows_by_key[key] = row
                doc_ids_to_fetch.add(row["doc_id"])

            # Get metadata using the async method
            doc_metadata_batch = await self._db._get_documents_metadata_async(list(doc_ids_to_fetch))

        results: List[QueryResult] = []
        for candidate in candidates:
            key = candidate.faiss_id if candidate.faiss_id is not None else candidate.chunk_rowid
            if key is None:
                continue
            chunk_row = rows_by_key.get(key)
            if not chunk_row:
                continue

            position = ChunkPosition(
                start=chunk_row["start_pos"],
                end=chunk_row["end_pos"],
                line=chunk_row["start_line"],
                column=chunk_row["start_col"],
                end_line=chunk_row["end_line"],
                end_column=chunk_row["end_col"],
            )
            doc_metadata = doc_metadata_batch.get(chunk_row["doc_id"], {})
            result = QueryResult(
                id=f"{chunk_row['document_id']}:{chunk_row['chunk_index']}",
                score=candidate.score,
                type="chunk",
                content=chunk_row["content"],
                metadata=doc_metadata,
                document_id=chunk_row["doc_id"],
                position=position,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Internal: document-level hydration
    # ------------------------------------------------------------------

    def _aggregate_to_documents(
        self, candidates: List[CursorCandidate], config: CursorConfig
    ) -> List[DocumentCandidate]:
        """
        Pre-aggregate chunk candidates into document-level scores.

        We need document_ids from SQLite for grouping, but only fetch the IDs,
        not full content. Scores come from FAISS/FTS (already available).
        """
        if not candidates:
            return []

        # We need to look up which document each candidate belongs to
        faiss_ids = [c.faiss_id for c in candidates if c.faiss_id is not None]
        chunk_rowids = [c.chunk_rowid for c in candidates if c.chunk_rowid is not None]

        candidate_doc_map: Dict[int, str] = {}

        with self._db.connection_pool.get_connection() as conn:
            if faiss_ids:
                placeholders = ",".join(["?"] * len(faiss_ids))
                cursor = conn.execute(
                    f"SELECT faiss_id, document_id FROM chunks WHERE faiss_id IN ({placeholders})",
                    faiss_ids,
                )
                for row in cursor.fetchall():
                    candidate_doc_map[row["faiss_id"]] = row["document_id"]

            if chunk_rowids:
                placeholders = ",".join(["?"] * len(chunk_rowids))
                cursor = conn.execute(
                    f"SELECT id, document_id FROM chunks WHERE id IN ({placeholders})",
                    chunk_rowids,
                )
                for row in cursor.fetchall():
                    candidate_doc_map[row["id"]] = row["document_id"]

        # Group scores by document
        from collections import defaultdict

        doc_scores: Dict[str, List[float]] = defaultdict(list)
        for candidate in candidates:
            key = candidate.faiss_id if candidate.faiss_id is not None else candidate.chunk_rowid
            if key is None:
                continue
            doc_id = candidate_doc_map.get(key)
            if doc_id:
                doc_scores[doc_id].append(candidate.score)

        # Compute aggregated document scores using the static method
        from localvectordb.database._search import SearchMixin

        method = config.document_scoring_method
        method_options = config.document_scoring_options or {}

        doc_candidates: List[DocumentCandidate] = []
        for doc_id, scores in doc_scores.items():
            # Use a simplified scoring: compute the aggregate score from the chunk scores
            doc_groups = {
                doc_id: [
                    QueryResult(id=f"{doc_id}:0", score=s, type="chunk", content="", document_id=doc_id) for s in scores
                ]
            }
            doc_content_map = {doc_id: "placeholder"}
            doc_metadata_batch: Dict[str, Dict[str, Any]] = {doc_id: {}}
            scored = SearchMixin._compute_document_scores(
                method, method_options, doc_groups, doc_content_map, doc_metadata_batch
            )
            if scored:
                doc_candidates.append(
                    DocumentCandidate(
                        document_id=doc_id,
                        score=scored[0].score,
                        chunk_scores=scores,
                    )
                )

        doc_candidates.sort(key=lambda d: d.score, reverse=True)
        return doc_candidates

    def _fetch_document_batch(self, batch_size: int) -> List[QueryResult]:
        """Hydrate a batch of document candidates from SQLite."""
        if self._position >= len(self._doc_candidates):
            self._exhausted = True
            return []

        end = min(self._position + batch_size, len(self._doc_candidates))
        batch = self._doc_candidates[self._position : end]
        self._position = end
        self._last_access = time.monotonic()

        doc_ids = [d.document_id for d in batch]
        scores = {d.document_id: d for d in batch}

        with self._db.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(doc_ids))
            cursor = conn.execute(
                f"SELECT id, content FROM documents WHERE id IN ({placeholders})",
                doc_ids,
            )
            doc_content = {row["id"]: row["content"] for row in cursor.fetchall()}
            doc_metadata_batch = self._db._get_documents_metadata_batch(conn, doc_ids)

        if self._config.filters:
            from localvectordb._filters import matches_metadata_filter

            doc_ids = [
                did for did in doc_ids if matches_metadata_filter(doc_metadata_batch.get(did, {}), self._config.filters)
            ]

        results: List[QueryResult] = []
        for doc_id in doc_ids:
            content = doc_content.get(doc_id, "")
            if not content:
                continue
            doc_cand = scores[doc_id]
            doc_metadata = doc_metadata_batch.get(doc_id, {})

            # Add scoring metadata
            doc_metadata["_scoring"] = {
                "_aggregation_method": self._config.document_scoring_method,
                "_chunk_count": len(doc_cand.chunk_scores),
                "_best_chunk_score": max(doc_cand.chunk_scores) if doc_cand.chunk_scores else 0.0,
                "_average_chunk_score": (
                    sum(doc_cand.chunk_scores) / len(doc_cand.chunk_scores) if doc_cand.chunk_scores else 0.0
                ),
            }

            results.append(
                QueryResult(
                    id=doc_id,
                    score=doc_cand.score,
                    type="document",
                    content=content,
                    metadata=doc_metadata,
                )
            )

        if not results and self._position < len(self._doc_candidates):
            return self._fetch_document_batch(batch_size)

        return results

    async def _fetch_document_batch_async(self, batch_size: int) -> List[QueryResult]:
        """Hydrate a batch of document candidates from SQLite asynchronously."""
        if self._position >= len(self._doc_candidates):
            self._exhausted = True
            return []

        end = min(self._position + batch_size, len(self._doc_candidates))
        batch = self._doc_candidates[self._position : end]
        self._position = end
        self._last_access = time.monotonic()

        doc_ids = [d.document_id for d in batch]
        scores = {d.document_id: d for d in batch}

        self._db._ensure_async_pool()
        assert self._db.async_connection_pool is not None

        async with self._db.async_connection_pool.get_connection_context() as conn:
            placeholders = ",".join(["?"] * len(doc_ids))
            cursor = await conn.execute(
                f"SELECT id, content FROM documents WHERE id IN ({placeholders})",
                doc_ids,
            )
            doc_content: Dict[str, str] = {}
            async for row in cursor:
                doc_content[row["id"]] = row["content"]

            doc_metadata_batch = await self._db._get_documents_metadata_async(doc_ids)

        if self._config.filters:
            from localvectordb._filters import matches_metadata_filter

            doc_ids = [
                did for did in doc_ids if matches_metadata_filter(doc_metadata_batch.get(did, {}), self._config.filters)
            ]

        results: List[QueryResult] = []
        for doc_id in doc_ids:
            content = doc_content.get(doc_id, "")
            if not content:
                continue
            doc_cand = scores[doc_id]
            doc_metadata = doc_metadata_batch.get(doc_id, {})
            doc_metadata["_scoring"] = {
                "_aggregation_method": self._config.document_scoring_method,
                "_chunk_count": len(doc_cand.chunk_scores),
                "_best_chunk_score": max(doc_cand.chunk_scores) if doc_cand.chunk_scores else 0.0,
                "_average_chunk_score": (
                    sum(doc_cand.chunk_scores) / len(doc_cand.chunk_scores) if doc_cand.chunk_scores else 0.0
                ),
            }
            results.append(
                QueryResult(
                    id=doc_id,
                    score=doc_cand.score,
                    type="document",
                    content=content,
                    metadata=doc_metadata,
                )
            )

        if not results and self._position < len(self._doc_candidates):
            return await self._fetch_document_batch_async(batch_size)

        return results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "closed" if self._closed else ("exhausted" if self._exhausted else "open")
        return (
            f"QueryCursor(status={status}, total={self.total_candidates}, "
            f"remaining={self.remaining}, return_type={self._config.return_type!r})"
        )
