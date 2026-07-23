"""
Query/search (sync + async), deduplication, context windows, and scoring.

This module keeps the original logic with minimal changes, grouped by purpose.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from abc import ABC
from collections import defaultdict
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Literal, Optional, Tuple

import faiss
import numpy as np

from localvectordb._filters import (
    FilterQueryBuilder,
    FTSQuerySanitization,
    matches_metadata_filter,
    validate_filter_spec,
)
from localvectordb.core import ChunkPosition, DocumentScoringMethod, GrepMatch, MetadataFieldType, QueryResult
from localvectordb.cursor import (
    CursorCandidate,
    CursorConfig,
    QueryCursor,
)
from localvectordb.database._utils import glob_escape
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.exceptions import _RERANK_STREAMING_UNSUPPORTED, DatabaseError, MetadataFilterError

if TYPE_CHECKING:
    from faiss import Index

    from localvectordb._pools import AsyncConnectionPool, ConnectionPool, ReadWriteLock
    from localvectordb.section_detection import SectionDetector

logger = logging.getLogger(__name__)

# Units in which a context/enriched budget may be expressed. ``"chunks"`` is the
# historical behaviour (``context_window`` counts whole chunks); the others treat
# ``context_window`` as an approximate budget on the assembled context content.
ContextUnit = Literal["chunks", "tokens", "words", "characters"]
CONTEXT_UNITS: Tuple[str, ...] = ("chunks", "tokens", "words", "characters")

# Separator inserted between adjacent chunks when assembling context/enriched text.
_CONTEXT_SEPARATOR = "\n\n---\n\n"

# Ceiling on the candidate pool fetched before reranking. A cross-encoder pass is
# O(pool) model calls, so an unbounded `rerank_k` would let a caller trigger an
# arbitrarily expensive rerank. 200 comfortably clears the `5*k` default for any
# reasonable `k` while capping the worst case.
_RERANK_K_MAX = 200

# Over-fetch factor when rolling section hits up to their parent documents. Several
# sections of one document collapse into a single document result, so fetching
# exactly `k` sections can yield far fewer than `k` documents -- the same starvation
# `_resolve_rerank_k` exists to prevent, one level up.
_SECTION_ROLLUP_OVERFETCH = 5

ReturnType = Literal["documents", "chunks", "sections", "context", "enriched"]

# The unit each search level answers in when the caller does not say. `return_type`
# defaults to None rather than "documents" so that "I want documents" is
# distinguishable from "I did not ask": a bare query(search_level="sections") wants
# sections, while an explicit return_type="documents" wants them rolled up.
_NATURAL_RETURN_TYPE: Dict[str, ReturnType] = {
    "chunks": "documents",
    "sections": "sections",
    "documents": "documents",
    "fused": "documents",
}


def _resolve_return_type(return_type: Optional[ReturnType], search_level: str) -> ReturnType:
    """Fill in the unit a search level answers in when the caller did not choose."""
    if return_type is not None:
        return return_type
    return _NATURAL_RETURN_TYPE.get(search_level, "documents")


def _resolve_rerank_k(rerank_k: Optional[int], k: int) -> int:
    """Size of the candidate pool to fetch before reranking down to ``k``.

    A reranker can only improve on the search legs if it sees candidates those
    legs ranked *below* position ``k`` -- fetching exactly ``k`` and reranking is
    a no-op on recall, which is the defect T1.2 fixes. The default over-fetch is
    ``5*k``, clamped to ``[k, _RERANK_K_MAX]`` (never below ``k``: reranking must
    not shrink the pool the caller asked for).
    """
    requested = rerank_k if rerank_k is not None else 5 * k
    return max(k, min(requested, _RERANK_K_MAX))


def _faiss_search_with_selector(
    index: Any,
    query_embedding: np.ndarray,
    k: int,
    faiss_ids: "np.ndarray",
) -> Tuple[np.ndarray, np.ndarray]:
    """Search ``index`` for the top ``k`` vectors *restricted to* ``faiss_ids``.

    Pushes a metadata filter into FAISS: only vectors whose id is in
    ``faiss_ids`` are considered, so the top ``k`` returned are the ``k``
    best-scoring *matching* vectors -- never starved by a fixed pre-filter pool.
    The caller must have verified ``supports_id_selector`` (IndexLSH rejects the
    selector). ``faiss_ids`` is kept referenced for the duration of the search
    because ``IDSelectorBatch`` reads from it.
    """
    selector = faiss.IDSelectorBatch(faiss_ids)
    params = faiss.SearchParameters()
    params.sel = selector
    distances, indices = index.search(query_embedding, k, params=params)
    return distances, indices


_token_encoder: Any = None


def _minmax_normalize(values: List[float]) -> List[float]:
    """Scale ``values`` into ``[0, 1]`` relative to their own min and max."""
    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span <= 1e-12:
        # A single candidate, or a pool of identical scores. There is nothing to
        # rank, so every member is equally the best of what was retrieved.
        return [1.0] * len(values)
    return [(value - low) / span for value in values]


def _relative_score_fusion(
    vector_scores: Dict[str, float],
    keyword_ranks: Dict[str, float],
    vector_weight: float,
) -> Dict[str, float]:
    """Fuse the vector and keyword legs after normalizing each within this query.

    The two legs arrive on incompatible, corpus-dependent scales: a bounded
    similarity (``1/(1+L2)`` or ``(ip+1)/2``) against raw BM25, whose range depends
    on the corpus and the query's term rarity. Summing them directly lets whichever
    leg happens to span the wider range decide the ranking, regardless of
    ``vector_weight``. Min-max within the query's own candidate pool is what makes
    ``vector_weight`` an actual blend.

    ``keyword_ranks`` must be *raw* BM25 (negative; more negative is better), never
    the output of ``_fts_rank_to_similarity``: that transform maps every decent
    match into a band ~2e-05 wide below 1.0, and saturates to exactly 1.0 once the
    rank drops past about -36. Normalizing it would normalize float noise.

    Two consequences worth knowing, both inherent to relative-score fusion:

    * A chunk retrieved by only one leg scores 0.0 on the other -- the same value the
      *worst* chunk of that leg normalizes to. Being retrieved last by a leg and not
      being retrieved by it at all are indistinguishable here.
    * Scores are relative to this query's candidate pool, so they are comparable within
      one result set but not across queries, and not across different ``k`` (which
      changes the pool size). ``score_threshold`` is therefore a threshold on rank
      position within the pool, not on absolute match quality.

    Returns ``{key: fused_score}`` in vector-leg-first insertion order, so that a
    subsequent stable sort breaks ties toward the vector ranking, as before.
    """
    normalized_vector = dict(zip(vector_scores, _minmax_normalize(list(vector_scores.values())), strict=True))
    # BM25 is negative-is-better, so negate before normalizing to get higher-is-better.
    normalized_keyword = dict(
        zip(keyword_ranks, _minmax_normalize([-rank for rank in keyword_ranks.values()]), strict=True)
    )

    fused: Dict[str, float] = {}
    for key in (*vector_scores, *keyword_ranks):
        if key in fused:
            continue
        fused[key] = vector_weight * normalized_vector.get(key, 0.0) + (1.0 - vector_weight) * normalized_keyword.get(
            key, 0.0
        )
    return fused


def _two_leg_minmax_fuse(
    primary: Dict[str, float],
    secondary: Dict[str, float],
    secondary_weight: float,
) -> Dict[str, float]:
    """Fuse two bounded-similarity legs after min-max normalizing each within its own pool.

    Both legs arrive as ``{key: similarity}`` on possibly different scales -- a chunk
    cosine against a section cosine that comes from a different index/metric. Min-max
    within each leg's own candidate pool puts them on a common ``[0, 1]`` scale, then
    they are blended:
    ``(1 - secondary_weight) * primary + secondary_weight * secondary``.
    ``secondary_weight`` is the weight on the *secondary* (section) leg: ``0.0`` is
    primary-only, ``1.0`` is secondary-only.

    This mirrors the measured harness fusion (``benchmarks/eval_hierarchical.fuse``).
    Unlike ``_relative_score_fusion``, both legs are higher-is-better similarities with
    no BM25 negation. A key in only one leg scores ``0.0`` on the other -- the same
    value that leg's worst candidate normalizes to. Primary-leg-first insertion order
    breaks ties toward the primary (chunk) ranking under a later stable sort.
    """
    norm_primary = dict(zip(primary, _minmax_normalize(list(primary.values())), strict=True))
    norm_secondary = dict(zip(secondary, _minmax_normalize(list(secondary.values())), strict=True))
    fused: Dict[str, float] = {}
    for key in (*primary, *secondary):
        if key in fused:
            continue
        fused[key] = (1.0 - secondary_weight) * norm_primary.get(key, 0.0) + secondary_weight * norm_secondary.get(
            key, 0.0
        )
    return fused


def _get_token_encoder() -> Any:
    """Lazily build (and cache) the tiktoken encoder used for token budgets.

    Chunks store their token count at ingest time, so this is only needed as a
    fallback (a chunk with a missing/zero count) and for hard token truncation.
    """
    global _token_encoder
    if _token_encoder is None:
        import tiktoken

        _token_encoder = tiktoken.get_encoding("cl100k_base")
    return _token_encoder


def _validate_context_unit(context_unit: str) -> None:
    """Validate a ``context_unit`` value, raising ValueError on an unknown unit."""
    if context_unit not in CONTEXT_UNITS:
        raise ValueError(f"Invalid context_unit {context_unit!r}; must be one of {CONTEXT_UNITS}")


def _measure_text(content: str, tokens: Optional[int], unit: str) -> int:
    """Measure ``content`` in ``unit`` (``tokens``/``words``/``characters``).

    For ``tokens`` the pre-computed per-chunk ``tokens`` count is preferred; if it
    is missing or zero the text is tokenised on the fly.
    """
    if unit == "tokens":
        if tokens:
            return int(tokens)
        if not content:
            return 0
        return len(_get_token_encoder().encode(content))
    if unit == "words":
        return len(content.split())
    # characters
    return len(content)


def _truncate_text_to_budget(text: str, budget: int, unit: str) -> str:
    """Hard-truncate ``text`` to at most ``budget`` units, never exceeding it.

    Truncation keeps the leading portion of the assembled context (the matched
    chunk sits at/near the front). Word/character cuts back off to a whitespace
    boundary to avoid slicing mid-word; token cuts are exact via tiktoken.
    """
    if budget <= 0:
        return ""
    if unit == "tokens":
        encoder = _get_token_encoder()
        token_ids = encoder.encode(text)
        if len(token_ids) <= budget:
            return text
        return str(encoder.decode(token_ids[:budget]))
    if unit == "words":
        matches = list(re.finditer(r"\S+", text))
        if len(matches) <= budget:
            return text
        return text[: matches[budget - 1].end()]
    # characters
    if len(text) <= budget:
        return text
    cut = text[:budget]
    # Back off to the last whitespace so we do not end mid-word, unless the
    # single leading token already fills the whole budget.
    last_ws = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
    if last_ws > 0:
        cut = cut[:last_ws]
    return cut.rstrip()


class SearchMixin(LocalVectorDBBase, ABC):

    # Redeclare attributes from LocalVectorDBBase and composed class as non-Optional.
    # At runtime these are always initialized before any mixin methods are called.
    _read_write_lock: "ReadWriteLock"
    connection_pool: "ConnectionPool"
    async_connection_pool: Optional["AsyncConnectionPool"]
    index: Optional["Index"]

    # Declare attributes from the composed class not on LocalVectorDBBase.
    _hierarchical_embeddings: bool
    _faiss_lock: "ReadWriteLock"
    section_index: Optional["Index"]
    document_index: Optional["Index"]
    _section_detector: Optional["SectionDetector"]

    # _distance_to_similarity is implemented in _core.py (LocalVectorDBCore).
    # Declared under TYPE_CHECKING so mypy sees it without shadowing at runtime.
    if TYPE_CHECKING:

        def _distance_to_similarity(self, distance: float, metric_type: Optional[str] = None) -> float: ...

        def _distances_to_similarities(
            self, distances: "np.ndarray", metric_type: Optional[str] = None
        ) -> "np.ndarray": ...

        @staticmethod
        def _detect_faiss_metric_type(index: Any) -> str: ...

        # Implemented on LocalVectorDBCore; declared here so mypy sees it without
        # a runtime stub shadowing the real method via the MRO.
        @classmethod
        def _index_supports_id_selector(cls, index: Any) -> bool: ...

    # -----------------
    # Helper methods
    # -----------------
    def _fts_rank_to_similarity(self, rank: float) -> float:
        """
        Convert FTS5 bm25 rank to similarity score with consistent formula.

        Parameters
        ----------
        rank : float
            The FTS5 bm25 score (negative values, lower/more negative is better)

        Returns
        -------
        float
            Similarity score between 0.0 and 1.0 (higher is better)
        """
        # BM25 scores are negative, more negative values indicate better matches
        # Convert to similarity where higher values are better
        # Use exponential mapping: for negative ranks, exp(rank) < 1, so 1-exp(rank) > 0
        return 1.0 - min(1.0, math.exp(float(rank)))

    # --------------------------------------
    # Metadata-filter pushdown (T1.3)
    # --------------------------------------
    def _build_filter_where(self, filters: Optional[Dict[str, Any]]) -> Optional[Tuple[str, List[Any]]]:
        """SQL ``(where_clause, params)`` for a metadata ``filters`` spec, or ``None``.

        ``None`` signals the filter cannot be expressed in SQL -- a dot-notation
        JSON path, or an operator the builder rejects -- so the caller keeps
        post-filtering in Python. The clause references ``documents`` columns and
        is parameterised; the Python matcher (``matches_metadata_filter``) stays
        the authority, so a SQL clause that is *broader* than the Python match
        (e.g. ``$like`` treating ``%`` as a wildcard) is harmless.
        """
        if not filters or not self.metadata_schema:
            return None
        builder = FilterQueryBuilder(self.metadata_schema)
        try:
            where_clause, params = builder.build_where_clause(filters)
        except (DatabaseError, TypeError, ValueError):
            # Not SQL-expressible (dotted JSON path, unsupported operator) or the
            # builder cannot coerce the operand (e.g. ``$in`` against a typed
            # column). Pushdown is an optimisation: decline it and let the caller
            # post-filter in Python rather than fail the query.
            return None
        if not where_clause:
            return None
        return where_clause, params

    def _matching_faiss_ids(
        self,
        filters: Optional[Dict[str, Any]],
        *,
        table: str = "chunks",
        id_column: str = "faiss_id",
        via_documents: bool = True,
    ) -> Optional[np.ndarray]:
        """FAISS ids in ``table`` whose owning document matches ``filters`` (SQL pushdown).

        Returns a sorted ``int64`` array of the matching ids, or ``None`` when the
        filter cannot be expressed in SQL (see :meth:`_build_filter_where`) -- the
        caller then falls back to post-filtering in Python. The set may be broader
        than the Python matcher but never narrower for supported operators, so
        restricting the FAISS search to it cannot drop a result the caller keeps.
        """
        built = self._build_filter_where(filters)
        if built is None:
            return None
        where_clause, params = built
        # ``table`` and ``id_column`` are internal literals; ``where_clause`` is
        # parameterised (``?`` placeholders bound from ``params``). No user text
        # is interpolated. bandit B608 is skipped project-wide for this reason.
        if via_documents:
            sql = (
                f"SELECT {id_column} FROM {table} "
                f"WHERE {id_column} IS NOT NULL "
                f"AND document_id IN (SELECT id FROM documents WHERE {where_clause})"
            )
        else:
            sql = f"SELECT {id_column} FROM {table} " f"WHERE {id_column} IS NOT NULL AND ({where_clause})"
        with self.connection_pool.get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        ids = np.fromiter((row[0] for row in rows), dtype=np.int64, count=len(rows))
        ids.sort()
        return ids

    def _warn_filter_starved(self, survivors: int, k: int) -> None:
        """Warn that a metadata filter could not be pushed into the index.

        Only fires on the fallback path -- an IndexLSH index (which rejects a
        FAISS id-selector) or a filter the SQL builder cannot express -- when
        fewer than ``k`` results survived post-filtering a fixed candidate pool.
        Matching results beyond that pool exist but were not searched. Flat
        indices with SQL-expressible filters take the exact pushdown path and
        never reach here.
        """
        logger.warning(
            "Metadata filter matched only %d of %d requested results within the "
            "candidate pool and could not be pushed into the index (IndexLSH or a "
            "filter SQL cannot express); further matches were not searched. Use a "
            "flat index (IndexFlatL2/IndexFlatIP) or a SQL-expressible filter for "
            "exact filtered retrieval.",
            survivors,
            k,
        )

    def _filtered_index_search(
        self,
        index: Any,
        query_embedding: np.ndarray,
        initial_k: int,
        filters: Optional[Dict[str, Any]],
        *,
        table: str = "chunks",
        id_column: str = "faiss_id",
        via_documents: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Search ``index`` for the top candidates, pushing ``filters`` into FAISS when possible.

        Returns ``(distances_row, indices_row, exact)``. ``exact`` is ``True`` when
        the returned pool is already restricted to the filter (an id-selector
        search, or no filter at all); ``False`` when the filter could not be
        pushed down and the caller must post-filter a fixed pool that may be
        starved -- the caller should warn if fewer than ``k`` survive.

        When a filter is pushed down, the pool is capped at the number of matching
        ids, so a selective filter returns *all* its matches (bounded by
        ``initial_k`` of the best-scoring ones) instead of whatever survived a
        fixed pre-filter budget.
        """
        # Normalize the query vector to match the stored vectors when ``index``
        # scores by inner product; a no-op for L2, so the baseline is unmoved.
        query_embedding = self._normalize_for_index(query_embedding, index)
        # Clamp the requested pool to the number of stored vectors. FAISS does
        # NOT clamp ``k`` to ``ntotal`` -- ``index.search(q, 2_000_000)`` on a
        # 5-vector index allocates a 2M-wide result array -- so an oversized
        # ``k`` (from a client or a bad caller) would amplify a tiny request into
        # gigabytes of allocation. We can never return more than ``ntotal`` hits
        # anyway, so this is semantically a no-op while closing that vector for
        # both local and remote callers.
        ntotal = int(getattr(index, "ntotal", 0))
        if ntotal <= 0:
            empty = np.empty(0, dtype=np.int64)
            return np.empty(0, dtype=np.float32), empty, True
        initial_k = min(initial_k, ntotal)
        if not filters:
            with self._faiss_lock.read_lock():
                distances, indices = index.search(query_embedding, initial_k)
            return distances[0], indices[0], True

        matching = self._matching_faiss_ids(filters, table=table, id_column=id_column, via_documents=via_documents)
        # Capability is per-index: the main index may be LSH (no selector) while
        # the hardcoded-flat section/document indices accept one, or vice versa.
        if matching is not None and self._index_supports_id_selector(index):
            if matching.size == 0:
                # Filter matches nothing: no candidates, and this is exact.
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64), True
            pool = min(int(matching.size), initial_k)
            with self._faiss_lock.read_lock():
                distances, indices = _faiss_search_with_selector(index, query_embedding, pool, matching)
            return distances[0], indices[0], True

        # Fallback: the index cannot take a selector (IndexLSH) or the filter is
        # not SQL-expressible. Search the fixed pool and let the caller post-filter.
        with self._faiss_lock.read_lock():
            distances, indices = index.search(query_embedding, initial_k)
        return distances[0], indices[0], False

    # Pure business logic helpers for DRY elimination
    def _build_metadata_field_search_sql(self, field_name: str) -> tuple[str, tuple[str]]:
        """Build SQL for searching metadata field embeddings (pure business logic)"""
        sql = """
            SELECT ce.faiss_id, ce.document_id, ce.chunk_index, d.content, d.created_at, d.updated_at
            FROM column_embeddings ce
            JOIN documents d ON ce.document_id = d.id
            WHERE ce.field_name = ?
        """
        return sql, (field_name,)

    def _calculate_embedding_similarities(
        self, query_embedding: np.ndarray, field_embeddings: np.ndarray
    ) -> np.ndarray:
        """Calculate similarities between query and field embeddings (pure business logic)

        Column embeddings are stored unnormalized on an L2 index (normalization is
        IP-only), so a raw dot product is unbounded -- ``(dot + 1) / 2`` then yields
        scores outside [0, 1] that sort incoherently against the [0, 1] content
        scores they are merged with. Compute a true cosine similarity (L2-normalize
        both sides, guarding zero-norm rows) so the mapped score is genuinely [0, 1].
        """
        query_vec = query_embedding.reshape(1, -1).astype(np.float64)
        field_vecs = field_embeddings.astype(np.float64)

        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        field_norms = np.linalg.norm(field_vecs, axis=1)
        safe_norms = np.where(field_norms > 0, field_norms, 1.0)
        field_vecs = field_vecs / safe_norms[:, None]

        cosine = np.dot(field_vecs, query_vec.T).flatten()
        cosine = np.clip(cosine, -1.0, 1.0)
        scores: np.ndarray = (cosine + 1) / 2  # cosine [-1, 1] -> [0, 1]
        return scores

    def _filter_and_sort_by_scores(self, scores: np.ndarray, score_threshold: float, k: int) -> np.ndarray:
        """Filter scores by threshold and return sorted indices (pure business logic)"""
        valid_indices = np.where(scores >= score_threshold)[0]
        if len(valid_indices) == 0:
            return np.array([], dtype=int)
        # Sort by score descending and limit
        sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]][:k]
        return sorted_indices

    def _create_metadata_search_result(
        self, row_data: Dict, field_name: str, score: float, doc_metadata: Dict[str, Any]
    ) -> QueryResult:
        """Create QueryResult for metadata field search (pure business logic)"""
        return QueryResult(
            id=f"{row_data['document_id']}:meta:{field_name}:{row_data['chunk_index']}",
            content=str(doc_metadata.get(field_name, "")),
            score=float(score),
            document_id=row_data["document_id"],
            metadata=doc_metadata,
            type="chunk",
        )

    # -----------------
    # Public search API
    # -----------------
    def grep(
        self,
        pattern: str,
        *,
        regex: bool = False,
        ignore_case: bool = False,
        whole_word: bool = False,
        context: int = 0,
        before_context: Optional[int] = None,
        after_context: Optional[int] = None,
        prefix: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
        max_count: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[GrepMatch]:
        """Lexical, line-oriented search over document content -- like ``grep``.

        This is exact/regex substring matching, deliberately separate from
        :meth:`query`. ``query`` does ranked semantic/keyword retrieval; ``grep``
        finds literal or regex matches and reports *where* they are (document id,
        line number, column span, and optional surrounding lines). Agents use it
        alongside vector and keyword search when they know a precise string or
        pattern to look for. Results are returned in document-id then line order,
        not by relevance.

        Parameters
        ----------
        pattern : str
            The text to search for. A literal substring by default; a regular
            expression when ``regex=True``.
        regex : bool
            Treat ``pattern`` as a Python regular expression. Default ``False``
            (literal match).
        ignore_case : bool
            Case-insensitive matching. Default ``False``.
        whole_word : bool
            Require the match to fall on word boundaries (wraps the pattern in
            ``\\b...\\b``). Default ``False``.
        context : int
            Number of adjacent lines to include both before and after each match
            (like ``grep -C``). Default ``0``.
        before_context : Optional[int]
            Lines of leading context (like ``grep -B``). Overrides ``context`` for
            the "before" side when set.
        after_context : Optional[int]
            Lines of trailing context (like ``grep -A``). Overrides ``context`` for
            the "after" side when set.
        prefix : Optional[str]
            Restrict the scan to documents whose id starts with this literal
            prefix (case-sensitive). Pair with :meth:`list_prefixes` to grep within
            a virtual "folder".
        where : Optional[Dict[str, Any]]
            Restrict the scan to documents matching this metadata filter (same
            syntax as :meth:`filter`).
        max_count : Optional[int]
            Stop after this many matches *per document* (like ``grep -m``).
        limit : Optional[int]
            Stop after this many matches in total across all documents.

        Returns
        -------
        List[GrepMatch]
            One entry per match, in document-id then line-number order.

        Notes
        -----
        - Matching runs line-by-line over the stored document content. Narrow the
          corpus with ``prefix`` / ``where`` on large databases, since every
          matched document is scanned.
        """
        if not pattern:
            raise ValueError("pattern must be a non-empty string")
        if limit is not None and limit <= 0:
            return []

        before_n = before_context if before_context is not None else context
        after_n = after_context if after_context is not None else context
        if before_n < 0 or after_n < 0:
            raise ValueError("context values must be non-negative")

        flags = re.IGNORECASE if ignore_case else 0
        body = pattern if regex else re.escape(pattern)
        if whole_word:
            body = rf"\b(?:{body})\b"
        try:
            matcher = re.compile(body, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        conditions: List[str] = []
        params: List[Any] = []
        if prefix:
            conditions.append("id GLOB ?")
            params.append(glob_escape(prefix) + "*")
        if where:
            try:
                where_clause, where_params = FilterQueryBuilder(self.metadata_schema).build_where_clause(where)
            except Exception as e:
                raise MetadataFilterError(f"Error building filter query: {e}") from e
            if where_clause:
                conditions.append(f"({where_clause})")
                params.extend(where_params)

        sql = "SELECT id, content FROM documents"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY id"

        matches: List[GrepMatch] = []
        with self._read_write_lock.read_lock():
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(sql, params)
                try:
                    rows = cursor.fetchall()
                finally:
                    cursor.close()

        for row in rows:
            doc_id, content = row["id"], row["content"]
            if content is None:
                continue
            lines = content.splitlines()
            per_doc = 0
            for line_index, line in enumerate(lines):
                match = matcher.search(line)
                if match is None:
                    continue
                before = lines[max(0, line_index - before_n) : line_index] if before_n else []
                after = lines[line_index + 1 : line_index + 1 + after_n] if after_n else []
                matches.append(
                    GrepMatch(
                        doc_id=doc_id,
                        line_number=line_index + 1,
                        line=line,
                        start=match.start(),
                        end=match.end(),
                        match=match.group(0),
                        before=before,
                        after=after,
                    )
                )
                per_doc += 1
                if limit is not None and len(matches) >= limit:
                    return matches
                if max_count is not None and per_doc >= max_count:
                    break

        return matches

    def query(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Optional[Literal["documents", "chunks", "sections", "context", "enriched"]] = None,
        search_level: Literal["chunks", "sections", "documents", "fused"] = "chunks",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        section_weight: float = 0.65,
        context_window: int = 2,
        context_unit: ContextUnit = "chunks",
        context_truncate: bool = False,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        reranker: Optional[Any] = None,
        reranker_config: Optional[Dict[str, Any]] = None,
        rerank_k: Optional[int] = None,
    ) -> List[QueryResult]:
        """
        Unified query interface for all search types

        Parameters
        ----------
        query : str
            Query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Optional[Literal['documents', 'chunks', 'sections', 'context', 'enriched']]
            The unit to report hits in: whole documents, individual chunks,
            sections, chunks with context, or enriched chunks with intra-document
            context. Defaults to ``None``, meaning "whatever unit
            ``search_level`` searched": documents for the default chunk search,
            sections for ``search_level='sections'``. Pass a value to override --
            notably ``return_type='documents'`` with ``search_level='sections'``
            ranks *documents* by their best-matching section.
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score to keep (0-1, higher=better). For ``search_type="hybrid"``
            each leg is min-max normalized *within this query's own candidate pool*,
            so scores are not comparable across queries or across different ``k``:
            the threshold cuts on rank position within the pool, not on absolute
            match quality, and is not a portable bar you can tune once and reuse.
        filters : Optional[Dict[str, Any]]
            Metadata filters. Filter fields must be declared in the metadata
            schema (or be reserved columns like ``id``/``created_at``);
            unknown fields or unsupported operators raise ``DatabaseError``.
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        search_level : Literal['chunks', 'sections', 'documents', 'fused']
            Which retrieval level to search. 'chunks' (default) is the normal path.
            'sections'/'documents' search the hierarchical indices directly. 'fused'
            blends chunk retrieval with section (raw-span) retrieval. All three
            require ``hierarchical_embeddings`` and raise ``ValueError`` without
            it. 'sections' and 'fused' report either 'sections' or 'documents';
            'documents' reports only 'documents'.
        section_weight : float
            Weight on the section leg when ``search_level='fused'`` (0-1): 0.0 is
            chunk-only, 1.0 is section-only. Default 0.65 (tuned on real long docs).
            Ignored for other search levels.
        context_window : int
            Size of the context to assemble for return_type='context'/'enriched'.
            Interpreted in the units given by ``context_unit``. When
            ``context_unit='chunks'`` (default): number of chunks before and after to
            include (context) or number of similar chunks to enrich. When
            ``context_unit`` is 'tokens'/'words'/'characters': an approximate budget
            for the assembled context content.
        context_unit : Literal['chunks', 'tokens', 'words', 'characters']
            Unit in which ``context_window`` is measured, by default 'chunks'.
            With a non-chunk unit, neighbouring/similar chunks are added whole,
            greedily, until the next one would exceed the budget (the matched chunk
            is always kept). Only applies to return_type='context'/'enriched'.
        context_truncate : bool
            When True and ``context_unit`` is a token/word/character budget, the
            assembled context is hard-truncated to exactly the budget (cutting the
            final chunk if needed). By default False (whole chunks only). This is the
            only way to guarantee the result never exceeds the budget when a single
            chunk is larger than it.
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores.
            One of: {"best", "average", "frequency_boost"}.
            For detailed explanations and guidance on selecting the appropriate method,
            see the Document Scoring documentation.
        document_scoring_options : dict, optional
            Parameters to pass to the scoring method function. For complete parameter
            documentation and examples, see the Document Scoring documentation.

            Common parameters by method:

            - frequency_boost
                frequency_bias : 0.0 - 1.0, default = 0.3
                    The ratio of the frequency multiplier to apply. Higher favors documents with more matching chunks
        reranker : object, optional
            A reranker instance whose ``rerank()`` re-scores the candidate pool.
        reranker_config : dict, optional
            Config from which the server/factory constructs a reranker, e.g.
            ``{"provider": "jina", "model": "jina-reranker-v2-base-multilingual"}``.
        rerank_k : int, optional
            Size of the candidate pool to fetch and hand to the reranker before
            truncating to ``k``. Only has an effect when a ``reranker`` or
            ``reranker_config`` is supplied. Defaults to ``5*k`` (clamped to at
            most 200); a reranker given only ``k`` candidates cannot improve
            recall, since it never sees the results ranked just below the cutoff.

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        _validate_context_unit(context_unit)
        return_type = _resolve_return_type(return_type, search_level)
        # A 'sections' return on a chunk-level search needs section data to group
        # into. On a non-hierarchical DB there is none, so the assembly below is
        # skipped and the user silently gets chunk results back. Fail loudly
        # instead -- consistent with the search_level='sections'/'fused' guards.
        if (
            return_type == "sections"
            and search_level not in ("fused", "sections", "documents")
            and not self._hierarchical_embeddings
        ):
            raise ValueError(
                "return_type='sections' requires a hierarchical database "
                "(create with hierarchical_embeddings=True), or use search_level='sections'."
            )
        if filters:
            validate_filter_spec(filters, self.metadata_schema)
        with self._read_write_lock.read_lock():
            # When a reranker is configured, over-fetch a larger candidate pool so
            # it can promote results the search legs ranked below `k`; otherwise the
            # rerank is a no-op on recall. `fetch_k == k` when no reranker, so the
            # non-rerank path is byte-for-byte unchanged -- including the fused and
            # hierarchical levels below, which are called with `fetch_k` too so the
            # shared rerank block (H8) can re-score their pool before truncating to k.
            reranking = reranker is not None or bool(reranker_config)
            fetch_k = _resolve_rerank_k(rerank_k, k) if reranking else k

            # Fused level: blend chunk retrieval with section (raw-span) retrieval.
            if search_level == "fused":
                if not self._hierarchical_embeddings:
                    raise ValueError("search_level='fused' requires hierarchical_embeddings=True")
                results = self._fused_search(
                    query,
                    return_type=return_type,
                    k=fetch_k,
                    score_threshold=score_threshold,
                    filters=filters,
                    section_weight=section_weight,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                )

            # Hierarchical search levels. Fail loudly on a non-hierarchical
            # database rather than quietly answering with chunk results: plausible
            # wrong-level results read as "the feature does nothing" instead of
            # "the feature is switched off". Matches 'fused' above.
            elif search_level in ("sections", "documents"):
                if not self._hierarchical_embeddings:
                    raise ValueError(f"search_level={search_level!r} requires hierarchical_embeddings=True")
                results = self._hierarchical_search(
                    query,
                    search_level=search_level,
                    return_type=return_type,
                    k=fetch_k,
                    score_threshold=score_threshold,
                    filters=filters,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                )

            elif search_type == "vector":
                results = self._vector_search(
                    query,
                    return_type if return_type != "sections" else "chunks",
                    fetch_k,
                    score_threshold,
                    filters,
                    context_window,
                    semantic_dedup_threshold,
                    document_scoring_method,
                    document_scoring_options,
                    context_unit,
                    context_truncate,
                )
            elif search_type == "keyword":
                results = self._keyword_search(
                    query,
                    return_type if return_type != "sections" else "chunks",
                    fetch_k,
                    score_threshold,
                    filters,
                    context_window,
                    semantic_dedup_threshold,
                    document_scoring_method,
                    document_scoring_options,
                    context_unit,
                    context_truncate,
                )
            elif search_type == "hybrid":
                results = self._hybrid_search(
                    query,
                    return_type if return_type != "sections" else "chunks",
                    fetch_k,
                    score_threshold,
                    filters,
                    vector_weight,
                    context_window,
                    semantic_dedup_threshold,
                    document_scoring_method,
                    document_scoring_options,
                    context_unit,
                    context_truncate,
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            # Post-process: if return_type='sections', group chunk results by section.
            # Keep the over-fetched pool intact here (fetch_k) so reranking still sees
            # the extra candidates; the rerank step below truncates to k. (The fused
            # and hierarchical branches above already return the requested unit.)
            if (
                return_type == "sections"
                and self._hierarchical_embeddings
                and search_level not in ("fused", "sections", "documents")
            ):
                results = self._assemble_section_results(results, fetch_k)

            # Apply reranking if configured. Runs for every search_level (H8): the
            # fused/hierarchical branches used to early-return above this block, so
            # a reranker passed with those levels was silently ignored.
            if reranker is not None:
                results = reranker.rerank(query, results, top_k=k)
            elif reranker_config:
                from localvectordb.reranking import RerankerRegistry

                _provider: str = reranker_config.get("provider", "")
                _reranker = RerankerRegistry.create_reranker(
                    _provider,
                    reranker_config.get("model"),
                    **{kk: v for kk, v in reranker_config.items() if kk not in ("provider", "model")},
                )
                results = _reranker.rerank(query, results, top_k=k)

            return results

    # -------------------------
    # Cursor / streaming API
    # -------------------------

    def _collect_vector_candidates(
        self,
        query_embedding: np.ndarray,
        initial_k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CursorCandidate]:
        """Run FAISS search and return scored candidates without SQLite hydration.

        A SQL-expressible ``filters`` spec is pushed into the index (id-selector)
        so a selective filter is not starved by the fixed ``initial_k`` pool
        before the cursor ever hydrates it (T1.3). The cursor still applies the
        filter as the authority during hydration.
        """
        assert self.index is not None
        distances_row, indices_row, _exact = self._filtered_index_search(
            self.index, query_embedding, initial_k, filters
        )

        candidates: List[CursorCandidate] = []
        for dist, idx in zip(distances_row, indices_row, strict=False):
            if idx == -1:
                continue
            score = self._distance_to_similarity(float(dist))
            if score < score_threshold:
                continue
            candidates.append(CursorCandidate(score=score, source="vector", faiss_id=int(idx)))
        return candidates

    def _collect_keyword_candidates(
        self,
        query: str,
        initial_k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CursorCandidate]:
        """Run FTS search and return scored candidates without SQLite hydration.

        A SQL-expressible ``filters`` spec is pushed into the FTS query so the
        ``LIMIT`` applies after filtering (T1.3).
        """
        if not self.fts_enabled:
            return []
        sanitized_query = FTSQuerySanitization.sanitize_fts_query(query)
        if not sanitized_query:
            return []

        built = self._build_filter_where(filters)
        if built is not None:
            where_clause, filter_params = built
            fts_sql = (
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "AND rowid IN (SELECT id FROM chunks WHERE document_id IN "
                f"(SELECT id FROM documents WHERE {where_clause})) "
                "ORDER BY rank ASC LIMIT ?"
            )
            fts_params: Tuple[Any, ...] = (sanitized_query, *filter_params, initial_k)
        else:
            fts_sql = (
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY rank ASC LIMIT ?"
            )
            fts_params = (sanitized_query, initial_k)

        candidates: List[CursorCandidate] = []
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(fts_sql, fts_params)
            for row in cursor.fetchall():
                score = self._fts_rank_to_similarity(row["rank"])
                if score < score_threshold:
                    continue
                candidates.append(
                    CursorCandidate(
                        score=score,
                        source="keyword",
                        chunk_rowid=row["rowid"],
                        raw_rank=float(row["rank"]),
                    )
                )
        return candidates

    def _collect_hybrid_candidates(
        self,
        query: str,
        query_embedding: np.ndarray,
        initial_k: int,
        score_threshold: float,
        vector_weight: float,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CursorCandidate]:
        """Run hybrid search and return merged/scored candidates."""
        if not self.fts_enabled:
            return self._collect_vector_candidates(query_embedding, initial_k, score_threshold, filters)

        vector_candidates = self._collect_vector_candidates(query_embedding, initial_k, 0.0, filters)
        keyword_candidates = self._collect_keyword_candidates(query, initial_k, 0.0, filters)

        if not vector_candidates and not keyword_candidates:
            return []

        return self._merge_hybrid_candidates(vector_candidates, keyword_candidates, vector_weight, score_threshold)

    def query_cursor(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Literal["documents", "chunks", "sections", "context", "enriched"] = "documents",
        search_level: Literal["chunks", "sections", "documents", "fused"] = "chunks",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        section_weight: float = 0.65,
        context_window: int = 2,
        context_unit: ContextUnit = "chunks",
        context_truncate: bool = False,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        reranker: Optional[Any] = None,
        reranker_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 50,
        cursor_ttl: float = 300.0,
    ) -> QueryCursor:
        """
        Create a QueryCursor for streaming results with lazy hydration.

        Performs the FAISS/FTS search once, caches scored candidates, and returns
        a cursor that lazily loads content/metadata from SQLite per batch.

        Parameters match ``query()`` with the addition of:

        Parameters
        ----------
        batch_size : int
            Default number of results per cursor batch (default 50).
        cursor_ttl : float
            Cursor time-to-live in seconds (default 300).

        Returns
        -------
        QueryCursor
            A cursor that can be iterated to fetch results in batches.

        Raises
        ------
        ValueError
            If a ``reranker`` or ``reranker_config`` is supplied. Reranking
            requires scoring the fully materialized result set, which is
            incompatible with lazy cursor hydration; use ``query()`` instead.
        """
        if reranker is not None or reranker_config:
            raise ValueError(_RERANK_STREAMING_UNSUPPORTED)
        if search_level == "fused":
            raise ValueError("search_level='fused' is not supported for streaming/cursor queries; use query()")
        _validate_context_unit(context_unit)

        with self._read_write_lock.read_lock():
            effective_return_type = return_type if return_type != "sections" else "chunks"
            initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == "documents" else k * 2)

            # Collect raw candidates. Push the filter down so a selective filter
            # is not starved before the cursor hydrates (T1.3); hydration still
            # applies the filter as the authority.
            if search_type == "vector":
                query_embeddings = self.embedding_provider.embed_sync([query])
                query_embedding = np.array(query_embeddings[0]).reshape(1, -1)
                candidates = self._collect_vector_candidates(query_embedding, initial_k, score_threshold, filters)
            elif search_type == "keyword":
                candidates = self._collect_keyword_candidates(query, initial_k, score_threshold, filters)
            elif search_type == "hybrid":
                query_embeddings = self.embedding_provider.embed_sync([query])
                query_embedding = np.array(query_embeddings[0]).reshape(1, -1)
                candidates = self._collect_hybrid_candidates(
                    query, query_embedding, initial_k, score_threshold, vector_weight, filters
                )
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            # Apply semantic deduplication (global operation, needs FAISS embeddings)
            if semantic_dedup_threshold is not None and candidates:
                faiss_ids = [c.faiss_id for c in candidates if c.faiss_id is not None]
                if faiss_ids:
                    # Build temporary QueryResults for dedup
                    temp_results = []
                    for c in candidates:
                        if c.faiss_id is not None:
                            temp_results.append(
                                QueryResult(
                                    id=f"_:{c.faiss_id}",
                                    score=c.score,
                                    type="chunk",
                                    content="",
                                    document_id="_",
                                )
                            )
                    deduped = self._apply_semantic_deduplication(temp_results, semantic_dedup_threshold)
                    kept_ids = {r.id for r in deduped}
                    candidates = [c for c in candidates if f"_:{c.faiss_id}" in kept_ids]

            candidates.sort(key=lambda c: c.score, reverse=True)

        config = CursorConfig(
            search_type=search_type,
            return_type=effective_return_type,
            search_level=search_level,
            score_threshold=score_threshold,
            filters=filters,
            vector_weight=vector_weight,
            context_window=context_window,
            context_unit=context_unit,
            context_truncate=context_truncate,
            semantic_dedup_threshold=semantic_dedup_threshold,
            document_scoring_method=document_scoring_method,
            document_scoring_options=document_scoring_options,
            total_k=k,
        )

        return QueryCursor(
            db=self,
            candidates=candidates,
            config=config,
            ttl_seconds=cursor_ttl,
            default_batch_size=batch_size,
        )

    async def query_cursor_async(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Literal["documents", "chunks", "sections", "context", "enriched"] = "documents",
        search_level: Literal["chunks", "sections", "documents", "fused"] = "chunks",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        section_weight: float = 0.65,
        context_window: int = 2,
        context_unit: ContextUnit = "chunks",
        context_truncate: bool = False,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        reranker: Optional[Any] = None,
        reranker_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 50,
        cursor_ttl: float = 300.0,
    ) -> QueryCursor:
        """Async version of query_cursor. Returns a QueryCursor for async iteration.

        Raises ``ValueError`` if a ``reranker``/``reranker_config`` is supplied;
        reranking is incompatible with lazy cursor hydration (use ``query_async()``).
        """
        if reranker is not None or reranker_config:
            raise ValueError(_RERANK_STREAMING_UNSUPPORTED)
        if search_level == "fused":
            raise ValueError("search_level='fused' is not supported for streaming/cursor queries; use query()")
        _validate_context_unit(context_unit)

        self._ensure_async_pool()
        await self._ensure_async_schema_initialized()

        loop = asyncio.get_event_loop()
        effective_return_type = return_type if return_type != "sections" else "chunks"
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == "documents" else k * 2)

        # Collect raw candidates (FAISS/FTS search). A SQL-expressible filter is
        # pushed into the index/FTS so a selective filter is not starved before
        # the cursor hydrates (T1.3); hydration remains the filter authority.
        if search_type == "vector":
            query_embedding = (await self.embedding_provider.embed_batch([query]))[0]
            query_embedding_np = np.array(query_embedding).reshape(1, -1)

            def protected_vector_search():
                return self._filtered_index_search(self.index, query_embedding_np, initial_k, filters)

            distances_row, indices_row, _exact = await loop.run_in_executor(None, protected_vector_search)
            candidates: List[CursorCandidate] = []
            for dist, idx in zip(distances_row, indices_row, strict=False):
                if idx == -1:
                    continue
                score = self._distance_to_similarity(float(dist))
                if score < score_threshold:
                    continue
                candidates.append(CursorCandidate(score=score, source="vector", faiss_id=int(idx)))

        elif search_type == "keyword":
            candidates = self._collect_keyword_candidates(query, initial_k, score_threshold, filters)

        elif search_type == "hybrid":
            # Run vector and keyword in parallel
            query_embedding = (await self.embedding_provider.embed_batch([query]))[0]
            query_embedding_np = np.array(query_embedding).reshape(1, -1)

            def protected_vector_search():
                return self._filtered_index_search(self.index, query_embedding_np, initial_k, filters)

            distances_row, indices_row, _exact = await loop.run_in_executor(None, protected_vector_search)
            vector_candidates: List[CursorCandidate] = []
            for dist, idx in zip(distances_row, indices_row, strict=False):
                if idx == -1:
                    continue
                score = self._distance_to_similarity(float(dist))
                vector_candidates.append(CursorCandidate(score=score, source="vector", faiss_id=int(idx)))

            keyword_candidates = self._collect_keyword_candidates(query, initial_k, 0.0, filters)

            # Merge using lightweight lookup
            candidates = self._merge_hybrid_candidates(
                vector_candidates, keyword_candidates, vector_weight, score_threshold
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        # Semantic dedup
        if semantic_dedup_threshold is not None and candidates:
            faiss_ids = [c.faiss_id for c in candidates if c.faiss_id is not None]
            if faiss_ids:
                temp_results = [
                    QueryResult(id=f"_:{c.faiss_id}", score=c.score, type="chunk", content="", document_id="_")
                    for c in candidates
                    if c.faiss_id is not None
                ]
                deduped = await self._apply_semantic_deduplication_async(temp_results, semantic_dedup_threshold)
                kept_ids = {r.id for r in deduped}
                candidates = [c for c in candidates if f"_:{c.faiss_id}" in kept_ids]

        candidates.sort(key=lambda c: c.score, reverse=True)

        config = CursorConfig(
            search_type=search_type,
            return_type=effective_return_type,
            search_level=search_level,
            score_threshold=score_threshold,
            filters=filters,
            vector_weight=vector_weight,
            context_window=context_window,
            context_unit=context_unit,
            context_truncate=context_truncate,
            semantic_dedup_threshold=semantic_dedup_threshold,
            document_scoring_method=document_scoring_method,
            document_scoring_options=document_scoring_options,
            total_k=k,
        )

        return QueryCursor(
            db=self,
            candidates=candidates,
            config=config,
            ttl_seconds=cursor_ttl,
            default_batch_size=batch_size,
        )

    def _merge_hybrid_candidates(
        self,
        vector_candidates: List[CursorCandidate],
        keyword_candidates: List[CursorCandidate],
        vector_weight: float,
        score_threshold: float,
    ) -> List[CursorCandidate]:
        """Merge vector and keyword candidates using a lightweight SQLite lookup.

        The single fusion point for the cursor and streaming paths;
        ``_collect_hybrid_candidates`` delegates here rather than repeating it.
        """
        faiss_ids = [c.faiss_id for c in vector_candidates if c.faiss_id is not None]
        chunk_rowids = [c.chunk_rowid for c in keyword_candidates if c.chunk_rowid is not None]

        faiss_to_key: Dict[int, str] = {}
        rowid_to_key: Dict[int, str] = {}
        key_to_faiss: Dict[str, int] = {}
        # Fallback rowid per key, used for keyword-only hits whose chunk has no
        # faiss_id yet (unembedded but FTS-indexed): without a rowid to hydrate
        # by, such a candidate would carry neither id and be silently dropped.
        key_to_rowid: Dict[str, int] = {}

        with self.connection_pool.get_connection() as conn:
            if faiss_ids:
                placeholders = ",".join(["?"] * len(faiss_ids))
                cursor = conn.execute(
                    f"SELECT faiss_id, document_id, chunk_index FROM chunks WHERE faiss_id IN ({placeholders})",
                    faiss_ids,
                )
                for row in cursor.fetchall():
                    key = f"{row['document_id']}:{row['chunk_index']}"
                    faiss_to_key[row["faiss_id"]] = key
                    key_to_faiss[key] = row["faiss_id"]

            if chunk_rowids:
                placeholders = ",".join(["?"] * len(chunk_rowids))
                cursor = conn.execute(
                    f"SELECT id, faiss_id, document_id, chunk_index FROM chunks WHERE id IN ({placeholders})",
                    chunk_rowids,
                )
                for row in cursor.fetchall():
                    key = f"{row['document_id']}:{row['chunk_index']}"
                    rowid_to_key[row["id"]] = key
                    key_to_rowid[key] = row["id"]
                    if row["faiss_id"] is not None:
                        key_to_faiss[key] = row["faiss_id"]

        vector_scores: Dict[str, float] = {}
        for c in vector_candidates:
            key = faiss_to_key.get(c.faiss_id)  # type: ignore[arg-type,assignment]
            if key:
                vector_scores[key] = c.score

        keyword_ranks: Dict[str, float] = {}
        for c in keyword_candidates:
            key = rowid_to_key.get(c.chunk_rowid)  # type: ignore[arg-type,assignment]
            if key and c.raw_rank is not None:
                keyword_ranks[key] = c.raw_rank

        candidates = []
        for key, score in _relative_score_fusion(vector_scores, keyword_ranks, vector_weight).items():
            if score < score_threshold:
                continue
            faiss_id = key_to_faiss.get(key)
            # Prefer hydrating by faiss_id; fall back to the chunk rowid for
            # keyword-only hits whose chunk isn't embedded (NULL faiss_id).
            chunk_rowid = key_to_rowid.get(key) if faiss_id is None else None
            candidates.append(CursorCandidate(score=score, source="hybrid", faiss_id=faiss_id, chunk_rowid=chunk_rowid))
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def query_stream(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Literal["documents", "chunks", "sections", "context", "enriched"] = "documents",
        search_level: Literal["chunks", "sections", "documents", "fused"] = "chunks",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        section_weight: float = 0.65,
        context_window: int = 2,
        context_unit: ContextUnit = "chunks",
        context_truncate: bool = False,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        batch_size: int = 50,
    ) -> Iterator[List[QueryResult]]:
        """
        Stream query results in batches. Convenience wrapper around ``query_cursor()``.

        Yields
        ------
        list of QueryResult
            Each yield is a batch of results.
        """
        cursor = self.query_cursor(
            query,
            search_type=search_type,
            return_type=return_type,
            search_level=search_level,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            vector_weight=vector_weight,
            context_window=context_window,
            context_unit=context_unit,
            context_truncate=context_truncate,
            semantic_dedup_threshold=semantic_dedup_threshold,
            document_scoring_method=document_scoring_method,
            document_scoring_options=document_scoring_options,
            batch_size=batch_size,
        )
        with cursor:
            yield from cursor.stream(batch_size)

    async def query_stream_async(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Literal["documents", "chunks", "sections", "context", "enriched"] = "documents",
        search_level: Literal["chunks", "sections", "documents", "fused"] = "chunks",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        section_weight: float = 0.65,
        context_window: int = 2,
        context_unit: ContextUnit = "chunks",
        context_truncate: bool = False,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        batch_size: int = 50,
    ) -> AsyncIterator[List[QueryResult]]:
        """
        Async stream query results in batches. Convenience wrapper around ``query_cursor_async()``.

        Yields
        ------
        list of QueryResult
            Each yield is a batch of results.
        """
        cursor = await self.query_cursor_async(
            query,
            search_type=search_type,
            return_type=return_type,
            search_level=search_level,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            vector_weight=vector_weight,
            context_window=context_window,
            context_unit=context_unit,
            context_truncate=context_truncate,
            semantic_dedup_threshold=semantic_dedup_threshold,
            document_scoring_method=document_scoring_method,
            document_scoring_options=document_scoring_options,
            batch_size=batch_size,
        )
        async with cursor:
            async for batch in cursor.stream_async(batch_size):
                yield batch

    # ---------------
    # Vector (sync)
    # ---------------
    def _vector_search(
        self,
        query: str,
        return_type: Literal["documents", "chunks", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
        query_embedding: Optional[np.ndarray] = None,
    ) -> List[QueryResult]:
        # A caller (the fused path) may pass a pre-computed query embedding to avoid
        # embedding the query twice; None preserves the default path byte-for-byte.
        if query_embedding is None:
            query_embeddings = self.embedding_provider.embed_sync([query])
            query_embedding = np.array(query_embeddings[0]).reshape(1, -1)
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == "documents" else k * 2)

        assert self.index is not None
        # Push the metadata filter into FAISS (id-selector) when the index and
        # filter allow it, so a selective filter returns its best matches rather
        # than whatever survives a fixed pre-filter pool (T1.3).
        distances_row, idx_row, exact = self._filtered_index_search(self.index, query_embedding, initial_k, filters)
        # Convert the whole result row at once, then filter with a numpy mask,
        # rather than calling _distance_to_similarity per candidate.
        sims = self._distances_to_similarities(distances_row)
        mask = (idx_row != -1) & (sims >= score_threshold)
        valid_faiss_ids = idx_row[mask].astype(int).tolist()
        valid_results = list(zip(valid_faiss_ids, sims[mask].tolist(), strict=False))
        if not valid_faiss_ids:
            return []
        chunk_results = []
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(valid_faiss_ids))
            cursor = conn.execute(
                f"""
                SELECT c.*, d.id as doc_id, d.content as doc_content
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.faiss_id IN ({placeholders})
            """,
                valid_faiss_ids,
            )
            faiss_id_to_row, doc_ids_to_fetch = {}, set()
            for row in cursor.fetchall():
                faiss_id_to_row[row["faiss_id"]] = row
                doc_ids_to_fetch.add(row["doc_id"])
            doc_metadata_batch = self._get_documents_metadata_batch(conn, list(doc_ids_to_fetch))
            for faiss_id, score in valid_results:
                row = faiss_id_to_row.get(faiss_id)
                if not row:
                    continue
                doc_metadata = doc_metadata_batch.get(row["doc_id"], {})
                if filters and not matches_metadata_filter(doc_metadata, filters):
                    continue
                position = ChunkPosition(
                    start=row["start_pos"],
                    end=row["end_pos"],
                    line=row["start_line"],
                    column=row["start_col"],
                    end_line=row["end_line"],
                    end_column=row["end_col"],
                )
                result = QueryResult(
                    id=f"{row['document_id']}:{row['chunk_index']}",
                    score=score,
                    type="chunk",
                    content=row["content"],
                    metadata=doc_metadata,
                    document_id=row["doc_id"],
                    position=position,
                )
                chunk_results.append(result)
        if not exact and filters and len(chunk_results) < k:
            self._warn_filter_starved(len(chunk_results), k)
        if semantic_dedup_threshold is not None:
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            chunk_results = self._apply_semantic_deduplication(chunk_results, semantic_dedup_threshold)
        if return_type == "context":
            final_results = self._add_context_window(chunk_results, context_window, context_unit, context_truncate)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == "enriched":
            final_results = self._enrich_with_intra_doc_context(
                chunk_results, context_window, context_unit, context_truncate
            )
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == "documents":
            document_results = self._aggregate_document_scores_with_method(
                chunk_results, document_scoring_method, document_scoring_options
            )
            return document_results[:k]
        else:
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]

    def _get_documents_metadata_batch(self, conn, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids or not self.metadata_schema:
            return {doc_id: {} for doc_id in doc_ids}
        metadata_columns = list(self.metadata_schema.keys())
        placeholders = ",".join(["?"] * len(doc_ids))
        cursor = conn.execute(
            f"SELECT id, {', '.join(metadata_columns)} FROM documents WHERE id IN ({placeholders})",
            doc_ids,
        )
        result = {}
        for row in cursor.fetchall():
            doc_id = row["id"]
            metadata = {}
            for col_name in metadata_columns:
                value = row[col_name]
                # Parse JSON fields if needed
                if value is not None and col_name in self.metadata_schema:
                    field_def = self.metadata_schema[col_name]
                    if (
                        isinstance(field_def.type, MetadataFieldType)
                        and field_def.type.name == "JSON"
                        and isinstance(value, str)
                    ):
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            # Leave the raw value in place if it is not valid JSON.
                            pass
                metadata[col_name] = value
            result[doc_id] = metadata
        for doc_id in doc_ids:
            if doc_id not in result:
                result[doc_id] = {}
        return result

    # ----------------
    # Keyword (sync)
    # ----------------
    def _keyword_chunk_hits(
        self,
        query: str,
        limit: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[List[QueryResult], Dict[str, float]]:
        """Hydrated chunk hits for the keyword leg, best-first, with their raw BM25 ranks.

        ``QueryResult.score`` carries the bounded ``[0, 1]`` similarity the public API
        promises. The second return value maps chunk key to *raw* BM25, which is what
        hybrid fusion normalizes -- see ``_relative_score_fusion``.

        Results come back in FTS rank order (best first). That is the same order a sort
        by descending similarity produces, because ``_fts_rank_to_similarity`` is
        monotone in the rank, so callers may truncate directly.
        """
        sanitized_query = FTSQuerySanitization.sanitize_fts_query(query)
        if not sanitized_query:
            return [], {}

        # Push a SQL-expressible metadata filter into the FTS query so ``LIMIT``
        # applies *after* filtering -- otherwise a selective filter starves the
        # fixed pool before the Python matcher ever sees the matches (T1.3). FTS
        # is pure SQL, so this works for any index type; only unpushable filters
        # (dot-notation JSON) fall through to the Python post-filter below.
        built = self._build_filter_where(filters)
        if built is not None:
            where_clause, filter_params = built
            # chunks_fts is contentless over chunks, so rowid == chunks.id.
            fts_sql = (
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "AND rowid IN (SELECT id FROM chunks WHERE document_id IN "
                f"(SELECT id FROM documents WHERE {where_clause})) "
                "ORDER BY rank ASC LIMIT ?"
            )
            fts_params: Tuple[Any, ...] = (sanitized_query, *filter_params, limit)
        else:
            fts_sql = (
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY rank ASC LIMIT ?"
            )
            fts_params = (sanitized_query, limit)

        chunk_results: List[QueryResult] = []
        raw_ranks: Dict[str, float] = {}
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(fts_sql, fts_params)
            valid_chunk_data, valid_chunk_ids = [], []
            for row in cursor.fetchall():
                score = self._fts_rank_to_similarity(row["rank"])
                if score < score_threshold:
                    continue
                chunk_id = row["rowid"]
                valid_chunk_data.append((chunk_id, score, float(row["rank"])))
                valid_chunk_ids.append(chunk_id)
            if not valid_chunk_ids:
                return [], {}
            placeholders = ",".join(["?"] * len(valid_chunk_ids))
            cursor = conn.execute(
                f"""
                SELECT c.*, d.id as doc_id, d.content as doc_content
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id IN ({placeholders})
            """,
                valid_chunk_ids,
            )
            chunk_id_to_row, doc_ids_to_fetch = {}, set()
            for row in cursor.fetchall():
                chunk_id_to_row[row["id"]] = row
                doc_ids_to_fetch.add(row["doc_id"])
            doc_metadata_batch = self._get_documents_metadata_batch(conn, list(doc_ids_to_fetch))
            for chunk_id, score, raw_rank in valid_chunk_data:
                row = chunk_id_to_row.get(chunk_id)
                if not row:
                    continue
                doc_metadata = doc_metadata_batch.get(row["doc_id"], {})
                if filters and not matches_metadata_filter(doc_metadata, filters):
                    continue
                position = ChunkPosition(
                    start=row["start_pos"],
                    end=row["end_pos"],
                    line=row["start_line"],
                    column=row["start_col"],
                    end_line=row["end_line"],
                    end_column=row["end_col"],
                )
                key = f"{row['document_id']}:{row['chunk_index']}"
                chunk_results.append(
                    QueryResult(
                        id=key,
                        score=score,
                        type="chunk",
                        content=row["content"],
                        metadata=doc_metadata,
                        document_id=row["doc_id"],
                        position=position,
                    )
                )
                raw_ranks[key] = raw_rank
        return chunk_results, raw_ranks

    def _keyword_search(
        self,
        query: str,
        return_type: Literal["documents", "chunks", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if not self.fts_enabled:
            logger.warning("FTS not available, returning empty results")
            return []
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == "documents" else k * 2)
        chunk_results, _ = self._keyword_chunk_hits(query, initial_k, score_threshold, filters)
        if not chunk_results:
            return []
        if semantic_dedup_threshold is not None:
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            chunk_results = self._apply_semantic_deduplication(chunk_results, semantic_dedup_threshold)
        if return_type == "context":
            final_results = self._add_context_window(chunk_results, context_window, context_unit, context_truncate)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == "enriched":
            final_results = self._enrich_with_intra_doc_context(
                chunk_results, context_window, context_unit, context_truncate
            )
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == "documents":
            document_results = self._aggregate_document_scores_with_method(
                chunk_results, document_scoring_method, document_scoring_options
            )
            return document_results[:k]
        else:
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]

    # ----------------
    # Hybrid (sync)
    # ----------------
    def _hybrid_search(
        self,
        query: str,
        return_type: Literal["documents", "chunks", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        vector_weight: float,
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if not self.fts_enabled:
            logger.info("FTS not available, falling back to vector search")
            return self._vector_search(
                query,
                return_type,
                k,
                score_threshold,
                filters,
                context_window,
                semantic_dedup_threshold,
                document_scoring_method,
                document_scoring_options,
                context_unit,
                context_truncate,
            )
        # Over-fetch k*4 for fusion/dedup headroom, ceiling 100 to bound work for
        # small k -- but never below k itself, or a large-k request (or the rerank
        # over-fetch that passes fetch_k in as k) is silently truncated / starved.
        search_k = max(k, min(k * 4, 100))
        vector_results = self._vector_search(query, "chunks", search_k, 0.0, filters, 0, None)
        # `search_k * 2` mirrors the `initial_k` over-fetch `_keyword_search` applies for
        # chunk results, so the keyword leg sees the same candidate pool it always has.
        keyword_results, keyword_ranks = self._keyword_chunk_hits(query, search_k * 2, 0.0, filters)
        keyword_results = keyword_results[:search_k]
        combined_results = self._combine_search_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            keyword_ranks=keyword_ranks,
            vector_weight=vector_weight,
            k=search_k,
            score_threshold=0.0,
        )
        if semantic_dedup_threshold is not None:
            combined_results = self._apply_semantic_deduplication(combined_results, semantic_dedup_threshold)
        combined_results = [r for r in combined_results if r.score >= score_threshold]
        if return_type == "context":
            final_results = self._add_context_window(combined_results, context_window, context_unit, context_truncate)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == "enriched":
            final_results = self._enrich_with_intra_doc_context(
                combined_results, context_window, context_unit, context_truncate
            )
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == "documents":
            document_results = self._aggregate_document_scores_with_method(
                combined_results, document_scoring_method, document_scoring_options
            )
            return document_results[:k]
        else:
            combined_results.sort(key=lambda x: x.score, reverse=True)
            return combined_results[:k]

    # ----------------------------
    # Embeddings access + dedup
    # ----------------------------
    # ---------------------------
    # Hierarchical search methods
    # ---------------------------
    def _hierarchical_search(
        self,
        query: str,
        *,
        search_level: str,
        return_type: str,
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
    ) -> List[QueryResult]:
        """Search using section or document FAISS indices.

        ``return_type`` chooses the unit the hits are reported in, exactly as it
        does for ``search_level='fused'``. It used to be accepted and ignored
        here, so ``return_type='documents'`` quietly handed back sections.
        """
        query_embeddings = self.embedding_provider.embed_sync([query])
        query_embedding = np.array(query_embeddings[0]).reshape(1, -1)

        if search_level == "sections":
            if return_type == "sections":
                return self._section_level_search(query_embedding, k, score_threshold, filters)
            if return_type == "documents":
                # Over-fetch before rolling up: k sections may live in far fewer
                # than k documents. Warn on starvation against the k the caller
                # actually asked for, not the inflated fetch.
                section_hits = self._section_level_search(
                    query_embedding,
                    k * _SECTION_ROLLUP_OVERFETCH,
                    score_threshold,
                    filters,
                    warn_k=k,
                )
                best_by_doc = self._reduce_to_best_per_key(section_hits, lambda r: r.document_id or r.id)
                # The threshold already cut the section hits; applying it again to
                # the rolled-up scores would be a no-op at best.
                return self._documents_from_scores(best_by_doc, k, 0.0)
            raise ValueError(
                f"search_level='sections' supports return_type 'sections' or 'documents', got {return_type!r}"
            )

        if search_level == "documents":
            if return_type != "documents":
                raise ValueError(f"search_level='documents' supports return_type 'documents', got {return_type!r}")
            return self._document_level_search(query_embedding, k, score_threshold, filters)
        return []

    def _section_level_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        warn_k: Optional[int] = None,
    ) -> List[QueryResult]:
        """Search the section FAISS index."""
        if self.section_index is None or self.section_index.ntotal == 0:
            return []

        initial_k = min(k * 2, self.section_index.ntotal)
        # Push the document-metadata filter into the section index (T1.3).
        distances_row, indices_row, exact = self._filtered_index_search(
            self.section_index, query_embedding, initial_k, filters, table="sections"
        )

        # Convert with the section index's OWN metric, not the main chunk index's
        # (T1.5). The section index is built independently -- historically always
        # IndexFlatL2 -- so auto-detecting off the main index applies the wrong
        # formula whenever the main index is IP.
        section_metric = self._detect_faiss_metric_type(self.section_index)
        valid_results = []
        for dist, idx in zip(distances_row, indices_row, strict=False):
            if idx == -1:
                continue
            score = self._distance_to_similarity(float(dist), section_metric)
            if score < score_threshold:
                continue
            valid_results.append((int(idx), score))

        if not valid_results:
            return []

        # Look up sections by faiss_id
        faiss_ids = [r[0] for r in valid_results]
        results = []
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(faiss_ids))
            cursor = conn.execute(
                f"""
                SELECT s.*, d.content as doc_content, d.id as doc_id
                FROM sections s
                JOIN documents d ON s.document_id = d.id
                WHERE s.faiss_id IN ({placeholders})
            """,
                faiss_ids,
            )
            faiss_to_section = {}
            doc_ids_to_fetch = set()
            for row in cursor.fetchall():
                faiss_to_section[row["faiss_id"]] = row
                doc_ids_to_fetch.add(row["doc_id"])

            doc_metadata_batch = self._get_documents_metadata_batch(conn, list(doc_ids_to_fetch))

            for faiss_id, score in valid_results:
                row = faiss_to_section.get(faiss_id)
                if not row:
                    continue
                doc_metadata = doc_metadata_batch.get(row["doc_id"], {})
                if filters and not matches_metadata_filter(doc_metadata, filters):
                    continue

                section_text = row["doc_content"][row["start_pos"] : row["end_pos"]]
                section_metadata = dict(doc_metadata)
                section_metadata["section_heading"] = row["heading"]
                section_metadata["section_level"] = row["heading_level"]
                section_metadata["section_index"] = row["section_index"]

                # Parse section-specific metadata
                if row["metadata"]:
                    try:
                        raw = row["metadata"]
                        section_meta = json.loads(raw) if isinstance(raw, str) else raw
                        section_metadata.update(section_meta)
                    except (json.JSONDecodeError, TypeError):
                        # Skip section metadata that is not valid JSON.
                        pass

                result = QueryResult(
                    id=f"{row['document_id']}:section:{row['section_index']}",
                    score=score,
                    type="section",
                    content=section_text,
                    metadata=section_metadata,
                    document_id=row["document_id"],
                    position=ChunkPosition(
                        start=row["start_pos"],
                        end=row["end_pos"],
                        line=row["start_line"] or 1,
                        column=1,
                        end_line=row["end_line"] or 1,
                        end_column=1,
                    ),
                )
                results.append(result)

        starve_k = warn_k if warn_k is not None else k
        if not exact and filters and len(results) < starve_k:
            self._warn_filter_starved(len(results), starve_k)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _document_level_search(
        self,
        query_embedding: np.ndarray,
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> List[QueryResult]:
        """Search the document FAISS index."""
        if self.document_index is None or self.document_index.ntotal == 0:
            return []

        initial_k = min(k * 2, self.document_index.ntotal)
        # Push the metadata filter into the document index. The filter is on the
        # documents table directly, so via_documents=False (T1.3).
        distances_row, indices_row, exact = self._filtered_index_search(
            self.document_index,
            query_embedding,
            initial_k,
            filters,
            table="documents",
            id_column="doc_faiss_id",
            via_documents=False,
        )

        # Convert with the document index's OWN metric, not the main chunk
        # index's (T1.5). See _section_level_search for the rationale.
        document_metric = self._detect_faiss_metric_type(self.document_index)
        valid_results = []
        for dist, idx in zip(distances_row, indices_row, strict=False):
            if idx == -1:
                continue
            score = self._distance_to_similarity(float(dist), document_metric)
            if score < score_threshold:
                continue
            valid_results.append((int(idx), score))

        if not valid_results:
            return []

        # Look up documents by doc_faiss_id
        faiss_ids = [r[0] for r in valid_results]
        results = []
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(faiss_ids))
            cursor = conn.execute(
                f"""
                SELECT id, content, doc_faiss_id
                FROM documents WHERE doc_faiss_id IN ({placeholders})
            """,
                faiss_ids,
            )
            faiss_to_doc = {}
            doc_ids = []
            for row in cursor.fetchall():
                faiss_to_doc[row["doc_faiss_id"]] = row
                doc_ids.append(row["id"])

            doc_metadata_batch = self._get_documents_metadata_batch(conn, doc_ids)

            for faiss_id, score in valid_results:
                row = faiss_to_doc.get(faiss_id)
                if not row:
                    continue
                doc_metadata = doc_metadata_batch.get(row["id"], {})
                if filters and not matches_metadata_filter(doc_metadata, filters):
                    continue

                result = QueryResult(
                    id=row["id"],
                    score=score,
                    type="document",
                    content=row["content"],
                    metadata=doc_metadata,
                )
                results.append(result)

        if not exact and filters and len(results) < k:
            self._warn_filter_starved(len(results), k)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _assemble_section_results(self, chunk_results: List[QueryResult], k: int) -> List[QueryResult]:
        """Group chunk results by section and return section-level results."""
        if not chunk_results:
            return []

        # Map chunks to their sections
        doc_chunk_pairs = []
        for result in chunk_results:
            if result.type == "chunk" and result.document_id:
                _, chunk_idx = self._split_chunk_id(result.id)
                doc_chunk_pairs.append((result.document_id, chunk_idx, result))

        if not doc_chunk_pairs:
            return chunk_results[:k]

        # Query sections for these chunks
        section_results = {}
        with self.connection_pool.get_connection() as conn:
            for doc_id, chunk_idx, result in doc_chunk_pairs:
                cursor = conn.execute(
                    """
                    SELECT s.*, d.content as doc_content
                    FROM sections s
                    JOIN chunks c ON c.section_id = s.id
                    JOIN documents d ON s.document_id = d.id
                    WHERE c.document_id = ? AND c.chunk_index = ?
                """,
                    (doc_id, chunk_idx),
                )
                row = cursor.fetchone()
                if row:
                    section_key = f"{row['document_id']}:section:{row['section_index']}"
                    if section_key not in section_results:
                        section_text = row["doc_content"][row["start_pos"] : row["end_pos"]]
                        section_metadata = dict(result.metadata)
                        section_metadata["section_heading"] = row["heading"]
                        section_metadata["section_level"] = row["heading_level"]
                        section_metadata["section_index"] = row["section_index"]

                        section_results[section_key] = QueryResult(
                            id=section_key,
                            score=result.score,
                            type="section",
                            content=section_text,
                            metadata=section_metadata,
                            document_id=row["document_id"],
                            position=ChunkPosition(
                                start=row["start_pos"],
                                end=row["end_pos"],
                                line=row["start_line"] or 1,
                                column=1,
                                end_line=row["end_line"] or 1,
                                end_column=1,
                            ),
                        )
                    else:
                        # Update score to best chunk score
                        if result.score > section_results[section_key].score:
                            section_results[section_key].score = result.score

        results = list(section_results.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _fused_search(
        self,
        query: str,
        *,
        return_type: str,
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        section_weight: float,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
    ) -> List[QueryResult]:
        """Fuse chunk retrieval with section (raw-span) retrieval.

        Runs a chunk vector search and a section-level search over the same query,
        maps each hit up to its target unit (document or section), and blends the two
        legs with ``_two_leg_minmax_fuse`` weighted by ``section_weight`` (the weight
        on the section leg). This is the shippable result of the hierarchical
        experiment (findings F3): fusing a section raw-span representation with chunk
        retrieval beats chunk-only on real, section-structured documents.

        Like hybrid search, fused scores are relative to this query's candidate pool,
        so ``score_threshold`` acts as a rank-position threshold, not an absolute
        match-quality gate.
        """
        # Embed once and share the vector across both legs.
        query_embedding = np.array(self.embedding_provider.embed_sync([query])[0]).reshape(1, -1)
        pool_k = k * 2

        chunk_hits = self._vector_search(
            query,
            "chunks",
            pool_k,
            0.0,
            filters,
            0,
            None,
            document_scoring_method,
            document_scoring_options,
            "chunks",
            False,
            query_embedding=query_embedding,
        )
        section_hits = self._section_level_search(query_embedding, pool_k, 0.0, filters)

        if return_type == "documents":
            return self._fuse_to_documents(chunk_hits, section_hits, k, score_threshold, section_weight)
        if return_type == "sections":
            return self._fuse_to_sections(chunk_hits, section_hits, pool_k, k, score_threshold, section_weight)
        raise ValueError(f"search_level='fused' supports return_type 'documents' or 'sections', got {return_type!r}")

    @staticmethod
    def _reduce_to_best_per_key(results: List[QueryResult], key_of: Any) -> Dict[str, float]:
        """Reduce hits to the max score per target key (a hit credits its parent unit)."""
        best: Dict[str, float] = {}
        for r in results:
            key = key_of(r)
            prev = best.get(key)
            if prev is None or r.score > prev:
                best[key] = r.score
        return best

    def _fuse_to_documents(
        self,
        chunk_hits: List[QueryResult],
        section_hits: List[QueryResult],
        k: int,
        score_threshold: float,
        section_weight: float,
    ) -> List[QueryResult]:
        """Fuse the two legs at the document target and return ``type="document"`` results."""
        chunk_by_doc = self._reduce_to_best_per_key(chunk_hits, lambda r: r.document_id or r.id)
        section_by_doc = self._reduce_to_best_per_key(section_hits, lambda r: r.document_id or r.id)
        fused = _two_leg_minmax_fuse(chunk_by_doc, section_by_doc, section_weight)
        return self._documents_from_scores(fused, k, score_threshold)

    def _documents_from_scores(
        self,
        scores: Dict[str, float],
        k: int,
        score_threshold: float,
    ) -> List[QueryResult]:
        """Materialise the top ``k`` scored document ids as ``type="document"`` results."""
        ranked = [(d, s) for d, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s >= score_threshold][:k]
        if not ranked:
            return []
        doc_ids = [d for d, _ in ranked]
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(doc_ids))
            cursor = conn.execute(f"SELECT id, content FROM documents WHERE id IN ({placeholders})", doc_ids)
            content_by_id = {row["id"]: row["content"] for row in cursor.fetchall()}
            metadata_batch = self._get_documents_metadata_batch(conn, doc_ids)

        results = []
        for doc_id, score in ranked:
            if doc_id not in content_by_id:
                continue
            results.append(
                QueryResult(
                    id=doc_id,
                    score=score,
                    type="document",
                    content=content_by_id[doc_id],
                    metadata=metadata_batch.get(doc_id, {}),
                )
            )
        return results

    def _fuse_to_sections(
        self,
        chunk_hits: List[QueryResult],
        section_hits: List[QueryResult],
        pool_k: int,
        k: int,
        score_threshold: float,
        section_weight: float,
    ) -> List[QueryResult]:
        """Fuse the two legs at the section target and return ``type="section"`` results."""
        # Chunk leg rolled up to sections (max chunk score per containing section).
        assembled = self._assemble_section_results(chunk_hits, pool_k)
        chunk_by_section = {r.id: r.score for r in assembled}
        section_by_section = {r.id: r.score for r in section_hits}
        fused = _two_leg_minmax_fuse(chunk_by_section, section_by_section, section_weight)

        # One representative object per section id; prefer the section-leg object
        # (raw-span content + section metadata), falling back to the chunk rollup.
        objs: Dict[str, QueryResult] = {r.id: r for r in assembled}
        for r in section_hits:
            objs[r.id] = r

        results = []
        for sid, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            if score < score_threshold:
                continue
            obj = objs.get(sid)
            if obj is None:
                continue
            obj.score = score
            results.append(obj)
            if len(results) >= k:
                break
        return results

    def get_chunk_embeddings(self, chunk_ids: str | List[str]) -> np.ndarray:
        """Returns embeddings for chunks given by `chunk_ids`"

        Parameters
        ----------
        chunk_ids : str | List[str]
            The chunk_ids for which to return embeddings

        Returns
        -------
        np.ndarray

        """
        chunk_ids_list: List[str] = [chunk_ids] if isinstance(chunk_ids, str) else list(chunk_ids)
        chunk_list = []
        for cid in chunk_ids_list:
            doc_id, chunk_idx = self._split_chunk_id(cid)
            if chunk_idx == -1:
                raise ValueError(f"Expected chunk ids (e.g. doc_1:1), found: {cid}")
            chunk_list.append((doc_id, chunk_idx))
        placeholders = ",".join(["(?,?)"] * len(chunk_list))
        query_str = (
            f"SELECT faiss_id, document_id, chunk_index FROM chunks "
            f"WHERE (document_id, chunk_index) IN ({placeholders})"
        )
        params = [item for pair in chunk_list for item in pair]
        with self._read_write_lock.read_lock():
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(query_str, params)
                rows = cursor.fetchall()
                faiss_ids = [row["faiss_id"] for row in rows]
            return self._reconstruct_embeddings_batch(faiss_ids)

    def _apply_semantic_deduplication(
        self, results: List[QueryResult], threshold: float, max_candidates: int = 1000
    ) -> List[QueryResult]:
        """
        Apply semantic deduplication to search results using FAISS index embeddings.

        Optimized version that minimizes database calls and uses batch FAISS operations.
        Includes scaling limits to prevent O(n²) behavior on large result sets.

        Parameters
        ----------
        results : List[QueryResult]
            Initial search results to deduplicate - MUST BE SORTED with highest score first
        threshold : float
            Similarity threshold (0-1, higher=more similar). Chunks above this threshold are considered duplicates.
        max_candidates : int, default=1000
            Maximum number of candidates to process for deduplication. If more chunks are provided,
            only the highest-scoring ones will be considered to prevent quadratic behavior.

        Returns
        -------
        List[QueryResult]
            Deduplicated results with highest-scored chunk from each similar group
        """
        if not results or threshold is None or threshold <= 0:
            return results
        chunk_results = [r for r in results if r.type == "chunk"]
        if len(chunk_results) <= 1:
            return results

        # Apply scaling limit to prevent O(n²) behavior
        if len(chunk_results) > max_candidates:
            logger.info(
                f"Limiting semantic deduplication to top {max_candidates} chunks "
                f"(from {len(chunk_results)} total) to prevent performance issues"
            )
            chunk_results = chunk_results[:max_candidates]

        chunk_identifiers = [(r.document_id, self._extract_chunk_index_from_id(r.id)) for r in chunk_results]
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["(?,?)"] * len(chunk_identifiers))
            query = f"""
                SELECT document_id, chunk_index, faiss_id
                FROM chunks
                WHERE (document_id, chunk_index) IN ({placeholders})
            """
            params = [item for pair in chunk_identifiers for item in pair]
            cursor = conn.execute(query, params)
            faiss_id_mapping = {(row["document_id"], row["chunk_index"]): row["faiss_id"] for row in cursor.fetchall()}
        faiss_ids = []
        result_mapping = {}
        for result in chunk_results:
            doc_id, chunk_idx = self._split_chunk_id(result.id)
            faiss_id = faiss_id_mapping.get((doc_id, chunk_idx))
            if faiss_id is not None:
                faiss_ids.append(faiss_id)
                result_mapping[faiss_id] = result

        if not faiss_ids:
            return results

        embeddings_matrix = self._reconstruct_embeddings_batch(faiss_ids)
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        normalized_embeddings = embeddings_matrix / np.maximum(norms, 1e-8)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        similar_pairs = similarity_matrix >= threshold
        np.fill_diagonal(similar_pairs, False)
        upper_tri = np.triu(similar_pairs, k=1)
        should_remove = np.any(upper_tri, axis=0)
        keep_mask = ~should_remove
        final_chunk_results = [result_mapping[faiss_ids[i]] for i in range(len(faiss_ids)) if keep_mask[i]]

        # Log deduplication statistics
        removed_count = len(chunk_results) - len(final_chunk_results)
        if removed_count > 0:
            logger.debug(
                f"Semantic deduplication removed {removed_count} similar chunks "
                f"({removed_count / len(chunk_results) * 100:.1f}% deduplication rate)"
            )

        return final_chunk_results

    @staticmethod
    def _split_chunk_id(chunk_id: str) -> tuple[str, int]:
        try:
            parts = chunk_id.rsplit(":", maxsplit=1)
            chunk_idx = int(parts[-1])
            doc_id = parts[0]
            return doc_id, chunk_idx
        except (ValueError, IndexError, TypeError):
            return chunk_id, -1

    @staticmethod
    def _extract_chunk_index_from_id(chunk_id: str) -> int:
        _, chunk_idx = SearchMixin._split_chunk_id(chunk_id)
        return chunk_idx

    # ---------------------
    # Context and enrichment
    # ---------------------
    def _add_context_window(
        self,
        results: List[QueryResult],
        context_window: int,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if context_window <= 0 or not results:
            return results
        if context_unit != "chunks":
            return self._add_context_window_budget(results, context_window, context_unit, context_truncate)
        context_results: List[QueryResult] = []
        doc_chunk_requests = defaultdict(list)
        for result in results:
            if result.type != "chunk":
                context_results.append(result)
                continue
            doc_id = result.document_id
            chunk_index = self._extract_chunk_index_from_id(result.id)
            doc_chunk_requests[doc_id].append((chunk_index, result))
        with self.connection_pool.get_connection() as conn:
            for doc_id, chunk_requests in doc_chunk_requests.items():
                ranges_needed = []
                for chunk_index, result in chunk_requests:
                    start_index = max(0, chunk_index - context_window)
                    end_index = chunk_index + context_window
                    ranges_needed.append((start_index, end_index, chunk_index, result))
                merged_ranges = self._merge_overlapping_ranges(ranges_needed)
                all_chunk_indices: set[int] = set()
                for start, end, _, _ in merged_ranges:
                    all_chunk_indices.update(range(start, end + 1))
                if not all_chunk_indices:
                    continue
                placeholders = ",".join(["?"] * len(all_chunk_indices))
                cursor = conn.execute(
                    f"""
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col
                    FROM chunks
                    WHERE document_id = ? AND chunk_index IN ({placeholders})
                    ORDER BY chunk_index
                """,
                    [doc_id] + list(all_chunk_indices),
                )
                chunks_by_index = {row["chunk_index"]: row for row in cursor.fetchall()}
                for chunk_index, result in chunk_requests:
                    context_chunks = []
                    start_context = max(0, chunk_index - context_window)
                    end_context = chunk_index + context_window
                    for i in range(start_context, end_context + 1):
                        if i in chunks_by_index:
                            context_chunks.append(chunks_by_index[i])
                    if not context_chunks:
                        context_results.append(result)
                        continue
                    combined_content = []
                    min_start_pos = float("inf")
                    max_end_pos = 0
                    min_start_line = float("inf")
                    min_start_col = float("inf")
                    max_end_line = 0
                    max_end_col = 0
                    for chunk_row in context_chunks:
                        combined_content.append(chunk_row["content"])
                        min_start_pos = min(min_start_pos, chunk_row["start_pos"])
                        max_end_pos = max(max_end_pos, chunk_row["end_pos"])
                        min_start_line = min(min_start_line, chunk_row["start_line"])
                        max_end_line = max(max_end_line, chunk_row["end_line"])
                        if chunk_row["start_line"] == min_start_line:
                            min_start_col = min(min_start_col, chunk_row["start_col"])
                        if chunk_row["end_line"] == max_end_line:
                            max_end_col = max(max_end_col, chunk_row["end_col"])
                    context_position = ChunkPosition(
                        start=int(min_start_pos),
                        end=int(max_end_pos),
                        line=int(min_start_line),
                        column=int(min_start_col),
                        end_line=int(max_end_line),
                        end_column=int(max_end_col),
                    )
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(combined_content)
                    context_result = QueryResult(
                        id=f"{doc_id}:context:{chunk_index}",
                        score=result.score,
                        type="context",
                        content=combined_text,
                        metadata=result.metadata.copy(),
                        document_id=doc_id,
                        position=context_position,
                    )
                    context_result.metadata["_context_window"] = context_window
                    context_result.metadata["_context_unit"] = "chunks"
                    context_result.metadata["_original_chunk_index"] = chunk_index
                    context_result.metadata["_context_chunk_count"] = len(context_chunks)
                    context_result.metadata["_context_start_index"] = start_context
                    context_result.metadata["_context_end_index"] = end_context
                    context_results.append(context_result)
        return context_results

    # -------------------------------------------------------------------------
    # Budget-based context/enrichment (context_unit in tokens/words/characters)
    # -------------------------------------------------------------------------
    @staticmethod
    def _select_context_indices_by_budget(
        chunks_by_index: Dict[int, Any], center: int, budget: int, unit: str
    ) -> List[int]:
        """Greedily grow a contiguous window around ``center`` within ``budget``.

        The matched (center) chunk is always included, even if it alone exceeds
        the budget. Expansion is symmetric: on each pass we try to add the next
        chunk on each side, stopping a side once its next chunk would overflow the
        budget or the document runs out. Returns the selected indices sorted.
        """
        if center not in chunks_by_index:
            return []
        center_row = chunks_by_index[center]
        # The assembled context joins chunks with a separator, which also counts
        # against the budget; each added chunk contributes its own size plus one
        # separator so the final assembled text stays within the budget.
        sep_size = _measure_text(_CONTEXT_SEPARATOR, None, unit)
        used = _measure_text(center_row["content"], center_row["tokens"], unit)
        selected = [center]
        lo, hi = center - 1, center + 1
        lo_open = hi_open = True
        while lo_open or hi_open:
            if lo_open:
                row = chunks_by_index.get(lo)
                if row is None:
                    lo_open = False
                else:
                    size = _measure_text(row["content"], row["tokens"], unit) + sep_size
                    if used + size <= budget:
                        selected.append(lo)
                        used += size
                        lo -= 1
                    else:
                        lo_open = False
            if hi_open:
                row = chunks_by_index.get(hi)
                if row is None:
                    hi_open = False
                else:
                    size = _measure_text(row["content"], row["tokens"], unit) + sep_size
                    if used + size <= budget:
                        selected.append(hi)
                        used += size
                        hi += 1
                    else:
                        hi_open = False
        return sorted(selected)

    @staticmethod
    def _merge_chunk_positions(ordered_rows: List[Any]) -> Tuple[int, int, int, int, int, int]:
        """Merge chunk row positions into a single (start, end, line, col, end_line, end_col)."""
        min_start_pos = float("inf")
        max_end_pos = 0
        min_start_line = float("inf")
        min_start_col = float("inf")
        max_end_line = 0
        max_end_col = 0
        for row in ordered_rows:
            min_start_pos = min(min_start_pos, row["start_pos"])
            max_end_pos = max(max_end_pos, row["end_pos"])
            min_start_line = min(min_start_line, row["start_line"])
            max_end_line = max(max_end_line, row["end_line"])
            if row["start_line"] == min_start_line:
                min_start_col = min(min_start_col, row["start_col"])
            if row["end_line"] == max_end_line:
                max_end_col = max(max_end_col, row["end_col"])
        return (
            int(min_start_pos),
            int(max_end_pos),
            int(min_start_line),
            int(min_start_col),
            int(max_end_line),
            int(max_end_col),
        )

    @classmethod
    def _build_budget_context_result(
        cls,
        doc_id: Optional[str],
        center_idx: int,
        base_result: QueryResult,
        ordered_rows: List[Any],
        context_window: int,
        context_unit: str,
        context_truncate: bool,
    ) -> QueryResult:
        combined_text = _CONTEXT_SEPARATOR.join(row["content"] for row in ordered_rows)
        start_pos, end_pos, start_line, start_col, end_line, end_col = cls._merge_chunk_positions(ordered_rows)
        truncated = False
        if context_truncate:
            new_text = _truncate_text_to_budget(combined_text, context_window, context_unit)
            if len(new_text) < len(combined_text):
                truncated = True
                combined_text = new_text
                # Content was cut; the character end is now approximate.
                end_pos = start_pos + len(combined_text)
        context_result = QueryResult(
            id=f"{doc_id}:context:{center_idx}",
            score=base_result.score,
            type="context",
            content=combined_text,
            metadata=base_result.metadata.copy(),
            document_id=doc_id,
            position=ChunkPosition(
                start=start_pos, end=end_pos, line=start_line, column=start_col, end_line=end_line, end_column=end_col
            ),
        )
        context_result.metadata["_context_window"] = context_window
        context_result.metadata["_context_unit"] = context_unit
        context_result.metadata["_original_chunk_index"] = center_idx
        context_result.metadata["_context_chunk_count"] = len(ordered_rows)
        context_result.metadata["_context_start_index"] = ordered_rows[0]["chunk_index"]
        context_result.metadata["_context_end_index"] = ordered_rows[-1]["chunk_index"]
        if truncated:
            context_result.metadata["_context_truncated"] = True
        return context_result

    @classmethod
    def _build_budget_enriched_result(
        cls,
        doc_id: Optional[str],
        ordered_rows: List[Any],
        best_score: float,
        combined_metadata: Dict[str, Any],
        chunk_similarities: List[float],
        matched_indices: List[int],
        sorted_indices: List[int],
        context_window: int,
        context_unit: str,
        context_truncate: bool,
    ) -> QueryResult:
        combined_text = _CONTEXT_SEPARATOR.join(row["content"] for row in ordered_rows)
        start_pos, end_pos, start_line, start_col, end_line, end_col = cls._merge_chunk_positions(ordered_rows)
        truncated = False
        if context_truncate:
            new_text = _truncate_text_to_budget(combined_text, context_window, context_unit)
            if len(new_text) < len(combined_text):
                truncated = True
                combined_text = new_text
                end_pos = start_pos + len(combined_text)
        enriched_result = QueryResult(
            id=f"{doc_id}:enriched",
            score=best_score,
            type="enriched",
            content=combined_text,
            metadata=dict(combined_metadata),
            document_id=doc_id,
            position=ChunkPosition(
                start=start_pos, end=end_pos, line=start_line, column=start_col, end_line=end_line, end_column=end_col
            ),
        )
        enriched_result.metadata["_enriched_chunk_count"] = len(ordered_rows)
        enriched_result.metadata["_matched_chunk_indices"] = matched_indices
        enriched_result.metadata["_all_chunk_indices"] = sorted_indices
        enriched_result.metadata["_similarity_scores"] = chunk_similarities
        enriched_result.metadata["_enrichment_method"] = "budget"
        enriched_result.metadata["_context_unit"] = context_unit
        if truncated:
            enriched_result.metadata["_context_truncated"] = True
        return enriched_result

    def _fetch_document_chunks_sync(self, conn: Any, doc_id: Optional[str]) -> Dict[int, Any]:
        cursor = conn.execute(
            """
            SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col, tokens
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            """,
            [doc_id],
        )
        return {row["chunk_index"]: row for row in cursor.fetchall()}

    def _rank_intra_doc_similarities(
        self, doc_chunks: Dict[int, Any], matched_indices: List[int]
    ) -> Tuple[List[int], Dict[int, float], List[Tuple[int, float]]]:
        """Rank a document's chunks by similarity to the matched chunk(s).

        Returns ``(present_matched, base_scores, candidates)`` where ``base_scores``
        maps each present matched chunk to 1.0 and ``candidates`` is the remaining
        chunks as ``(index, best_similarity)`` sorted by similarity descending.
        """
        present_matched = [i for i in matched_indices if i in doc_chunks]
        base_scores: Dict[int, float] = {i: 1.0 for i in present_matched}
        if not present_matched:
            return present_matched, base_scores, []
        faiss_ids = [row["faiss_id"] for row in doc_chunks.values()]
        embeddings = self._reconstruct_embeddings_batch(faiss_ids)
        emb_by_faiss = {fid: emb for fid, emb in zip(faiss_ids, embeddings, strict=False)}
        best_sim: Dict[int, float] = {}
        for m in present_matched:
            target = emb_by_faiss[doc_chunks[m]["faiss_id"]]
            target_norm = float(np.linalg.norm(target))
            for idx, row in doc_chunks.items():
                if idx in base_scores:
                    continue
                other = emb_by_faiss[row["faiss_id"]]
                denom = target_norm * float(np.linalg.norm(other))
                sim = float(np.dot(target, other) / denom) if denom else 0.0
                if idx not in best_sim or sim > best_sim[idx]:
                    best_sim[idx] = sim
        candidates = sorted(best_sim.items(), key=lambda kv: kv[1], reverse=True)
        return present_matched, base_scores, candidates

    @staticmethod
    def _select_enriched_indices_by_budget(
        doc_chunks: Dict[int, Any],
        present_matched: List[int],
        base_scores: Dict[int, float],
        candidates: List[Tuple[int, float]],
        budget: int,
        unit: str,
    ) -> Tuple[List[int], Dict[int, float]]:
        """Select matched chunks plus the most-similar others that fit the budget.

        Matched chunks are always kept; remaining chunks are added in descending
        similarity order while they fit (non-adjacent inclusion is allowed, so we
        keep scanning smaller candidates rather than stopping at the first overflow).
        """
        selected = set(present_matched)
        similarity_scores: Dict[int, float] = dict(base_scores)
        sep_size = _measure_text(_CONTEXT_SEPARATOR, None, unit)
        used = sum(_measure_text(doc_chunks[i]["content"], doc_chunks[i]["tokens"], unit) for i in present_matched)
        used += max(0, len(present_matched) - 1) * sep_size
        for idx, sim in candidates:
            size = _measure_text(doc_chunks[idx]["content"], doc_chunks[idx]["tokens"], unit) + sep_size
            if used + size <= budget:
                selected.add(idx)
                similarity_scores[idx] = sim
                used += size
        return sorted(selected), similarity_scores

    def _add_context_window_budget(
        self, results: List[QueryResult], context_window: int, context_unit: str, context_truncate: bool
    ) -> List[QueryResult]:
        context_results: List[QueryResult] = []
        doc_chunk_requests = defaultdict(list)
        for result in results:
            if result.type != "chunk":
                context_results.append(result)
                continue
            chunk_index = self._extract_chunk_index_from_id(result.id)
            doc_chunk_requests[result.document_id].append((chunk_index, result))
        with self.connection_pool.get_connection() as conn:
            for doc_id, chunk_requests in doc_chunk_requests.items():
                chunks_by_index = self._fetch_document_chunks_sync(conn, doc_id)
                for chunk_index, result in chunk_requests:
                    selected = self._select_context_indices_by_budget(
                        chunks_by_index, chunk_index, context_window, context_unit
                    )
                    if not selected:
                        context_results.append(result)
                        continue
                    ordered_rows = [chunks_by_index[i] for i in selected]
                    context_results.append(
                        self._build_budget_context_result(
                            doc_id, chunk_index, result, ordered_rows, context_window, context_unit, context_truncate
                        )
                    )
        return context_results

    def _enrich_with_intra_doc_context_budget(
        self, results: List[QueryResult], context_window: int, context_unit: str, context_truncate: bool
    ) -> List[QueryResult]:
        enriched_results: List[QueryResult] = []
        doc_chunks_to_enrich = defaultdict(list)
        with self.connection_pool.get_connection() as conn:
            for result in results:
                if result.type != "chunk":
                    enriched_results.append(result)
                    continue
                chunk_index = self._extract_chunk_index_from_id(result.id)
                doc_chunks_to_enrich[result.document_id].append((chunk_index, result))
            for doc_id, chunk_requests in doc_chunks_to_enrich.items():
                cursor = conn.execute(
                    """
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col,
                           end_line, end_col, tokens, faiss_id
                    FROM chunks
                    WHERE document_id = ? AND faiss_id IS NOT NULL
                    ORDER BY chunk_index
                    """,
                    [doc_id],
                )
                doc_chunks = {row["chunk_index"]: row for row in cursor.fetchall()}
                matched_indices = [ci for ci, _ in chunk_requests]
                matched_results = [r for _, r in chunk_requests]
                present_matched, base_scores, candidates = self._rank_intra_doc_similarities(
                    doc_chunks, matched_indices
                )
                if not present_matched:
                    enriched_results.extend(matched_results)
                    continue
                sorted_indices, similarity_scores = self._select_enriched_indices_by_budget(
                    doc_chunks, present_matched, base_scores, candidates, context_window, context_unit
                )
                ordered_rows = [doc_chunks[i] for i in sorted_indices]
                chunk_similarities = [similarity_scores[i] for i in sorted_indices]
                best_score = max(r.score for r in matched_results)
                combined_metadata: Dict[str, Any] = {}
                for r in matched_results:
                    combined_metadata.update(r.metadata)
                enriched_results.append(
                    self._build_budget_enriched_result(
                        doc_id,
                        ordered_rows,
                        best_score,
                        combined_metadata,
                        chunk_similarities,
                        matched_indices,
                        sorted_indices,
                        context_window,
                        context_unit,
                        context_truncate,
                    )
                )
        return enriched_results

    @staticmethod
    def _merge_overlapping_ranges(ranges_needed: List[Tuple[int, int, int, Any]]) -> List[Tuple[int, int, int, Any]]:
        if not ranges_needed:
            return []
        sorted_ranges = sorted(ranges_needed, key=lambda x: x[0])
        merged = []
        current_start, current_end, first_chunk, first_result = sorted_ranges[0]
        for start, end, chunk_idx, result in sorted_ranges[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end, first_chunk, first_result))
                current_start, current_end, first_chunk, first_result = start, end, chunk_idx, result
        merged.append((current_start, current_end, first_chunk, first_result))
        return merged

    def _enrich_with_intra_doc_context(
        self,
        results: List[QueryResult],
        context_window: int,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if context_window <= 0 or not results:
            return results
        if context_unit != "chunks":
            return self._enrich_with_intra_doc_context_budget(results, context_window, context_unit, context_truncate)
        enriched_results: List[QueryResult] = []
        doc_chunks_to_enrich = defaultdict(list)
        with self.connection_pool.get_connection() as conn:
            for result in results:
                if result.type != "chunk":
                    enriched_results.append(result)
                    continue
                doc_id = result.document_id
                chunk_index = self._extract_chunk_index_from_id(result.id)
                doc_chunks_to_enrich[doc_id].append((chunk_index, result))
            for doc_id, chunk_requests in doc_chunks_to_enrich.items():
                cursor = conn.execute(
                    """
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col,
                           end_line, end_col, faiss_id
                    FROM chunks
                    WHERE document_id = ? AND faiss_id IS NOT NULL
                    ORDER BY chunk_index
                """,
                    [doc_id],
                )
                doc_chunks = {row["chunk_index"]: row for row in cursor.fetchall()}
                doc_faiss_ids = [row["faiss_id"] for row in doc_chunks.values()]
                if len(doc_chunks) <= 1:
                    for chunk_index, result in chunk_requests:
                        if chunk_index not in doc_chunks:
                            enriched_results.append(result)
                            continue
                        target_chunk = doc_chunks[chunk_index]
                        enriched_result = QueryResult(
                            id=f"{doc_id}:enriched:{chunk_index}",
                            score=result.score,
                            type="enriched",
                            content=target_chunk["content"],
                            metadata=result.metadata.copy(),
                            document_id=doc_id,
                            position=ChunkPosition(
                                start=int(target_chunk["start_pos"]),
                                end=int(target_chunk["end_pos"]),
                                line=int(target_chunk["start_line"]),
                                column=int(target_chunk["start_col"]),
                                end_line=int(target_chunk["end_line"]),
                                end_column=int(target_chunk["end_col"]),
                            ),
                        )
                        enriched_result.metadata["_enriched_chunk_count"] = 1
                        enriched_result.metadata["_original_chunk_index"] = chunk_index
                        enriched_result.metadata["_similarity_scores"] = [1.0]
                        enriched_result.metadata["_enrichment_method"] = "single_chunk"
                        enriched_result.metadata["_context_unit"] = "chunks"
                        enriched_results.append(enriched_result)
                    continue
                doc_embeddings = self._reconstruct_embeddings_batch(doc_faiss_ids)
                faiss_id_to_embedding = {fid: emb for fid, emb in zip(doc_faiss_ids, doc_embeddings, strict=False)}
                all_matched_indices = [chunk_index for chunk_index, _ in chunk_requests]
                all_matched_results = [result for _, result in chunk_requests]
                relevant_chunks = set()
                similarity_scores: Dict[int, float] = {}
                for chunk_index in all_matched_indices:
                    if chunk_index not in doc_chunks:
                        continue
                    target_chunk = doc_chunks[chunk_index]
                    target_faiss_id = target_chunk["faiss_id"]
                    target_embedding = faiss_id_to_embedding[target_faiss_id]
                    relevant_chunks.add(chunk_index)
                    similarity_scores[chunk_index] = max(similarity_scores.get(chunk_index, 0), 1.0)
                    similarities = []
                    for other_chunk_idx, other_chunk in doc_chunks.items():
                        if other_chunk_idx == chunk_index:
                            continue
                        other_embedding = faiss_id_to_embedding[other_chunk["faiss_id"]]
                        similarity = float(
                            np.dot(target_embedding, other_embedding)
                            / (np.linalg.norm(target_embedding) * np.linalg.norm(other_embedding))
                        )
                        similarities.append((similarity, other_chunk_idx))
                    similarities.sort(reverse=True, key=lambda x: x[0])
                    for similarity, other_chunk_idx in similarities[: context_window - 1]:
                        relevant_chunks.add(other_chunk_idx)
                        similarity_scores[other_chunk_idx] = max(similarity_scores.get(other_chunk_idx, 0), similarity)
                if relevant_chunks:
                    sorted_chunk_indices = sorted(relevant_chunks)
                    combined_content = []
                    chunk_similarities = []
                    min_start_pos = float("inf")
                    max_end_pos = 0
                    min_start_line = float("inf")
                    min_start_col = float("inf")
                    max_end_line = 0
                    max_end_col = 0
                    for chunk_idx in sorted_chunk_indices:
                        chunk_data = doc_chunks[chunk_idx]
                        combined_content.append(chunk_data["content"])
                        chunk_similarities.append(similarity_scores[chunk_idx])
                        min_start_pos = min(min_start_pos, chunk_data["start_pos"])
                        max_end_pos = max(max_end_pos, chunk_data["end_pos"])
                        min_start_line = min(min_start_line, chunk_data["start_line"])
                        max_end_line = max(max_end_line, chunk_data["end_line"])
                        if chunk_data["start_line"] == min_start_line:
                            min_start_col = min(min_start_col, chunk_data["start_col"])
                        if chunk_data["end_line"] == max_end_line:
                            max_end_col = max(max_end_col, chunk_data["end_col"])
                    enriched_position = ChunkPosition(
                        start=int(min_start_pos),
                        end=int(max_end_pos),
                        line=int(min_start_line),
                        column=int(min_start_col),
                        end_line=int(max_end_line),
                        end_column=int(max_end_col),
                    )
                    best_score = max(result.score for result in all_matched_results)
                    combined_metadata: Dict[str, Any] = {}
                    for result in all_matched_results:
                        combined_metadata.update(result.metadata)
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(combined_content)
                    enriched_result = QueryResult(
                        id=f"{doc_id}:enriched",
                        score=best_score,
                        type="enriched",
                        content=combined_text,
                        metadata=combined_metadata,
                        document_id=doc_id,
                        position=enriched_position,
                    )
                    enriched_result.metadata["_enriched_chunk_count"] = len(relevant_chunks)
                    enriched_result.metadata["_matched_chunk_indices"] = all_matched_indices
                    enriched_result.metadata["_all_chunk_indices"] = sorted_chunk_indices
                    enriched_result.metadata["_similarity_scores"] = chunk_similarities
                    enriched_result.metadata["_enrichment_method"] = "intra_document_similarity"
                    enriched_result.metadata["_context_unit"] = "chunks"
                    enriched_results.append(enriched_result)
        return enriched_results

    # -------------------------------
    # Aggregation/scoring (sync/async)
    # -------------------------------
    def _aggregate_document_scores_with_method(
        self,
        chunk_results: List[QueryResult],
        method: DocumentScoringMethod = "frequency_boost",
        method_options: Optional[dict] = None,
    ) -> List[QueryResult]:
        if not chunk_results:
            return []
        method_options = method_options or {}
        doc_groups: Dict[str, List[QueryResult]] = defaultdict(list)
        for result in chunk_results:
            doc_id = result.document_id if result.type == "chunk" else result.id
            if doc_id is not None:
                doc_groups[doc_id].append(result)
        all_doc_ids = list(doc_groups.keys())
        if not all_doc_ids:
            return []
        with self.connection_pool.get_connection() as conn:
            placeholders = ",".join(["?"] * len(all_doc_ids))
            cursor = conn.execute(
                f"""
                SELECT id, content
                FROM documents
                WHERE id IN ({placeholders})
                """,
                all_doc_ids,
            )
            doc_content_map = {row["id"]: row["content"] for row in cursor.fetchall()}
            doc_metadata_batch = self._get_documents_metadata_batch(conn, all_doc_ids)
        scored_results: List[QueryResult] = self._compute_document_scores(
            method, method_options, doc_groups, doc_content_map, doc_metadata_batch
        )
        return scored_results

    @staticmethod
    def _compute_document_scores(method, method_options, doc_groups, doc_content_map, doc_metadata_batch):
        import math

        document_results: List[QueryResult] = []
        for doc_id, chunks in doc_groups.items():
            doc_content = doc_content_map.get(doc_id)
            if not doc_content:
                continue
            scores = [chunk.score for chunk in chunks]
            method_metadata = {}
            if method == "best":
                final_score = max(scores)
            elif method == "average":
                final_score = sum(scores) / len(scores)
            elif method == "frequency_boost":
                best_score = max(scores)
                if best_score == 0:
                    quality_weights = [1.0 for _ in scores]
                else:
                    quality_weights = [score / best_score for score in scores]
                effective_chunk_count = sum(quality_weights)
                frequency_multiplier = 1.0 + (math.log2(2 + effective_chunk_count) - 1) * method_options.get(
                    "frequency_bias", 0.3
                )
                method_metadata["effective_chunk_count"] = effective_chunk_count
                method_metadata["frequency_multiplier"] = frequency_multiplier
                final_score = min(1.0, best_score * frequency_multiplier)
            else:
                raise ValueError(
                    f"Unknown document_scoring_method: {method!r}. "
                    "Valid methods are 'best', 'average', 'frequency_boost'."
                )
            doc_metadata = doc_metadata_batch.get(doc_id, {})
            method_metadata["_aggregation_method"] = method
            method_metadata["_chunk_count"] = len(chunks)
            method_metadata["_best_chunk_score"] = max(scores)
            method_metadata["_average_chunk_score"] = sum(scores) / len(scores)
            doc_metadata["_scoring"] = method_metadata
            doc_result = QueryResult(
                id=doc_id,
                score=final_score,
                type="document",
                content=doc_content_map.get(doc_id, ""),
                metadata=doc_metadata,
            )
            document_results.append(doc_result)
        document_results.sort(key=lambda x: x.score, reverse=True)
        return document_results

    # ----------------------
    # Multi-column search
    # ----------------------
    def query_multi_column(
        self,
        query: str,
        *,
        columns: Optional[List[str]] = None,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Literal["documents", "chunks", "enriched"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
    ) -> List[QueryResult]:
        """
        Query across multiple columns (main content + embedding-enabled metadata fields)

        Parameters
        ----------
        query : str
            Query text
        columns : Optional[List[str]]
            Specific columns to search. If None, searches all embedding-enabled fields
            plus main content. Use 'content' for main document content.
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score to keep (0-1, higher=better). For ``search_type="hybrid"``
            each leg is min-max normalized *within this query's own candidate pool*,
            so scores are not comparable across queries or across different ``k``:
            the threshold cuts on rank position within the pool, not on absolute
            match quality, and is not a portable bar you can tune once and reuse.
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply. Filter fields must be declared in the
            metadata schema; unknown fields or unsupported operators raise
            ``DatabaseError``.
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters for the scoring method

        Returns
        -------
        List[QueryResult]
            Search results with column attribution
        """

        if filters:
            validate_filter_spec(filters, self.metadata_schema)
        with self._read_write_lock.read_lock():
            embedding_enabled_fields = self._get_embedding_enabled_fields()
            if columns is None:
                search_columns = ["content"] + list(embedding_enabled_fields.keys())
            else:
                search_columns = []
                for col in columns:
                    if col == "content" or col in embedding_enabled_fields:
                        search_columns.append(col)
                    else:
                        logger.warning(f"Column '{col}' is not embedding-enabled, skipping")
                if not search_columns:
                    logger.warning("No valid columns specified for search")
                    return []
            all_results: List[QueryResult] = []
            if "content" in search_columns:
                content_results = self.query(
                    query=query,
                    search_type=search_type,
                    return_type="chunks",
                    k=k * 2,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                )
                for result in content_results:
                    result.metadata = result.metadata or {}
                    result.metadata["_search_column"] = "content"
                    all_results.append(result)
            metadata_columns = [col for col in search_columns if col != "content"]
            if metadata_columns and search_type in ["vector", "hybrid"]:
                for field_name in metadata_columns:
                    field_results = self._search_metadata_field(
                        query=query, field_name=field_name, k=k * 2, score_threshold=score_threshold, filters=filters
                    )
                    for result in field_results:
                        result.metadata = result.metadata or {}
                        result.metadata["_search_column"] = field_name
                        all_results.append(result)
            all_results.sort(key=lambda x: x.score, reverse=True)
            limited_results = all_results[:k]
            if return_type == "documents":
                return self._aggregate_document_scores_with_method(
                    limited_results, document_scoring_method, document_scoring_options
                )
            else:
                return limited_results

    def _search_metadata_field(
        self, query: str, field_name: str, k: int, score_threshold: float, filters: Optional[Dict[str, Any]]
    ) -> List[QueryResult]:
        # Generate query embedding
        if hasattr(self.embedding_provider, "embed_query"):
            query_embedding = self.embedding_provider.embed_query(query)
        else:
            query_embedding = self.embedding_provider.embed_sync([query])[0]

        # Use shared business logic for SQL construction
        sql, params = self._build_metadata_field_search_sql(field_name)
        # Restrict the scored field embeddings to filter-matching documents before
        # the top-k truncation below, so a selective filter is not starved (T1.3).
        # Unpushable filters (dot-notation) fall through to the Python matcher.
        query_params: Tuple[Any, ...] = params
        built = self._build_filter_where(filters)
        if built is not None:
            where_clause, filter_params = built
            sql = sql + f" AND ce.document_id IN (SELECT id FROM documents WHERE {where_clause})"
            query_params = (*params, *filter_params)

        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(sql, query_params)
            field_embedding_data = cursor.fetchall()

            if not field_embedding_data:
                return []

            # Extract FAISS IDs and get embeddings
            faiss_ids = [row["faiss_id"] for row in field_embedding_data]
            if not faiss_ids:
                return []

            field_embeddings = self._reconstruct_embeddings_batch(faiss_ids)
            if field_embeddings.size == 0:
                return []

            # Use shared business logic for similarity calculation
            scores = self._calculate_embedding_similarities(query_embedding, field_embeddings)

            # Use shared business logic for filtering and sorting
            sorted_indices = self._filter_and_sort_by_scores(scores, score_threshold, k)
            if len(sorted_indices) == 0:
                return []

            # Build results using shared business logic
            results: List[QueryResult] = []
            for idx in sorted_indices:
                row_data = field_embedding_data[idx]
                doc_metadata = self._get_document_metadata(conn, row_data["document_id"])
                result = self._create_metadata_search_result(row_data, field_name, scores[idx], doc_metadata)
                results.append(result)

            # Apply filters if provided
            if filters:
                results = [r for r in results if matches_metadata_filter(r.metadata, filters)]

            return results

    def _get_document_metadata(self, conn, document_id: str) -> Dict[str, Any]:
        if not self.metadata_schema:
            return {}
        columns = ["id"] + list(self.metadata_schema.keys())
        cursor = conn.execute(f"SELECT {', '.join(columns)} FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        if not row:
            return {}
        return {col: row[col] for col in columns[1:]}

    @staticmethod
    def _combine_search_results(
        vector_results: List[QueryResult],
        keyword_results: List[QueryResult],
        keyword_ranks: Dict[str, float],
        vector_weight: float,
        k: int,
        score_threshold: float,
    ) -> List[QueryResult]:
        """Fuse the two legs into one ranking. ``keyword_ranks`` maps chunk key to raw BM25."""
        by_key: Dict[str, QueryResult] = {}
        vector_scores: Dict[str, float] = {}
        for result in vector_results:
            by_key[result.id] = result
            vector_scores[result.id] = result.score

        ranks: Dict[str, float] = {}
        for result in keyword_results:
            by_key.setdefault(result.id, result)
            ranks[result.id] = keyword_ranks[result.id]

        final_results: List[QueryResult] = []
        for key, final_score in _relative_score_fusion(vector_scores, ranks, vector_weight).items():
            if final_score >= score_threshold:
                result = by_key[key]
                result.score = final_score
                final_results.append(result)
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]

    # ----------------
    # Async Search API
    # ----------------
    async def query_async(
        self,
        query: str,
        *,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Optional[Literal["documents", "chunks", "sections", "context", "enriched"]] = None,
        search_level: Literal["chunks", "sections", "documents", "fused"] = "chunks",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        section_weight: float = 0.65,
        context_window: int = 2,
        context_unit: ContextUnit = "chunks",
        context_truncate: bool = False,
        semantic_dedup_threshold: Optional[float] = None,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        reranker: Optional[Any] = None,
        reranker_config: Optional[Dict[str, Any]] = None,
        rerank_k: Optional[int] = None,
    ) -> List[QueryResult]:
        """
        Async query the database using vector, keyword, or hybrid search

        Parameters
        ----------
        query : str
            Search query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform, by default 'hybrid'
        return_type : Optional[Literal['documents', 'chunks', 'sections', 'context', 'enriched']]
            The unit to report hits in. Defaults to ``None``, meaning "whatever
            unit ``search_level`` searched" -- documents for the default chunk
            search, sections for ``search_level='sections'``. See ``query()``.
        k : int
            Maximum number of results to return, by default 10
        score_threshold : float
            Minimum score to keep (0-1, higher=better). For ``search_type="hybrid"``
            each leg is min-max normalized *within this query's own candidate pool*,
            so scores are not comparable across queries or across different ``k``:
            the threshold cuts on rank position within the pool, not on absolute
            match quality, and is not a portable bar you can tune once and reuse., by default 0.0
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply, by default None. Filter fields must be
            declared in the metadata schema; unknown fields or unsupported
            operators raise ``DatabaseError``.
        vector_weight : float
            Weight for vector search in hybrid mode (0-1), by default 0.7
        context_window : int
            Size of the assembled context for return_type='context'/'enriched',
            measured in ``context_unit`` (chunks before/after or similar-chunk count
            when 'chunks'; an approximate token/word/character budget otherwise),
            by default 2
        context_unit : Literal['chunks', 'tokens', 'words', 'characters']
            Unit in which ``context_window`` is measured, by default 'chunks'.
        context_truncate : bool
            Hard-truncate the assembled context to exactly the token/word/character
            budget (only applies with a non-chunk ``context_unit``), by default False.
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar), by default None
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores, by default "frequency_boost"
            For detailed explanations and guidance on selecting the appropriate method,
            see the Document Scoring documentation.
        document_scoring_options : dict, optional
            Parameters for the document_scoring_method (to choose overall scores for documents from chunk results).
            For complete parameter documentation and examples, see the Document Scoring documentation.
        rerank_k : int, optional
            Size of the candidate pool to fetch and hand to the reranker before
            truncating to ``k``. Only has an effect when a ``reranker`` or
            ``reranker_config`` is supplied. Defaults to ``5*k`` (clamped to at
            most 200). See ``query()`` for the rationale.

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        _validate_context_unit(context_unit)
        return_type = _resolve_return_type(return_type, search_level)
        # See the sync path: a 'sections' return on a chunk-level search over a
        # non-hierarchical DB would silently degrade to chunk results.
        if (
            return_type == "sections"
            and search_level not in ("fused", "sections", "documents")
            and not self._hierarchical_embeddings
        ):
            raise ValueError(
                "return_type='sections' requires a hierarchical database "
                "(create with hierarchical_embeddings=True), or use search_level='sections'."
            )
        if filters:
            validate_filter_spec(filters, self.metadata_schema)
        self._ensure_async_pool()
        await self._ensure_async_schema_initialized()

        # Hierarchical + fused search levels: delegate to sync for now. Every
        # non-chunk level delegates unconditionally, so the sync path is the one
        # place that decides whether the DB can serve the level (and raises if not).
        if search_level in ("fused", "sections", "documents"):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.query(
                    query,
                    search_type=search_type,
                    return_type=return_type,
                    search_level=search_level,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    section_weight=section_weight,
                    context_window=context_window,
                    context_unit=context_unit,
                    context_truncate=context_truncate,
                    semantic_dedup_threshold=semantic_dedup_threshold,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options,
                    # Forward reranking so fused/hierarchical levels rerank remotely
                    # too (H8) instead of silently dropping the reranker.
                    reranker=reranker,
                    reranker_config=reranker_config,
                    rerank_k=rerank_k,
                ),
            )

        # Over-fetch a larger pool when reranking so the reranker can promote
        # candidates the legs ranked below `k`; `fetch_k == k` otherwise, leaving
        # the non-rerank path unchanged. Mirrors the sync `query()`.
        reranking = reranker is not None or bool(reranker_config)
        fetch_k = _resolve_rerank_k(rerank_k, k) if reranking else k

        query_embedding: Optional[np.ndarray] = None
        if search_type in ["vector", "hybrid"]:
            query_embedding = (await self.embedding_provider.embed_batch([query]))[0]
        results = await self._search_with_embedding_async(
            query,
            query_embedding,
            search_type,
            return_type if return_type != "sections" else "chunks",
            fetch_k,
            score_threshold,
            filters,
            vector_weight,
            context_window,
            semantic_dedup_threshold,
            document_scoring_method,
            document_scoring_options,
            context_unit,
            context_truncate,
        )

        # Apply reranking if configured
        if reranker is not None:
            results = await reranker.rerank_async(query, results, top_k=k)
        elif reranker_config:
            from localvectordb.reranking import RerankerRegistry

            _async_provider: str = reranker_config.get("provider", "")
            _reranker = RerankerRegistry.create_reranker(
                _async_provider,
                reranker_config.get("model"),
                **{kk: v for kk, v in reranker_config.items() if kk not in ("provider", "model")},
            )
            results = await _reranker.rerank_async(query, results, top_k=k)

        return results

    async def _search_with_embedding_async(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        search_type: Literal["vector", "keyword", "hybrid"],
        return_type: Literal["documents", "chunks", "sections", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        vector_weight: float,
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        effective_return_type: Literal["documents", "chunks", "context", "enriched"] = (
            "chunks" if return_type == "sections" else return_type
        )
        if search_type == "vector":
            assert query_embedding is not None
            results = await self._vector_search_with_embedding_async(
                query_embedding,
                effective_return_type,
                k,
                score_threshold,
                filters,
                context_window,
                semantic_dedup_threshold,
                document_scoring_method,
                document_scoring_options,
                context_unit,
                context_truncate,
            )
        elif search_type == "keyword":
            results = await self._keyword_search_async(
                query,
                effective_return_type,
                k,
                score_threshold,
                filters,
                context_window,
                semantic_dedup_threshold,
                document_scoring_method,
                document_scoring_options,
                context_unit,
                context_truncate,
            )
        elif search_type == "hybrid":
            assert query_embedding is not None
            results = await self._hybrid_search_with_embedding_async(
                query,
                query_embedding,
                effective_return_type,
                k,
                score_threshold,
                filters,
                vector_weight,
                context_window,
                semantic_dedup_threshold,
                document_scoring_method,
                document_scoring_options,
                context_unit,
                context_truncate,
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        return results

    async def _vector_search_with_embedding_async(
        self,
        query_embedding: np.ndarray,
        return_type: Literal["documents", "chunks", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:

        loop = asyncio.get_event_loop()
        # Ceiling the over-fetch at 100 bounds work for small k, but never below
        # k (nor the rerank fetch_k passed in as k), which would truncate/starve.
        search_k = max(k, min(k * 2, 100))

        # Push the metadata filter into FAISS (id-selector) when the index and
        # filter allow it, so a selective filter is not starved by the fixed
        # ``search_k`` pool (T1.3). The id lookup and FAISS search both run off
        # the event loop. The Python matcher below remains the authority.
        def protected_search():
            return self._filtered_index_search(self.index, query_embedding.reshape(1, -1), search_k, filters)

        distances_row, idx_row, exact = await loop.run_in_executor(None, protected_search)
        distances = distances_row.tolist()
        indices = idx_row.tolist()
        valid_results = [(dist, idx) for dist, idx in zip(distances, indices, strict=False) if idx != -1]
        if not valid_results:
            return []
        chunk_faiss_ids = [idx for _, idx in valid_results]
        chunks_data = await self._get_chunks_by_faiss_ids_async(chunk_faiss_ids)
        if filters:
            chunks_data = await self._apply_metadata_filters_async(chunks_data, filters)
        query_results: List[QueryResult] = []
        faiss_id_to_distance = {idx: dist for dist, idx in valid_results}
        for chunk_data in chunks_data:
            faiss_id = chunk_data["faiss_id"]
            if faiss_id in faiss_id_to_distance:
                distance = faiss_id_to_distance[faiss_id]
                similarity = self._distance_to_similarity(distance)
                if similarity >= score_threshold:
                    query_results.append(
                        QueryResult(
                            id=chunk_data["chunk_id"],
                            score=similarity,
                            type="chunk",
                            content=chunk_data["content"],
                            metadata=chunk_data.get("metadata", {}),
                            document_id=chunk_data["document_id"],
                            position=chunk_data.get("position"),
                        )
                    )
        if not exact and filters and len(query_results) < k:
            self._warn_filter_starved(len(query_results), k)
        query_results.sort(key=lambda x: x.score, reverse=True)
        if semantic_dedup_threshold is not None:
            query_results = await self._apply_semantic_deduplication_async(query_results, semantic_dedup_threshold)
        final_results = await self._process_search_results_async(
            query_results,
            return_type,
            document_scoring_method,
            document_scoring_options,
            context_window,
            context_unit,
            context_truncate,
        )
        return final_results[:k]

    async def _keyword_chunk_hits_async(
        self,
        query: str,
        limit: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[List[QueryResult], Dict[str, float]]:
        """Async counterpart of ``_keyword_chunk_hits``: hits best-first, plus raw BM25 ranks.

        Mirrors the sync path's filter contract (H10): a SQL-expressible filter is
        pushed into the FTS query via ``_build_filter_where`` -- which *declines
        gracefully* for a filter SQL cannot express (a dot-notation JSON path, an
        operator the builder rejects) instead of raising -- and
        ``matches_metadata_filter`` is always applied afterwards as the authority,
        so async neither crashes on an unpushable filter nor returns rows a
        broader-than-Python SQL clause would leak.
        """
        sanitized_query = FTSQuerySanitization.sanitize_fts_query(query)
        if not sanitized_query:
            return [], {}

        built = self._build_filter_where(filters)
        if built is not None:
            where_clause, filter_params = built
            # chunks_fts is contentless over chunks, so rowid == chunks.id.
            fts_sql = (
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? "
                "AND rowid IN (SELECT id FROM chunks WHERE document_id IN "
                f"(SELECT id FROM documents WHERE {where_clause})) "
                "ORDER BY rank ASC LIMIT ?"
            )
            fts_params: List[Any] = [sanitized_query, *filter_params, limit]
        else:
            fts_sql = (
                "SELECT rowid, bm25(chunks_fts) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY rank ASC LIMIT ?"
            )
            fts_params = [sanitized_query, limit]

        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(fts_sql, fts_params)
            fts_rows = await cursor.fetchall()

            valid_chunk_data: List[Tuple[int, float, float]] = []
            valid_chunk_ids: List[int] = []
            for row in fts_rows:
                score = self._fts_rank_to_similarity(row["rank"])
                if score < score_threshold:
                    continue
                valid_chunk_ids.append(row["rowid"])
                valid_chunk_data.append((row["rowid"], score, float(row["rank"])))
            if not valid_chunk_ids:
                return [], {}

            placeholders = ",".join(["?"] * len(valid_chunk_ids))
            cursor = await conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                valid_chunk_ids,
            )
            chunk_id_to_row: Dict[int, Any] = {}
            doc_ids_to_fetch: set[str] = set()
            async for row in cursor:
                chunk_id_to_row[row["id"]] = row
                doc_ids_to_fetch.add(row["document_id"])

        # Fetched outside the connection context above: the async pool is small
        # and this opens its own connection.
        doc_metadata_batch = await self._get_documents_metadata_async(list(doc_ids_to_fetch))

        query_results: List[QueryResult] = []
        raw_ranks: Dict[str, float] = {}
        for chunk_id, score, raw_rank in valid_chunk_data:
            chunk_row = chunk_id_to_row.get(chunk_id)
            if chunk_row is None:
                continue
            doc_metadata = doc_metadata_batch.get(chunk_row["document_id"], {})
            # Python authority: never return a row the matcher would reject.
            if filters and not matches_metadata_filter(doc_metadata, filters):
                continue
            position = ChunkPosition(
                start=chunk_row["start_pos"],
                end=chunk_row["end_pos"],
                line=chunk_row["start_line"],
                column=chunk_row["start_col"],
                end_line=chunk_row["end_line"],
                end_column=chunk_row["end_col"],
            )
            key = f"{chunk_row['document_id']}:{chunk_row['chunk_index']}"
            query_results.append(
                QueryResult(
                    id=key,
                    score=score,
                    type="chunk",
                    content=chunk_row["content"],
                    metadata=doc_metadata,
                    document_id=chunk_row["document_id"],
                    position=position,
                )
            )
            raw_ranks[key] = raw_rank
        return query_results, raw_ranks

    async def _keyword_search_async(
        self,
        query: str,
        return_type: Literal["documents", "chunks", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if not self.fts_enabled:
            logger.warning("FTS not enabled, returning empty results")
            return []
        query_results, _ = await self._keyword_chunk_hits_async(query, k * 2, score_threshold, filters)
        if semantic_dedup_threshold is not None:
            query_results = await self._apply_semantic_deduplication_async(query_results, semantic_dedup_threshold)
        final_results = await self._process_search_results_async(
            query_results,
            return_type,
            document_scoring_method,
            document_scoring_options,
            context_window,
            context_unit,
            context_truncate,
        )
        return final_results[:k]

    async def _hybrid_search_with_embedding_async(
        self,
        query: str,
        query_embedding: np.ndarray,
        return_type: Literal["documents", "chunks", "context", "enriched"],
        k: int,
        score_threshold: float,
        filters: Optional[Dict[str, Any]],
        vector_weight: float,
        context_window: int,
        semantic_dedup_threshold: Optional[float],
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if not self.fts_enabled:
            return await self._vector_search_with_embedding_async(
                query_embedding,
                return_type,
                k,
                score_threshold,
                filters,
                context_window,
                semantic_dedup_threshold,
                document_scoring_method,
                document_scoring_options,
                context_unit,
                context_truncate,
            )
        # Ceiling the over-fetch at 100 bounds work for small k, but never below
        # k (nor the rerank fetch_k passed in as k), which would truncate/starve.
        search_k = max(k, min(k * 4, 100))
        vector_task = asyncio.create_task(
            self._vector_search_with_embedding_async(query_embedding, "chunks", search_k, 0.0, filters, 0, None, "best")
        )
        # `search_k * 2` mirrors the over-fetch `_keyword_search_async` applies, so the
        # keyword leg sees the same candidate pool it always has.
        keyword_task = asyncio.create_task(self._keyword_chunk_hits_async(query, search_k * 2, 0.0, filters))
        vector_results, (keyword_results, keyword_ranks) = await asyncio.gather(vector_task, keyword_task)
        combined_results = await self._combine_search_results_async(
            vector_results=vector_results,
            keyword_results=keyword_results[:search_k],
            keyword_ranks=keyword_ranks,
            vector_weight=vector_weight,
            k=search_k,
            score_threshold=0.0,
        )
        if semantic_dedup_threshold is not None:
            combined_results = await self._apply_semantic_deduplication_async(
                combined_results, semantic_dedup_threshold
            )
        if score_threshold > 0.0:
            combined_results = [r for r in combined_results if r.score >= score_threshold]
        final_results = await self._process_search_results_async(
            combined_results,
            return_type,
            document_scoring_method,
            document_scoring_options,
            context_window,
            context_unit,
            context_truncate,
        )
        return final_results[:k]

    async def _get_chunks_by_faiss_ids_async(self, faiss_ids: List[int]) -> List[Dict[str, Any]]:
        if not faiss_ids:
            return []
        metadata_columns = list(self.metadata_schema.keys())
        chunk_columns = [
            "c.document_id",
            "c.chunk_index",
            "c.content",
            "c.faiss_id",
            "c.start_pos",
            "c.end_pos",
            "c.start_line",
            "c.start_col",
            "c.end_line",
            "c.end_col",
            "d.content as doc_content",
        ]
        # Only add metadata columns, not base document columns
        for col in metadata_columns:
            chunk_columns.append(f"d.{col}")
        placeholders = ",".join(["?" for _ in faiss_ids])
        sql = f"""
            SELECT {', '.join(chunk_columns)}
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.faiss_id IN ({placeholders})
        """
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(sql, faiss_ids)
            rows = await cursor.fetchall()
        chunks_data: List[Dict[str, Any]] = []
        for row in rows:
            position = ChunkPosition(
                start=row["start_pos"],
                end=row["end_pos"],
                line=row["start_line"],
                column=row["start_col"],
                end_line=row["end_line"],
                end_column=row["end_col"],
            )
            metadata: Dict[str, Any] = {}
            # Only include actual metadata fields, not base columns
            for field_name in metadata_columns:
                value = row[field_name]
                if value is not None:
                    field_def = self.metadata_schema[field_name]
                    if (
                        isinstance(field_def.type, MetadataFieldType)
                        and field_def.type.name == "JSON"
                        and isinstance(value, str)
                    ):
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            # Leave the raw value in place if it is not valid JSON.
                            pass
                    metadata[field_name] = value
            chunks_data.append(
                {
                    "chunk_id": f"{row['document_id']}:{row['chunk_index']}",
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "faiss_id": row["faiss_id"],
                    "position": position,
                    "metadata": metadata,
                }
            )
        return chunks_data

    async def _apply_metadata_filters_async(
        self, chunks_data: List[Dict[str, Any]], filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        if not filters or not chunks_data:
            return chunks_data
        doc_ids = list(set(chunk["document_id"] for chunk in chunks_data))
        if not doc_ids:
            return chunks_data
        doc_metadata = await self._get_documents_metadata_async(doc_ids)
        filtered_doc_ids = set()
        for doc_id, metadata in doc_metadata.items():
            if matches_metadata_filter(metadata, filters):
                filtered_doc_ids.add(doc_id)
        return [chunk for chunk in chunks_data if chunk["document_id"] in filtered_doc_ids]

    async def _get_documents_metadata_async(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not doc_ids:
            return {}
        # Get all columns but only put metadata fields in the metadata dict
        base_columns = ["id"]
        metadata_columns = list(self.metadata_schema.keys())
        all_columns = base_columns + metadata_columns
        if not all_columns:
            return {doc_id: {} for doc_id in doc_ids}
        sql = f"SELECT {', '.join(all_columns)} FROM documents WHERE id IN ({','.join(['?' for _ in doc_ids])})"
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(sql, doc_ids)
            rows = await cursor.fetchall()
        doc_metadata: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            doc_id = row["id"]
            metadata: Dict[str, Any] = {}
            # Only include actual metadata fields, not base columns
            for field_name in metadata_columns:
                value = row[field_name]
                if value is not None:
                    field_def = self.metadata_schema[field_name]
                    if (
                        isinstance(field_def.type, MetadataFieldType)
                        and field_def.type.name == "JSON"
                        and isinstance(value, str)
                    ):
                        try:
                            value = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            # Leave the raw value in place if it is not valid JSON.
                            pass
                metadata[field_name] = value
            doc_metadata[doc_id] = metadata
        return doc_metadata

    async def _combine_search_results_async(
        self,
        vector_results: List[QueryResult],
        keyword_results: List[QueryResult],
        keyword_ranks: Dict[str, float],
        vector_weight: float,
        k: int,
        score_threshold: float,
    ) -> List[QueryResult]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._combine_search_results,
            vector_results,
            keyword_results,
            keyword_ranks,
            vector_weight,
            k,
            score_threshold,
        )

    async def _process_search_results_async(
        self,
        results: List[QueryResult],
        return_type: Literal["documents", "chunks", "context", "enriched"],
        document_scoring_method: DocumentScoringMethod,
        document_scoring_options: Optional[dict],
        context_window: int,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if return_type == "chunks":
            return results
        if return_type == "documents":
            return await self._aggregate_document_scores_with_method_async(
                results, document_scoring_method, document_scoring_options or {}
            )
        elif return_type == "enriched":
            return await self._enrich_with_intra_doc_context_async(
                results, context_window, context_unit, context_truncate
            )
        else:
            return await self._add_context_window_async(results, context_window, context_unit, context_truncate)

    async def _apply_semantic_deduplication_async(
        self, results: List[QueryResult], threshold: float, max_candidates: int = 1000
    ) -> List[QueryResult]:
        if not results or threshold is None or threshold <= 0:
            return results
        chunk_results = [r for r in results if r.type == "chunk"]
        if len(chunk_results) <= 1:
            return results

        # Apply scaling limit to prevent O(n²) behavior
        if len(chunk_results) > max_candidates:
            logger.info(
                f"Limiting semantic deduplication to top {max_candidates} chunks "
                f"(from {len(chunk_results)} total) to prevent performance issues"
            )
            chunk_results = chunk_results[:max_candidates]

        chunk_identifiers = [(r.document_id, self._extract_chunk_index_from_id(r.id)) for r in chunk_results]
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ",".join(["(?,?)"] * len(chunk_identifiers))
            query = f"""
                SELECT document_id, chunk_index, faiss_id
                FROM chunks
                WHERE (document_id, chunk_index) IN ({placeholders})
            """
            params = [item for pair in chunk_identifiers for item in pair]
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            faiss_id_mapping = {(row["document_id"], row["chunk_index"]): row["faiss_id"] for row in rows}
        faiss_ids, result_mapping = [], {}
        for result in chunk_results:
            doc_id, chunk_idx = self._split_chunk_id(result.id)
            faiss_id = faiss_id_mapping.get((doc_id, chunk_idx))
            if faiss_id is not None:
                faiss_ids.append(faiss_id)
                result_mapping[faiss_id] = result
        if not faiss_ids:
            return results
        loop = asyncio.get_event_loop()
        embeddings_matrix = await loop.run_in_executor(None, self._reconstruct_embeddings_batch, faiss_ids)

        def compute_keep_mask():
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            normalized_embeddings = embeddings_matrix / np.maximum(norms, 1e-8)
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            similar_pairs = similarity_matrix >= threshold
            import numpy as _np

            _np.fill_diagonal(similar_pairs, False)
            upper_tri = _np.triu(similar_pairs, k=1)
            should_remove = _np.any(upper_tri, axis=0)
            keep_mask = ~should_remove
            return keep_mask

        keep_mask = await loop.run_in_executor(None, compute_keep_mask)
        final_chunk_results = [result_mapping[faiss_ids[i]] for i in range(len(faiss_ids)) if keep_mask[i]]

        # Log deduplication statistics
        removed_count = len(chunk_results) - len(final_chunk_results)
        if removed_count > 0:
            logger.debug(
                f"Semantic deduplication removed {removed_count} similar chunks "
                f"({removed_count / len(chunk_results) * 100:.1f}% deduplication rate)"
            )

        return final_chunk_results

    async def _aggregate_document_scores_with_method_async(
        self,
        chunk_results: List[QueryResult],
        method: DocumentScoringMethod = "frequency_boost",
        method_options: Optional[dict] = None,
    ) -> List[QueryResult]:
        if not chunk_results:
            return []
        method_options = method_options or {}
        doc_groups: Dict[str, List[QueryResult]] = defaultdict(list)
        for result in chunk_results:
            doc_id = result.document_id if result.type == "chunk" else result.id
            if doc_id is not None:
                doc_groups[doc_id].append(result)
        all_doc_ids = list(doc_groups.keys())
        if not all_doc_ids:
            return []
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ",".join(["?"] * len(all_doc_ids))
            cursor = await conn.execute(
                f"""
                SELECT id, content
                FROM documents
                WHERE id IN ({placeholders})
                """,
                all_doc_ids,
            )
            doc_content_map: Dict[str, str] = {}
            async for row in cursor:
                doc_content_map[row["id"]] = row["content"]
            doc_metadata_batch = await self._get_documents_metadata_async(all_doc_ids)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._compute_document_scores, method, method_options, doc_groups, doc_content_map, doc_metadata_batch
        )

    async def _fetch_document_chunks_async(
        self, conn: Any, doc_id: Optional[str], with_faiss: bool = False
    ) -> Dict[int, Any]:
        faiss_col = ", faiss_id" if with_faiss else ""
        faiss_filter = " AND faiss_id IS NOT NULL" if with_faiss else ""
        cursor = await conn.execute(
            f"""
            SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col, tokens{faiss_col}
            FROM chunks
            WHERE document_id = ?{faiss_filter}
            ORDER BY chunk_index
            """,
            [doc_id],
        )
        chunks_by_index: Dict[int, Any] = {}
        async for row in cursor:
            chunks_by_index[row["chunk_index"]] = row
        return chunks_by_index

    async def _add_context_window_budget_async(
        self, results: List[QueryResult], context_window: int, context_unit: str, context_truncate: bool
    ) -> List[QueryResult]:
        context_results: List[QueryResult] = []
        doc_chunk_requests = defaultdict(list)
        for result in results:
            if result.type != "chunk":
                context_results.append(result)
                continue
            chunk_index = self._extract_chunk_index_from_id(result.id)
            doc_chunk_requests[result.document_id].append((chunk_index, result))
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            for doc_id, chunk_requests in doc_chunk_requests.items():
                chunks_by_index = await self._fetch_document_chunks_async(conn, doc_id)
                for chunk_index, result in chunk_requests:
                    selected = self._select_context_indices_by_budget(
                        chunks_by_index, chunk_index, context_window, context_unit
                    )
                    if not selected:
                        context_results.append(result)
                        continue
                    ordered_rows = [chunks_by_index[i] for i in selected]
                    context_results.append(
                        self._build_budget_context_result(
                            doc_id, chunk_index, result, ordered_rows, context_window, context_unit, context_truncate
                        )
                    )
        return context_results

    async def _enrich_with_intra_doc_context_budget_async(
        self, results: List[QueryResult], context_window: int, context_unit: str, context_truncate: bool
    ) -> List[QueryResult]:
        enriched_results: List[QueryResult] = []
        doc_chunks_to_enrich = defaultdict(list)
        loop = asyncio.get_event_loop()
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            for result in results:
                if result.type != "chunk":
                    enriched_results.append(result)
                    continue
                chunk_index = self._extract_chunk_index_from_id(result.id)
                doc_chunks_to_enrich[result.document_id].append((chunk_index, result))
            for doc_id, chunk_requests in doc_chunks_to_enrich.items():
                doc_chunks = await self._fetch_document_chunks_async(conn, doc_id, with_faiss=True)
                matched_indices = [ci for ci, _ in chunk_requests]
                matched_results = [r for _, r in chunk_requests]
                present_matched, base_scores, candidates = await loop.run_in_executor(
                    None, self._rank_intra_doc_similarities, doc_chunks, matched_indices
                )
                if not present_matched:
                    enriched_results.extend(matched_results)
                    continue
                sorted_indices, similarity_scores = self._select_enriched_indices_by_budget(
                    doc_chunks, present_matched, base_scores, candidates, context_window, context_unit
                )
                ordered_rows = [doc_chunks[i] for i in sorted_indices]
                chunk_similarities = [similarity_scores[i] for i in sorted_indices]
                best_score = max(r.score for r in matched_results)
                combined_metadata: Dict[str, Any] = {}
                for r in matched_results:
                    combined_metadata.update(r.metadata)
                enriched_results.append(
                    self._build_budget_enriched_result(
                        doc_id,
                        ordered_rows,
                        best_score,
                        combined_metadata,
                        chunk_similarities,
                        matched_indices,
                        sorted_indices,
                        context_window,
                        context_unit,
                        context_truncate,
                    )
                )
        return enriched_results

    async def _add_context_window_async(
        self,
        results: List[QueryResult],
        context_window: int,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if context_window <= 0 or not results:
            return results
        if context_unit != "chunks":
            return await self._add_context_window_budget_async(results, context_window, context_unit, context_truncate)
        context_results: List[QueryResult] = []
        doc_chunk_requests = defaultdict(list)
        for result in results:
            if result.type != "chunk":
                context_results.append(result)
                continue
            doc_id = result.document_id
            chunk_index = self._extract_chunk_index_from_id(result.id)
            doc_chunk_requests[doc_id].append((chunk_index, result))
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            for doc_id, chunk_requests in doc_chunk_requests.items():
                ranges_needed = []
                for chunk_index, result in chunk_requests:
                    start_index = max(0, chunk_index - context_window)
                    end_index = chunk_index + context_window
                    ranges_needed.append((start_index, end_index, chunk_index, result))
                merged_ranges = self._merge_overlapping_ranges(ranges_needed)
                all_chunk_indices: set[int] = set()
                for start, end, _, _ in merged_ranges:
                    all_chunk_indices.update(range(start, end + 1))
                if not all_chunk_indices:
                    continue
                placeholders = ",".join(["?"] * len(all_chunk_indices))
                cursor = await conn.execute(
                    f"""
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col
                    FROM chunks
                    WHERE document_id = ? AND chunk_index IN ({placeholders})
                    ORDER BY chunk_index
                """,
                    [doc_id] + list(all_chunk_indices),
                )
                chunks_by_index: Dict[int, Any] = {}
                async for row in cursor:
                    chunks_by_index[row["chunk_index"]] = row
                for chunk_index, result in chunk_requests:
                    context_chunks = []
                    start_context = max(0, chunk_index - context_window)
                    end_context = chunk_index + context_window
                    for i in range(start_context, end_context + 1):
                        if i in chunks_by_index:
                            context_chunks.append(chunks_by_index[i])
                    if not context_chunks:
                        context_results.append(result)
                        continue
                    combined_content = []
                    min_start_pos = float("inf")
                    max_end_pos = 0
                    min_start_line = float("inf")
                    min_start_col = float("inf")
                    max_end_line = 0
                    max_end_col = 0
                    for chunk_row in context_chunks:
                        combined_content.append(chunk_row["content"])
                        min_start_pos = min(min_start_pos, chunk_row["start_pos"])
                        max_end_pos = max(max_end_pos, chunk_row["end_pos"])
                        min_start_line = min(min_start_line, chunk_row["start_line"])
                        max_end_line = max(max_end_line, chunk_row["end_line"])
                        if chunk_row["start_line"] == min_start_line:
                            min_start_col = min(min_start_col, chunk_row["start_col"])
                        if chunk_row["end_line"] == max_end_line:
                            max_end_col = max(max_end_col, chunk_row["end_col"])
                    context_position = ChunkPosition(
                        start=int(min_start_pos),
                        end=int(max_end_pos),
                        line=int(min_start_line),
                        column=int(min_start_col),
                        end_line=int(max_end_line),
                        end_column=int(max_end_col),
                    )
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(combined_content)
                    context_result = QueryResult(
                        id=f"{doc_id}:context:{chunk_index}",
                        score=result.score,
                        type="context",
                        content=combined_text,
                        metadata=result.metadata.copy(),
                        document_id=doc_id,
                        position=context_position,
                    )
                    context_result.metadata.update(
                        {
                            "_context_window": context_window,
                            "_context_unit": "chunks",
                            "_original_chunk_index": chunk_index,
                            "_context_chunk_count": len(context_chunks),
                            "_context_start_index": start_context,
                            "_context_end_index": end_context,
                        }
                    )
                    context_results.append(context_result)
        return context_results

    async def _enrich_with_intra_doc_context_async(
        self,
        results: List[QueryResult],
        context_window: int,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ) -> List[QueryResult]:
        if context_window <= 0 or not results:
            return results
        if context_unit != "chunks":
            return await self._enrich_with_intra_doc_context_budget_async(
                results, context_window, context_unit, context_truncate
            )
        enriched_results: List[QueryResult] = []
        doc_chunks_to_enrich = defaultdict(list)
        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            for result in results:
                if result.type != "chunk":
                    enriched_results.append(result)
                    continue
                doc_id = result.document_id
                chunk_index = self._extract_chunk_index_from_id(result.id)
                doc_chunks_to_enrich[doc_id].append((chunk_index, result))
            for doc_id, chunk_requests in doc_chunks_to_enrich.items():
                cursor = await conn.execute(
                    """
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col,
                           end_line, end_col, faiss_id
                    FROM chunks
                    WHERE document_id = ? AND faiss_id IS NOT NULL
                    ORDER BY chunk_index
                """,
                    [doc_id],
                )
                rows = await cursor.fetchall()
                doc_chunks = {row["chunk_index"]: row for row in rows}
                doc_faiss_ids = [row["faiss_id"] for row in doc_chunks.values()]
                if len(doc_chunks) <= 1:
                    for chunk_index, result in chunk_requests:
                        if chunk_index not in doc_chunks:
                            enriched_results.append(result)
                            continue
                        target_chunk = doc_chunks[chunk_index]
                        enriched_result = QueryResult(
                            id=f"{doc_id}:enriched:{chunk_index}",
                            score=result.score,
                            type="enriched",
                            content=target_chunk["content"],
                            metadata=result.metadata.copy(),
                            document_id=doc_id,
                            position=ChunkPosition(
                                start=int(target_chunk["start_pos"]),
                                end=int(target_chunk["end_pos"]),
                                line=int(target_chunk["start_line"]),
                                column=int(target_chunk["start_col"]),
                                end_line=int(target_chunk["end_line"]),
                                end_column=int(target_chunk["end_col"]),
                            ),
                        )
                        enriched_result.metadata["_enriched_chunk_count"] = 1
                        enriched_result.metadata["_original_chunk_index"] = chunk_index
                        enriched_result.metadata["_similarity_scores"] = [1.0]
                        enriched_result.metadata["_enrichment_method"] = "single_chunk"
                        enriched_result.metadata["_context_unit"] = "chunks"
                        enriched_results.append(enriched_result)
                    continue
                loop = asyncio.get_event_loop()
                doc_embeddings = await loop.run_in_executor(None, self._reconstruct_embeddings_batch, doc_faiss_ids)
                faiss_id_to_embedding = {fid: emb for fid, emb in zip(doc_faiss_ids, doc_embeddings, strict=False)}
                all_matched_indices = [chunk_index for chunk_index, _ in chunk_requests]
                all_matched_results = [result for _, result in chunk_requests]
                relevant_chunks = set()
                similarity_scores: Dict[int, float] = {}
                for chunk_index in all_matched_indices:
                    if chunk_index not in doc_chunks:
                        continue
                    target_chunk = doc_chunks[chunk_index]
                    target_faiss_id = target_chunk["faiss_id"]
                    target_embedding = faiss_id_to_embedding[target_faiss_id]
                    relevant_chunks.add(chunk_index)
                    similarity_scores[chunk_index] = max(similarity_scores.get(chunk_index, 0), 1.0)

                    def calc_sims(
                        doc_chunks=doc_chunks,
                        chunk_index=chunk_index,
                        faiss_id_to_embedding=faiss_id_to_embedding,
                        target_embedding=target_embedding,
                    ):
                        sims = []
                        import numpy as _np

                        for other_chunk_idx, other_chunk in doc_chunks.items():
                            if other_chunk_idx == chunk_index:
                                continue
                            other_embedding = faiss_id_to_embedding[other_chunk["faiss_id"]]
                            similarity = float(
                                _np.dot(target_embedding, other_embedding)
                                / (_np.linalg.norm(target_embedding) * _np.linalg.norm(other_embedding))
                            )
                            sims.append((similarity, other_chunk_idx, other_chunk))
                        return sims

                    similarities_data = await loop.run_in_executor(None, calc_sims)
                    similarities_data.sort(reverse=True, key=lambda x: x[0])
                    for similarity, other_chunk_idx, _ in similarities_data[: context_window - 1]:
                        relevant_chunks.add(other_chunk_idx)
                        similarity_scores[other_chunk_idx] = max(similarity_scores.get(other_chunk_idx, 0), similarity)
                if relevant_chunks:
                    sorted_chunk_indices = sorted(relevant_chunks)
                    combined_content = []
                    chunk_similarities: List[float] = []
                    min_start_pos = float("inf")
                    max_end_pos = 0
                    min_start_line = float("inf")
                    min_start_col = float("inf")
                    max_end_line = 0
                    max_end_col = 0
                    for chunk_idx in sorted_chunk_indices:
                        chunk_data = doc_chunks[chunk_idx]
                        combined_content.append(chunk_data["content"])
                        chunk_similarities.append(similarity_scores[chunk_idx])
                        min_start_pos = min(min_start_pos, chunk_data["start_pos"])
                        max_end_pos = max(max_end_pos, chunk_data["end_pos"])
                        min_start_line = min(min_start_line, chunk_data["start_line"])
                        max_end_line = max(max_end_line, chunk_data["end_line"])
                        if chunk_data["start_line"] == min_start_line:
                            min_start_col = min(min_start_col, chunk_data["start_col"])
                        if chunk_data["end_line"] == max_end_line:
                            max_end_col = max(max_end_col, chunk_data["end_col"])
                    enriched_position = ChunkPosition(
                        start=int(min_start_pos),
                        end=int(max_end_pos),
                        line=int(min_start_line),
                        column=int(min_start_col),
                        end_line=int(max_end_line),
                        end_column=int(max_end_col),
                    )
                    best_score = max(result.score for result in all_matched_results)
                    combined_metadata: Dict[str, Any] = {}
                    for result in all_matched_results:
                        combined_metadata.update(result.metadata)
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(combined_content)
                    enriched_result = QueryResult(
                        id=f"{doc_id}:enriched",
                        score=best_score,
                        type="enriched",
                        content=combined_text,
                        metadata=combined_metadata,
                        document_id=doc_id,
                        position=enriched_position,
                    )
                    enriched_result.metadata["_enriched_chunk_count"] = len(relevant_chunks)
                    enriched_result.metadata["_matched_chunk_indices"] = all_matched_indices
                    enriched_result.metadata["_all_chunk_indices"] = sorted_chunk_indices
                    enriched_result.metadata["_similarity_scores"] = chunk_similarities
                    enriched_result.metadata["_enrichment_method"] = "intra_document_similarity"
                    enriched_result.metadata["_context_unit"] = "chunks"
                    enriched_results.append(enriched_result)
        return enriched_results

    async def _search_metadata_field_async(
        self, query: str, field_name: str, k: int, score_threshold: float, filters: Optional[Dict[str, Any]]
    ) -> List[QueryResult]:
        """
        Async search a specific metadata field's embeddings

        Parameters
        ----------
        query : str
            Query text
        field_name : str
            Name of metadata field to search
        k : int
            Maximum results to return
        score_threshold : float
            Minimum score threshold
        filters : Optional[Dict[str, Any]]
            Metadata filters

        Returns
        -------
        List[QueryResult]
            Search results for this field
        """
        # Generate query embedding asynchronously
        embeddings = await self.embedding_provider.embed_batch([query])
        query_embedding = embeddings[0]

        # Use shared business logic for SQL construction
        sql, params = self._build_metadata_field_search_sql(field_name)
        # Restrict the scored field embeddings to filter-matching documents before
        # the top-k truncation below, so a selective filter is not starved (T1.3).
        query_params: Tuple[Any, ...] = params
        built = self._build_filter_where(filters)
        if built is not None:
            where_clause, filter_params = built
            sql = sql + f" AND ce.document_id IN (SELECT id FROM documents WHERE {where_clause})"
            query_params = (*params, *filter_params)

        assert self.async_connection_pool is not None
        async with self.async_connection_pool.get_connection_context() as conn:
            # Get all metadata field embeddings
            cursor = await conn.execute(sql, query_params)
            field_embedding_data_rows = list(await cursor.fetchall())

            if not field_embedding_data_rows:
                return []

            # Extract FAISS IDs for this field
            faiss_ids = [row["faiss_id"] for row in field_embedding_data_rows]

            if not faiss_ids:
                return []

            # Get embeddings for these FAISS IDs (run in executor as FAISS is not async)
            loop = asyncio.get_event_loop()
            field_embeddings = await loop.run_in_executor(None, self._reconstruct_embeddings_batch, faiss_ids)

            if field_embeddings.size == 0:
                return []

            # Use shared business logic for similarity calculation (run in executor)
            scores = await loop.run_in_executor(
                None, self._calculate_embedding_similarities, query_embedding, field_embeddings
            )

            # Use shared business logic for filtering and sorting (run in executor)
            sorted_indices = await loop.run_in_executor(
                None, self._filter_and_sort_by_scores, scores, score_threshold, k
            )

            if len(sorted_indices) == 0:
                return []

            results = []

            # Get document metadata for all results in batch
            doc_ids = [field_embedding_data_rows[idx]["document_id"] for idx in sorted_indices]
            doc_metadata_batch = await self._get_documents_metadata_async(doc_ids)

            # Build results using shared business logic
            for idx in sorted_indices:
                row_data = field_embedding_data_rows[idx]
                doc_metadata = doc_metadata_batch.get(row_data["document_id"], {})
                result = self._create_metadata_search_result(row_data, field_name, scores[idx], doc_metadata)
                results.append(result)

            # Apply metadata filters if provided
            if filters:
                # Use the existing sync filter function in executor
                def apply_filters():
                    return [r for r in results if matches_metadata_filter(r.metadata, filters)]

                results = await loop.run_in_executor(None, apply_filters)

            return results

    async def query_multi_column_async(
        self,
        query: str,
        *,
        columns: Optional[List[str]] = None,
        search_type: Literal["vector", "keyword", "hybrid"] = "hybrid",
        return_type: Literal["documents", "chunks", "context", "enriched"] = "documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.5,
        document_scoring_method: DocumentScoringMethod = "frequency_boost",
        document_scoring_options: Optional[dict] = None,
    ) -> List[QueryResult]:
        """
        Async query across multiple columns (main content + embedding-enabled metadata fields)

        Parameters
        ----------
        query : str
            Query text
        columns : Optional[List[str]]
            Specific columns to search. If None, searches all embedding-enabled fields
            plus main content. Use 'content' for main document content.
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score to keep (0-1, higher=better). For ``search_type="hybrid"``
            each leg is min-max normalized *within this query's own candidate pool*,
            so scores are not comparable across queries or across different ``k``:
            the threshold cuts on rank position within the pool, not on absolute
            match quality, and is not a portable bar you can tune once and reuse.
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply. Filter fields must be declared in the
            metadata schema; unknown fields or unsupported operators raise
            ``DatabaseError``.
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters for the scoring method

        Returns
        -------
        List[QueryResult]
            Search results with column attribution
        """
        self._ensure_async_pool()

        # Determine which columns to search
        embedding_enabled_fields = self._get_embedding_enabled_fields()

        if columns is None:
            # Search all embedding-enabled fields plus main content
            search_columns = ["content"] + list(embedding_enabled_fields.keys())
        else:
            # Validate requested columns
            search_columns = []
            for col in columns:
                if col == "content":
                    search_columns.append(col)
                elif col in embedding_enabled_fields:
                    search_columns.append(col)
                else:
                    logger.warning(f"Column '{col}' is not embedding-enabled, skipping")

            if not search_columns:
                logger.warning("No valid columns specified for search")
                return []

        all_results = []

        # Search main content if requested
        if "content" in search_columns:
            content_results = await self.query_async(
                query=query,
                search_type=search_type,
                return_type="chunks",  # Always get chunks for multi-column
                k=k * 2,  # Get more results to allow for proper ranking
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                document_scoring_method=document_scoring_method,
                document_scoring_options=document_scoring_options,
            )

            # Add column attribution
            for result in content_results:
                result.metadata = result.metadata or {}
                result.metadata["_search_column"] = "content"
                all_results.append(result)

        # Search metadata fields
        metadata_columns = [col for col in search_columns if col != "content"]
        if metadata_columns and search_type in ["vector", "hybrid"]:
            # Create tasks for concurrent metadata field searches
            metadata_search_tasks = []
            for field_name in metadata_columns:
                task = asyncio.create_task(
                    self._search_metadata_field_async(
                        query=query, field_name=field_name, k=k * 2, score_threshold=score_threshold, filters=filters
                    )
                )
                metadata_search_tasks.append((field_name, task))

            # Wait for all metadata searches to complete
            for field_name, task in metadata_search_tasks:
                field_results = await task

                # Add column attribution
                for result in field_results:
                    result.metadata = result.metadata or {}
                    result.metadata["_search_column"] = field_name
                    all_results.append(result)

        # Sort all results by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        limited_results = all_results[:k]

        if return_type == "documents":
            # Aggregate chunks into documents
            return await self._aggregate_document_scores_with_method_async(
                limited_results, document_scoring_method, document_scoring_options
            )
        else:
            return limited_results
