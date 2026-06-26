"""
LocalVectorDB Query Builder - SQL-like interface for vector database queries.

This module provides a high-level, fluent interface for building complex queries against
vector databases. It combines the power of vector similarity search with traditional database
operations like filtering, grouping, aggregation, and sorting, all through a SQL-like API.

The QueryBuilder supports multiple search modes (vector, keyword, hybrid), semantic filtering,
result reranking, and both synchronous and asynchronous execution.

Core Features
-------------
- **Multi-modal Search**: Vector similarity, keyword (FTS), and hybrid search
- **Advanced Filtering**: Exact filters with operators, semantic similarity filters
- **SQL-like Operations**: GROUP BY, aggregations (COUNT, SUM, AVG, etc.), HAVING, ORDER BY
- **Result Processing**: Pagination, deduplication, reranking, context windows
- **Async Support**: Native async execution with optimized semantic filtering
- **Performance**: Query planning, execution hints, batch processing
- **Debugging**: Query explanation, execution statistics, SQL-like previews

Examples
--------

**Basic Vector Search**::

    # Simple semantic search
    results = (db.query_builder()
        .search("machine learning algorithms")
        .limit(10)
        .execute())

**Advanced Filtering and Search Combinations**::

    # Complex multi-field query with exact and semantic filters
    results = (db.query_builder()
        .search("deep learning frameworks")
        .filter("year", gte_=2020, lt_=2024)          # Year range
        .filter("category", "AI")                     # Exact match
        .semantic_filter("methodology", "supervised learning", threshold=0.75)
        .order_by("year", "desc")
        .limit(50)
        .execute())

**Asynchronous Execution**::

    # Async query with semantic filtering
    async def search_research_papers():
        results = await (db.query_builder()
            .search("quantum computing algorithms")
            .filter("year", gte_=2022)
            .semantic_filter("approach", "quantum machine learning", threshold=0.8)
            .order_by("citation_count", "desc")
            .limit(25)
            .execute_async())
        return results
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

import numpy as np

from localvectordb._filters import FILTER_OPERATORS
from localvectordb.core import Document, DocumentScoringMethod, QueryResult
from localvectordb.cursor import QueryCursor
from localvectordb.database.base import BaseVectorDB
from localvectordb.utils import parse_iso8601

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics for semantic filtering."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class SearchClause:
    """Represents a search operation on a specific field."""

    field: str  # Currently only 'content' is supported
    query: str
    weight: float = 1.0
    search_type: Literal["vector", "keyword", "hybrid"] = "vector"
    score_threshold: Optional[float] = None


@dataclass
class SemanticFilter:
    """Semantic filtering based on conceptual similarity with async support."""

    field: str
    concept: str
    threshold: float
    metric: SimilarityMetric = SimilarityMetric.COSINE
    embedding_model: Optional[str] = None

    async def apply_async(self, documents: List[Document], db: "BaseVectorDB") -> List[Document]:
        """Apply semantic filtering with async embedding generation."""
        if not documents:
            return documents

        # Extract field contents efficiently
        field_contents = []
        valid_docs = []

        for doc in documents:
            content = self._extract_field_content(doc, self.field)
            if content:
                field_contents.append(content)
                valid_docs.append(doc)

        if not field_contents:
            return []

        try:
            # Use the database's embedding provider directly
            embedding_provider = db.embedding_provider

            # Generate embeddings - prefer async methods if available

            concept_embedding = (await embedding_provider.embed_batch([self.concept]))[0]
            field_embeddings = await embedding_provider.embed_batch(field_contents)

        except Exception as e:
            logger.error(f"Async embedding generation failed: {e}")
            raise e

        if len(field_embeddings) != len(valid_docs):
            raise ValueError(
                f"Embedding provider returned {len(field_embeddings)} embeddings for "
                f"{len(valid_docs)} documents; cannot align semantic-filter scores."
            )

        # Apply similarity filtering
        filtered_docs = []
        for doc, field_embedding in zip(valid_docs, field_embeddings, strict=True):
            similarity = self._calculate_similarity(concept_embedding, field_embedding)

            if similarity >= self.threshold:
                # Add semantic score to metadata for transparency
                doc.metadata["_semantic_scores"] = doc.metadata.get("_semantic_scores", {})
                doc.metadata["_semantic_scores"][f"{self.field}_{self.concept}"] = float(similarity)
                filtered_docs.append(doc)

        return filtered_docs

    def apply(self, documents: List[Document], db: BaseVectorDB) -> List[Document]:
        """Apply semantic filtering synchronously."""
        if not documents:
            return documents

        # Extract field contents efficiently
        field_contents = []
        valid_docs = []

        for doc in documents:
            content = self._extract_field_content(doc, self.field)
            if content:
                field_contents.append(content)
                valid_docs.append(doc)

        if not field_contents:
            return []

        # Use embedding provider directly
        concept_embedding = db.embedding_provider.embed_sync([self.concept])[0]
        field_embeddings = db.embedding_provider.embed_sync(field_contents)

        if len(field_embeddings) != len(valid_docs):
            raise ValueError(
                f"Embedding provider returned {len(field_embeddings)} embeddings for "
                f"{len(valid_docs)} documents; cannot align semantic-filter scores."
            )

        # Apply similarity filtering
        filtered_docs = []
        for doc, field_embedding in zip(valid_docs, field_embeddings, strict=True):
            similarity = self._calculate_similarity(concept_embedding, field_embedding)

            if similarity >= self.threshold:
                # Add semantic score to metadata for transparency
                doc.metadata["_semantic_scores"] = doc.metadata.get("_semantic_scores", {})
                doc.metadata["_semantic_scores"][f"{self.field}_{self.concept}"] = float(similarity)
                filtered_docs.append(doc)

        return filtered_docs

    @staticmethod
    def _extract_field_content(doc: Document, field: str) -> Optional[str]:
        """Extract content from document field with support for nested fields."""
        if field == "content":
            return doc.content

        # Support dot notation for nested metadata
        if "." in field:
            parts = field.split(".")
            value = doc.metadata
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return str(value) if value is not None else None

        # Simple metadata field
        if field in doc.metadata:
            value = doc.metadata[field]
            return str(value) if value is not None else None

        return None

    def _calculate_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate similarity using the specified metric."""
        a_flat = a.flatten()
        b_flat = b.flatten()

        if self.metric == SimilarityMetric.COSINE:
            denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
            if denom == 0.0:
                return 0.0
            return float(np.dot(a_flat, b_flat) / denom)
        elif self.metric == SimilarityMetric.DOT_PRODUCT:
            return float(np.dot(a_flat, b_flat))
        elif self.metric == SimilarityMetric.EUCLIDEAN:
            # Convert distance to similarity (0-1 range)
            distance = float(np.linalg.norm(a_flat - b_flat))
            return float(1.0 / (1.0 + distance))
        elif self.metric == SimilarityMetric.MANHATTAN:
            # Convert L1 distance to similarity
            distance = np.sum(np.abs(a_flat - b_flat))
            return float(1.0 / (1.0 + distance))
        else:
            raise ValueError(f"Unsupported similarity metric: {self.metric}")


@dataclass
class AggregationClause:
    """Represents an aggregation operation."""

    field: str
    function: Literal["count", "sum", "avg", "min", "max", "std", "var"]
    alias: Optional[str] = None


# Remove the preceding '$', used in QueryBuilder.filter
FILTER_OPERATOR_NAMES = tuple(map(lambda x: x[1:], FILTER_OPERATORS))


class QueryBuilder:
    """
    Fluent interface for building complex vector database queries with async support.

    This class provides a SQL-like interface for building sophisticated search
    and filter operations against vector databases.
    """

    def __init__(self, db: BaseVectorDB):
        self._db = db
        self._search_clauses: List[SearchClause] = []
        self._exact_filters: List[Dict[str, Any]] = []
        self._semantic_filters: List[SemanticFilter] = []
        self._search_type: Literal["vector", "keyword", "hybrid"] = "hybrid"
        self._vector_weight: float = 0.7
        self._return_type: Literal["documents", "chunks", "sections", "context"] = "documents"
        self._search_level: Literal["chunks", "sections", "documents"] = "chunks"
        self._limit: int = 10
        self._offset: int = 0
        self._order_by: List[tuple[str, str]] = []  # (field, direction)
        self._group_by: List[str] = []
        self._aggregations: List[AggregationClause] = []
        self._having_clauses: List[Dict[str, Any]] = []
        self._rerank_config: Optional[Dict[str, Any]] = None
        self._explain: bool = False
        self._context_window: int = 2
        self._semantic_dedup_threshold: Optional[float] = None
        self._document_scoring_method: DocumentScoringMethod = "frequency_boost"
        self._document_scoring_options: Optional[dict] = None

    def clone(self) -> "QueryBuilder":
        """Create a copy of this QueryBuilder for chaining."""
        new_builder = QueryBuilder(self._db)
        new_builder._search_clauses = self._search_clauses.copy()
        new_builder._exact_filters = self._exact_filters.copy()
        new_builder._semantic_filters = self._semantic_filters.copy()
        new_builder._search_type = self._search_type
        new_builder._vector_weight = self._vector_weight
        new_builder._return_type = self._return_type
        new_builder._search_level = self._search_level
        new_builder._limit = self._limit
        new_builder._offset = self._offset
        new_builder._order_by = self._order_by.copy()
        new_builder._group_by = self._group_by.copy()
        new_builder._aggregations = self._aggregations.copy()
        new_builder._having_clauses = self._having_clauses.copy()
        new_builder._rerank_config = self._rerank_config
        new_builder._explain = self._explain
        new_builder._context_window = self._context_window
        new_builder._semantic_dedup_threshold = self._semantic_dedup_threshold
        new_builder._document_scoring_method = self._document_scoring_method
        new_builder._document_scoring_options = self._document_scoring_options
        return new_builder

    # Core search methods
    def search(self, query: str, search_type=None, vector_weight=None, score_threshold=None) -> "QueryBuilder":
        """Search the content for query text."""
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if search_type and search_type not in ("vector", "hybrid", "keyword"):
            raise ValueError("`search_type` must be one of 'vector', 'hybrid', 'keyword'.")

        if isinstance(vector_weight, float) and (vector_weight < 0.0 or vector_weight > 1.0):
            raise ValueError("`vector_weight` must be a float between 0.0 and 1.0")

        builder = self.clone()
        builder._search_clauses.append(
            SearchClause("content", query, 1.0, search_type or self._search_type, score_threshold)
        )

        if vector_weight:
            builder._vector_weight = vector_weight

        return builder

    def search_field(self, field: str, query: str) -> "QueryBuilder":
        """Find records where field contains query."""
        if not field or not isinstance(field, str):
            raise ValueError("Field must be a non-empty string")

        builder = self.clone()
        # Use ilike for string values, exact match for others
        if isinstance(query, str):
            filter_dict = {field: {"$ilike": query}}
        else:
            filter_dict = {field: query}
        builder._exact_filters.append(filter_dict)
        return builder

    def filter(self, field: Optional[str] = None, value: Any = None, **kwargs: Any) -> "QueryBuilder":
        """
        Add exact filter conditions.

        Parameters
        ----------
        field : str, optional
            Field to filter on
        value : Any, optional
            Value to filter for
        kwargs : dict
            Key-value pairs for filtering multiple fields,
            or operator suffixes for advanced filtering

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with filter conditions added

        Examples
        --------
        Basic filtering::

            # Simple equality filter
            builder.filter("year", 2023)

            # Multiple fields
            builder.filter(year=2023, category="AI")

        Advanced filtering::

            # Range conditions
            builder.filter("year", gt_=2020, lt_=2024)

            # Field exists
            builder.filter("tags", exists_=True)

        Raises
        ------
        ValueError
            If field is not a string when provided directly
        """
        if field is not None and (not isinstance(field, str) or not field):
            raise ValueError("Field must be a non-empty string")

        # Validate operator suffixes in kwargs
        for key in kwargs:
            if key.endswith("_") and key[:-1] not in FILTER_OPERATOR_NAMES:
                raise ValueError(f"Invalid operator suffix in '{key}'")

        builder = self.clone()

        if field is not None and value is not None:
            builder._exact_filters.append({field: value})

        for key, val in kwargs.items():
            if key.endswith("_"):
                operator = key[:-1]
                if field is None:
                    raise ValueError("Field must be specified when using operators")
                builder._exact_filters.append({field: {f"${operator}": val}})
            else:
                if key in FILTER_OPERATOR_NAMES:
                    if field is None:
                        raise ValueError("Field must be specified when using operators")
                    builder._exact_filters.append({field: {f"${key}": val}})
                else:
                    builder._exact_filters.append({key: val})

        return builder

    def semantic_filter(
        self, field: str, concept: str, threshold: float = 0.8, metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> "QueryBuilder":
        """Add semantic filtering based on conceptual similarity."""
        if not field or not isinstance(field, str):
            raise ValueError("Field must be a non-empty string")

        if not concept or not isinstance(concept, str):
            raise ValueError("Concept must be a non-empty string")

        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")

        builder = self.clone()
        semantic_filter = SemanticFilter(field, concept, threshold, metric)
        builder._semantic_filters.append(semantic_filter)
        return builder

    def limit(self, n: int) -> "QueryBuilder":
        """
        Limit the number of results.

        Parameters
        ----------
        n : int
            Maximum number of results to return

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with limit applied

        Raises
        ------
        ValueError
            If n is not positive
        """
        if n <= 0:
            raise ValueError("Limit must be positive")
        builder = self.clone()
        builder._limit = n
        return builder

    def offset(self, n: int) -> "QueryBuilder":
        """
        Skip the first n results.

        Parameters
        ----------
        n : int
            Number of results to skip

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with offset applied

        Raises
        ------
        ValueError
            If n is negative
        """
        if n < 0:
            raise ValueError("Offset must be non-negative")
        builder = self.clone()
        builder._offset = n
        return builder

    # Search type convenience methods
    def vector(self, query, score_threshold=None) -> "QueryBuilder":
        """Use vector search."""
        return self.search(query, search_type="vector", score_threshold=score_threshold)

    def keyword(self, query, score_threshold=None) -> "QueryBuilder":
        """Use keyword search."""
        return self.search(query, search_type="keyword", score_threshold=score_threshold)

    def hybrid(self, query, vector_weight: float = 0.7, score_threshold=None) -> "QueryBuilder":
        """Use hybrid search with specified vector weight."""
        return self.search(query, search_type="hybrid", vector_weight=vector_weight, score_threshold=score_threshold)

    # Return type configuration
    def semantic_dedup(self, threshold: float) -> "QueryBuilder":
        """Enable semantic deduplication with similarity threshold."""
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Semantic deduplication threshold must be between 0.0 and 1.0")
        builder = self.clone()
        builder._semantic_dedup_threshold = threshold
        return builder

    def documents(
        self, scoring_method: DocumentScoringMethod = "frequency_boost", scoring_options: Optional[dict] = None
    ) -> "QueryBuilder":
        """Return full documents in results (default)."""
        builder = self.clone()
        builder._return_type = "documents"
        builder._document_scoring_method = scoring_method
        builder._document_scoring_options = scoring_options
        return builder

    def chunks(self) -> "QueryBuilder":
        """Return individual chunks in results with position information."""
        builder = self.clone()
        builder._return_type = "chunks"
        return builder

    def sections(self) -> "QueryBuilder":
        """Return section-level results (requires hierarchical_embeddings)."""
        builder = self.clone()
        builder._return_type = "sections"
        return builder

    def search_level(self, level: Literal["chunks", "sections", "documents"]) -> "QueryBuilder":
        """Set the FAISS index to search ('chunks', 'sections', or 'documents').

        Parameters
        ----------
        level : str
            Which FAISS index to search.
        """
        if level not in ("chunks", "sections", "documents"):
            raise ValueError("`level` must be 'chunks', 'sections', or 'documents'")
        builder = self.clone()
        builder._search_level = level
        return builder

    def context(self, window_size: int = 2) -> "QueryBuilder":
        """Return chunks with context window."""
        builder = self.clone()
        builder._return_type = "context"
        builder._context_window = window_size
        return builder

    # Ordering methods
    def order_by(self, field: str, direction: str = "desc") -> "QueryBuilder":
        """Add ordering by specified field."""
        if direction.lower() not in ["asc", "desc"]:
            raise ValueError("`direction` must be 'asc' or 'desc'")
        if not field:
            raise ValueError("`field` must be a non-empty string")

        builder = self.clone()
        builder._order_by.append((field, direction.lower()))
        return builder

    def order_by_score(self, direction: str = "desc") -> "QueryBuilder":
        """Order results by relevance score."""
        return self.order_by("score", direction)

    def clear_ordering(self) -> "QueryBuilder":
        """Remove all ordering clauses."""
        builder = self.clone()
        builder._order_by.clear()
        return builder

    # Grouping and aggregation methods
    def group_by(self, *fields: str) -> "QueryBuilder":
        """Group results by one or more fields."""
        if not fields:
            raise ValueError("At least one field must be specified for grouping")

        for field in fields:
            if not isinstance(field, str) or not field.strip():
                raise ValueError("All group_by fields must be non-empty strings")

        builder = self.clone()
        builder._group_by.extend(fields)
        return builder

    def aggregate(
        self,
        field: str,
        function: Literal["count", "sum", "avg", "min", "max", "std", "var"],
        alias: Optional[str] = None,
    ) -> "QueryBuilder":
        """Add an aggregation function."""
        valid_functions = ["count", "sum", "avg", "min", "max", "std", "var"]
        if function not in valid_functions:
            raise ValueError(f"function must be one of: {', '.join(valid_functions)}")

        builder = self.clone()
        aggregation = AggregationClause(field=field, function=function, alias=alias)
        builder._aggregations.append(aggregation)
        return builder

    def count_by(self, field: str = "*", alias: Optional[str] = None) -> "QueryBuilder":
        """Count documents/chunks, optionally grouped by field."""
        return self.aggregate(field, "count", alias or "count")

    def sum_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Sum numeric values in a field."""
        return self.aggregate(field, "sum", alias or f"sum_{field}")

    def avg_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Calculate average of numeric values in a field."""
        return self.aggregate(field, "avg", alias or f"avg_{field}")

    def min_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """
        Find minimum value in a field.

        Parameters
        ----------
        field : str
            Field name containing values
        alias : str, optional
            Alias for the minimum result

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with minimum aggregation
        """
        return self.aggregate(field, "min", alias or f"min_{field}")

    def max_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """
        Find maximum value in a field.

        Parameters
        ----------
        field : str
            Field name containing values
        alias : str, optional
            Alias for the maximum result

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with maximum aggregation
        """
        return self.aggregate(field, "max", alias or f"max_{field}")

    def having(self, field: str, operator: str, value: Any) -> "QueryBuilder":
        """Add HAVING clause for post-aggregation filtering.

        Note
        ----
        Unlike :meth:`filter`, ``having`` supports only the comparison operators
        ``eq``, ``ne``, ``gt``, ``gte``, ``lt`` and ``lte``. Set/pattern operators
        such as ``in``, ``like`` or ``contains`` are not available for HAVING clauses.
        """
        valid_operators = ["eq", "ne", "gt", "gte", "lt", "lte"]
        if operator not in valid_operators:
            raise ValueError(f"operator must be one of: {', '.join(valid_operators)}")

        builder = self.clone()
        having_clause = {field: {f"${operator}": value}}
        builder._having_clauses.append(having_clause)
        return builder

    def having_count(self, operator: str, value: int, alias: str = "count") -> "QueryBuilder":
        """Add HAVING clause for count aggregations."""
        return self.having(alias, operator, value)

    # Reranking methods
    def rerank(self, method: str, **config) -> "QueryBuilder":
        """Add reranking configuration for result post-processing."""
        valid_methods = ["relevance", "recency", "diversity", "custom", "cross_encoder"]
        if method not in valid_methods:
            raise ValueError(f"rerank method must be one of: {', '.join(valid_methods)}")

        builder = self.clone()
        builder._rerank_config = {"method": method, **config}
        return builder

    def rerank_by_recency(self, date_field: str = "updated_at", weight: float = 1.0) -> "QueryBuilder":
        """Rerank results by recency (newer documents ranked higher)."""
        return self.rerank("recency", date_field=date_field, weight=weight)

    def rerank_by_diversity(self, field: str, weight: float = 1.0) -> "QueryBuilder":
        """Rerank results to promote diversity in specified field."""
        return self.rerank("diversity", field=field, weight=weight)

    def rerank_by_model(
        self, provider: str, model: Optional[str] = None, top_k: Optional[int] = None, **config
    ) -> "QueryBuilder":
        """Rerank results using a cross-encoder or reranking model.

        Parameters
        ----------
        provider : str
            Reranker provider name (e.g., "sentence_transformers", "jina", "huggingface", "mock").
        model : str, optional
            Model name. If None, provider default is used.
        top_k : int, optional
            Maximum results to keep after reranking.
        **config
            Additional configuration passed to the reranker.
        """
        return self.rerank("cross_encoder", provider=provider, model=model, top_k=top_k, **config)

    # Debug and validation methods
    def explain(self, detailed: bool = False, return_plan: bool = False) -> Union["QueryBuilder", Dict[str, Any]]:
        """
        Enable query explanation or return execution plan directly.

        Parameters
        ----------
        detailed : bool, optional
            If True, includes additional details in explanation/plan
        return_plan : bool, optional
            If True, returns the execution plan dict instead of QueryBuilder.
            If False (default), returns QueryBuilder with explanation enabled.

        Returns
        -------
        Union[QueryBuilder, Dict[str, Any]]
            If return_plan=False: QueryBuilder with explanation enabled
            If return_plan=True: Execution plan dictionary

        Examples
        --------
        Traditional usage (returns QueryBuilder with explain enabled)::

            results = (db.query_builder()
                .search("machine learning")
                .explain(detailed=True)
                .execute())

        New usage (returns execution plan directly)::

            plan = (db.query_builder()
                .search("machine learning")
                .explain(detailed=True, return_plan=True))
            print(f"Query will execute: {plan['steps']}")
        """
        if return_plan:
            return self.get_execution_plan(detailed=detailed)

        # Traditional behavior: return QueryBuilder with explain enabled
        builder = self.clone()
        builder._explain = True
        return builder

    def validate(self) -> Dict[str, Any]:
        """Validate the current query configuration and return validation results."""
        issues = []
        warnings = []
        recommendations = []

        # Check for common issues
        if self._aggregations and not self._group_by:
            warnings.append("Aggregations without GROUP BY may not behave as expected")

        if self._having_clauses and not self._aggregations:
            issues.append("HAVING clauses require aggregations to be meaningful")

        if self._limit > 1000:
            recommendations.append("Consider using streaming for large result sets")

        if self._semantic_filters and not self._search_clauses:
            warnings.append("Semantic filters without search may be slow")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "query_complexity": self._estimate_complexity(),
        }

    def _estimate_complexity(self) -> str:
        """Estimate query complexity for optimization hints."""
        score = 0
        score += len(self._search_clauses)
        score += len(self._semantic_filters) * 2  # Semantic filters are expensive
        score += len(self._exact_filters)
        score += len(self._aggregations) * 2
        score += len(self._group_by)

        if score <= 3:
            return "low"
        elif score <= 8:
            return "medium"
        else:
            return "high"

    def debug_info(self) -> Dict[str, Any]:
        """Get detailed debugging information about the current query state."""
        return {
            "search_clauses": len(self._search_clauses),
            "exact_filters": len(self._exact_filters),
            "semantic_filters": len(self._semantic_filters),
            "search_type": self._search_type,
            "return_type": self._return_type,
            "limit": self._limit,
            "offset": self._offset,
            "order_by": self._order_by,
            "group_by": self._group_by,
            "aggregations": len(self._aggregations),
            "having_clauses": len(self._having_clauses),
            "rerank_config": self._rerank_config,
            "explain_enabled": self._explain,
        }

    # Execution methods
    def execute(
        self, *, streaming: bool = False, batch_size: int = 100
    ) -> Union[List[QueryResult], Iterator[List[QueryResult]]]:
        """
        Execute the query and return results.

        Parameters
        ----------
        streaming : bool, default = False
            If True, return an iterator that yields batches of results instead of all results at once
        batch_size : int, default = 100
            Size of each batch when streaming is enabled

        Returns
        -------
        Union[List[QueryResult], Iterator[List[QueryResult]]]
            Query results. Returns List when streaming=False, Iterator[List] when streaming=True
        """
        if streaming:
            return self.stream(batch_size)
        else:
            executor = QueryExecutor(self)
            return executor.execute()

    async def execute_async(
        self, *, streaming: bool = False, batch_size: int = 100
    ) -> Union[List[QueryResult], Iterator[List[QueryResult]]]:
        """
        Execute the query asynchronously with native async support.

        Parameters
        ----------
        streaming : bool, default = False
            If True, return an iterator that yields batches of results instead of all results at once
        batch_size : int, default = 100
            Size of each batch when streaming is enabled

        Returns
        -------
        Union[List[QueryResult], Iterator[List[QueryResult]]]
            Query results. Returns List when streaming=False, Iterator[List] when streaming=True
        """
        if streaming:
            result: Union[List[QueryResult], Iterator[List[QueryResult]]] = self.stream_async(batch_size)
            return result
        else:
            executor = AsyncQueryExecutor(self)
            return await executor.execute()

    def cursor(self, batch_size: int = 50, cursor_ttl: float = 300.0) -> "QueryCursor":
        """
        Create a QueryCursor for this query configuration.

        The cursor performs FAISS/FTS search once and lazily loads
        content/metadata from SQLite in batches as you iterate.

        Parameters
        ----------
        batch_size : int
            Default number of results per batch (default 50).
        cursor_ttl : float
            Cursor time-to-live in seconds (default 300).

        Returns
        -------
        QueryCursor
        """
        executor = QueryExecutor(self)
        return executor.cursor(batch_size=batch_size, cursor_ttl=cursor_ttl)

    async def cursor_async(self, batch_size: int = 50, cursor_ttl: float = 300.0) -> "QueryCursor":
        """
        Create a QueryCursor asynchronously.

        Parameters
        ----------
        batch_size : int
            Default number of results per batch (default 50).
        cursor_ttl : float
            Cursor time-to-live in seconds (default 300).

        Returns
        -------
        QueryCursor
        """
        executor = AsyncQueryExecutor(self)
        return await executor.cursor(batch_size=batch_size, cursor_ttl=cursor_ttl)

    def stream(self, batch_size: int = 100) -> Iterator[List[QueryResult]]:
        """Stream results in batches using a cursor (single FAISS/FTS search)."""
        c = self.cursor(batch_size=batch_size)
        with c:
            yield from c.stream(batch_size)

    async def stream_async(self, batch_size: int = 100):
        """Stream results in batches asynchronously using a cursor."""
        c = await self.cursor_async(batch_size=batch_size)
        async with c:
            async for batch in c.stream_async(batch_size):
                yield batch

    def count(self) -> int:
        """Get count of matching results without returning them."""
        executor = QueryExecutor(self)
        return executor.count()

    async def count_async(self) -> int:
        """Get count of matching results asynchronously."""
        executor = AsyncQueryExecutor(self)
        return await executor.count()

    def get_execution_plan(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get the execution plan for this query without executing it.

        This method allows you to preview how the query will be executed,
        including the steps, estimated cost, and optimizations that will be applied.

        Parameters
        ----------
        detailed : bool, optional
            If True, includes additional details like field usage and optimization hints

        Returns
        -------
        Dict[str, Any]
            Execution plan containing:
            - steps: List of execution steps
            - estimated_cost: Relative cost estimate
            - query_type: Type of query (search, filter, hybrid)
            - optimizations: List of applied optimizations
            - details: Additional details if detailed=True

        Examples
        --------
        >>> plan = (db.query_builder()
        ...     .search("machine learning")
        ...     .filter("year", gte=2020)
        ...     .get_execution_plan())
        >>> print(f"Query type: {plan['query_type']}")
        >>> print(f"Steps: {plan['steps']}")
        """
        executor = QueryExecutor(self)
        plan = executor._generate_execution_plan()

        if detailed:
            plan["details"] = {
                "search_clauses": len(self._search_clauses),
                "exact_filters": len(self._exact_filters),
                "semantic_filters": len(self._semantic_filters),
                "group_by_fields": self._group_by,
                "aggregations": [{"field": agg.field, "function": agg.function} for agg in self._aggregations],
                "order_by": self._order_by,
                "limit": self._limit,
                "offset": self._offset,
                "return_type": self._return_type,
                "vector_weight": self._vector_weight,
            }

        return plan

    async def get_execution_plan_async(self, detailed: bool = False) -> Dict[str, Any]:
        """
        Get the execution plan for this query without executing it (async version).

        Parameters
        ----------
        detailed : bool, optional
            If True, includes additional details like field usage and optimization hints

        Returns
        -------
        Dict[str, Any]
            Execution plan with same structure as get_execution_plan()
        """
        executor = AsyncQueryExecutor(self)
        plan = await executor._generate_execution_plan()

        if detailed:
            plan["details"] = {
                "search_clauses": len(self._search_clauses),
                "exact_filters": len(self._exact_filters),
                "semantic_filters": len(self._semantic_filters),
                "group_by_fields": self._group_by,
                "aggregations": [{"field": agg.field, "function": agg.function} for agg in self._aggregations],
                "order_by": self._order_by,
                "limit": self._limit,
                "offset": self._offset,
                "return_type": self._return_type,
                "vector_weight": self._vector_weight,
            }

        return plan


class _QueryExecutorBase:
    """Base class with shared methods for sync and async query executors."""

    def __init__(self, query_builder: "QueryBuilder"):
        self.builder = query_builder
        self.db = query_builder._db

    def _combine_exact_filters(self) -> Dict[str, Any]:
        """Combine all exact filters into a single filter dictionary."""
        if not self.builder._exact_filters:
            return {}

        combined: Dict[str, Any] = {}
        and_clauses: List[Dict[str, Any]] = []

        for filter_dict in self.builder._exact_filters:
            if len(filter_dict) == 1:
                key, value = next(iter(filter_dict.items()))
                if key in combined:
                    and_clauses.append({key: combined[key]})
                    and_clauses.append({key: value})
                    combined.pop(key)
                else:
                    combined[key] = value
            else:
                and_clauses.append(filter_dict)

        if and_clauses:
            if combined:
                and_clauses.append(combined)
            return {"$and": and_clauses}

        return combined

    def _build_order_by_clause(self) -> Optional[str]:
        """Build ORDER BY clause for database queries."""
        if not self.builder._order_by:
            return None

        field, direction = self.builder._order_by[0]
        return f"{field} {direction.upper()}"

    def _apply_sorting(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply sorting to results."""
        for field, direction in reversed(self.builder._order_by):
            reverse = direction.lower() == "desc"

            if field == "score":
                results.sort(key=lambda x: x.score, reverse=reverse)
            else:

                def sort_key(result: QueryResult, field: str = field, reverse: bool = reverse) -> Any:
                    value = result.metadata.get(field)
                    if value is None:
                        return float("-inf") if reverse else float("inf")
                    return value

                results.sort(key=sort_key, reverse=reverse)

        return results

    def _apply_aggregations_and_grouping(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply GROUP BY and aggregation operations to results."""
        if not (self.builder._group_by or self.builder._aggregations):
            return results

        # Group results
        if self.builder._group_by:
            grouped_results = self._group_results(results)
        else:
            grouped_results = {"_all": results}

        # Apply aggregations
        aggregated_results: List[QueryResult] = []
        for group_key, group_results in grouped_results.items():
            aggregation_data: Dict[str, Any] = {}

            for agg in self.builder._aggregations:
                agg_value = self._calculate_aggregation(group_results, agg)
                alias = agg.alias or f"{agg.function}_{agg.field}"
                aggregation_data[alias] = agg_value

            # Create result for this group
            if self.builder._group_by:
                group_metadata = dict(
                    zip(
                        self.builder._group_by, group_key if isinstance(group_key, tuple) else [group_key], strict=False
                    )
                )
                group_metadata.update(aggregation_data)

                result = QueryResult(
                    id=f"group_{hash(group_key)}",
                    score=1.0,
                    type="group",
                    content=f"Group: {group_key}",
                    metadata=group_metadata,
                )
            else:
                result = QueryResult(
                    id="aggregation_result",
                    score=1.0,
                    type="aggregation",
                    content="Aggregation Result",
                    metadata=aggregation_data,
                )

            aggregated_results.append(result)

        # Apply HAVING clauses
        if self.builder._having_clauses:
            aggregated_results = self._apply_having_clauses(aggregated_results)

        return aggregated_results

    def _group_results(self, results: List[QueryResult]) -> Dict[Any, List[QueryResult]]:
        """Group results by specified fields."""
        grouped: Dict[Any, List[QueryResult]] = defaultdict(list)

        for result in results:
            if len(self.builder._group_by) == 1:
                field = self.builder._group_by[0]
                key = result.metadata.get(field, "NULL")
            else:
                key = tuple(result.metadata.get(field, "NULL") for field in self.builder._group_by)

            grouped[key].append(result)

        return dict(grouped)

    def _calculate_aggregation(self, results: List[QueryResult], agg: AggregationClause) -> Union[int, float]:
        """Calculate aggregation value for a group of results."""
        if agg.function == "count":
            return len(results)

        # Extract values for numeric aggregations
        values: List[Union[int, float]] = []
        for result in results:
            if agg.field == "score":
                values.append(result.score)
            elif agg.field in result.metadata:
                value = result.metadata[agg.field]
                if isinstance(value, (int, float)):
                    values.append(value)

        if not values:
            return 0

        if agg.function == "sum":
            return sum(values)
        elif agg.function == "avg":
            return sum(values) / len(values)
        elif agg.function == "min":
            return min(values)
        elif agg.function == "max":
            return max(values)
        elif agg.function == "std":
            return statistics.stdev(values) if len(values) > 1 else 0
        elif agg.function == "var":
            return statistics.variance(values) if len(values) > 1 else 0

        return 0

    def _apply_having_clauses(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply HAVING clauses to aggregated results."""
        filtered_results: List[QueryResult] = []

        for result in results:
            passes_having = True

            for having_clause in self.builder._having_clauses:
                for field, condition in having_clause.items():
                    if field not in result.metadata:
                        passes_having = False
                        break

                    value = result.metadata[field]

                    if isinstance(condition, dict):
                        for operator, target_value in condition.items():
                            if not self._check_condition(value, operator, target_value):
                                passes_having = False
                                break
                    else:
                        if value != condition:
                            passes_having = False
                            break

                if not passes_having:
                    break

            if passes_having:
                filtered_results.append(result)

        return filtered_results

    def _check_condition(self, value: Any, operator: str, target: Any) -> bool:
        """Check if a value satisfies a condition."""
        if operator == "$eq":
            return bool(value == target)
        elif operator == "$ne":
            return bool(value != target)
        elif operator == "$gt":
            return bool(value > target)
        elif operator == "$gte":
            return bool(value >= target)
        elif operator == "$lt":
            return bool(value < target)
        elif operator == "$lte":
            return bool(value <= target)
        return False

    def _apply_reranking(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply reranking based on configuration."""
        if not self.builder._rerank_config:
            return results

        method = self.builder._rerank_config["method"]

        if method == "recency":
            date_field = self.builder._rerank_config["date_field"]
            weight = self.builder._rerank_config.get("weight", 1.0)

            current_time = datetime.now()
            for result in results:
                if date_field in result.metadata:
                    try:
                        date_value = result.metadata[date_field]
                        if isinstance(date_value, str):
                            date_obj = parse_iso8601(date_value)
                        elif isinstance(date_value, datetime):
                            date_obj = date_value
                        else:
                            continue

                        days_ago = (current_time - date_obj).days
                        recency_score = 1.0 / (1.0 + days_ago / 365.0)
                        result.score = (result.score * (1 - weight)) + (recency_score * weight)
                    except (ValueError, TypeError):
                        continue

            results.sort(key=lambda x: x.score, reverse=True)

        elif method == "diversity":
            field = self.builder._rerank_config["field"]
            weight = self.builder._rerank_config.get("weight", 1.0)

            seen_values: set[Any] = set()
            for result in results:
                field_value = result.metadata.get(field)
                if field_value not in seen_values:
                    result.score *= 1.0 + weight
                    seen_values.add(field_value)

            results.sort(key=lambda x: x.score, reverse=True)

        elif method == "cross_encoder":
            from localvectordb.reranking import RerankerRegistry

            config = self.builder._rerank_config
            provider: str = config.get("provider", "")
            model = config.get("model")
            top_k = config.get("top_k")

            reranker_kwargs = {
                k: v for k, v in config.items() if k not in ("method", "provider", "model", "top_k") and v is not None
            }

            reranker = RerankerRegistry.create_reranker(provider, model, **reranker_kwargs)

            # Extract query text from search clauses
            query_text = ""
            if self.builder._search_clauses:
                query_text = self.builder._search_clauses[0].query

            results = reranker.rerank(query_text, results, top_k=top_k)

        return results


class QueryExecutor(_QueryExecutorBase):
    """Synchronous executor for QueryBuilder queries."""

    def execute(self) -> List[QueryResult]:
        """Execute the query synchronously with full feature support."""
        start_time = time.time()

        try:
            if self.builder._explain:
                execution_plan = self._generate_execution_plan()

            # Execute base query
            if self.builder._search_clauses:
                results = self._execute_search_query()
            else:
                results = self._execute_filter_only_query()

            # Apply post-processing
            results = self._apply_post_processing(results)

            # Apply aggregations and grouping if specified
            if self.builder._aggregations or self.builder._group_by:
                results = self._apply_aggregations_and_grouping(results)

            # Apply reranking if configured
            if self.builder._rerank_config:
                results = self._apply_reranking(results)

            execution_time = time.time() - start_time
            logger.debug(f"Query executed in {execution_time:.3f}s, returned {len(results)} results")

            # Add execution info if explain is enabled
            if self.builder._explain:
                for result in results:
                    result.metadata["_execution_plan"] = execution_plan
                    result.metadata["_execution_time"] = execution_time

            return results

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def _execute_search_query(self) -> List[QueryResult]:
        """Execute search-based query."""
        clause = self.builder._search_clauses[0]  # Use first search clause
        filters = self._combine_exact_filters()

        results = self.db.query(
            query=clause.query,
            search_type=clause.search_type,
            return_type=self.builder._return_type,
            search_level=self.builder._search_level,
            k=self.builder._limit + self.builder._offset,
            score_threshold=clause.score_threshold or 0.0,
            filters=filters,
            vector_weight=self.builder._vector_weight,
            context_window=self.builder._context_window,
            semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
            document_scoring_method=self.builder._document_scoring_method,
            document_scoring_options=self.builder._document_scoring_options,
        )

        if self.builder._semantic_filters:
            results = self._apply_semantic_filters(results)

        return results

    def _execute_filter_only_query(self) -> List[QueryResult]:
        """Execute a filter-only query."""
        filters = self._combine_exact_filters()

        documents = self.db.filter(
            where=filters,
            limit=self.builder._limit + self.builder._offset,
            offset=0,
            order_by=self._build_order_by_clause(),
        )

        results = [
            QueryResult(id=doc.id, score=1.0, type="document", content=doc.content, metadata=doc.metadata)
            for doc in documents
        ]

        if self.builder._semantic_filters:
            results = self._apply_semantic_filters(results)

        return results

    def _apply_semantic_filters(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply semantic filtering to results."""
        if not self.builder._semantic_filters:
            return results

        documents = []
        for result in results:
            doc = Document(id=result.id, content=result.content, metadata=result.metadata.copy())
            doc.metadata["_original_score"] = result.score
            documents.append(doc)

        for semantic_filter in self.builder._semantic_filters:
            documents = semantic_filter.apply(documents, self.db)

        filtered_results = []
        for doc in documents:
            original_score = doc.metadata.pop("_original_score", 1.0)
            result = QueryResult(
                id=doc.id, score=original_score, type="document", content=doc.content, metadata=doc.metadata
            )
            filtered_results.append(result)

        return filtered_results

    def _apply_post_processing(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply post-processing steps."""
        if self.builder._order_by and not self.builder._aggregations:
            results = self._apply_sorting(results)

        if self.builder._offset > 0:
            results = results[self.builder._offset :]

        if len(results) > self.builder._limit:
            results = results[: self.builder._limit]

        return results

    def count(self) -> int:
        """Get count of matching results."""
        if self.builder._search_clauses:
            results = self._execute_search_query()
            return len(results)
        else:
            # Use database count if available, otherwise execute filter
            if hasattr(self.db, "count"):
                filters = self._combine_exact_filters()
                return self.db.count(filters=filters)
            else:
                results = self._execute_filter_only_query()
                return len(results)

    def cursor(self, batch_size: int = 50, cursor_ttl: float = 300.0) -> "QueryCursor":
        """Create a QueryCursor by delegating to db.query_cursor()."""
        if not self.builder._search_clauses:
            raise ValueError("cursor() requires a search clause; use .search() first")

        clause = self.builder._search_clauses[0]
        filters = self._combine_exact_filters()

        return self.db.query_cursor(
            query=clause.query,
            search_type=clause.search_type,
            return_type=self.builder._return_type,
            search_level=self.builder._search_level,
            k=self.builder._limit + self.builder._offset,
            score_threshold=clause.score_threshold or 0.0,
            filters=filters,
            vector_weight=self.builder._vector_weight,
            context_window=self.builder._context_window,
            semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
            document_scoring_method=self.builder._document_scoring_method,
            document_scoring_options=self.builder._document_scoring_options,
            batch_size=batch_size,
            cursor_ttl=cursor_ttl,
        )

    def stream(self, batch_size: int = 100) -> Iterator[List[QueryResult]]:
        """Stream results in batches using a cursor (single FAISS/FTS search)."""
        c = self.cursor(batch_size=batch_size)
        with c:
            yield from c.stream(batch_size)

    def _generate_execution_plan(self) -> Dict[str, Any]:
        """Generate execution plan for explain functionality."""
        steps: List[str] = []
        estimated_cost: int = 0
        query_type: str = (
            "hybrid"
            if self.builder._search_clauses and self.builder._exact_filters
            else "search" if self.builder._search_clauses else "filter"
        )
        optimizations: List[str] = []

        if self.builder._search_clauses:
            steps.append("vector_search")
            estimated_cost += len(self.builder._search_clauses) * 10

        if self.builder._exact_filters:
            steps.append("exact_filtering")
            estimated_cost += len(self.builder._exact_filters) * 2

        if self.builder._semantic_filters:
            steps.append("semantic_filtering")
            estimated_cost += len(self.builder._semantic_filters) * 50

        if self.builder._group_by:
            steps.append("grouping")
            estimated_cost += 20

        if self.builder._aggregations:
            steps.append("aggregation")
            estimated_cost += len(self.builder._aggregations) * 5

        if self.builder._having_clauses:
            steps.append("having_filter")
            estimated_cost += len(self.builder._having_clauses) * 3

        if self.builder._order_by:
            steps.append("sorting")
            estimated_cost += 10

        if self.builder._rerank_config:
            steps.append("reranking")
            estimated_cost += 30

        return {
            "steps": steps,
            "estimated_cost": estimated_cost,
            "query_type": query_type,
            "optimizations": optimizations,
        }


class AsyncQueryExecutor(_QueryExecutorBase):
    """Asynchronous executor for QueryBuilder queries with native database async support."""

    async def _generate_execution_plan(self) -> Dict[str, Any]:
        """Generate execution plan for explain functionality."""
        steps: List[str] = []
        estimated_cost: int = 0
        query_type: str = (
            "hybrid"
            if self.builder._search_clauses and self.builder._exact_filters
            else "search" if self.builder._search_clauses else "filter"
        )
        optimizations: List[str] = []

        if self.builder._search_clauses:
            steps.append("vector_search")
            estimated_cost += len(self.builder._search_clauses) * 10

        if self.builder._exact_filters:
            steps.append("exact_filtering")
            estimated_cost += len(self.builder._exact_filters) * 2

        if self.builder._semantic_filters:
            steps.append("semantic_filtering")
            estimated_cost += len(self.builder._semantic_filters) * 50

        if self.builder._group_by:
            steps.append("grouping")
            estimated_cost += 20

        if self.builder._aggregations:
            steps.append("aggregation")
            estimated_cost += len(self.builder._aggregations) * 5

        if self.builder._having_clauses:
            steps.append("having_filter")
            estimated_cost += len(self.builder._having_clauses) * 3

        if self.builder._order_by:
            steps.append("sorting")
            estimated_cost += 10

        if self.builder._rerank_config:
            steps.append("reranking")
            estimated_cost += 30

        return {
            "steps": steps,
            "estimated_cost": estimated_cost,
            "query_type": query_type,
            "optimizations": optimizations,
        }

    async def execute(self) -> List[QueryResult]:
        """Execute the query asynchronously with full feature support."""
        start_time = time.time()

        try:
            execution_plan = None
            if self.builder._explain:
                execution_plan = await self._generate_execution_plan()

            # Execute base query
            if self.builder._search_clauses:
                results = await self._execute_search_query_async()
            else:
                results = await self._execute_filter_only_query_async()

            # Apply post-processing
            results = await self._apply_post_processing_async(results)

            # Apply aggregations and grouping if specified
            if self.builder._aggregations or self.builder._group_by:
                # These are CPU-bound operations, can use sync version
                results = self._apply_aggregations_and_grouping(results)

            # Apply reranking if configured
            if self.builder._rerank_config:
                results = self._apply_reranking(results)

            execution_time = time.time() - start_time
            logger.debug(f"Async query executed in {execution_time:.3f}s, returned {len(results)} results")

            # Add execution info if explain is enabled
            if self.builder._explain:
                for result in results:
                    result.metadata["_execution_time"] = execution_time
                    result.metadata["_execution_plan"] = execution_plan

            return results

        except Exception as e:
            logger.error(f"Async query execution failed: {e}")
            raise

    async def _execute_search_query_async(self) -> List[QueryResult]:
        """Execute search-based query asynchronously."""
        clause = self.builder._search_clauses[0]
        filters = self._combine_exact_filters()

        results = await self.db.query_async(
            query=clause.query,
            search_type=clause.search_type,
            return_type=self.builder._return_type,
            search_level=self.builder._search_level,
            k=self.builder._limit + self.builder._offset,
            score_threshold=clause.score_threshold or 0.0,
            filters=filters,
            vector_weight=self.builder._vector_weight,
            context_window=self.builder._context_window,
            semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
            document_scoring_method=self.builder._document_scoring_method,
            document_scoring_options=self.builder._document_scoring_options,
        )

        if self.builder._semantic_filters:
            results = await self._apply_semantic_filters_async(results)

        return results

    async def _execute_filter_only_query_async(self) -> List[QueryResult]:
        """Execute filter-only query asynchronously."""
        filters = self._combine_exact_filters()

        documents = await self.db.filter_async(
            where=filters,
            limit=self.builder._limit + self.builder._offset,
            offset=0,
            order_by=self._build_order_by_clause(),
        )
        results = [
            QueryResult(id=doc.id, score=1.0, type="document", content=doc.content, metadata=doc.metadata)
            for doc in documents
        ]

        if self.builder._semantic_filters:
            results = await self._apply_semantic_filters_async(results)

        return results

    async def _apply_semantic_filters_async(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply semantic filtering to results asynchronously."""
        if not self.builder._semantic_filters:
            return results

        documents = []
        for result in results:
            doc = Document(id=result.id, content=result.content, metadata=result.metadata.copy())
            doc.metadata["_original_score"] = result.score
            documents.append(doc)

        for semantic_filter in self.builder._semantic_filters:
            documents = await semantic_filter.apply_async(documents, self.db)

        filtered_results = []
        for doc in documents:
            original_score = doc.metadata.pop("_original_score", 1.0)
            result = QueryResult(
                id=doc.id, score=original_score, type="document", content=doc.content, metadata=doc.metadata
            )
            filtered_results.append(result)

        return filtered_results

    async def _apply_post_processing_async(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply post-processing steps asynchronously."""
        # These operations are CPU-bound, so we can use the sync version
        if self.builder._order_by and not self.builder._aggregations:
            results = self._apply_sorting(results)

        if self.builder._offset > 0:
            results = results[self.builder._offset :]

        if len(results) > self.builder._limit:
            results = results[: self.builder._limit]

        return results

    async def count(self) -> int:
        """Get count of matching results asynchronously."""
        # Use database's async count method if available

        if self.builder._search_clauses:
            # For search queries, we need to execute and count
            results = await self._execute_search_query_async()
            return len(results)
        else:
            filters = self._combine_exact_filters()
            return await self.db.count_async(filters)

    async def cursor(self, batch_size: int = 50, cursor_ttl: float = 300.0) -> "QueryCursor":
        """Create a QueryCursor asynchronously by delegating to db.query_cursor_async()."""
        if not self.builder._search_clauses:
            raise ValueError("cursor() requires a search clause; use .search() first")

        clause = self.builder._search_clauses[0]
        filters = self._combine_exact_filters()

        return await self.db.query_cursor_async(
            query=clause.query,
            search_type=clause.search_type,
            return_type=self.builder._return_type,
            search_level=self.builder._search_level,
            k=self.builder._limit + self.builder._offset,
            score_threshold=clause.score_threshold or 0.0,
            filters=filters,
            vector_weight=self.builder._vector_weight,
            context_window=self.builder._context_window,
            semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
            document_scoring_method=self.builder._document_scoring_method,
            document_scoring_options=self.builder._document_scoring_options,
            batch_size=batch_size,
            cursor_ttl=cursor_ttl,
        )

    async def stream(self, batch_size: int = 100):
        """Stream results in batches asynchronously using a cursor."""
        c = await self.cursor(batch_size=batch_size)
        async with c:
            async for batch in c.stream_async(batch_size):
                yield batch

    # Shared methods inherited from _QueryExecutorBase
