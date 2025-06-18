# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb/query_builder.py
"""
LocalVectorDB Query Builder - SQL-like interface for vector database queries.

This module provides a sophisticated, fluent interface for building complex queries against
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

Classes
-------
QueryBuilder : Main query builder with fluent interface
SearchClause : Represents a search operation on a field
SemanticFilter : Semantic filtering based on conceptual similarity
AggregationClause : Represents aggregation operations
QueryExecutor : Synchronous query execution engine
AsyncQueryExecutor : Asynchronous query execution engine
SimilarityMetric : Enum for semantic similarity metrics

Examples
--------

**Basic Vector Search**::

    # Simple semantic search
    results = (db.query_builder()
        .search("machine learning algorithms")
        .limit(10)
        .execute())

    # Vector search with score threshold
    results = (db.query_builder()
        .vector("neural networks", score_threshold=0.8)
        .limit(5)
        .execute())

**Hybrid and Keyword Search**::

    # Balanced hybrid search (default 70% vector, 30% keyword)
    results = (db.query_builder()
        .hybrid("artificial intelligence", vector_weight=0.6)
        .limit(20)
        .execute())

    # Pure keyword search with FTS
    results = (db.query_builder()
        .keyword("machine learning")
        .filter("category", "research")
        .execute())

**Advanced Filtering and Search Combinations**::

    # Complex multi-field query with exact and semantic filters
    results = (db.query_builder()
        .search("deep learning frameworks")
        .filter("year", gte_=2020, lt_=2024)          # Year range
        .filter("category", "AI")                     # Exact match
        .filter("tags", contains_="pytorch")          # Tag filtering
        .semantic_filter("methodology", "supervised learning", threshold=0.75)
        .order_by("year", "desc")
        .limit(50)
        .execute())

    # Multi-term search with weights
    results = (db.query_builder()
        .search("neural networks", weight=0.7)
        .search("computer vision", weight=0.3)
        .filter("published", exists_=True)
        .semantic_dedup(threshold=0.9)               # Remove near-duplicates
        .execute())

**Document Analysis and Aggregation**::

    # Research paper analysis by category and year
    stats = (db.query_builder()
        .search("machine learning")
        .filter("type", "research_paper")
        .group_by("category", "year")
        .count_by("*", "paper_count")
        .avg_by("citation_count", "avg_citations")
        .having_count("gt", 5)                       # Groups with >5 papers
        .order_by("avg_citations", "desc")
        .execute())

    # Author productivity analysis
    top_authors = (db.query_builder()
        .filter("year", gte_=2020)
        .group_by("author")
        .count_by("*", "publication_count")
        .sum_by("citation_count", "total_citations")
        .having("publication_count", "gte", 10)
        .order_by("total_citations", "desc")
        .limit(25)
        .execute())

**Advanced Result Processing**::

    # Diverse results with recency boost and context
    results = (db.query_builder()
        .search("artificial intelligence trends")
        .filter("status", "published")
        .semantic_filter("topic", "emerging technology", threshold=0.7)
        .context(window_size=3)                      # Include surrounding chunks
        .rerank_by_recency("published_date", weight=0.4)
        .rerank_by_diversity("source", weight=0.2)
        .semantic_dedup(0.85)
        .order_by_score()
        .limit(30)
        .execute())

    # Return individual chunks instead of full documents
    chunks = (db.query_builder()
        .search("implementation details")
        .filter("document_type", "tutorial")
        .chunks()                                    # Return chunks, not documents
        .order_by("score", "desc")
        .limit(100)
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

    # Async streaming for large result sets
    async def process_large_dataset():
        async for batch in (db.query_builder()
            .search("climate change")
            .filter("source", "scientific_journals")
            .stream_async(batch_size=50)):

            # Process each batch
            await process_batch(batch)

**Performance and Debugging**::

    # Query with performance explanation
    results = (db.query_builder()
        .search("machine learning optimization")
        .filter("complexity", lt_=100)
        .explain(detailed=True)                      # Include execution stats
        .execute())

    # Validation before execution
    builder = (db.query_builder()
        .search("data science")
        .group_by("department")
        .avg_by("performance_score"))

    validation = builder.validate()
    if validation["valid"]:
        results = builder.execute()
    else:
        print("Query issues:", validation["issues"])

    # Debug query structure
    debug_info = builder.debug_info()
    print("SQL Preview:", debug_info["sql_preview"])
    print("Complexity:", debug_info["query_complexity"])

**Specialized Search Patterns**::

    # Research literature review workflow
    papers = (db.query_builder()
        .search("transformer architecture")
        .filter("venue", in_=["ICML", "NeurIPS", "ICLR"])
        .filter("year", gte_=2017)
        .semantic_filter("contribution", "attention mechanism", threshold=0.8)
        .rerank_by_recency("published_date", weight=0.3)
        .semantic_dedup(0.9)
        .documents(scoring_method="frequency_boost")
        .order_by("citation_count", "desc")
        .limit(100)
        .execute())

    # Content recommendation system
    recommendations = (db.query_builder()
        .semantic_filter("content", user_interests, threshold=0.7)
        .filter("age_rating", lte_=user_age)
        .filter("language", user_language)
        .rerank_by_diversity("genre", weight=0.4)
        .order_by("popularity_score", "desc")
        .limit(20)
        .execute())

    # Duplicate detection and analysis
    potential_duplicates = (db.query_builder()
        .filter("status", "new")
        .semantic_dedup(threshold=0.95)
        .group_by("semantic_cluster")               # Hypothetical clustering
        .count_by("*", "cluster_size")
        .having_count("gt", 1)                      # Multiple items per cluster
        .order_by("cluster_size", "desc")
        .execute())

Notes
-----
- The QueryBuilder uses a fluent interface where each method returns a new builder instance
- Semantic filtering requires embedding generation and can be computationally expensive
- Async execution is recommended for I/O bound operations and large datasets
- Query validation helps catch configuration errors before expensive operations
- The explain() method provides insights into query execution and performance

See Also
--------
localvectordb.core : Core database classes and data structures
localvectordb._filters : Filter operators and implementations
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Iterator,
    Literal, Union
)

import numpy as np

from localvectordb.core import QueryResult, Document, AnyVectorDB

logger = logging.getLogger(__name__)

DocumentScoringMethod = Literal["best", "worst", "average", "weighted_average", "frequency_boost"]
"""
Methods for aggregating chunk scores into document scores.

- best: Use the highest chunk score
- worst: Use the lowest chunk score
- average: Use simple average of all chunk scores
- weighted_average: Use weighted average based on position and relevance
- frequency_boost: Boost by frequency of matching chunks (default)
"""

class SimilarityMetric(Enum):
    """Supported similarity metrics for semantic filtering."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


@dataclass
class SearchClause:
    """Represents a search operation on a specific field."""
    field: str  # The only possible field right now is 'content'
    query: str
    weight: float = 1.0
    search_type: Literal["vector", "keyword", "hybrid"] = "vector"
    score_threshold: float = None


@dataclass
class SemanticFilter:
    """Semantic filtering based on conceptual similarity with async support."""
    field: str
    concept: str
    threshold: float
    metric: SimilarityMetric = SimilarityMetric.COSINE
    embedding_model: Optional[str] = None

    async def apply_async(self, documents: List[Document], db: AnyVectorDB) -> List[Document]:
        """
        Apply semantic filtering with async embedding generation.

        Parameters
        ----------
        documents : List[Document]
            Documents to filter
        db : AnyVectorDB
            Vector database for embedding generation

        Returns
        -------
        List[Document]
            Documents that meet the similarity threshold
        """
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
            # Try multiple async methods in order of preference
            embedding_provider = db.embedding_provider

            # Method 1: Database's own async embedding method
            if hasattr(db, '_generate_embeddings_async'):
                concept_embedding = (await db._generate_embeddings_async([self.concept]))[0]
                field_embeddings = await db._generate_embeddings_async(field_contents)

            # Method 2: Provider's embed_batch method (if async)
            elif (hasattr(embedding_provider, 'embed_batch')
                  and asyncio.iscoroutinefunction(embedding_provider.embed_batch)):
                concept_embedding = (await embedding_provider.embed_batch([self.concept]))[0]
                field_embeddings = await embedding_provider.embed_batch(field_contents)

            # Method 3: Provider's embed_async method
            elif hasattr(embedding_provider, 'embed_async'):
                concept_embedding = (await embedding_provider.embed_async([self.concept]))[0]
                field_embeddings = await embedding_provider.embed_async(field_contents)

            # Method 4: Fall back to sync in thread pool
            else:
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    concept_embedding = await loop.run_in_executor(
                        executor, lambda: embedding_provider.embed_sync([self.concept])[0]
                    )
                    field_embeddings = await loop.run_in_executor(
                        executor, lambda: embedding_provider.embed_sync(field_contents)
                    )

        except Exception as e:
            logger.error(f"Async embedding generation failed: {e}")
            # Fall back to sync version
            return self.apply(documents, db)

        # Apply similarity filtering
        filtered_docs = []
        for doc, field_embedding in zip(valid_docs, field_embeddings):
            similarity = self._calculate_similarity(concept_embedding, field_embedding)

            if similarity >= self.threshold:
                # Add semantic score to metadata for transparency
                doc.metadata["_semantic_scores"] = doc.metadata.get("_semantic_scores", {})
                doc.metadata["_semantic_scores"][f"{self.field}_{self.concept}"] = float(similarity)
                filtered_docs.append(doc)

        return filtered_docs

    def apply(self, documents: List[Document], db: AnyVectorDB) -> List[Document]:
        """
        Apply semantic filtering synchronously.

        Parameters
        ----------
        documents : List[Document]
            Documents to filter
        db : AnyVectorDB
            Vector database for embedding generation

        Returns
        -------
        List[Document]
            Documents that meet the similarity threshold
        """
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

        # Use embedding provider directly for sync version too
        concept_embedding = db.embedding_provider.embed_sync([self.concept])[0]
        field_embeddings = db.embedding_provider.embed_sync(field_contents)

        # Apply similarity filtering
        filtered_docs = []
        for doc, field_embedding in zip(valid_docs, field_embeddings):
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
            return float(np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)))
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

from localvectordb._filters import FILTER_OPERATORS
# Removes the preceding '$', used in QueryBuilder.filter
FILTER_OPERATOR_NAMES = tuple(map(lambda x: x[1:], FILTER_OPERATORS))

class QueryBuilder:
    """
    Fluent interface for building complex vector database queries with async support.

    This class provides a SQL-like interface for building sophisticated search
    and filter operations against vector databases, with support for:

    - Vector, keyword, and hybrid search
    - Exact and semantic filters
    - Grouping and aggregations
    - Sorting and pagination
    - Result reranking
    - Execution plan generation

    Parameters
    ----------
    db : AnyVectorDB
        Database instance to query
    """

    def __init__(self, db: AnyVectorDB):
        self._db = db
        self._search_clauses: List[SearchClause] = []
        self._exact_filters: List[Dict[str, Any]] = []
        self._semantic_filters: List[SemanticFilter] = []
        self._search_type: Literal["vector", "keyword", "hybrid"] = "hybrid"
        self._vector_weight: float = 0.7
        self._return_type: Literal["documents", "chunks", "context"] = "documents"
        self._limit: int = 10
        self._offset: int = 0
        self._order_by: List[tuple[str, str]] = []  # (field, direction)
        self._group_by: List[str] = []
        self._aggregations: List[AggregationClause] = []
        self._having_clauses: List[Dict[str, Any]] = []
        self._rerank_config: Optional[Dict[str, Any]] = None
        self._explain: bool = False
        self._hints: Dict[str, Any] = {}
        self._context_window: int = 2
        self._semantic_dedup_threshold: Optional[float] = None
        self._document_scoring_method: DocumentScoringMethod = "frequency_boost"

        # Performance optimization flags
        self._batch_size: int = 100


    def clone(self) -> "QueryBuilder":
        """
        Create a copy of this QueryBuilder for chaining.

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with same configuration
        """
        new_builder = QueryBuilder(self._db)
        new_builder._search_clauses = self._search_clauses.copy()
        new_builder._exact_filters = self._exact_filters.copy()
        new_builder._semantic_filters = self._semantic_filters.copy()
        new_builder._search_type = self._search_type
        new_builder._vector_weight = self._vector_weight
        new_builder._return_type = self._return_type
        new_builder._limit = self._limit
        new_builder._offset = self._offset
        new_builder._order_by = self._order_by.copy()
        new_builder._group_by = self._group_by.copy()
        new_builder._aggregations = self._aggregations.copy()
        new_builder._having_clauses = self._having_clauses.copy()
        new_builder._rerank_config = self._rerank_config
        new_builder._explain = self._explain
        new_builder._hints = self._hints.copy()
        new_builder._batch_size = self._batch_size
        # new_builder._use_cache = self._use_cache
        # new_builder._parallel_semantic_filtering = self._parallel_semantic_filtering
        new_builder._context_window = self._context_window
        new_builder._semantic_dedup_threshold = self._semantic_dedup_threshold
        new_builder._document_scoring_method = self._document_scoring_method

        return new_builder

    # Core search methods
    def search(self, query: str, search_type=None, vector_weight=None, score_threshold=None) -> "QueryBuilder":
        """
        Search the content for `query`

        Parameters
        ----------
        query : str
            Query text
        search_type : str, optional, one of "vector", "hybrid", "keyword"
            Type of search to conduct.
        vector_weight : float, optional, between 0.0-1.0 (inclusive)
            Ignored unless `search_type="hybrid"`, controls balance of vector vs. keyword search on rank.
        score_threshold : float, optional, between 0.0-1.0
            Optionally filter results with similarity score lower than `score_threshold`

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with search clause added

        Raises
        ------
        ValueError
            If query or field is empty or not a string
        """
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
        """Find records where `field` contains `query`"""

        if not field or not isinstance(field, str):
            raise ValueError("Field must be a non-empty string")

        builder = self.clone()
        # Do ilike for str, exact match for others.
        if isinstance(query, str):
            filter = {field: {"$ilike": query}}
        else:
            filter = {field: query}
        builder._exact_filters.append(filter)

        return builder

    def filter(self, field: str = None, value=None, **kwargs) -> "QueryBuilder":
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
        if field is not None:
            if not isinstance(field, str) or not field:
                raise ValueError("Field must be a non-empty string")

        # Validate operator suffixes in kwargs
        for key in kwargs:
            if key.endswith('_') and key[:-1] not in FILTER_OPERATOR_NAMES:
                raise ValueError(f"Invalid operator suffix in '{key}'")

        builder = self.clone()

        if field is not None and value is not None:
            builder._exact_filters.append({field: value})

        for key, val in kwargs.items():
            if key.endswith('_'):
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
            self,
            field: str,
            concept: str,
            threshold: float = 0.8,
            metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> "QueryBuilder":
        """
        Add semantic filtering based on conceptual similarity.

        This allows you to further refine a set of documents after an initial search by
        using semantic searching across any fields. Calculates embeddings of field data and
        compares to `concept`

        Parameters
        ----------
        field : str
            Field to apply semantic filtering to
        concept : str
            Concept to measure similarity against
        threshold : float, default 0.8
            Minimum similarity threshold (0-1, higher = more similar)
        metric : SimilarityMetric, default SimilarityMetric.COSINE
            Similarity metric to use

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with semantic filter added

        Raises
        ------
        ValueError
            If field or concept is empty, or threshold is not in [0,1]
        """
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

    def vector(self, query, score_threshold=None) -> "QueryBuilder":
        """Use vector search."""
        return self.search(query, search_type="vector", score_threshold=score_threshold)

    def keyword(self, query, score_threshold=None) -> "QueryBuilder":
        """Use keyword search."""
        return self.search(query, search_type="keyword", score_threshold=score_threshold)

    def hybrid(self, query, vector_weight: float = 0.7, score_threshold=None) -> "QueryBuilder":
        """Use hybrid search with specified vector weight."""
        return self.search(query, search_type="hybrid", vector_weight=vector_weight, score_threshold=score_threshold)

    def semantic_dedup(self, threshold: float) -> "QueryBuilder":
        """Enable semantic deduplication with similarity threshold."""
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Semantic deduplication threshold must be between 0.0 and 1.0")
        builder = self.clone()
        builder._semantic_dedup_threshold = threshold
        return builder

    def documents(self, scoring_method: DocumentScoringMethod = "frequency_boost") -> "QueryBuilder":
        """Return full documents in results (default)."""
        builder = self.clone()
        builder._return_type = "documents"
        builder._document_scoring_method = scoring_method
        return builder

    def chunks(self) -> "QueryBuilder":
        """Return individual chunks in results with position information."""
        builder = self.clone()
        builder._return_type = "chunks"
        return builder

    def return_type(self, return_type: Literal["documents", "chunks", "context"]) -> "QueryBuilder":
        """Set the return type explicitly."""
        if return_type not in ["documents", "chunks", "context"]:
            raise ValueError("`return_type` must be 'documents' or 'chunks'")
        builder = self.clone()
        builder._return_type = return_type
        return builder

    def context(self, window_size: int = 2) -> "QueryBuilder":
        """Return chunks with context window."""
        builder = self.clone()
        builder._return_type = "context"
        builder._context_window = window_size
        return builder


    def order_by(self, field: str, direction: str = "desc") -> "QueryBuilder":
        """
        Add ordering by specified field.

        Parameters
        ----------
        field : str
            Field name to order by (supports 'score' and metadata fields)
        direction : str, default "desc"
            Sort direction: "desc" or "asc"

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with ordering applied

        Examples
        --------
        Order by relevance score (default)::

            results = db.query_builder().search("AI").order_by("score").execute()

        Order by date field::

            results = db.query_builder().search("news").order_by("published_date", "desc").execute()

        Multiple ordering (last added has priority)::

            results = (db.query_builder()
                        .search("research")
                        .order_by("year", "desc")
                        .order_by("score", "desc")
                        .execute())
        """
        if direction.lower() not in ["asc", "desc"]:
            raise ValueError("`direction` must be 'asc' or 'desc'")
        if not field:
            raise ValueError("`field` must be a non-empty string")

        builder = self.clone()
        builder._order_by.append((field, direction.lower()))
        return builder

    def order_by_score(self, direction: str = "desc") -> "QueryBuilder":
        """
        Order results by relevance score.

        Parameters
        ----------
        direction : str, default "desc"
            Sort direction: "desc" (best first) or "asc" (worst first)

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with score ordering applied
        """
        return self.order_by("score", direction)

    def clear_ordering(self) -> "QueryBuilder":
        """
        Remove all ordering clauses.

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with no ordering
        """
        builder = self.clone()
        builder._order_by.clear()
        return builder

    def group_by(self, *fields: str) -> "QueryBuilder":
        """
        Group results by one or more fields.

        Parameters
        ----------
        *fields : str
            Field names to group by

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with grouping applied

        Examples
        --------
        Group by single field::

            results = db.query_builder().search("AI").group_by("category").execute()

        Group by multiple fields::

            results = db.query_builder().search("research").group_by("year", "category").execute()
        """
        if not fields:
            raise ValueError("At least one field must be specified for grouping")

        for field in fields:
            if not isinstance(field, str) or not field.strip():
                raise ValueError("All group_by fields must be non-empty strings")

        builder = self.clone()
        builder._group_by.extend(fields)
        return builder

    def clear_grouping(self) -> "QueryBuilder":
        """
        Remove all grouping clauses.

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with no grouping
        """
        builder = self.clone()
        builder._group_by.clear()
        return builder

    def aggregate(self, field: str, function: str, alias: Optional[str] = None) -> "QueryBuilder":
        """
        Add an aggregation function.

        Parameters
        ----------
        field : str
            Field name to aggregate
        function : str
            Aggregation function: "count", "sum", "avg", "min", "max", "std", "var"
        alias : str, optional
            Alias for the aggregation result

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with aggregation applied

        Examples
        --------
        Count documents by category::

            results = (db.query_builder()
                        .search("AI")
                        .group_by("category")
                        .aggregate("*", "count", "doc_count")
                        .execute())
        """
        valid_functions = ["count", "sum", "avg", "min", "max", "std", "var"]
        if function not in valid_functions:
            raise ValueError(f"function must be one of: {', '.join(valid_functions)}")

        builder = self.clone()
        aggregation = AggregationClause(field=field, function=function, alias=alias)
        builder._aggregations.append(aggregation)
        return builder

    def count_by(self, field: str = "*", alias: Optional[str] = None) -> "QueryBuilder":
        """
        Count documents/chunks, optionally grouped by field.

        Parameters
        ----------
        field : str, default "*"
            Field to count by ("*" for total count)
        alias : str, optional
            Alias for the count result

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with count aggregation
        """
        return self.aggregate(field, "count", alias or "count")

    def sum_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """
        Sum numeric values in a field.

        Parameters
        ----------
        field : str
            Field name containing numeric values
        alias : str, optional
            Alias for the sum result

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with sum aggregation
        """
        return self.aggregate(field, "sum", alias or f"sum_{field}")

    def avg_by(self, field: str, alias: Optional[str] = None) -> "QueryBuilder":
        """
        Calculate average of numeric values in a field.

        Parameters
        ----------
        field : str
            Field name containing numeric values
        alias : str, optional
            Alias for the average result

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with average aggregation
        """
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
        """
        Add HAVING clause for post-aggregation filtering.

        Parameters
        ----------
        field : str
            Aggregated field name or alias
        operator : str
            Comparison operator: "eq", "ne", "gt", "gte", "lt", "lte"
        value : Any
            Value to compare against

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with HAVING clause

        Examples
        --------
        Filter groups with count > 5::

            results = (db.query_builder()
                        .search("AI")
                        .group_by("category")
                        .count_by("*", "doc_count")
                        .having("doc_count", "gt", 5)
                        .execute())
        """
        valid_operators = ["eq", "ne", "gt", "gte", "lt", "lte"]
        if operator not in valid_operators:
            raise ValueError(f"operator must be one of: {', '.join(valid_operators)}")

        builder = self.clone()
        having_clause = {field: {f"${operator}": value}}
        builder._having_clauses.append(having_clause)
        return builder

    def having_count(self, operator: str, value: int, alias: str = "count") -> "QueryBuilder":
        """
        Add HAVING clause for count aggregations.

        Parameters
        ----------
        operator : str
            Comparison operator: "eq", "ne", "gt", "gte", "lt", "lte"
        value : int
            Count value to compare against
        alias : str, default "count"
            Alias used for the count aggregation

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with count HAVING clause
        """
        return self.having(alias, operator, value)

    def explain(self, detailed: bool = False) -> "QueryBuilder":
        """
        Enable query explanation to understand execution plan and performance.

        Parameters
        ----------
        detailed : bool, default False
            Include detailed execution statistics

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with explanation enabled

        Notes
        -----
        When explain is enabled, query results will include additional metadata
        about execution time, steps taken, and optimization decisions.
        """
        builder = self.clone()
        builder._explain = True
        builder._hints["detailed_explain"] = detailed
        return builder

    def rerank(self, method: str, **config) -> "QueryBuilder":
        """
        Add reranking configuration for result post-processing.

        Parameters
        ----------
        method : str
            Reranking method: "relevance", "recency", "diversity", "custom"
        **config
            Method-specific configuration parameters

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with reranking configured

        Examples
        --------
        Rerank by recency::

            results = (db.query_builder()
                        .search("news")
                        .rerank("recency", date_field="published_date", weight=0.3)
                        .execute())
        """
        valid_methods = ["relevance", "recency", "diversity", "custom"]
        if method not in valid_methods:
            raise ValueError(f"rerank method must be one of: {', '.join(valid_methods)}")

        builder = self.clone()
        builder._rerank_config = {"method": method, **config}
        return builder

    def rerank_by_relevance(self, weight: float = 1.0) -> "QueryBuilder":
        """
        Rerank results by relevance score.

        Parameters
        ----------
        weight : float, default 1.0
            Weight factor for relevance in reranking

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with relevance reranking
        """
        return self.rerank("relevance", weight=weight)

    def rerank_by_recency(self, date_field: str = "updated_at", weight: float = 1.0) -> "QueryBuilder":
        """
        Rerank results by recency (newer documents ranked higher).

        Parameters
        ----------
        date_field : str
            Metadata field containing date information
        weight : float, default 1.0
            Weight factor for recency in reranking

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with recency reranking
        """
        return self.rerank("recency", date_field=date_field, weight=weight)

    def rerank_by_diversity(self, field: str, weight: float = 1.0) -> "QueryBuilder":
        """
        Rerank results to promote diversity in specified field.

        Parameters
        ----------
        field : str
            Metadata field to diversify by
        weight : float, default 1.0
            Weight factor for diversity in reranking

        Returns
        -------
        QueryBuilder
            New QueryBuilder instance with diversity reranking
        """
        return self.rerank("diversity", field=field, weight=weight)

    def validate(self) -> Dict[str, Any]:
        """
        Validate the current query configuration and return validation results.

        Returns
        -------
        Dict[str, Any]
            Validation results including warnings and recommendations

        Examples
        --------
        Check query before execution::

            builder = db.query_builder().search("AI").group_by("category")
            validation = builder.validate()
            if validation["valid"]:
                results = builder.execute()
        """
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
            "query_complexity": self._estimate_complexity()
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

    def _generate_sql_preview(self) -> str:
        """
        Generate a human-readable SQL-like preview of the query.
        This is a simplified representation for debugging, not actual SQL.
        """
        parts = []

        # Select clause
        if self._aggregations:
            agg_fields = [f"{agg.function}({agg.field}) AS {agg.alias or f'{agg.function}_{agg.field}'}"
                          for agg in self._aggregations]
            if self._group_by:
                select_fields = self._group_by + agg_fields
            else:
                select_fields = agg_fields
            parts.append(f"SELECT {', '.join(select_fields)}")
        else:
            parts.append(f"SELECT * FROM {self._return_type}")

        # Vector search
        if self._search_clauses:
            search_terms = [f"MATCH({clause.field}) AGAINST('{clause.query}' WEIGHT {clause.weight})"
                            for clause in self._search_clauses]
            if self._search_type == "vector":
                parts.append(f"USING VECTOR SEARCH {' AND '.join(search_terms)}")
            elif self._search_type == "keyword":
                parts.append(f"USING KEYWORD SEARCH {' AND '.join(search_terms)}")
            else:  # hybrid
                parts.append(
                    f"USING HYBRID SEARCH (VECTOR {self._vector_weight:.3f}, KEYWORD {1 - self._vector_weight:.3f}) {' AND '.join(search_terms)}")

        # Where clause for filters
        if self._exact_filters:
            filter_exprs = []
            for filter_dict in self._exact_filters:
                for field, value in filter_dict.items():
                    if isinstance(value, dict):  # Operator filter
                        for op, op_value in value.items():
                            filter_exprs.append(f"{field} {op} {repr(op_value)}")
                    else:  # Equality filter
                        filter_exprs.append(f"{field} = {repr(value)}")

            parts.append(f"WHERE {' AND '.join(filter_exprs)}")

        # Semantic filters
        if self._semantic_filters:
            semantic_exprs = [f"SEMANTIC_MATCH({filter.field}, '{filter.concept}') > {filter.threshold}"
                              for filter in self._semantic_filters]

            if "WHERE" in parts:
                parts.append(f"AND {' AND '.join(semantic_exprs)}")
            else:
                parts.append(f"WHERE {' AND '.join(semantic_exprs)}")

        # Group by
        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        # Having
        if self._having_clauses:
            having_exprs = []
            for having in self._having_clauses:
                for field, condition in having.items():
                    if isinstance(condition, dict):
                        for op, value in condition.items():
                            having_exprs.append(f"{field} {op} {repr(value)}")
                    else:
                        having_exprs.append(f"{field} = {repr(condition)}")

            parts.append(f"HAVING {' AND '.join(having_exprs)}")

        # Order by
        if self._order_by:
            order_terms = [f"{field} {direction.upper()}" for field, direction in self._order_by]
            parts.append(f"ORDER BY {', '.join(order_terms)}")

        # Limit and offset
        if self._limit:
            parts.append(f"LIMIT {self._limit}")

        if self._offset > 0:
            parts.append(f"OFFSET {self._offset}")

        return " ".join(parts)

    def debug_info(self) -> Dict[str, Any]:
        """
        Get detailed debugging information about the current query state.

        Returns
        -------
        Dict[str, Any]
            Complete query state information for debugging
        """
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
            "sql_preview": self._generate_sql_preview(),
            "hints": self._hints,
            "performance_flags": {
                # "use_cache": self._use_cache,
                # "parallel_semantic_filtering": self._parallel_semantic_filtering,
                "batch_size": self._batch_size
            }
        }

    # Execution methods with proper async support
    def execute(self) -> List[QueryResult]:
        """Execute the query and return results."""
        # Check if this is an async database and warn user
        if self.is_async_database():
            logger.warning(
                "Using sync execute() on async database. "
                "Consider using execute_async() for better performance."
            )
        executor = QueryExecutor(self)
        return executor.execute()

    def is_async_database(self) -> bool:
        """Check if the underlying database is async."""
        return (
                hasattr(self, '_is_async_db') and self._is_async_db or
                hasattr(self._db, 'is_async_database') and self._db.is_async_database() or
                asyncio.iscoroutinefunction(getattr(self._db, 'query', None))
        )

    async def execute_async(self) -> List[QueryResult]:
        """Execute the query asynchronously with native async support."""
        executor = AsyncQueryExecutor(self)
        return await executor.execute()

    def stream(self, batch_size: int = 100) -> Iterator[List[QueryResult]]:
        """Stream results in batches."""
        executor = QueryExecutor(self)
        return executor.stream(batch_size)

    async def stream_async(self, batch_size: int = 100):
        """Stream results in batches asynchronously."""
        executor = AsyncQueryExecutor(self)
        async for batch in executor.stream(batch_size):
            yield batch

    def count(self) -> int:
        """Get count of matching results without returning them."""
        executor = QueryExecutor(self)
        return executor.count()

    async def count_async(self) -> int:
        """Get count of matching results asynchronously."""
        executor = AsyncQueryExecutor(self)
        return await executor.count()


# Keep the original QueryExecutor for sync databases
class QueryExecutor:
    """Executor for QueryBuilder queries.

    This class handles the actual execution of queries built by QueryBuilder,
    including result processing, aggregation, and optimization.

    Parameters
    ----------
    query_builder : QueryBuilder
        QueryBuilder instance to execute

    Attributes
    ----------
    builder : QueryBuilder
        Reference to the QueryBuilder
    db : AnyVectorDB
        Reference to the database
    """

    def __init__(self, query_builder: "QueryBuilder"):
        self.builder = query_builder
        self.db = query_builder._db

    def execute(self) -> List[QueryResult]:
        """
        Execute the query synchronously with full feature support.

        Returns
        -------
        List[QueryResult]
            Query results

        Raises
        ------
        Exception
            If query execution fails
        """
        start_time = time.time()

        try:
            # Generate execution plan if explain is enabled
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
                    result.metadata["_execution_time"] = execution_time
                    result.metadata["_execution_plan"] = execution_plan
                    if self.builder._hints.get("detailed_explain", False):
                        result.metadata["_detailed_stats"] = self._get_detailed_stats(results)

            return results

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def _generate_execution_plan(self) -> Dict[str, Any]:
        """Generate execution plan for explain functionality."""
        plan = {
            "steps": [],
            "estimated_cost": 0,
            "query_type": "hybrid" if self.builder._search_clauses and self.builder._exact_filters else
            "search" if self.builder._search_clauses else "filter",
            "optimizations": []
        }

        if self.builder._search_clauses:
            plan["steps"].append("vector_search")
            plan["estimated_cost"] += len(self.builder._search_clauses) * 10

        if self.builder._exact_filters:
            plan["steps"].append("exact_filtering")
            plan["estimated_cost"] += len(self.builder._exact_filters) * 2

        if self.builder._semantic_filters:
            plan["steps"].append("semantic_filtering")
            plan["estimated_cost"] += len(self.builder._semantic_filters) * 50

        if self.builder._group_by:
            plan["steps"].append("grouping")
            plan["estimated_cost"] += 20

        if self.builder._aggregations:
            plan["steps"].append("aggregation")
            plan["estimated_cost"] += len(self.builder._aggregations) * 5

        if self.builder._having_clauses:
            plan["steps"].append("having_filter")
            plan["estimated_cost"] += len(self.builder._having_clauses) * 3

        if self.builder._order_by:
            plan["steps"].append("sorting")
            plan["estimated_cost"] += 10

        if self.builder._rerank_config:
            plan["steps"].append("reranking")
            plan["estimated_cost"] += 30

        # Add optimization recommendations
        # if self.builder._use_cache:
        #     plan["optimizations"].append("result_caching_enabled")

        # if self.builder._parallel_semantic_filtering and self.builder._semantic_filters:
        #     plan["optimizations"].append("parallel_semantic_processing")

        return plan

    def _apply_aggregations_and_grouping(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply GROUP BY and aggregation operations to results."""
        if not (self.builder._group_by or self.builder._aggregations):
            return results

        # Group results
        if self.builder._group_by:
            grouped_results = self._group_results(results)
        else:
            # Single group for aggregations without GROUP BY
            grouped_results = {"_all": results}

        # Apply aggregations
        aggregated_results = []

        for group_key, group_results in grouped_results.items():
            # Calculate aggregations for this group
            aggregation_data = {}

            for agg in self.builder._aggregations:
                agg_value = self._calculate_aggregation(group_results, agg)
                alias = agg.alias or f"{agg.function}_{agg.field}"
                aggregation_data[alias] = agg_value

            # Create result for this group
            if self.builder._group_by:
                # Group result with group key information
                group_metadata = dict(zip(self.builder._group_by,
                                          group_key if isinstance(group_key, tuple) else [group_key]))
                group_metadata.update(aggregation_data)

                result = QueryResult(
                    id=f"group_{hash(group_key)}",
                    score=1.0,  # Groups don't have meaningful scores
                    type="group",
                    content=f"Group: {group_key}",
                    metadata=group_metadata
                )
            else:
                # Single aggregation result
                result = QueryResult(
                    id="aggregation_result",
                    score=1.0,
                    type="aggregation",
                    content="Aggregation Result",
                    metadata=aggregation_data
                )

            aggregated_results.append(result)

        # Apply HAVING clauses
        if self.builder._having_clauses:
            aggregated_results = self._apply_having_clauses(aggregated_results)

        return aggregated_results

    def _group_results(self, results: List[QueryResult]) -> Dict[Any, List[QueryResult]]:
        """Group results by specified fields."""
        grouped = defaultdict(list)

        for result in results:
            # Extract group key
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
        values = []
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
        filtered_results = []

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
            return value == target
        elif operator == "$ne":
            return value != target
        elif operator == "$gt":
            return value > target
        elif operator == "$gte":
            return value >= target
        elif operator == "$lt":
            return value < target
        elif operator == "$lte":
            return value <= target

        return False

    # TODO: to implement AI-based reranking.
    def _apply_reranking(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply reranking based on configuration."""
        if not self.builder._rerank_config:
            return results

        method = self.builder._rerank_config["method"]

        if method == "relevance":
            # Already sorted by relevance, just apply weight
            weight = self.builder._rerank_config.get("weight", 1.0)
            for result in results:
                result.score *= weight

        elif method == "recency":
            date_field = self.builder._rerank_config["date_field"]
            weight = self.builder._rerank_config.get("weight", 1.0)

            # Calculate recency scores
            current_time = datetime.now()
            for result in results:
                if date_field in result.metadata:
                    try:
                        date_value = result.metadata[date_field]
                        if isinstance(date_value, str):
                            date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                        elif isinstance(date_value, datetime):
                            date_obj = date_value
                        else:
                            continue

                        # Calculate recency score (newer = higher)
                        days_ago = (current_time - date_obj).days
                        recency_score = 1.0 / (1.0 + days_ago / 365.0)  # Decay over year

                        # Combine with existing score
                        result.score = (result.score * (1 - weight)) + (recency_score * weight)
                    except (ValueError, TypeError):
                        continue

            # Re-sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)

        elif method == "diversity":
            # Implement diversity reranking (promote variety in specified field)
            field = self.builder._rerank_config["field"]
            weight = self.builder._rerank_config.get("weight", 1.0)

            seen_values = set()
            reranked_results = []

            for result in results:
                field_value = result.metadata.get(field)
                if field_value not in seen_values:
                    # Boost score for diversity
                    result.score *= (1.0 + weight)
                    seen_values.add(field_value)

                reranked_results.append(result)

            # Re-sort by new scores
            reranked_results.sort(key=lambda x: x.score, reverse=True)
            results = reranked_results

        return results

    def _get_detailed_stats(self, results: List[QueryResult]) -> Dict[str, Any]:
        """Get detailed execution statistics for explain mode."""
        return {
            "result_count": len(results),
            "score_distribution": {
                "min": min(r.score for r in results) if results else 0,
                "max": max(r.score for r in results) if results else 0,
                "avg": sum(r.score for r in results) / len(results) if results else 0
            },
            "query_components": {
                "search_clauses": len(self.builder._search_clauses),
                "exact_filters": len(self.builder._exact_filters),
                "semantic_filters": len(self.builder._semantic_filters),
                "aggregations": len(self.builder._aggregations),
                "group_by_fields": len(self.builder._group_by)
            },
            "optimization_flags": {
                # "cache_enabled": self.builder._use_cache,
                # "parallel_semantic": self.builder._parallel_semantic_filtering,
                "batch_size": self.builder._batch_size
            }
        }

    def _execute_search_query(self) -> List[QueryResult]:
        """Execute search-based query."""
        if len(self.builder._search_clauses) == 1:
            return self._execute_single_search()
        else:
            return self._execute_multi_search()

    def _execute_single_search(self) -> List[QueryResult]:
        """Execute a single search clause."""
        clause = self.builder._search_clauses[0]
        filters = self._combine_exact_filters()

        results = self.db.query(
            query=clause.query,
            search_type=clause.search_type,
            return_type=self.builder._return_type,
            k=self.builder._limit + self.builder._offset,
            score_threshold=clause.score_threshold,
            filters=filters,
            vector_weight=self.builder._vector_weight,
            context_window=self.builder._context_window,
            semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
            document_scoring_method=self.builder._document_scoring_method
        )

        if self.builder._semantic_filters:
            results = self._apply_semantic_filters(results)

        return results

    def _execute_multi_search(self) -> List[QueryResult]:
        """Execute multiple search clauses and combine results."""
        all_results = []
        filters = self._combine_exact_filters()

        for clause in self.builder._search_clauses:
            clause_results = self.db.query(
                query=clause.query,
                search_type=clause.search_type,
                return_type=self.builder._return_type,
                k=self.builder._limit * 2,
                score_threshold=clause.score_threshold,
                filters=filters,
                vector_weight=self.builder._vector_weight,
                context_window=self.builder._context_window,
                semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
                document_scoring_method=self.builder._document_scoring_method
            )

            for result in clause_results:
                result.score *= clause.weight

            all_results.extend(clause_results)

        merged_results = self._merge_search_results(all_results)

        if self.builder._semantic_filters:
            merged_results = self._apply_semantic_filters(merged_results)

        return merged_results

    def _execute_filter_only_query(self) -> List[QueryResult]:
        """Execute a filter-only query."""
        filters = self._combine_exact_filters()

        documents = self.db.filter(
            where=filters,
            limit=self.builder._limit,
            offset=self.builder._offset,
            order_by=self._build_order_by_clause()
        )

        results = [
            QueryResult(
                id=doc.id,
                score=1.0,
                type="document",
                content=doc.content,
                metadata=doc.metadata
            )
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
            doc = Document(
                id=result.id,
                content=result.content,
                metadata=result.metadata.copy()
            )
            doc.metadata["_original_score"] = result.score
            documents.append(doc)

        for semantic_filter in self.builder._semantic_filters:
            documents = semantic_filter.apply(documents, self.db)

        filtered_results = []
        for doc in documents:
            original_score = doc.metadata.pop("_original_score", 1.0)
            result = QueryResult(
                id=doc.id,
                score=original_score,
                type="document",
                content=doc.content,
                metadata=doc.metadata
            )
            filtered_results.append(result)

        return filtered_results

    def _apply_post_processing(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply post-processing steps."""
        if self.builder._order_by and not self.builder._aggregations:
            results = self._apply_sorting(results)

        if self.builder._offset > 0:
            results = results[self.builder._offset:]

        if len(results) > self.builder._limit:
            results = results[:self.builder._limit]

        return results

    def count(self) -> int:
        """Get count of matching results."""
        count_builder = self.builder.clone()
        count_builder._limit = 999999
        count_builder._offset = 0

        if count_builder._search_clauses:
            results = self._execute_search_query()
            return len(results)
        else:
            results = self._execute_filter_only_query()
            return len(results)

    def stream(self, batch_size: int = 100) -> Iterator[List[QueryResult]]:
        """Stream results in batches."""
        original_limit = self.builder._limit
        original_offset = self.builder._offset
        current_offset = original_offset

        try:
            while True:
                batch_builder = self.builder.clone()
                batch_builder._limit = min(batch_size, original_limit - (current_offset - original_offset))
                batch_builder._offset = current_offset

                batch_executor = QueryExecutor(batch_builder)
                batch_results = batch_executor.execute()

                if not batch_results:
                    break

                yield batch_results
                current_offset += len(batch_results)

                if current_offset - original_offset >= original_limit:
                    break

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            raise

    # Helper methods (same as AsyncQueryExecutor)
    def _combine_exact_filters(self) -> Dict[str, Any]:
        """Combine all exact filters into a single filter dictionary."""
        if not self.builder._exact_filters:
            return {}

        combined = {}
        and_clauses = []

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

    def _merge_search_results(self, all_results: List[QueryResult]) -> List[QueryResult]:
        """Merge and deduplicate search results from multiple clauses."""
        result_groups = defaultdict(list)
        for result in all_results:
            result_groups[result.id].append(result)

        merged_results = []
        for result_id, results in result_groups.items():
            if len(results) == 1:
                merged_results.append(results[0])
            else:
                total_score = sum(r.score for r in results)
                avg_score = total_score / len(results)
                merged_result = results[0]
                merged_result.score = avg_score
                merged_results.append(merged_result)

        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results

    def _apply_sorting(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply sorting to results."""
        for field, direction in reversed(self.builder._order_by):
            reverse = (direction.lower() == "desc")

            if field == "score":
                results.sort(key=lambda x: x.score, reverse=reverse)
            else:
                def sort_key(result):
                    value = result.metadata.get(field)
                    if value is None:
                        return float('-inf') if reverse else float('inf')
                    return value

                results.sort(key=sort_key, reverse=reverse)

        return results

    def _build_order_by_clause(self) -> Optional[str]:
        """Build ORDER BY clause for database queries."""
        if not self.builder._order_by:
            return None

        field, direction = self.builder._order_by[0]
        return f"{field} {direction.upper()}"


class AsyncQueryExecutor:
    """Enhanced async query executor with full SQL-like functionality."""

    def __init__(self, query_builder: "QueryBuilder"):
        self.builder = query_builder
        self.db = query_builder._db

    # The async executor can reuse most methods from the sync version
    # since aggregation, grouping, and reranking are CPU-bound operations
    _generate_execution_plan = QueryExecutor._generate_execution_plan
    _apply_aggregations_and_grouping = QueryExecutor._apply_aggregations_and_grouping
    _group_results = QueryExecutor._group_results
    _calculate_aggregation = QueryExecutor._calculate_aggregation
    _apply_having_clauses = QueryExecutor._apply_having_clauses
    _check_condition = QueryExecutor._check_condition
    _apply_reranking = QueryExecutor._apply_reranking
    _get_detailed_stats = QueryExecutor._get_detailed_stats
    _apply_post_processing = QueryExecutor._apply_post_processing
    _combine_exact_filters = QueryExecutor._combine_exact_filters
    _merge_search_results = QueryExecutor._merge_search_results
    _apply_sorting = QueryExecutor._apply_sorting
    _build_order_by_clause = QueryExecutor._build_order_by_clause

    async def execute(self) -> List[QueryResult]:
        """Execute the query asynchronously with full feature support."""
        start_time = time.time()

        try:
            # Generate execution plan if explain is enabled
            if self.builder._explain:
                execution_plan = self._generate_execution_plan()

            # Execute base query
            if self.builder._search_clauses:
                results = await self._execute_search_query_async()
            else:
                results = await self._execute_filter_only_query_async()

            # Apply post-processing
            results = await self._apply_post_processing_async(results)

            # Apply aggregations and grouping if specified
            if self.builder._aggregations or self.builder._group_by:
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
                    if self.builder._hints.get("detailed_explain", False):
                        result.metadata["_detailed_stats"] = self._get_detailed_stats(results)

            return results

        except Exception as e:
            logger.error(f"Async query execution failed: {e}")
            raise

    async def _execute_search_query_async(self) -> List[QueryResult]:
        """Execute search-based query asynchronously."""
        # Implementation would go here - this depends on the specific database backend
        # For now, we'll indicate this needs to be implemented based on the database type
        if hasattr(self.db, 'query_async'):
            # Use database's async query method
            clause = self.builder._search_clauses[0]  # Simplified for example
            filters = self._combine_exact_filters()

            results = await self.db.query_async(
                query=clause.query,
                search_type=clause.search_type,
                return_type=self.builder._return_type,
                k=self.builder._limit + self.builder._offset,
                score_threshold=clause.score_threshold,
                filters=filters,
                vector_weight=self.builder._vector_weight,
                context_window=self.builder._context_window,
                semantic_dedup_threshold=self.builder._semantic_dedup_threshold,
                document_scoring_method=self.builder._document_scoring_method
            )
        else:
            # Fall back to sync version in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                results = await loop.run_in_executor(
                    executor, self._execute_search_query_sync
                )

        # Apply semantic filters asynchronously
        if self.builder._semantic_filters:
            results = await self._apply_semantic_filters_async(results)

        return results

    async def _execute_filter_only_query_async(self) -> List[QueryResult]:
        """Execute filter-only query asynchronously."""
        # Similar pattern as search query
        if hasattr(self.db, 'filter_async'):
            filters = self._combine_exact_filters()
            order_by = self._build_order_by_clause()

            documents = await self.db.filter_async(
                where=filters,
                order_by=order_by,
                limit=self.builder._limit + self.builder._offset,
                offset=0
            )

            # Convert to QueryResults
            results = [
                QueryResult(
                    id=doc.id,
                    score=1.0,
                    type="document",
                    content=doc.content,
                    metadata=doc.metadata
                )
                for doc in documents
            ]
        else:
            # Fall back to sync version
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                results = await loop.run_in_executor(
                    executor, self._execute_filter_only_query_sync
                )

        return results

    def _execute_search_query_sync(self) -> List[QueryResult]:
        """Sync fallback for search query."""
        # This would call the existing sync implementation
        sync_executor = QueryExecutor(self.builder)
        return sync_executor._execute_search_query()

    def _execute_filter_only_query_sync(self) -> List[QueryResult]:
        """Sync fallback for filter-only query."""
        # This would call the existing sync implementation
        sync_executor = QueryExecutor(self.builder)
        return sync_executor._execute_filter_only_query()
