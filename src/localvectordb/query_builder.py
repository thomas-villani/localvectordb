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

from __future__ import annotations
import asyncio
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
    Dict, List, Union, Optional, Any, Iterator,
    Literal, Callable
)
import numpy as np
from enum import Enum

from localvectordb.core import QueryResult, Document, AnyVectorDB

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
    field: str
    query: str
    weight: float = 1.0
    search_type: Literal["vector", "keyword", "hybrid"] = "vector"


@dataclass
class SemanticFilter:
    """Semantic filtering based on conceptual similarity with async support."""
    field: str
    concept: str
    threshold: float
    metric: SimilarityMetric = SimilarityMetric.COSINE
    embedding_model: Optional[str] = None
    cache_key: Optional[str] = None

    def __post_init__(self):
        if self.cache_key is None:
            self.cache_key = f"semantic:{self.field}:{hash(self.concept)}:{self.embedding_model}"

    async def apply_async(self, documents: List[Document], db: AnyVectorDB) -> List[Document]:
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
        """Sync version of semantic filtering - also uses embedding provider directly."""
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
        embedding_provider = db.embedding_provider
        concept_embedding = embedding_provider.embed_sync([self.concept])[0]
        field_embeddings = embedding_provider.embed_sync(field_contents)

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

    def _extract_field_content(self, doc: Document, field: str) -> Optional[str]:
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
            distance = np.linalg.norm(a_flat - b_flat)
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


class QueryBuilder:
    """Fluent interface for building complex vector database queries with async support."""

    def __init__(self, db: AnyVectorDB):
        self._db = db
        self._search_clauses: List[SearchClause] = []
        self._exact_filters: List[Dict[str, Any]] = []
        self._semantic_filters: List[SemanticFilter] = []
        self._search_type: Literal["vector", "keyword", "hybrid"] = "vector"
        self._vector_weight: float = 0.7
        self._return_type: Literal["documents", "chunks"] = "documents"
        self._limit: int = 10
        self._offset: int = 0
        self._order_by: List[tuple[str, str]] = []  # (field, direction)
        self._group_by: List[str] = []
        self._aggregations: List[AggregationClause] = []
        self._having_clauses: List[Dict[str, Any]] = []
        self._rerank_config: Optional[Dict[str, Any]] = None
        self._explain: bool = False
        self._hints: Dict[str, Any] = {}
        self._score_threshold: float = 0.0

        # Performance optimization flags
        self._batch_size: int = 100
        self._use_cache: bool = True
        self._parallel_semantic_filtering: bool = True

    # ... (keep all the existing builder methods like search, filter, etc.)

    def clone(self) -> "QueryBuilder":
        """Create a copy of this QueryBuilder for chaining."""
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
        new_builder._score_threshold = self._score_threshold
        new_builder._batch_size = self._batch_size
        new_builder._use_cache = self._use_cache
        new_builder._parallel_semantic_filtering = self._parallel_semantic_filtering
        return new_builder

    # Core search methods
    def search(self, query: str, field: str = "content") -> "QueryBuilder":
        """Add a search clause for the specified field."""
        builder = self.clone()
        builder._search_clauses.append(SearchClause(field, query, 1.0, self._search_type))
        return builder

    def search_field(self, field: str, query: str, weight: float = 1.0) -> "QueryBuilder":
        """Add a weighted search clause for a specific field."""
        builder = self.clone()
        builder._search_clauses.append(SearchClause(field, query, weight, self._search_type))
        return builder

    # Add all other builder methods (filter, semantic_filter, limit, etc.)
    def filter(self, field: str = None, value=None, **kwargs) -> "QueryBuilder":
        """Add exact filter conditions."""
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
                builder._exact_filters.append({key: val})

        return builder

    def semantic_filter(
            self,
            field: str,
            concept: str,
            threshold: float = 0.8,
            metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> "QueryBuilder":
        """Add semantic filtering based on conceptual similarity."""
        builder = self.clone()
        semantic_filter = SemanticFilter(field, concept, threshold, metric)
        builder._semantic_filters.append(semantic_filter)
        return builder

    def limit(self, n: int) -> "QueryBuilder":
        """Limit the number of results."""
        if n <= 0:
            raise ValueError("Limit must be positive")
        builder = self.clone()
        builder._limit = n
        return builder

    def offset(self, n: int) -> "QueryBuilder":
        """Skip the first n results."""
        if n < 0:
            raise ValueError("Offset must be non-negative")
        builder = self.clone()
        builder._offset = n
        return builder

    def vector(self) -> "QueryBuilder":
        """Use vector search."""
        builder = self.clone()
        builder._search_type = "vector"
        return builder

    def keyword(self) -> "QueryBuilder":
        """Use keyword search."""
        builder = self.clone()
        builder._search_type = "keyword"
        return builder

    def hybrid(self, vector_weight: float = 0.7) -> "QueryBuilder":
        """Use hybrid search with specified vector weight."""
        builder = self.clone()
        builder._search_type = "hybrid"
        builder._vector_weight = vector_weight
        return builder

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


class AsyncQueryExecutor:
    """Async query executor that leverages native async database methods."""

    def __init__(self, query_builder: "QueryBuilder"):
        self.builder = query_builder
        self.db = query_builder._db

    async def execute(self) -> List[QueryResult]:
        """Execute the query asynchronously."""
        start_time = time.time()

        try:
            # Determine if this is an async database
            is_async_db = hasattr(self.db, 'query') and asyncio.iscoroutinefunction(self.db.query)

            if self.builder._search_clauses:
                results = await self._execute_search_query_async()
            else:
                results = await self._execute_filter_only_query_async()

            # Apply post-processing
            results = await self._apply_post_processing_async(results)

            execution_time = time.time() - start_time
            logger.debug(f"Async query executed in {execution_time:.3f}s, returned {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Async query execution failed: {e}")
            raise

    async def _execute_search_query_async(self) -> List[QueryResult]:
        """Execute search query using async database methods."""
        if len(self.builder._search_clauses) == 1:
            return await self._execute_single_search_async()
        else:
            return await self._execute_multi_search_async()

    async def _execute_single_search_async(self) -> List[QueryResult]:
        """Execute a single search clause asynchronously."""
        clause = self.builder._search_clauses[0]
        filters = self._combine_exact_filters()

        # Check if database has async query method
        if hasattr(self.db, 'query') and asyncio.iscoroutinefunction(self.db.query):
            results = await self.db.query(
                query=clause.query,
                search_type=clause.search_type,
                return_type=self.builder._return_type,
                k=self.builder._limit + self.builder._offset,
                score_threshold=self.builder._score_threshold,
                filters=filters,
                vector_weight=self.builder._vector_weight
            )
        else:
            # Fallback to sync in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                results = await loop.run_in_executor(
                    executor,
                    lambda: self.db.query(
                        query=clause.query,
                        search_type=clause.search_type,
                        return_type=self.builder._return_type,
                        k=self.builder._limit + self.builder._offset,
                        score_threshold=self.builder._score_threshold,
                        filters=filters,
                        vector_weight=self.builder._vector_weight
                    )
                )

        # Apply semantic filters asynchronously
        if self.builder._semantic_filters:
            results = await self._apply_semantic_filters_async(results)

        return results

    async def _execute_multi_search_async(self) -> List[QueryResult]:
        """Execute multiple search clauses and combine results asynchronously."""
        tasks = []
        filters = self._combine_exact_filters()

        for clause in self.builder._search_clauses:
            if hasattr(self.db, 'query') and asyncio.iscoroutinefunction(self.db.query):
                task = self.db.query(
                    query=clause.query,
                    search_type=clause.search_type,
                    return_type=self.builder._return_type,
                    k=self.builder._limit * 2,
                    score_threshold=self.builder._score_threshold,
                    filters=filters,
                    vector_weight=self.builder._vector_weight
                )
            else:
                # Create async wrapper for sync query
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    task = loop.run_in_executor(
                        executor,
                        lambda c=clause: self.db.query(
                            query=c.query,
                            search_type=c.search_type,
                            return_type=self.builder._return_type,
                            k=self.builder._limit * 2,
                            score_threshold=self.builder._score_threshold,
                            filters=filters,
                            vector_weight=self.builder._vector_weight
                        )
                    )
            tasks.append((clause, task))

        # Execute all searches concurrently
        all_results = []
        for clause, task in tasks:
            clause_results = await task
            # Weight the scores
            for result in clause_results:
                result.score *= clause.weight
            all_results.extend(clause_results)

        # Merge and deduplicate results
        merged_results = self._merge_search_results(all_results)

        # Apply semantic filters
        if self.builder._semantic_filters:
            merged_results = await self._apply_semantic_filters_async(merged_results)

        return merged_results

    async def _execute_filter_only_query_async(self) -> List[QueryResult]:
        """Execute filter-only query asynchronously."""
        filters = self._combine_exact_filters()

        if hasattr(self.db, 'filter') and asyncio.iscoroutinefunction(self.db.filter):
            documents = await self.db.filter(
                where=filters,
                limit=self.builder._limit,
                offset=self.builder._offset,
                order_by=self._build_order_by_clause()
            )
        else:
            # Fallback to sync in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                documents = await loop.run_in_executor(
                    executor,
                    lambda: self.db.filter(
                        where=filters,
                        limit=self.builder._limit,
                        offset=self.builder._offset,
                        order_by=self._build_order_by_clause()
                    )
                )

        # Convert documents to QueryResults
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

        # Apply semantic filters
        if self.builder._semantic_filters:
            results = await self._apply_semantic_filters_async(results)

        return results

    async def _apply_semantic_filters_async(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply semantic filtering to results asynchronously."""
        if not self.builder._semantic_filters:
            return results

        # Convert QueryResults to Documents for semantic filtering
        documents = []
        for result in results:
            doc = Document(
                id=result.id,
                content=result.content,
                metadata=result.metadata.copy()
            )
            doc.metadata["_original_score"] = result.score
            documents.append(doc)

        # Apply each semantic filter asynchronously
        for semantic_filter in self.builder._semantic_filters:
            try:
                # Check if database has async embedding support
                if hasattr(self.db, '_generate_embeddings_async'):
                    # Use database's async embedding method
                    documents = await semantic_filter.apply_async(documents, self.db)
                elif hasattr(self.db, 'embedding_provider'):
                    # Use embedding provider directly
                    documents = await semantic_filter.apply_async(documents, self.db)
                else:
                    # Fall back to sync version in thread pool
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        documents = await loop.run_in_executor(
                            executor, semantic_filter.apply, documents, self.db
                        )
            except Exception as e:
                logger.warning(f"Semantic filter failed, skipping: {e}")
                continue

        # Convert back to QueryResults
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

    async def _apply_post_processing_async(self, results: List[QueryResult]) -> List[QueryResult]:
        """Apply post-processing steps asynchronously."""
        # Apply sorting if specified
        if self.builder._order_by and not self.builder._aggregations:
            results = self._apply_sorting(results)

        # Apply pagination
        if self.builder._offset > 0:
            results = results[self.builder._offset:]

        if len(results) > self.builder._limit:
            results = results[:self.builder._limit]

        return results

    async def count(self) -> int:
        """Get count of matching results asynchronously."""
        # Create a modified builder for counting
        count_builder = self.builder.clone()
        count_builder._limit = 999999
        count_builder._offset = 0

        if count_builder._search_clauses:
            results = await self._execute_search_query_async()
            return len(results)
        else:
            results = await self._execute_filter_only_query_async()
            return len(results)

    async def stream(self, batch_size: int = 100):
        """Stream results in batches asynchronously."""
        original_limit = self.builder._limit
        original_offset = self.builder._offset
        current_offset = original_offset

        try:
            while True:
                # Create a modified builder for this batch
                batch_builder = self.builder.clone()
                batch_builder._limit = min(batch_size, original_limit - (current_offset - original_offset))
                batch_builder._offset = current_offset

                # Execute batch
                batch_executor = AsyncQueryExecutor(batch_builder)
                batch_results = await batch_executor.execute()

                if not batch_results:
                    break

                yield batch_results
                current_offset += len(batch_results)

                # Check if we've reached the original limit
                if current_offset - original_offset >= original_limit:
                    break

        except Exception as e:
            logger.error(f"Error during async streaming: {e}")
            raise

    # Helper methods (same as sync version)
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


# Keep the original QueryExecutor for sync databases
class QueryExecutor:
    """Original sync query executor."""

    def __init__(self, query_builder: "QueryBuilder"):
        self.builder = query_builder
        self.db = query_builder._db

    def execute(self) -> List[QueryResult]:
        """Execute the query synchronously."""
        start_time = time.time()

        try:
            if self.builder._search_clauses:
                results = self._execute_search_query()
            else:
                results = self._execute_filter_only_query()

            results = self._apply_post_processing(results)

            execution_time = time.time() - start_time
            logger.debug(f"Query executed in {execution_time:.3f}s, returned {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

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
            score_threshold=self.builder._score_threshold,
            filters=filters,
            vector_weight=self.builder._vector_weight
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
                score_threshold=self.builder._score_threshold,
                filters=filters,
                vector_weight=self.builder._vector_weight
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