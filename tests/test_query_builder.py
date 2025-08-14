# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/test_query_builder.py
"""
Tests for localvectordb.query_builder module.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from localvectordb.query_builder import (
    QueryBuilder, SearchClause, SemanticFilter, AggregationClause, SimilarityMetric, QueryExecutor
)
from localvectordb.core import QueryResult, Document


class TestSearchClause:
    """Test SearchClause dataclass."""

    def test_search_clause_creation(self):
        """Test creating SearchClause with all parameters."""
        clause = SearchClause(
            field="content",
            query="machine learning",
            weight=0.8,
            search_type="hybrid"
        )

        assert clause.field == "content"
        assert clause.query == "machine learning"
        assert clause.weight == 0.8
        assert clause.search_type == "hybrid"

    def test_search_clause_defaults(self):
        """Test SearchClause default values."""
        clause = SearchClause(field="content", query="test")

        assert clause.weight == 1.0
        assert clause.search_type == "vector"


class TestSemanticFilter:
    """Test SemanticFilter dataclass."""

    def test_semantic_filter_creation(self):
        """Test creating SemanticFilter with all parameters."""
        semantic_filter = SemanticFilter(
            field="category",
            concept="artificial intelligence",
            threshold=0.8,
            metric=SimilarityMetric.COSINE,
            embedding_model="test-model"
        )

        assert semantic_filter.field == "category"
        assert semantic_filter.concept == "artificial intelligence"
        assert semantic_filter.threshold == 0.8
        assert semantic_filter.metric == SimilarityMetric.COSINE
        assert semantic_filter.embedding_model == "test-model"

    def test_semantic_filter_defaults(self):
        """Test SemanticFilter default values."""
        semantic_filter = SemanticFilter("field", "concept", 0.7)

        assert semantic_filter.metric == SimilarityMetric.COSINE
        assert semantic_filter.embedding_model is None

    def test_similarity_metrics(self):
        """Test all similarity metric types."""
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"
        assert SimilarityMetric.MANHATTAN.value == "manhattan"


class TestAggregationClause:
    """Test AggregationClause dataclass."""

    def test_aggregation_clause_creation(self):
        """Test creating AggregationClause with all parameters."""
        agg = AggregationClause(
            field="rating",
            function="avg",
            alias="average_rating"
        )

        assert agg.field == "rating"
        assert agg.function == "avg"
        assert agg.alias == "average_rating"

    def test_aggregation_clause_without_alias(self):
        """Test AggregationClause without alias."""
        agg = AggregationClause(field="count", function="sum")

        assert agg.alias is None


class TestQueryBuilderInitialization:
    """Test QueryBuilder initialization and basic properties."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        db.supports_async_embeddings.return_value = False
        return db

    def test_initialization(self, mock_db):
        """Test QueryBuilder initialization with default values."""
        builder = QueryBuilder(mock_db)

        assert builder._db is mock_db
        assert builder._search_clauses == []
        assert builder._exact_filters == []
        assert builder._semantic_filters == []
        assert builder._search_type == "hybrid"
        assert builder._vector_weight == 0.7
        assert builder._return_type == "documents"
        assert builder._limit == 10
        assert builder._offset == 0
        assert builder._order_by == []
        assert builder._group_by == []
        assert builder._aggregations == []
        assert builder._having_clauses == []
        assert builder._rerank_config is None
        assert builder._explain is False
        assert builder._context_window == 2
        assert builder._semantic_dedup_threshold is None
        assert builder._document_scoring_method == "frequency_boost"
        assert builder._batch_size == 100

    def test_clone(self, mock_db):
        """Test cloning QueryBuilder preserves all state."""
        original = QueryBuilder(mock_db)
        original._search_clauses.append(SearchClause("content", "test"))
        original._exact_filters.append({"field": "value"})
        original._limit = 20
        original._offset = 5
        original._search_type = "vector"
        original._explain = True

        cloned = original.clone()

        # Verify it's a different instance
        assert cloned is not original
        assert cloned._db is original._db

        # Verify all properties are copied
        assert cloned._search_clauses == original._search_clauses
        assert cloned._exact_filters == original._exact_filters
        assert cloned._limit == original._limit
        assert cloned._offset == original._offset
        assert cloned._search_type == original._search_type
        assert cloned._explain == original._explain

        # Verify they're independent copies
        cloned._limit = 30
        assert original._limit == 20


class TestQueryBuilderSearchMethods:
    """Test QueryBuilder search-related methods."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_search_basic(self, builder):
        """Test basic search functionality."""
        result = builder.search("machine learning")

        assert len(result._search_clauses) == 1
        clause = result._search_clauses[0]
        assert clause.field == "content"
        assert clause.query == "machine learning"
        assert clause.weight == 1.0
        assert clause.search_type == "hybrid"  # Default from builder

    def test_search_with_type(self, builder):
        """Test search with specified search type."""
        result = builder.search("AI", search_type="vector")

        clause = result._search_clauses[0]
        assert clause.search_type == "vector"

    def test_search_with_vector_weight(self, builder):
        """Test search with vector weight."""
        result = builder.search("AI", vector_weight=0.8)

        assert result._vector_weight == 0.8

    def test_search_validation_empty_query(self, builder):
        """Test search with empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            builder.search("")

        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            builder.search(None)

    def test_search_validation_invalid_type(self, builder):
        """Test search with invalid search type."""
        with pytest.raises(ValueError, match="search_type.*must be one of"):
            builder.search("test", search_type="invalid")

    def test_search_field(self, builder):
        """Test search_field functionality."""
        result = builder.search_field("author", "John Doe")

        assert len(result._exact_filters) == 1
        filter_dict = result._exact_filters[0]
        assert "author" in filter_dict
        assert filter_dict["author"] == {"$ilike": "John Doe"}

    def test_search_field_non_string_value(self, builder):
        """Test search_field with non-string value."""
        result = builder.search_field("rating", 5)

        filter_dict = result._exact_filters[0]
        assert filter_dict["rating"] == 5

    def test_search_field_validation(self, builder):
        """Test search_field validation."""
        with pytest.raises(ValueError, match="Field must be a non-empty string"):
            builder.search_field("", "value")

        with pytest.raises(ValueError, match="Field must be a non-empty string"):
            builder.search_field(None, "value")

    def test_vector_search_shortcut(self, builder):
        """Test vector search shortcut method."""
        result = builder.vector("test query")

        clause = result._search_clauses[0]
        assert clause.search_type == "vector"

    def test_keyword_search_shortcut(self, builder):
        """Test keyword search shortcut method."""
        result = builder.keyword("test query")

        clause = result._search_clauses[0]
        assert clause.search_type == "keyword"

    def test_hybrid_search_shortcut(self, builder):
        """Test hybrid search shortcut method."""
        result = builder.hybrid("test query")

        clause = result._search_clauses[0]
        assert clause.search_type == "hybrid"


class TestQueryBuilderFilterMethods:
    """Test QueryBuilder filtering methods."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_filter_field_value(self, builder):
        """Test basic field-value filtering."""
        result = builder.filter("category", "AI")

        assert len(result._exact_filters) == 1
        assert result._exact_filters[0] == {"category": "AI"}

    def test_filter_kwargs(self, builder):
        """Test filtering with keyword arguments."""
        result = builder.filter(category="AI", rating=5)

        assert len(result._exact_filters) == 2
        assert {"category": "AI"} in result._exact_filters
        assert {"rating": 5} in result._exact_filters

    def test_filter_operators(self, builder):
        """Test filtering with operators."""
        result = builder.filter("rating", gt_=4, lt_=10)

        assert len(result._exact_filters) == 2
        assert {"rating": {"$gt": 4}} in result._exact_filters
        assert {"rating": {"$lt": 10}} in result._exact_filters

    def test_filter_validation_invalid_field(self, builder):
        """Test filter validation with invalid field."""
        with pytest.raises(ValueError, match="Field must be a non-empty string"):
            builder.filter("", "value")

    def test_filter_validation_invalid_operator(self, builder):
        """Test filter validation with invalid operator."""
        with pytest.raises(ValueError, match="Invalid operator suffix"):
            builder.filter(invalid_op_="value")

    def test_semantic_filter(self, builder):
        """Test semantic filtering."""
        result = builder.semantic_filter("category", "machine learning", 0.8)

        assert len(result._semantic_filters) == 1
        sem_filter = result._semantic_filters[0]
        assert sem_filter.field == "category"
        assert sem_filter.concept == "machine learning"
        assert sem_filter.threshold == 0.8
        assert sem_filter.metric == SimilarityMetric.COSINE

    def test_semantic_filter_with_metric(self, builder):
        """Test semantic filtering with specific metric."""
        result = builder.semantic_filter(
            "content", "AI", 0.9, metric=SimilarityMetric.EUCLIDEAN
        )

        sem_filter = result._semantic_filters[0]
        assert sem_filter.metric == SimilarityMetric.EUCLIDEAN

    def test_semantic_filter_validation(self, builder):
        """Test semantic filter validation."""
        with pytest.raises(ValueError, match="Field must be a non-empty string"):
            builder.semantic_filter("", "concept", 0.8)

        with pytest.raises(ValueError, match="Concept must be a non-empty string"):
            builder.semantic_filter("field", "", 0.8)

        with pytest.raises(ValueError, match="Threshold must be between 0 and 1"):
            builder.semantic_filter("field", "concept", 1.5)


class TestQueryBuilderPaginationAndSorting:
    """Test QueryBuilder pagination and sorting methods."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_limit(self, builder):
        """Test limit functionality."""
        result = builder.limit(20)

        assert result._limit == 20

    def test_limit_validation(self, builder):
        """Test limit validation."""
        with pytest.raises(ValueError, match="Limit must be positive"):
            builder.limit(0)

        with pytest.raises(ValueError, match="Limit must be positive"):
            builder.limit(-5)

    def test_offset(self, builder):
        """Test offset functionality."""
        result = builder.offset(10)

        assert result._offset == 10

    def test_offset_validation(self, builder):
        """Test offset validation."""
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            builder.offset(-1)

    def test_order_by(self, builder):
        """Test order_by functionality."""
        result = builder.order_by("created_at", "desc")

        assert len(result._order_by) == 1
        assert result._order_by[0] == ("created_at", "desc")

    def test_order_by_defaults(self, builder):
        """Test order_by with default direction."""
        result = builder.order_by("title")

        assert result._order_by[0] == ("title", "desc")

    def test_order_by_validation(self, builder):
        """Test order_by validation."""
        with pytest.raises(ValueError, match="`field` must be a non-empty string"):
            builder.order_by("", "asc")

        with pytest.raises(ValueError, match="`direction` must be"):
            builder.order_by("field", "invalid")

    def test_multiple_order_by(self, builder):
        """Test multiple order_by clauses."""
        result = builder.order_by("category", "asc").order_by("rating", "desc")

        assert len(result._order_by) == 2
        assert result._order_by[0] == ("category", "asc")
        assert result._order_by[1] == ("rating", "desc")


class TestQueryBuilderAggregationMethods:
    """Test QueryBuilder aggregation and grouping methods."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_group_by(self, builder):
        """Test group_by functionality."""
        result = builder.group_by("category")

        assert "category" in result._group_by

    def test_group_by_validation(self, builder):
        """Test group_by validation."""
        with pytest.raises(ValueError, match="All group_by fields must be non-empty strings"):
            builder.group_by("")

    def test_aggregate(self, builder):
        """Test aggregate functionality."""
        result = builder.aggregate("rating", "avg", "average_rating")

        assert len(result._aggregations) == 1
        agg = result._aggregations[0]
        assert agg.field == "rating"
        assert agg.function == "avg"
        assert agg.alias == "average_rating"

    def test_aggregate_validation(self, builder):
        """Test aggregate validation."""
        with pytest.raises(ValueError, match="function must be one of"):
            builder.aggregate("field", "invalid")

    def test_count_by(self, builder):
        """Test count_by functionality."""
        result = builder.count_by("*", "doc_count")

        agg = result._aggregations[0]
        assert agg.function == "count"
        assert agg.alias == "doc_count"

    def test_sum_by(self, builder):
        """Test sum_by functionality."""
        result = builder.sum_by("rating")

        agg = result._aggregations[0]
        assert agg.function == "sum"
        assert agg.alias == "sum_rating"

    def test_avg_by(self, builder):
        """Test avg_by functionality."""
        result = builder.avg_by("rating", "average")

        agg = result._aggregations[0]
        assert agg.function == "avg"
        assert agg.alias == "average"

    def test_min_by(self, builder):
        """Test min_by functionality."""
        result = builder.min_by("rating")

        agg = result._aggregations[0]
        assert agg.function == "min"

    def test_max_by(self, builder):
        """Test max_by functionality."""
        result = builder.max_by("rating")

        agg = result._aggregations[0]
        assert agg.function == "max"

    def test_having(self, builder):
        """Test having functionality."""
        result = builder.having("count", "gt", 5)

        assert len(result._having_clauses) == 1
        having = result._having_clauses[0]
        assert having == {"count": {"$gt": 5}}

    def test_having_validation(self, builder):
        """Test having validation."""
        with pytest.raises(ValueError, match="operator must be one of"):
            builder.having("field", "invalid", 5)

    def test_having_count(self, builder):
        """Test having_count functionality."""
        result = builder.having_count("gt", 10)

        having = result._having_clauses[0]
        assert having == {"count": {"$gt": 10}}


class TestQueryBuilderChaining:
    """Test QueryBuilder method chaining."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_complex_chaining(self, builder):
        """Test complex method chaining."""
        result = (builder
                  .search("machine learning")
                  .filter("category", "AI")
                  .semantic_filter("methodology", "supervised", 0.8)
                  .limit(20)
                  .offset(10)
                  .order_by("rating", "desc")
                  .group_by("author")
                  .count_by("*", "doc_count")
                  .having("doc_count", "gt", 2))

        # Verify all operations were applied
        assert len(result._search_clauses) == 1
        assert len(result._exact_filters) == 1
        assert len(result._semantic_filters) == 1
        assert result._limit == 20
        assert result._offset == 10
        assert len(result._order_by) == 1
        assert len(result._group_by) == 1
        assert len(result._aggregations) == 1
        assert len(result._having_clauses) == 1

    def test_chaining_independence(self, builder):
        """Test that chained operations create independent instances."""
        base = builder.search("test")
        branch1 = base.filter("category", "A")
        branch2 = base.filter("category", "B")

        assert len(base._exact_filters) == 0
        assert len(branch1._exact_filters) == 1
        assert len(branch2._exact_filters) == 1
        assert branch1._exact_filters != branch2._exact_filters


class TestQueryBuilderUtilityMethods:
    """Test QueryBuilder utility and configuration methods."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_explain(self, builder):
        """Test explain functionality."""
        result = builder.explain()

        assert result._explain is True

    def test_explain_detailed(self, builder):
        """Test explain with detailed flag."""
        result = builder.explain(detailed=True)

        assert result._explain is True

    def test_return_type(self, builder):
        """Test return_type functionality."""
        result = builder.return_type("chunks")

        assert result._return_type == "chunks"

    def test_return_type_validation(self, builder):
        """Test return_type validation."""
        with pytest.raises(ValueError, match="`return_type` must be 'documents' or 'chunks'"):
            builder.return_type("invalid")

    def test_debug_info(self, builder):
        """Test debug_info functionality."""
        builder_with_data = (builder
                             .search("test")
                             .filter("category", "AI")
                             .limit(20))

        debug_info = builder_with_data.debug_info()

        assert isinstance(debug_info, dict)
        assert debug_info["search_clauses"] == 1
        assert debug_info["exact_filters"] == 1
        assert debug_info["limit"] == 20
        assert "sql_preview" in debug_info
        assert "performance_flags" in debug_info

    def test_generate_sql_preview(self, builder):
        """Test SQL preview generation."""
        query = (builder
                 .search("machine learning")
                 .filter("category", "AI")
                 .order_by("rating", "desc")
                 .limit(10)
                 .offset(5))

        sql = query._generate_sql_preview()

        assert "SELECT" in sql
        assert "machine learning" in sql
        assert "WHERE" in sql
        assert "ORDER BY rating DESC" in sql
        assert "LIMIT 10" in sql
        assert "OFFSET 5" in sql


class TestQueryBuilderAsyncDetection:
    """Test QueryBuilder async database detection."""

    def test_is_async_database_with_async_method(self):
        """Test async detection with async query method."""
        mock_db = Mock()

        async def mock_query():
            pass

        mock_db.query = mock_query
        mock_db.is_async_database.return_value = False

        builder = QueryBuilder(mock_db)
        assert builder.is_async_database() is True

    def test_is_async_database_with_flag(self):
        """Test async detection with explicit flag."""
        mock_db = Mock()
        mock_db.is_async_database.return_value = True

        builder = QueryBuilder(mock_db)
        assert builder.is_async_database() is True

    def test_is_async_database_sync(self):
        """Test sync database detection."""
        mock_db = Mock()
        mock_db.is_async_database.return_value = False
        mock_db.query = Mock()  # Regular sync method

        builder = QueryBuilder(mock_db)
        assert builder.is_async_database() is False


class TestQueryExecutor:
    """Test QueryExecutor functionality."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False

        # Mock query result
        mock_result = QueryResult(
            id="doc1",
            score=0.9,
            type="document",
            content="Test content",
            metadata={"category": "AI"}
        )
        db.query.return_value = [mock_result]
        db.filter.return_value = []

        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    def test_execute_search_query(self, builder):
        """Test executing a search query."""
        query = builder.search("machine learning")

        with patch('localvectordb.query_builder.QueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = []
            mock_executor_class.return_value = mock_executor

            results = query.execute()

            mock_executor_class.assert_called_once_with(query)
            mock_executor.execute.assert_called_once()

    def test_execute_filter_only_query(self, builder):
        """Test executing a filter-only query."""
        query = builder.filter("category", "AI")

        with patch('localvectordb.query_builder.QueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.execute.return_value = []
            mock_executor_class.return_value = mock_executor

            results = query.execute()

            mock_executor.execute.assert_called_once()

    def test_count(self, builder):
        """Test count functionality."""
        query = builder.search("test")

        with patch('localvectordb.query_builder.QueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor.count.return_value = 5
            mock_executor_class.return_value = mock_executor

            count = query.count()

            assert count == 5
            mock_executor.count.assert_called_once()

    def test_stream(self, builder):
        """Test stream functionality."""
        query = builder.search("test")

        with patch('localvectordb.query_builder.QueryExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_stream = Mock()
            mock_executor.stream.return_value = mock_stream
            mock_executor_class.return_value = mock_executor

            stream = query.stream(batch_size=50)

            assert stream is mock_stream
            mock_executor.stream.assert_called_once_with(50)


class TestAsyncQueryExecutor:
    """Test AsyncQueryExecutor functionality."""

    @pytest.fixture(scope="function")
    def mock_async_db(self):
        """Create a mock async database for testing."""
        db = Mock()
        db.is_async_database.return_value = True

        async def mock_query(**kwargs):
            return []

        db.query = mock_query
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_async_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_async_db)

    @pytest.mark.asyncio
    async def test_execute_async(self, builder):
        """Test async execution."""
        query = builder.search("machine learning")

        with patch('localvectordb.query_builder.AsyncQueryExecutor') as mock_executor_class:
            mock_executor = AsyncMock()

            async def mock_execute():
                return []

            mock_executor.execute = mock_execute
            mock_executor_class.return_value = mock_executor

            results = await query.execute_async()

            assert results == []
            mock_executor_class.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_count_async(self, builder):
        """Test async count functionality."""
        query = builder.search("test")

        with patch('localvectordb.query_builder.AsyncQueryExecutor') as mock_executor_class:
            mock_executor = Mock()

            async def mock_count():
                return 10

            mock_executor.count = mock_count
            mock_executor_class.return_value = mock_executor

            count = await query.count_async()

            assert count == 10

    @pytest.mark.asyncio
    async def test_stream_async(self, builder):
        """Test async stream functionality."""
        query = builder.search("test")

        with patch('localvectordb.query_builder.AsyncQueryExecutor') as mock_executor_class:
            mock_executor = Mock()

            async def mock_stream(batch_size):
                yield []

            mock_executor.stream = mock_stream
            mock_executor_class.return_value = mock_executor

            async for batch in query.stream_async(batch_size=25):
                assert batch == []
                break


class TestQueryBuilderIntegration:
    """Integration tests for QueryBuilder with mock database."""

    @pytest.fixture(scope="function")
    def mock_documents(self):
        """Create mock documents for testing."""
        return [
            Document(
                id="doc1",
                content="Machine learning and AI research",
                metadata={"category": "AI", "rating": 4.5, "author": "John Doe"}
            ),
            Document(
                id="doc2",
                content="Deep learning neural networks",
                metadata={"category": "AI", "rating": 4.8, "author": "Jane Smith"}
            ),
            Document(
                id="doc3",
                content="Data analysis with Python",
                metadata={"category": "Programming", "rating": 4.2, "author": "Bob Johnson"}
            )
        ]

    @pytest.fixture(scope="function")
    def mock_db(self, mock_documents):
        """Create a comprehensive mock database."""
        db = Mock()
        db.is_async_database.return_value = False

        # Mock query results
        def mock_query(**kwargs):
            query_text = kwargs.get('query', '')
            if 'machine learning' in query_text.lower():
                return [
                    QueryResult(
                        id="doc1",
                        score=0.95,
                        type="document",
                        content=mock_documents[0].content,
                        metadata=mock_documents[0].metadata
                    )
                ]
            return []

        def mock_filter(**kwargs):
            where = kwargs.get('where', {})
            if where.get('category') == 'AI':
                return mock_documents[:2]
            return mock_documents

        db.query = mock_query
        db.filter = mock_filter

        return db

    def test_search_and_filter_integration(self, mock_db):
        """Test combining search and filter operations."""
        builder = QueryBuilder(mock_db)

        query = (builder
                 .search("machine learning")
                 .filter("category", "AI")
                 .limit(10))

        # Verify query structure
        assert len(query._search_clauses) == 1
        assert len(query._exact_filters) == 1
        assert query._limit == 10

        # Test SQL preview generation
        sql = query._generate_sql_preview()
        assert "machine learning" in sql
        assert "category" in sql

    def test_complex_aggregation_query(self, mock_db):
        """Test complex query with aggregations."""
        builder = QueryBuilder(mock_db)

        query = (builder
                 .search("AI")
                 .group_by("category")
                 .count_by("*", "doc_count")
                 .avg_by("rating", "avg_rating")
                 .having("doc_count", "gt", 1)
                 .order_by("avg_rating", "desc"))

        # Verify query structure
        assert len(query._group_by) == 1
        assert len(query._aggregations) == 2
        assert len(query._having_clauses) == 1
        assert len(query._order_by) == 1

        # Check aggregations
        count_agg = next(agg for agg in query._aggregations if agg.function == "count")
        avg_agg = next(agg for agg in query._aggregations if agg.function == "avg")

        assert count_agg.alias == "doc_count"
        assert avg_agg.alias == "avg_rating"

    def test_semantic_filtering_integration(self, mock_db):
        """Test semantic filtering integration."""
        builder = QueryBuilder(mock_db)

        query = (builder
                 .search("machine learning")
                 .semantic_filter("methodology", "supervised learning", 0.8)
                 .filter("rating", gte_=4.0)
                 .return_type("chunks"))

        assert len(query._semantic_filters) == 1
        assert query._return_type == "chunks"

        sem_filter = query._semantic_filters[0]
        assert sem_filter.concept == "supervised learning"
        assert sem_filter.threshold == 0.8


class TestQueryBuilderErrorHandling:
    """Test QueryBuilder error handling and edge cases."""

    @pytest.fixture(scope="function")
    def mock_db(self):
        """Create a mock database for testing."""
        db = Mock()
        db.is_async_database.return_value = False
        return db

    @pytest.fixture(scope="function")
    def builder(self, mock_db):
        """Create a QueryBuilder instance for testing."""
        return QueryBuilder(mock_db)

    @patch('localvectordb.query_builder.QueryExecutor')
    def test_empty_query_execution(self, mock_executor_class, builder):
        """Test executing empty query."""
        # Empty query should still work (filter-only)
        # with patch('localvectordb.query_builder.QueryExecutor') as mock_executor_class:
        mock_executor = Mock()
        mock_executor.execute.return_value = []
        mock_executor_class.return_value = mock_executor

        results = builder.execute()
        assert results == []

    def test_multiple_search_clauses(self, builder):
        """Test multiple search clauses."""
        query = (builder
                 .search("machine learning")
                 .search("artificial intelligence"))

        assert len(query._search_clauses) == 2

    def test_conflicting_operations(self, builder):
        """Test potentially conflicting operations."""
        # This should work - return_type can be set multiple times
        query = (builder
                 .return_type("documents")
                 .return_type("chunks"))

        assert query._return_type == "chunks"

    def test_large_limit_values(self, builder):
        """Test with large limit values."""
        query = builder.limit(10000)
        assert query._limit == 10000

