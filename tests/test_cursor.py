"""Tests for QueryCursor and streaming query functionality."""

import pytest

from localvectordb.core import MetadataField, MetadataFieldType, QueryResult
from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import CursorExpiredError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def db_with_docs(tmp_path):
    """Create a LocalVectorDB with test documents inserted."""
    db = LocalVectorDB(
        name="cursor_test",
        base_path=str(tmp_path),
        embedding_provider="mock",
        embedding_model="test-model",
        embedding_config={"dimension": 384},
        metadata_schema={
            "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "priority": MetadataField(type=MetadataFieldType.INTEGER),
        },
        chunk_size=100,
        chunk_overlap=0,
        enable_fts=True,
    )

    docs = [
        "The quick brown fox jumps over the lazy dog. A simple test of the vector database system.",
        "Machine learning and artificial intelligence are transforming modern technology.",
        "Python is a high-level programming language known for its simplicity and readability.",
        "Vector databases store embeddings and enable semantic search over documents.",
        "Natural language processing uses computational methods to analyze and understand text.",
        "Deep learning neural networks have achieved remarkable results in computer vision.",
        "Data science combines statistics, programming, and domain expertise for insights.",
        "Cloud computing provides scalable infrastructure for modern applications.",
    ]
    metadata = [
        {"category": "test", "priority": 1},
        {"category": "ai", "priority": 5},
        {"category": "programming", "priority": 3},
        {"category": "database", "priority": 4},
        {"category": "ai", "priority": 4},
        {"category": "ai", "priority": 5},
        {"category": "data", "priority": 3},
        {"category": "infrastructure", "priority": 2},
    ]

    db.upsert(documents=docs, metadata=metadata)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Unit tests for QueryCursor lifecycle
# ---------------------------------------------------------------------------


class TestCursorLifecycle:
    def test_cursor_close(self, db_with_docs):
        cursor = db_with_docs.query_cursor("test", search_type="vector", return_type="chunks", k=5)
        assert not cursor.closed
        cursor.close()
        assert cursor.closed

    def test_cursor_context_manager(self, db_with_docs):
        with db_with_docs.query_cursor("test", search_type="vector", return_type="chunks", k=5) as cursor:
            assert not cursor.closed
        assert cursor.closed

    def test_cursor_expired_raises(self, db_with_docs):
        cursor = db_with_docs.query_cursor("test", search_type="vector", return_type="chunks", k=5, cursor_ttl=0.001)
        import time

        time.sleep(0.05)  # Ensure TTL is well past
        with pytest.raises(CursorExpiredError):
            cursor.fetch_batch()

    def test_cursor_closed_raises(self, db_with_docs):
        cursor = db_with_docs.query_cursor("test", search_type="vector", return_type="chunks", k=5)
        cursor.close()
        with pytest.raises(CursorExpiredError):
            cursor.fetch_batch()

    def test_cursor_repr(self, db_with_docs):
        cursor = db_with_docs.query_cursor("test", search_type="vector", return_type="chunks", k=5)
        r = repr(cursor)
        assert "open" in r
        assert "return_type='chunks'" in r
        cursor.close()
        r = repr(cursor)
        assert "closed" in r


# ---------------------------------------------------------------------------
# Unit tests for QueryCursor chunk streaming
# ---------------------------------------------------------------------------


class TestCursorChunkStreaming:
    def test_fetch_batch_returns_results(self, db_with_docs):
        with db_with_docs.query_cursor(
            "machine learning", search_type="vector", return_type="chunks", k=5, batch_size=2
        ) as cursor:
            batch = cursor.fetch_batch()
            assert len(batch) > 0
            for r in batch:
                assert isinstance(r, QueryResult)
                assert r.type == "chunk"
                assert r.score > 0

    def test_fetch_batch_respects_batch_size(self, db_with_docs):
        with db_with_docs.query_cursor(
            "test", search_type="vector", return_type="chunks", k=10, batch_size=2
        ) as cursor:
            batch = cursor.fetch_batch(2)
            assert len(batch) <= 2

    def test_stream_yields_all_results(self, db_with_docs):
        with db_with_docs.query_cursor(
            "programming", search_type="vector", return_type="chunks", k=10, batch_size=3
        ) as cursor:
            total = 0
            for batch in cursor.stream(3):
                assert len(batch) > 0
                total += len(batch)
            assert total > 0
            assert cursor.is_exhausted

    def test_stream_individual(self, db_with_docs):
        with db_with_docs.query_cursor("programming", search_type="vector", return_type="chunks", k=5) as cursor:
            results = list(cursor.stream_individual())
            assert len(results) > 0
            for r in results:
                assert isinstance(r, QueryResult)

    def test_fetch_all(self, db_with_docs):
        with db_with_docs.query_cursor("data", search_type="vector", return_type="chunks", k=5) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0
            # Should be exhausted now
            assert cursor.is_exhausted
            # Fetching again should return empty
            assert cursor.fetch_batch() == []

    def test_remaining_decreases(self, db_with_docs):
        with db_with_docs.query_cursor(
            "test", search_type="vector", return_type="chunks", k=10, batch_size=3
        ) as cursor:
            initial = cursor.remaining
            assert initial > 0
            cursor.fetch_batch(3)
            assert cursor.remaining < initial


# ---------------------------------------------------------------------------
# Tests for document return type
# ---------------------------------------------------------------------------


class TestCursorDocumentStreaming:
    def test_document_return_type(self, db_with_docs):
        with db_with_docs.query_cursor(
            "machine learning", search_type="vector", return_type="documents", k=5, batch_size=2
        ) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0
            for r in results:
                assert r.type == "document"
                assert r.content  # Has content
                assert r.score > 0

    def test_document_scoring_metadata(self, db_with_docs):
        with db_with_docs.query_cursor("AI", search_type="vector", return_type="documents", k=5) as cursor:
            results = cursor.fetch_all()
            if results:
                r = results[0]
                assert "_scoring" in r.metadata
                scoring = r.metadata["_scoring"]
                assert "_aggregation_method" in scoring
                assert "_chunk_count" in scoring


# ---------------------------------------------------------------------------
# Tests for different search types
# ---------------------------------------------------------------------------


class TestCursorSearchTypes:
    def test_vector_search(self, db_with_docs):
        with db_with_docs.query_cursor("programming", search_type="vector", return_type="chunks", k=5) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0

    def test_keyword_search(self, db_with_docs):
        with db_with_docs.query_cursor("programming", search_type="keyword", return_type="chunks", k=5) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0

    def test_hybrid_search(self, db_with_docs):
        with db_with_docs.query_cursor(
            "programming language", search_type="hybrid", return_type="chunks", k=5
        ) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0


# ---------------------------------------------------------------------------
# Tests for context/enriched return types
# ---------------------------------------------------------------------------


class TestCursorContextEnriched:
    def test_context_return_type(self, db_with_docs):
        with db_with_docs.query_cursor(
            "machine learning", search_type="vector", return_type="context", k=5, batch_size=2
        ) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0
            for r in results:
                assert r.type == "context"

    def test_enriched_return_type(self, db_with_docs):
        with db_with_docs.query_cursor(
            "machine learning", search_type="vector", return_type="enriched", k=5, batch_size=2
        ) as cursor:
            results = cursor.fetch_all()
            assert len(results) > 0
            for r in results:
                assert r.type == "enriched"


# ---------------------------------------------------------------------------
# Tests for query_stream convenience methods
# ---------------------------------------------------------------------------


class TestQueryStream:
    def test_query_stream_sync(self, db_with_docs):
        total = 0
        for batch in db_with_docs.query_stream("test", search_type="vector", return_type="chunks", k=10, batch_size=3):
            assert isinstance(batch, list)
            total += len(batch)
        assert total > 0

    def test_query_stream_documents(self, db_with_docs):
        total = 0
        for batch in db_with_docs.query_stream("AI", search_type="vector", return_type="documents", k=5, batch_size=2):
            for r in batch:
                assert r.type == "document"
            total += len(batch)
        assert total > 0


# ---------------------------------------------------------------------------
# Tests for async cursor
# ---------------------------------------------------------------------------


class TestCursorAsync:
    @pytest.mark.asyncio
    async def test_query_cursor_async(self, db_with_docs):
        cursor = await db_with_docs.query_cursor_async("test", search_type="vector", return_type="chunks", k=5)
        async with cursor:
            results = await cursor.fetch_batch_async()
            assert len(results) > 0
            for r in results:
                assert isinstance(r, QueryResult)

    @pytest.mark.asyncio
    async def test_stream_async(self, db_with_docs):
        cursor = await db_with_docs.query_cursor_async(
            "test", search_type="vector", return_type="chunks", k=5, batch_size=2
        )
        total = 0
        async with cursor:
            async for batch in cursor.stream_async(2):
                total += len(batch)
        assert total > 0

    @pytest.mark.asyncio
    async def test_stream_individual_async(self, db_with_docs):
        cursor = await db_with_docs.query_cursor_async("test", search_type="vector", return_type="chunks", k=5)
        results = []
        async with cursor:
            async for result in cursor.stream_individual_async():
                results.append(result)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_fetch_all_async(self, db_with_docs):
        cursor = await db_with_docs.query_cursor_async("AI", search_type="vector", return_type="documents", k=5)
        async with cursor:
            results = await cursor.fetch_all_async()
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_stream_async(self, db_with_docs):
        total = 0
        async for batch in db_with_docs.query_stream_async(
            "test", search_type="vector", return_type="chunks", k=5, batch_size=2
        ):
            total += len(batch)
        assert total > 0


# ---------------------------------------------------------------------------
# Tests for consistency: cursor results should match regular query results
# ---------------------------------------------------------------------------


class TestCursorConsistency:
    def test_cursor_matches_query_vector_chunks(self, db_with_docs):
        """Cursor results should contain the same items as a regular query."""
        regular = db_with_docs.query("machine learning", search_type="vector", return_type="chunks", k=5)
        with db_with_docs.query_cursor("machine learning", search_type="vector", return_type="chunks", k=5) as cursor:
            streamed = cursor.fetch_all()

        # Both should return results
        assert len(regular) > 0
        assert len(streamed) > 0

        # The IDs should overlap significantly (order may vary slightly due to batching)
        regular_ids = {r.id for r in regular}
        streamed_ids = {r.id for r in streamed}
        overlap = regular_ids & streamed_ids
        assert len(overlap) > 0

    def test_cursor_matches_query_keyword_chunks(self, db_with_docs):
        regular = db_with_docs.query("programming language", search_type="keyword", return_type="chunks", k=5)
        with db_with_docs.query_cursor(
            "programming language", search_type="keyword", return_type="chunks", k=5
        ) as cursor:
            streamed = cursor.fetch_all()

        assert len(regular) > 0
        assert len(streamed) > 0

        regular_ids = {r.id for r in regular}
        streamed_ids = {r.id for r in streamed}
        overlap = regular_ids & streamed_ids
        assert len(overlap) > 0


# ---------------------------------------------------------------------------
# Tests for QueryBuilder cursor integration
# ---------------------------------------------------------------------------


class TestQueryBuilderCursor:
    def test_query_builder_cursor(self, db_with_docs):
        cursor = db_with_docs.query_builder().search("machine learning").limit(5).cursor(batch_size=2)
        with cursor:
            results = cursor.fetch_all()
            assert len(results) > 0

    def test_query_builder_stream(self, db_with_docs):
        total = 0
        for batch in db_with_docs.query_builder().search("test").limit(5).stream(batch_size=2):
            total += len(batch)
        assert total > 0

    @pytest.mark.asyncio
    async def test_query_builder_cursor_async(self, db_with_docs):
        cursor = await db_with_docs.query_builder().search("test").limit(5).cursor_async(batch_size=2)
        async with cursor:
            results = await cursor.fetch_all_async()
            assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_builder_stream_async(self, db_with_docs):
        total = 0
        async for batch in db_with_docs.query_builder().search("test").limit(5).stream_async(batch_size=2):
            total += len(batch)
        assert total > 0
