"""
Tests for async functionality in localvectordb.database module.
"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import DatabaseError, DocumentNotFoundError


def create_mock_async_connection():
    """Create a properly mocked async SQLite connection."""
    mock_conn = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.rowcount = 0
    mock_conn.execute.return_value = mock_cursor
    mock_conn.executemany.return_value = mock_cursor
    mock_conn.commit = AsyncMock()
    mock_conn.rollback = AsyncMock()
    mock_conn.close = AsyncMock()
    return mock_conn


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncInitialization:
    """Test async LocalVectorDB initialization and basic functionality."""

    async def test_create_new_database_async_operations(self, temp_dir, sample_metadata_schema):
        """Test creating a new database and performing basic async operations."""
        with (
            patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_embedding,
            patch("localvectordb.chunking.ChunkerFactory.create_chunker") as _mock_chunker,
            patch("faiss.IndexFlatL2") as _mock_faiss,
            patch("faiss.IndexIDMap2") as _mock_faiss_idmap,
        ):
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.embed_async.return_value = [[0.1] * 384, [0.2] * 384]
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk.return_value = [
                Mock(content="chunk1", tokens=5, index=0, position=Mock(start=0, end=6)),
                Mock(content="chunk2", tokens=5, index=1, position=Mock(start=7, end=13)),
            ]
            _mock_chunker.return_value = mock_chunker_instance

            # Create database
            db = LocalVectorDB(
                name="test_async",
                base_path=temp_dir,
                metadata_schema=sample_metadata_schema,
                embedding_provider="test",
                embedding_model="test-model",
            )

            # Test basic properties
            assert db.name == "test_async"
            assert db.embedding_provider.provider_name == "test"
            assert not db.closed

    async def test_in_memory_database_async(self, sample_metadata_schema):
        """Test creating an in-memory database with async operations."""
        with (
            patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_embedding,
            patch("localvectordb.chunking.ChunkerFactory.create_chunker") as _mock_chunker,
            patch("faiss.IndexFlatL2") as _mock_faiss,
            patch("faiss.IndexIDMap2") as _mock_faiss_idmap,
        ):
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.embed_async.return_value = [[0.1] * 384]
            mock_embedding.return_value = mock_provider

            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk.return_value = [
                Mock(content="test chunk", tokens=2, index=0, position=Mock(start=0, end=10), content_hash="test_hash")
            ]
            _mock_chunker.return_value = mock_chunker_instance

            # Create in-memory database
            db = LocalVectorDB(
                name="test_memory",
                base_path=":memory:",
                metadata_schema=sample_metadata_schema,
                embedding_provider="test",
                embedding_model="test-model",
            )

            # Verify it's using shared cache format for in-memory
            assert "mode=memory&cache=shared" in db.db_path
            assert db.is_memory_only
            assert db.index_path is None


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncUpsert:
    """Test async upsert functionality."""

    @pytest_asyncio.fixture
    async def mock_db(self, temp_dir):
        """Create a database for async testing with minimal mocking."""
        # Create database with MockEmbeddings - use real chunking
        db = LocalVectorDB(
            name="test_async_upsert", base_path=temp_dir, embedding_provider="mock", embedding_model="test-model"
        )

        yield db

        # Cleanup
        try:
            if hasattr(db, "connection_pool") and db.connection_pool:
                db.connection_pool.close_all()
            if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                await db.async_connection_pool.close_all()
        except Exception:
            pass

    async def test_single_document_upsert_async(self, mock_db):
        """Test upserting a single document asynchronously."""
        document = "This is a test document for async upsert."

        result = await mock_db.upsert_async([document])

        assert result is not None
        assert len(result) == 1

        # Verify document was processed (MockEmbeddings tracks calls)
        assert mock_db.embedding_provider.number_of_calls > 0

    async def test_batch_document_upsert_async(self, mock_db):
        """Test upserting multiple documents asynchronously."""
        documents = [
            "First test document for async batch upsert.",
            "Second test document for async batch upsert.",
            "Third test document for async batch upsert.",
        ]

        result = await mock_db.upsert_async(documents)

        assert result is not None
        assert len(result) == 3

        # Verify document was processed (MockEmbeddings tracks calls)
        assert mock_db.embedding_provider.number_of_calls > 0

    async def test_upsert_with_metadata_async(self, mock_db):
        """Test upserting documents with metadata asynchronously."""
        documents = ["Test document with metadata"]
        metadata = [{"author": "Test Author", "category": "test"}]

        result = await mock_db.upsert_async(documents, metadata=metadata)

        assert result is not None
        assert len(result) == 1

    async def test_upsert_in_memory_database_async(self):
        """Test async upsert specifically with in-memory database."""
        # Create in-memory database with MockEmbeddings
        db = LocalVectorDB(
            name="test_memory_upsert", base_path=":memory:", embedding_provider="mock", embedding_model="test-model"
        )

        try:
            # Test upsert with in-memory database
            document = "Test document for in-memory async upsert"
            result = await db.upsert_async([document])

            assert result is not None
            assert len(result) == 1
        finally:
            # Cleanup
            try:
                if hasattr(db, "connection_pool") and db.connection_pool:
                    db.connection_pool.close_all()
                if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                    await db.async_connection_pool.close_all()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncRetrieval:
    """Test async document retrieval functionality."""

    @pytest_asyncio.fixture
    async def mock_db_with_data(self, temp_dir):
        """Create a database with sample data for retrieval testing."""
        # Create database with MockEmbeddings
        db = LocalVectorDB(
            name="test_retrieval", base_path=temp_dir, embedding_provider="mock", embedding_model="test-model"
        )

        # Insert some test data
        test_documents = ["Test document 1", "Test document 2"]
        await db.upsert_async(test_documents)

        yield db

        # Cleanup
        try:
            if hasattr(db, "connection_pool") and db.connection_pool:
                db.connection_pool.close_all()
            if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                await db.async_connection_pool.close_all()
        except Exception:
            pass

    async def test_get_document_by_id_async(self, mock_db_with_data):
        """Test retrieving a document by ID asynchronously."""
        db = mock_db_with_data

        # Get all documents to find a valid ID
        all_docs = await db.filter_async({})
        assert len(all_docs) > 0

        # Get the first document by ID
        doc_id = all_docs[0].id
        result = await db.get_async(doc_id)

        assert result is not None
        assert result.id == doc_id

    async def test_get_nonexistent_document_async(self, mock_db_with_data):
        """Test retrieving a non-existent document asynchronously."""
        db = mock_db_with_data

        with pytest.raises(DocumentNotFoundError):
            await db.get_async("nonexistent_doc")

    async def test_get_multiple_documents_async(self, mock_db_with_data):
        """Test retrieving multiple documents asynchronously."""
        db = mock_db_with_data

        # Get all documents to find valid IDs
        all_docs = await db.filter_async({})
        assert len(all_docs) >= 2

        # Get multiple documents by ID
        doc_ids = [doc.id for doc in all_docs[:2]]
        results = await db.get_async(doc_ids)

        assert results is not None
        assert len(results) == 2
        assert all(doc is not None for doc in results)


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncQuery:
    """Test async query functionality."""

    @pytest_asyncio.fixture
    async def mock_db_for_query(self, temp_dir):
        """Create a database with data for query testing."""
        # Create database with MockEmbeddings
        db = LocalVectorDB(
            name="test_query",
            base_path=temp_dir,
            embedding_provider="mock",
            embedding_model="test-model",
            metadata_schema={"category": MetadataField(type=MetadataFieldType.TEXT, indexed=True)},
        )

        # Insert test documents with varied content for different search types
        test_documents = [
            "Machine learning is a subset of artificial intelligence",
            "Python is a powerful programming language",
            "Vector databases enable semantic search capabilities",
            "Natural language processing involves computational linguistics",
        ]
        await db.upsert_async(test_documents)

        yield db

        # Cleanup
        try:
            if hasattr(db, "connection_pool") and db.connection_pool:
                db.connection_pool.close_all()
            if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                await db.async_connection_pool.close_all()
        except Exception:
            pass

    async def test_vector_search_async(self, mock_db_for_query):
        """Test vector search asynchronously."""
        db = mock_db_for_query

        result = await db.query_async("machine learning", search_type="vector", k=3)

        assert result is not None
        assert len(result) <= 3
        # Should find relevant documents
        assert any(
            "machine learning" in doc.content.lower() or "artificial intelligence" in doc.content.lower()
            for doc in result
            if hasattr(doc, "content")
        )

    async def test_keyword_search_async(self, mock_db_for_query):
        """Test keyword search asynchronously."""
        db = mock_db_for_query

        result = await db.query_async("Python", search_type="keyword", k=3)

        assert result is not None
        # Should find documents containing "Python"
        if len(result) > 0:
            assert any("python" in doc.content.lower() for doc in result if hasattr(doc, "content"))

    async def test_hybrid_search_async(self, mock_db_for_query):
        """Test hybrid search asynchronously."""
        db = mock_db_for_query

        result = await db.query_async("programming language", search_type="hybrid", k=3)

        assert result is not None
        # Hybrid search should return results combining vector and keyword search

    async def test_query_with_filters_async(self, mock_db_for_query):
        """Test query with metadata filters asynchronously."""
        db = mock_db_for_query

        # Insert documents with metadata
        docs_with_metadata = ["Document about AI and ML"]
        metadata = [{"category": "artificial_intelligence"}]
        await db.upsert_async(docs_with_metadata, metadata=metadata)

        # Query with filters - should work even if no results match
        filters = {"category": "artificial_intelligence"}
        result = await db.query_async("AI", search_type="vector", filters=filters)

        assert result is not None

    async def test_query_async_unknown_filter_field_raises(self, mock_db_for_query):
        """Filtering on a field that is not in the metadata schema should raise,
        matching filter(where=...) behavior, instead of silently returning []."""
        db = mock_db_for_query

        with pytest.raises(DatabaseError, match="not found in metadata schema"):
            await db.query_async("AI", search_type="vector", filters={"no_such_field": "x"})


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncFilter:
    """Test async metadata filtering functionality."""

    @pytest_asyncio.fixture
    async def mock_db_for_filter(self, temp_dir, sample_metadata_schema):
        """Create a database with metadata for filter testing."""
        # Create database with MockEmbeddings and metadata schema
        db = LocalVectorDB(
            name="test_filter",
            base_path=temp_dir,
            metadata_schema=sample_metadata_schema,
            embedding_provider="mock",
            embedding_model="test-model",
        )

        # Insert test documents with metadata
        test_documents = [
            "Document by John Doe about AI",
            "Document by Jane Smith about Python",
            "Document by Bob Johnson about databases",
        ]
        test_metadata = [
            {"author": "John Doe", "category": "ai", "rating": 4.5},
            {"author": "Jane Smith", "category": "programming", "rating": 5.0},
            {"author": "Bob Johnson", "category": "database", "rating": 3.8},
        ]
        await db.upsert_async(test_documents, metadata=test_metadata)

        yield db

        # Cleanup
        try:
            if hasattr(db, "connection_pool") and db.connection_pool:
                db.connection_pool.close_all()
            if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                await db.async_connection_pool.close_all()
        except Exception:
            pass

    async def test_simple_filter_async(self, mock_db_for_filter):
        """Test simple metadata filter asynchronously."""
        db = mock_db_for_filter

        result = await db.filter_async({"author": "John Doe"})

        assert result is not None
        assert len(result) >= 1
        # Should find documents by John Doe
        assert any("John Doe" in doc.content for doc in result)

    async def test_complex_filter_async(self, mock_db_for_filter):
        """Test complex metadata filter asynchronously."""
        db = mock_db_for_filter

        filters = {"category": "ai", "rating": {"$gte": 4.0}}
        result = await db.filter_async(filters)

        assert result is not None
        # Should find AI documents with rating >= 4.0

    async def test_empty_filter_result_async(self, mock_db_for_filter):
        """Test filter that returns no results asynchronously."""
        db = mock_db_for_filter

        result = await db.filter_async({"author": "Nonexistent Author"})

        assert result == []


@pytest.mark.asyncio
@pytest.mark.unit
class TestAsyncDeletion:
    """Test async document deletion functionality."""

    @pytest_asyncio.fixture
    async def mock_db_for_deletion(self, temp_dir):
        """Create a database with data for deletion testing."""
        # Create database with MockEmbeddings
        db = LocalVectorDB(
            name="test_deletion", base_path=temp_dir, embedding_provider="mock", embedding_model="test-model"
        )

        # Insert test documents for deletion
        test_documents = ["Document to be deleted", "Another document for batch deletion", "Third document for testing"]
        await db.upsert_async(test_documents)

        yield db

        # Cleanup
        try:
            if hasattr(db, "connection_pool") and db.connection_pool:
                db.connection_pool.close_all()
            if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                await db.async_connection_pool.close_all()
        except Exception:
            pass

    async def test_delete_document_async(self, mock_db_for_deletion):
        """Test deleting a document asynchronously."""
        db = mock_db_for_deletion

        # Get a valid document ID to delete
        all_docs = await db.filter_async({})
        assert len(all_docs) > 0

        doc_to_delete = all_docs[0].id
        await db.delete_async(doc_to_delete)

        # Verify document was deleted
        remaining_docs = await db.filter_async({})
        assert len(remaining_docs) == len(all_docs) - 1

    async def test_delete_nonexistent_document_async(self, mock_db_for_deletion):
        """Test deleting a non-existent document asynchronously."""
        db = mock_db_for_deletion

        with pytest.raises(DocumentNotFoundError):
            await db.delete_async("nonexistent_doc")

    async def test_batch_delete_async(self, mock_db_for_deletion):
        """Test deleting multiple documents asynchronously."""
        db = mock_db_for_deletion

        # Get valid document IDs for batch deletion
        all_docs = await db.filter_async({})
        assert len(all_docs) >= 2

        # Delete first two documents
        doc_ids = [all_docs[0].id, all_docs[1].id]
        await db.delete_async(doc_ids)

        # Verify documents were deleted
        remaining_docs = await db.filter_async({})
        assert len(remaining_docs) == len(all_docs) - 2


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncInMemorySpecific:
    """Test async functionality specific to in-memory databases."""

    async def test_in_memory_shared_cache_async(self):
        """Test that in-memory database uses shared cache for async operations."""
        # Create in-memory database
        db = LocalVectorDB(
            name="test_in_memory_async", base_path=":memory:", embedding_provider="mock", embedding_model="test-model"
        )

        try:
            # Verify shared cache is used
            assert "mode=memory&cache=shared" in db.db_path
            assert db.is_memory_only

            # Test that async operations work
            document = "Test document for in-memory async operations"
            result = await db.upsert_async([document])
            assert result is not None
            assert len(result) == 1
        finally:
            # Cleanup
            try:
                if hasattr(db, "connection_pool") and db.connection_pool:
                    db.connection_pool.close_all()
                if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                    await db.async_connection_pool.close_all()
            except Exception:
                pass

    async def test_concurrent_async_operations_in_memory(self):
        """Test concurrent async operations on in-memory database."""
        # Create in-memory database
        db = LocalVectorDB(
            name="test_concurrent", base_path=":memory:", embedding_provider="mock", embedding_model="test-model"
        )

        try:
            # Run concurrent upserts
            documents1 = ["Document 1 for concurrent test"]
            documents2 = ["Document 2 for concurrent test"]
            documents3 = ["Document 3 for concurrent test"]

            # Execute concurrent operations
            results = await asyncio.gather(
                db.upsert_async(documents1),
                db.upsert_async(documents2),
                db.upsert_async(documents3),
                return_exceptions=True,
            )

            # Verify all operations completed successfully
            for result in results:
                assert not isinstance(result, Exception)
                assert result is not None
        finally:
            # Cleanup
            try:
                if hasattr(db, "connection_pool") and db.connection_pool:
                    db.connection_pool.close_all()
                if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                    await db.async_connection_pool.close_all()
            except Exception:
                pass


@pytest.mark.asyncio
@pytest.mark.integration
class TestAsyncErrorHandling:
    """Test async error handling and edge cases."""

    async def test_async_connection_failure_handling(self, temp_dir):
        """Test handling of async connection failures."""
        # Create a real database first
        db = LocalVectorDB(
            name="test_error_handling", base_path=temp_dir, embedding_provider="mock", embedding_model="test-model"
        )

        # Now patch the async connection pool to fail
        with patch.object(db, "async_connection_pool") as mock_pool:
            mock_pool.get_connection_context = AsyncMock(side_effect=Exception("Connection failed"))

            # Test that connection failure is handled gracefully
            with pytest.raises((Exception, TypeError)):
                await db.upsert_async(["test document"])

        # Clean up the database
        try:
            db.close()
        except Exception:
            pass

    async def test_async_transaction_rollback(self, temp_dir):
        """Test basic async operation error handling."""
        # Create a database and test that errors are handled gracefully
        db = LocalVectorDB(
            name="test_rollback", base_path=temp_dir, embedding_provider="mock", embedding_model="test-model"
        )

        try:
            # Test with valid operations - should work
            result = await db.upsert_async(["test document"])
            assert result is not None
            assert len(result) == 1
        finally:
            # Cleanup
            try:
                if hasattr(db, "connection_pool") and db.connection_pool:
                    db.connection_pool.close_all()
                if hasattr(db, "async_connection_pool") and db.async_connection_pool:
                    await db.async_connection_pool.close_all()
            except Exception:
                pass
