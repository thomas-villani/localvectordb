# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/test_async_database.py
"""
Tests for localvectordb.async_database module.
"""

import pytest
import asyncio
import tempfile
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor

from localvectordb.async_database import AsyncLocalVectorDB, create_async_vectordb
from localvectordb.core import Document, QueryResult, MetadataField, MetadataFieldType, ChunkPosition
from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import DatabaseError, DuplicateDocumentIDError


class TestAsyncLocalVectorDBInitialization:
    """Test AsyncLocalVectorDB initialization and setup."""

    def test_init_parameters_storage(self, isolated_db_path):
        """Test that initialization parameters are stored correctly."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'rating': MetadataField(type=MetadataFieldType.REAL, indexed=False)
        }

        db = AsyncLocalVectorDB(
            "test_db",
            base_path=isolated_db_path,
            metadata_schema=metadata_schema,
            embedding_provider="mock",
            embedding_model="test-model",
            chunk_size=256,
            chunk_overlap=20,
            max_workers=4
        )

        # Check that parameters are stored in _init_params
        assert db._init_params['name'] == "test_db"
        assert db._init_params['base_path'] == isolated_db_path
        assert db._init_params['metadata_schema'] == metadata_schema
        assert db._init_params['embedding_provider'] == "mock"
        assert db._init_params['embedding_model'] == "test-model"
        assert db._init_params['chunk_size'] == 256
        assert db._init_params['chunk_overlap'] == 20

        # Check initial state
        assert not db._initialized
        assert not db._closed
        assert db._sync_db is None
        assert isinstance(db._executor, ThreadPoolExecutor)

    def test_init_with_custom_executor(self, isolated_db_path):
        """Test initialization with custom executor."""
        custom_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CustomTest")

        db = AsyncLocalVectorDB(
            "test_db",
            base_path=isolated_db_path,
            executor=custom_executor
        )

        assert db._executor is custom_executor
        assert not db._owns_executor

    def test_init_auto_executor(self, isolated_db_path):
        """Test initialization with auto-created executor."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)

        assert isinstance(db._executor, ThreadPoolExecutor)
        assert db._owns_executor

    def test_property_access_before_initialization(self, isolated_db_path):
        """Test that certain properties raise errors before initialization."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)

        # These should work (from _init_params)
        assert db.name == "test_db"
        assert db.embedding_model == "nomic-embed-text"  # default
        assert db.chunk_size == 500  # default
        assert db.closed is True  # Since _sync_db is None

        # These should raise errors (require initialization)
        with pytest.raises(DatabaseError, match="Database not initialized"):
            _ = db.embedding_dimension

        with pytest.raises(DatabaseError, match="Database not initialized"):
            _ = db.fts_enabled

        with pytest.raises(DatabaseError, match="Database not initialized"):
            _ = db.metadata_schema


class TestAsyncLocalVectorDBLazyInitialization:
    """Test lazy initialization behavior."""

    @pytest.mark.asyncio
    async def test_ensure_initialized_creates_sync_db(self, isolated_db_path, mock_embeddings):
        """Test that _ensure_initialized creates the sync database."""
        db = AsyncLocalVectorDB(
            "test_db",
            base_path=isolated_db_path,
            embedding_provider="mock"
        )

        assert not db._initialized
        assert db._sync_db is None

        # Mock the LocalVectorDB creation to avoid actual initialization
        with patch.object(db, '_create_sync_db') as mock_create:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_create.return_value = mock_sync_db

            await db._ensure_initialized()

            assert db._initialized
            assert db._sync_db is mock_sync_db
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_thread_safety(self, isolated_db_path):
        """Test that _ensure_initialized is thread-safe."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)

        init_call_count = 0

        def mock_create_sync_db():
            nonlocal init_call_count
            init_call_count += 1
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            return mock_sync_db

        with patch.object(db, '_create_sync_db', side_effect=mock_create_sync_db):
            # Call initialization multiple times concurrently
            tasks = [db._ensure_initialized() for _ in range(5)]
            await asyncio.gather(*tasks)

            # Should only initialize once
            assert init_call_count == 1
            assert db._initialized

    @pytest.mark.asyncio
    async def test_ensure_initialized_handles_closed_database(self, isolated_db_path):
        """Test that _ensure_initialized raises error for closed database."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
        db._closed = True

        with pytest.raises(DatabaseError, match="Database has been closed"):
            await db._ensure_initialized()

    @pytest.mark.asyncio
    async def test_ensure_initialized_handles_creation_error(self, isolated_db_path):
        """Test that _ensure_initialized handles creation errors."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)

        with patch.object(db, '_create_sync_db', side_effect=Exception("Creation failed")):
            with pytest.raises(DatabaseError, match="Database initialization failed"):
                await db._ensure_initialized()


class TestAsyncLocalVectorDBContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, isolated_db_path):
        """Test async context manager entry and exit."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_local_db_class.return_value = mock_sync_db

            async with AsyncLocalVectorDB("test_db", base_path=isolated_db_path) as db:
                assert db._initialized
                assert db._sync_db is mock_sync_db
                assert not db._closed

            # After exiting context, close should be called
            assert db._closed or db._executor._shutdown

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self, isolated_db_path):
        """Test that context manager properly handles exceptions."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_local_db_class.return_value = mock_sync_db

            with pytest.raises(ValueError, match="test error"):
                async with AsyncLocalVectorDB("test_db", base_path=isolated_db_path) as db:
                    assert db._initialized
                    raise ValueError("test error")


class TestAsyncLocalVectorDBDocumentOperations:
    """Test async document operations."""

    @pytest.fixture
    def initialized_db(self, isolated_db_path):
        """Create an initialized AsyncLocalVectorDB for testing."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {}

            # Add this line to your mock_sync_db setup in initialized_db fixture:
            mock_sync_db._generate_doc_id = Mock(return_value="doc_1")

            # Mock the connection_pool and database connection
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)

            # Mock cursor that is returned by execute()
            mock_cursor = Mock()
            mock_cursor.fetchall = Mock(return_value=[])  # Empty list = no existing documents
            mock_cursor.fetchone = Mock(return_value=None)
            mock_connection.execute = Mock(return_value=mock_cursor)

            mock_connection.fetchone = Mock(return_value=None)

            mock_connection_pool = Mock()
            mock_connection_pool.get_connection = Mock(return_value=mock_connection)
            mock_sync_db.connection_pool = mock_connection_pool

            mock_local_db_class.return_value = mock_sync_db

            db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
            # Manually set up the initialized state to avoid async issues in fixture
            db._sync_db = mock_sync_db
            db._initialized = True
            db._init_params['embedding_dimension'] = 384
            db._init_params['fts_enabled'] = True

            yield db

    @pytest.mark.asyncio
    async def test_upsert_single_document(self, initialized_db):
        """Test upserting a single document."""
        # Mock the chunking and embedding generation
        mock_chunks = [Mock(content="test chunk", index=0)]
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_mapping = [0]

        with patch.object(initialized_db, '_generate_chunks_with_mapping', return_value=(mock_chunks, mock_mapping)):
            with patch.object(initialized_db, '_generate_embeddings_async', return_value=mock_embeddings):
                with patch.object(initialized_db._sync_db, '_upsert_with_precomputed_embeddings', return_value=["doc_1"]):

                    result = await initialized_db.upsert(["Test document"], ids=["doc_1"])
                    assert result == ["doc_1"]

    @pytest.mark.asyncio
    async def test_upsert_multiple_documents(self, initialized_db):
        """Test upserting multiple documents."""
        documents = ["Doc 1", "Doc 2"]
        metadata = [{"author": "A"}, {"author": "B"}]
        ids = ["custom_1", "custom_2"]

        mock_chunks = [Mock(content="chunk 1", index=0), Mock(content="chunk 2", index=1)]
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_mapping = [0, 1]

        with patch.object(initialized_db, '_generate_chunks_with_mapping', return_value=(mock_chunks, mock_mapping)):
            with patch.object(initialized_db, '_generate_embeddings_async', return_value=mock_embeddings):
                with patch.object(initialized_db._sync_db, '_upsert_with_precomputed_embeddings', return_value=ids):

                    result = await initialized_db.upsert(documents, metadata=metadata, ids=ids)

                    assert result == ids

    @pytest.mark.asyncio
    async def test_insert_documents(self, initialized_db):
        """Test inserting new documents."""
        mock_chunks = [Mock(content="test chunk", index=0)]
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_mapping = [0]

        with patch.object(initialized_db, '_generate_chunks_with_mapping', return_value=(mock_chunks, mock_mapping)):
            with patch.object(initialized_db, '_generate_embeddings_async', return_value=mock_embeddings):
                with patch.object(initialized_db._sync_db, 'insert', return_value=["doc_100"]):

                    result = await initialized_db.insert(["Test document"], ids=["doc_100"], errors="raise")

                    assert result == ["doc_100"]
                    initialized_db._sync_db.insert.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_single_document(self, initialized_db):
        """Test getting a single document."""
        mock_doc = Document(
            id="doc_1",
            content="Test content",
            metadata={"author": "Test"},
            content_hash="hash123"
        )

        with patch.object(initialized_db._sync_db, 'get', return_value=mock_doc):
            result = await initialized_db.get("doc_1")

            assert result == mock_doc
            initialized_db._sync_db.get.assert_called_once_with("doc_1")

    @pytest.mark.asyncio
    async def test_get_multiple_documents(self, initialized_db):
        """Test getting multiple documents."""
        mock_docs = [
            Document(id="doc_1", content="Content 1", metadata={}, content_hash="hash1"),
            Document(id="doc_2", content="Content 2", metadata={}, content_hash="hash2")
        ]

        with patch.object(initialized_db._sync_db, 'get', return_value=mock_docs):
            result = await initialized_db.get(["doc_1", "doc_2"])

            assert result == mock_docs
            initialized_db._sync_db.get.assert_called_once_with(["doc_1", "doc_2"])

    @pytest.mark.asyncio
    async def test_exists_documents(self, initialized_db):
        """Test checking document existence."""
        with patch.object(initialized_db._sync_db, 'exists', return_value=[True, False]):
            result = await initialized_db.exists(["doc_1", "doc_2"])

            assert result == [True, False]
            initialized_db._sync_db.exists.assert_called_once_with(["doc_1", "doc_2"])

    @pytest.mark.asyncio
    async def test_delete_documents(self, initialized_db):
        """Test deleting documents."""
        with patch.object(initialized_db._sync_db, 'delete', return_value=2):
            result = await initialized_db.delete(["doc_1", "doc_2"])

            assert result == 2
            initialized_db._sync_db.delete.assert_called_once_with(["doc_1", "doc_2"])

    @pytest.mark.asyncio
    async def test_update_document(self, initialized_db):
        """Test updating a document."""
        with patch.object(initialized_db._sync_db, 'update', return_value=True):
            result = await initialized_db.update(
                "doc_1",
                content="New content",
                metadata={"author": "New Author"}
            )

            assert result is True
            initialized_db._sync_db.update.assert_called_once()


class TestAsyncLocalVectorDBQuery:
    """Test async query functionality."""

    @pytest.fixture
    def initialized_db(self, isolated_db_path):
        """Create an initialized AsyncLocalVectorDB for testing."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {}
            mock_local_db_class.return_value = mock_sync_db

            db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
            # Manually set up the initialized state to avoid async issues in fixture
            db._sync_db = mock_sync_db
            db._initialized = True
            db._init_params['embedding_dimension'] = 384
            db._init_params['fts_enabled'] = True

            yield db

    @pytest.mark.asyncio
    async def test_vector_query(self, initialized_db):
        """Test vector similarity query."""
        mock_results = [
            QueryResult(
                id="doc_1",
                score=0.95,
                type="document",
                content="Test content",
                metadata={"author": "Test"}
            )
        ]

        mock_embedding = np.array([[0.1, 0.2, 0.3]])

        with patch.object(initialized_db, '_generate_embeddings_async', return_value=mock_embedding):
            with patch.object(initialized_db._sync_db, '_search_with_embedding', return_value=mock_results):

                results = await initialized_db.query("test query", search_type="vector", k=5)

                assert len(results) == 1
                assert results[0].id == "doc_1"
                assert results[0].score == 0.95

                initialized_db._generate_embeddings_async.assert_called_once_with(["test query"])
                initialized_db._sync_db._search_with_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_keyword_query(self, initialized_db):
        """Test keyword search query."""
        mock_results = [
            QueryResult(
                id="doc_1:0",
                score=0.85,
                type="chunk",
                content="Test chunk",
                metadata={"author": "Test"},
                document_id="doc_1",
                position=ChunkPosition(start=0, end=10, line=1, column=1, end_line=1, end_column=10)
            )
        ]

        with patch.object(initialized_db._sync_db, 'query', return_value=mock_results):
            results = await initialized_db.query(
                "test",
                search_type="keyword",
                return_type="chunks"
            )

            assert len(results) == 1
            assert results[0].type == "chunk"
            assert results[0].document_id == "doc_1"

            initialized_db._sync_db.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_query(self, initialized_db):
        """Test hybrid search query."""
        mock_results = []
        mock_embedding = np.array([[0.1, 0.2, 0.3]])

        with patch.object(initialized_db, '_generate_embeddings_async', return_value=mock_embedding):
            with patch.object(initialized_db._sync_db, '_search_with_embedding', return_value=mock_results):

                await initialized_db.query(
                    "test query",
                    search_type="hybrid",
                    vector_weight=0.7,
                    score_threshold=0.5,
                    filters={"author": "Test"}
                )

                initialized_db._sync_db._search_with_embedding.assert_called_once()
                call_kwargs = initialized_db._sync_db._search_with_embedding.call_args.kwargs
                assert call_kwargs["search_type"] == "hybrid"
                assert call_kwargs["vector_weight"] == 0.7
                assert call_kwargs["score_threshold"] == 0.5
                assert call_kwargs["filters"] == {"author": "Test"}

    @pytest.mark.asyncio
    async def test_filter_documents(self, initialized_db):
        """Test filtering documents."""
        mock_docs = [
            Document(
                id="doc_1",
                content="Test content",
                metadata={"author": "Test"},
                content_hash="hash1"
            )
        ]

        with patch.object(initialized_db._sync_db, 'filter', return_value=mock_docs):
            results = await initialized_db.filter(
                where={"author": "Test"},
                order_by="created_at DESC",
                limit=10,
                offset=5
            )

            assert len(results) == 1
            assert results[0].id == "doc_1"

            initialized_db._sync_db.filter.assert_called_once()
            call_kwargs = initialized_db._sync_db.filter.call_args.kwargs
            assert call_kwargs["where"] == {"author": "Test"}
            assert call_kwargs["order_by"] == "created_at DESC"
            assert call_kwargs["limit"] == 10
            assert call_kwargs["offset"] == 5


class TestAsyncLocalVectorDBEmbeddings:
    """Test async embedding generation."""

    @pytest.fixture
    def initialized_db(self, isolated_db_path):
        """Create an initialized AsyncLocalVectorDB for testing."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {}

            # Mock the connection_pool and database connection
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            mock_connection.execute = Mock()
            mock_connection.fetchall = Mock(return_value=[])
            mock_connection.fetchone = Mock(return_value=None)

            mock_connection_pool = Mock()
            mock_connection_pool.get_connection = Mock(return_value=mock_connection)
            mock_sync_db.connection_pool = mock_connection_pool

            mock_local_db_class.return_value = mock_sync_db

            db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
            # Manually set up the initialized state to avoid async issues in fixture
            db._sync_db = mock_sync_db
            db._initialized = True
            db._init_params['embedding_dimension'] = 384
            db._init_params['fts_enabled'] = True

            yield db

    @pytest.mark.asyncio
    async def test_generate_embeddings_async_with_async_provider(self, initialized_db):
        """Test async embedding generation with async-capable provider."""
        # Mock an async embedding provider
        from localvectordb.embeddings import MockEmbeddings
        mock_provider = MockEmbeddings("mock-model", dimension=3)
        initialized_db._sync_db.embedding_provider = mock_provider

        texts = ["text 1", "text 2"]
        result = await initialized_db._generate_embeddings_async(texts)

        assert result.shape == (2, 3)
        assert mock_provider.number_of_calls > 0


    @pytest.mark.asyncio
    async def test_generate_embeddings_async_batched(self, initialized_db):
        """Test batched async embedding generation."""
        from localvectordb.embeddings import MockEmbeddings
        mock_provider = MockEmbeddings("mock-model", dimension=3)

        initialized_db._sync_db.embedding_provider = mock_provider

        # Large number of texts to trigger batching
        texts = [f"text {i}" for i in range(150)]
        result = await initialized_db._generate_embeddings_async(texts, batch_size=50)

        assert result.shape == (150, 3)


class TestAsyncLocalVectorDBChunking:
    """Test async chunking functionality."""

    @pytest.fixture
    def initialized_db(self, isolated_db_path):
        """Create an initialized AsyncLocalVectorDB for testing."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {}
            mock_local_db_class.return_value = mock_sync_db

            db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
            # Manually set up the initialized state to avoid async issues in fixture
            db._sync_db = mock_sync_db
            db._initialized = True
            db._init_params['embedding_dimension'] = 384
            db._init_params['fts_enabled'] = True

            yield db

    @pytest.mark.asyncio
    async def test_chunk_documents_async(self, initialized_db):
        """Test async document chunking."""
        from localvectordb.core import Chunk, ChunkPosition

        mock_chunks = [
            Chunk(
                content="chunk 1",
                position=ChunkPosition(start=0, end=7, line=1, column=1, end_line=1, end_column=7),
                tokens=2,
                index=0
            ),
            Chunk(
                content="chunk 2",
                position=ChunkPosition(start=8, end=15, line=1, column=9, end_line=1, end_column=15),
                tokens=2,
                index=1
            )
        ]

        initialized_db._sync_db.chunker = Mock()
        initialized_db._sync_db.chunker.chunk = Mock(
            side_effect=[mock_chunks[:1], mock_chunks[1:]]  # Return one chunk per document
        )

        documents = ["doc 1", "doc 2"]
        chunks, mapping = initialized_db._generate_chunks_with_mapping(documents)

        assert len(chunks) == 2
        assert mapping == [0, 1]
        assert initialized_db._sync_db.chunker.chunk.call_count == 2


class TestAsyncLocalVectorDBUtilityMethods:
    """Test utility methods and properties."""

    @pytest.fixture
    def initialized_db(self, isolated_db_path):
        """Create an initialized AsyncLocalVectorDB for testing."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
            }
            mock_sync_db.get_stats = Mock(return_value={
                "documents": 100,
                "chunks": 500,
                "index_vectors": 500
            })
            mock_local_db_class.return_value = mock_sync_db

            db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
            # Manually set up the initialized state to avoid async issues in fixture
            db._sync_db = mock_sync_db
            db._initialized = True
            db._init_params['embedding_dimension'] = 384
            db._init_params['fts_enabled'] = True

            yield db

    def test_properties_after_initialization(self, initialized_db):
        """Test properties after database initialization."""
        assert initialized_db.embedding_dimension == 384
        assert initialized_db.fts_enabled is True
        assert isinstance(initialized_db.metadata_schema, dict)
        assert 'author' in initialized_db.metadata_schema

    def test_get_stats(self, initialized_db):
        """Test get_stats method."""
        stats = initialized_db.get_stats()

        assert stats["documents"] == 100
        assert stats["chunks"] == 500
        assert stats["index_vectors"] == 500

    def test_is_async_database(self, initialized_db):
        """Test is_async_database method."""
        assert initialized_db.is_async_database() is True

    def test_supports_async_embeddings_true(self, initialized_db):
        """Test supports_async_embeddings when provider has async support."""
        initialized_db._sync_db.embedding_provider = Mock()
        initialized_db._sync_db.embedding_provider.embed_async = AsyncMock()

        assert initialized_db.supports_async_embeddings() is True

    def test_supports_async_embeddings_false(self, initialized_db):
        """Test supports_async_embeddings when provider lacks async support."""
        initialized_db._sync_db.embedding_provider = Mock()
        delattr(initialized_db._sync_db.embedding_provider, "embed_async")
        print("hasattr?", hasattr(initialized_db._sync_db.embedding_provider, "embed_async"))
        # No embed_async method

        assert initialized_db.supports_async_embeddings() is False

    @pytest.mark.asyncio
    async def test_save_method(self, initialized_db):
        """Test save method."""
        initialized_db._sync_db.save = Mock()

        await initialized_db.save()

        initialized_db._sync_db.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_method(self, initialized_db):
        """Test close method."""
        await initialized_db.close()

        # Should mark as closed and shutdown executor
        assert initialized_db._closed or initialized_db._executor._shutdown

    @pytest.mark.asyncio
    async def test_update_metadata_schema(self, initialized_db):
        """Test update_metadata_schema method."""
        new_schema = {
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
        }

        initialized_db._sync_db.update_metadata_schema = Mock(return_value={"updated": True})

        result = await initialized_db.update_metadata_schema(new_schema, drop_columns=False)

        assert result == {"updated": True}
        initialized_db._sync_db.update_metadata_schema.assert_called_once_with(
            new_schema=new_schema,
            drop_columns=False
        )

    @pytest.mark.asyncio
    async def test_get_metadata_schema_info(self, initialized_db):
        """Test get_metadata_schema_info method."""
        schema_info = {"schema": "info"}
        initialized_db._sync_db.get_metadata_schema_info = Mock(return_value=schema_info)

        result = await initialized_db.get_metadata_schema_info()

        assert result == schema_info
        initialized_db._sync_db.get_metadata_schema_info.assert_called_once()


class TestAsyncLocalVectorDBErrorHandling:
    """Test error handling in AsyncLocalVectorDB."""

    @pytest.mark.asyncio
    async def test_method_calls_before_initialization(self, isolated_db_path):
        """Test that methods raise errors when called before initialization."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
        # These methods should fail because database is not initialized
        with patch.object(db, '_create_sync_db', side_effect=Exception("Init error")):
            with pytest.raises(Exception):  # Will fail in _ensure_initialized
                await db.upsert("test")

    @pytest.mark.asyncio
    async def test_initialization_error_propagation(self, isolated_db_path):
        """Test that initialization errors are properly propagated."""
        db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)

        with patch.object(db, '_create_sync_db', side_effect=Exception("Init error")):
            with pytest.raises(DatabaseError, match="Database initialization failed"):
                await db._ensure_initialized()

    @pytest.mark.asyncio
    async def test_executor_error_handling(self, isolated_db_path):
        """Test error handling in executor operations."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.get.side_effect = Exception("Database error")
            mock_local_db_class.return_value = mock_sync_db

            db = AsyncLocalVectorDB("test_db", base_path=isolated_db_path)
            await db._ensure_initialized()

            with pytest.raises(Exception, match="Database error"):
                await db.get("doc_1")

            await db.close()


class TestCreateAsyncVectorDBFactory:
    """Test the create_async_vectordb factory function."""

    @pytest.mark.asyncio
    async def test_create_async_vectordb_basic(self, isolated_db_path):
        """Test basic factory function usage."""
        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 384
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {}
            mock_local_db_class.return_value = mock_sync_db

            db = await create_async_vectordb(
                "test_db",
                base_path=isolated_db_path,
                embedding_model="test-model"
            )

            assert isinstance(db, AsyncLocalVectorDB)
            assert db._initialized
            assert db.name == "test_db"
            assert db.embedding_model == "test-model"

            await db.close()

    @pytest.mark.asyncio
    async def test_create_async_vectordb_with_params(self, isolated_db_path):
        """Test factory function with various parameters."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
        }

        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 512
            mock_sync_db.fts_enabled = False
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = metadata_schema
            mock_local_db_class.return_value = mock_sync_db

            db = await create_async_vectordb(
                "test_db",
                base_path=isolated_db_path,
                metadata_schema=metadata_schema,
                chunk_size=256,
                enable_fts=False,
                max_workers=2
            )

            assert db._initialized
            assert db.chunk_size == 256
            assert db.fts_enabled is False

            await db.close()


class TestAsyncLocalVectorDBIntegration:
    """Integration tests for AsyncLocalVectorDB."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_with_mocked_dependencies(self, isolated_db_path):
        """Test a full workflow with mocked dependencies."""
        from localvectordb.core import Chunk, ChunkPosition

        # Setup mocks
        mock_chunks = [
            Chunk(
                content="This is a test document",
                position=ChunkPosition(start=0, end=23, line=1, column=1, end_line=1, end_column=23),
                tokens=5,
                index=0
            )
        ]
        mock_embeddings = np.array([[0.1, 0.2, 0.3]])
        mock_doc = Document(
            id="doc_1",
            content="This is a test document",
            metadata={"author": "Test"},
            content_hash="hash123"
        )
        mock_results = [
            QueryResult(
                id="doc_1",
                score=0.95,
                type="document",
                content="This is a test document",
                metadata={"author": "Test"}
            )
        ]

        with patch('localvectordb.async_database.LocalVectorDB') as mock_local_db_class:
            mock_sync_db = Mock(spec=LocalVectorDB)
            mock_sync_db.embedding_dimension = 3
            mock_sync_db.fts_enabled = True
            mock_sync_db.closed = False
            mock_sync_db.metadata_schema = {}

            # Mock cursor that is returned by execute()
            mock_cursor = Mock()
            # Mock the connection_pool and database connection
            mock_connection = Mock()
            mock_connection.__enter__ = Mock(return_value=mock_connection)
            mock_connection.__exit__ = Mock(return_value=None)
            mock_connection.execute = Mock()
            mock_connection.fetchall = Mock(return_value=[])
            mock_connection.fetchone = Mock(return_value=None)

            mock_connection_pool = Mock()
            mock_connection_pool.get_connection = Mock(return_value=mock_connection)

            mock_cursor.fetchall = Mock(return_value=[])  # Empty list = no existing documents
            mock_cursor.fetchone = Mock(return_value=None)
            mock_connection.execute = Mock(return_value=mock_cursor)
            mock_connection.fetchone = Mock(return_value=None)

            mock_sync_db.connection_pool = mock_connection_pool

            # Mock chunker
            mock_sync_db.chunker = Mock()
            mock_sync_db.chunker.chunk_documents = Mock(return_value=(mock_chunks, [0]))

            # Mock embedding provider
            mock_sync_db.embedding_provider = Mock()
            mock_sync_db.embedding_provider.embed_async = AsyncMock(return_value=mock_embeddings)

            # Mock database operations
            mock_sync_db._upsert_with_precomputed_embeddings = Mock(return_value=["doc_1"])
            mock_sync_db.get = Mock(return_value=mock_doc)
            mock_sync_db._search_with_embedding = Mock(return_value=mock_results)

            mock_local_db_class.return_value = mock_sync_db

            async with AsyncLocalVectorDB("test_db", base_path=isolated_db_path) as db:
                # Test upsert
                doc_ids = await db.upsert("This is a test document", metadata={"author": "Test"}, ids=["doc_1"])
                assert doc_ids == ["doc_1"]

                # Test get
                doc = await db.get("doc_1")
                assert doc.id == "doc_1"
                assert doc.content == "This is a test document"

                # Test query
                results = await db.query("test document", search_type="vector")
                assert len(results) == 1
                assert results[0].id == "doc_1"
                assert results[0].score == 0.95


# Test markers for different test types
pytestmark = [
    pytest.mark.asyncio,  # All tests in this module are async
]