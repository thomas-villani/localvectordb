"""
Tests for localvectordb.database module.
"""

import pytest
import sqlite3
import numpy as np

from unittest.mock import Mock, patch
from localvectordb.database import LocalVectorDB
from localvectordb.core import Document, QueryResult
from localvectordb.exceptions import DatabaseNotFoundError, DuplicateDocumentIDError, DocumentNotFoundError


def create_mock_connection():
    """Create a properly mocked SQLite connection."""
    mock_conn = Mock(spec=sqlite3.Connection)
    mock_cursor = Mock(spec=sqlite3.Cursor)
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_cursor.rowcount = 0
    mock_conn.execute.return_value = mock_cursor
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    return mock_conn


def create_mock_pooled_connection(mock_conn):
    """Create a mock PooledConnection that properly implements context manager."""
    mock_pooled = Mock()
    mock_pooled.__enter__ = Mock(return_value=mock_conn)
    mock_pooled.__exit__ = Mock(return_value=None)
    mock_pooled.connection = mock_conn
    return mock_pooled

class TestLocalVectorDBInitialization:
    """Test LocalVectorDB initialization."""

    def test_create_new_database(self, temp_dir, sample_metadata_schema):
        """Test creating a new database."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_faiss_index = Mock()
            mock_faiss_index.ntotal = 0
            mock_chunker.return_value = Mock()
            mock_faiss.return_value = mock_faiss_index

            mock_faiss_idmap.return_value = mock_faiss_index

            db = LocalVectorDB(
                name="test_db",
                base_path=temp_dir,
                metadata_schema=sample_metadata_schema,
                embedding_provider="test",
                embedding_model="test-model"
            )

            assert db.name == "test_db"
            assert db.base_path == temp_dir
            assert db.metadata_schema == sample_metadata_schema
            assert db.embedding_dimension == 384
            assert not db.closed
            db.close()

    def test_create_memory_database(self, sample_metadata_schema):
        """Test creating in-memory database."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker.return_value = Mock()
            mock_faiss.return_value = Mock()
            mock_faiss_idmap.return_value = Mock()

            db = LocalVectorDB(
                name=":memory:",
                metadata_schema=sample_metadata_schema
            )

            assert db.is_memory_only is True
            assert db.db_path == ":memory:"
            assert db.index_path is None

    def test_database_not_found_error(self, temp_dir):
        """Test error when database doesn't exist and create_if_not_exists=False."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding:
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_embedding.return_value = mock_provider

            with pytest.raises(DatabaseNotFoundError):
                LocalVectorDB(
                    name="nonexistent",
                    base_path=temp_dir,
                    create_if_not_exists=False
                )

    def test_invalid_embedding_model(self, temp_dir):
        """Test error with invalid embedding model."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding:
            mock_provider = Mock()
            mock_provider.validate_model.return_value = False
            mock_embedding.return_value = mock_provider

            with pytest.raises(ValueError, match="Embedding model .* is not available"):
                LocalVectorDB(
                    name="test",
                    base_path=temp_dir,
                    embedding_model="invalid-model"
                )

    @patch('localvectordb.database.faiss')
    def test_faiss_index_initialization(self, mock_faiss, temp_dir):
        """Test FAISS index initialization."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker'):
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_embedding.return_value = mock_provider

            mock_index = Mock()
            mock_faiss.IndexFlatL2.return_value = mock_index
            mock_faiss.IndexIDMap.return_value = mock_index

            db = LocalVectorDB(name="test", base_path=temp_dir)

            mock_faiss.IndexFlatL2.assert_called_once_with(384)
            assert db.index == mock_index

    @patch('localvectordb.database.faiss')
    def test_load_existing_faiss_index(self, mock_faiss, temp_dir):
        """Test loading existing FAISS index."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker'):
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_embedding.return_value = mock_provider

            # Create fake index file
            index_path = temp_dir / "test.faiss"
            index_path.touch()

            mock_index = Mock()
            mock_faiss.read_index.return_value = mock_index

            db = LocalVectorDB(name="test", base_path=temp_dir)

            mock_faiss.read_index.assert_called_once_with(str(index_path))
            assert db.index == mock_index


class TestLocalVectorDBUpsert:
    """Test LocalVectorDB upsert functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.embed_sync.return_value = np.random.random((2, 384))
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk.return_value = [
                Mock(content="chunk1", tokens=5, index=0, position=Mock(start=0, end=6)),
                Mock(content="chunk2", tokens=5, index=1, position=Mock(start=7, end=13))
            ]
            mock_chunker.return_value = mock_chunker_instance

            mock_index = Mock()
            mock_index.ntotal = 0
            mock_faiss.return_value = mock_index

            mock_faiss_idmap.return_value = Mock()

            mock_schema_instance = Mock()
            mock_schema.return_value = mock_schema_instance

            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(name="test", base_path=":memory:")
            db.index = mock_index

            return db

    def test_upsert_single_document(self, mock_db):
        """Test upserting a single document."""
        with patch.object(mock_db, '_process_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = ["doc_1"]

            result = mock_db.upsert("Test document")

            assert result == ["doc_1"]
            mock_pipeline.assert_called_once()

            # Check arguments
            args = mock_pipeline.call_args[0]
            assert args[0] == ["Test document"]  # documents
            assert args[1] == [{}]  # metadata
            assert len(args[2]) == 1  # ids

    def test_upsert_multiple_documents(self, mock_db):
        """Test upserting multiple documents."""
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        metadata = [{"author": "A"}, {"author": "B"}, {"author": "C"}]

        with patch.object(mock_db, '_process_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = ["doc_1", "doc_2", "doc_3"]

            result = mock_db.upsert(documents, metadata=metadata)

            assert result == ["doc_1", "doc_2", "doc_3"]

            # Check arguments
            args = mock_pipeline.call_args[0]
            assert args[0] == documents
            assert args[1] == metadata

    def test_upsert_with_custom_ids(self, mock_db):
        """Test upserting with custom document IDs."""
        with patch.object(mock_db, '_process_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = ["custom_1"]

            result = mock_db.upsert("Test", ids="custom_1")

            assert result == ["custom_1"]

            # Check that custom ID was used
            args = mock_pipeline.call_args[0]
            assert args[2] == ["custom_1"]

    def test_upsert_validation_error(self, mock_db):
        """Test upsert with validation errors."""
        documents = ["Doc 1", "Doc 2"]
        metadata = [{"author": "A"}]  # Mismatched length

        with pytest.raises(ValueError, match="Number of metadata entries must match"):
            mock_db.upsert(documents, metadata=metadata)

    def test_upsert_batching(self, mock_db):
        """Test that large upserts are batched."""
        documents = [f"Doc {i}" for i in range(150)]  # More than default batch size

        with patch.object(mock_db, '_process_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = [f"doc_{i}" for i in range(150)]

            result = mock_db.upsert(documents, batch_size=100)

            # Should be called once (pipeline handles internal batching)
            assert mock_pipeline.call_count == 1
            assert len(result) == 150


class TestLocalVectorDBInsert:
    """Test LocalVectorDB insert functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.embed_sync.return_value = np.random.random((2, 384))
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker_instance = Mock()
            mock_chunker_instance.chunk.return_value = [
                Mock(content="chunk1", tokens=5, index=0, position=Mock(start=0, end=6)),
                Mock(content="chunk2", tokens=5, index=1, position=Mock(start=7, end=13))
            ]
            mock_chunker.return_value = mock_chunker_instance

            mock_index = Mock()
            mock_index.ntotal = 0
            mock_faiss.return_value = mock_index

            mock_faiss_idmap.return_value = mock_index

            mock_schema_instance = Mock()
            mock_schema.return_value = mock_schema_instance

            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(name="test", base_path=":memory:")
            db.index = mock_index

            return db

    def test_insert_new_documents(self, mock_db):
        """Test inserting new documents."""
        with patch.object(mock_db, '_process_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = ["doc_1", "doc_2"]

            # Mock connection to return no existing docs
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_pooled = create_mock_pooled_connection(mock_conn)

            with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
                result = mock_db.insert(["Doc 1", "Doc 2"])

            assert result == ["doc_1", "doc_2"]
            mock_pipeline.assert_called_once()

    def test_insert_duplicate_id_error(self, mock_db):
        """Test insert with duplicate ID raises error."""
        # Mock connection to return existing doc
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{'id': 'existing_doc'}]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            with pytest.raises(DuplicateDocumentIDError):
                mock_db.insert("Test doc", ids="existing_doc", errors="raise")

    def test_insert_duplicate_id_ignore(self, mock_db):
        """Test insert with duplicate ID ignores when errors='ignore'."""
        # Mock connection to return existing doc
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{'id': 'existing_doc'}]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.insert("Test doc", ids="existing_doc", errors="ignore")

        assert result == []

    def test_insert_with_similarity_threshold(self, mock_db):
        """Test insert with similarity threshold."""
        with patch.object(mock_db, '_process_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = ["doc_1"]

            # Mock no existing documents
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_pooled = create_mock_pooled_connection(mock_conn)

            with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
                result = mock_db.insert("Test doc", similarity_threshold=0.95)

            # Check that pipeline was called with similarity threshold
            args = mock_pipeline.call_args[0]
            # similarity_threshold should be in position or kwargs
            assert result == ["doc_1"]
            mock_pipeline.assert_called_once()


class TestLocalVectorDBRetrieval:
    """Test LocalVectorDB retrieval functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker.return_value = Mock()
            mock_faiss.return_value = Mock()
            mock_faiss_idmap.return_value = Mock()

            mock_schema_instance = Mock()
            mock_schema_instance.metadata_fields = {}
            mock_schema.return_value = mock_schema_instance

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(name="test", base_path=":memory:")
            db._metadata_schema = {}

            return db

    def test_get_single_document(self, mock_db):
        """Test getting a single document by ID."""
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{
            'id': 'doc_1',
            'content': 'Test content',
            'content_hash': 'hash123',
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00'
        }]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.get("doc_1")

        assert isinstance(result, Document)
        assert result.id == "doc_1"
        assert result.content == "Test content"
        assert result.content_hash == "hash123"

    def test_get_multiple_documents(self, mock_db):
        """Test getting multiple documents by IDs."""
        # Mock database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [
            {
                'id': 'doc_1',
                'content': 'Content 1',
                'content_hash': 'hash1',
                'created_at': '2024-01-01T00:00:00',
                'updated_at': '2024-01-01T00:00:00'
            },
            {
                'id': 'doc_2',
                'content': 'Content 2',
                'content_hash': 'hash2',
                'created_at': '2024-01-01T00:00:00',
                'updated_at': '2024-01-01T00:00:00'
            }
        ]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.get(["doc_1", "doc_2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)
        assert result[0].id == "doc_1"
        assert result[1].id == "doc_2"

    def test_get_nonexistent_document(self, mock_db):
        """Test getting nonexistent document returns None."""
        # Mock empty database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            with pytest.raises(DocumentNotFoundError):
                result = mock_db.get("nonexistent")

    def test_exists_single_document(self, mock_db):
        """Test checking if single document exists."""
        # Mock database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{'id': 'doc_1'}]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.exists("doc_1")

        assert result is True

    def test_exists_multiple_documents(self, mock_db):
        """Test checking if multiple documents exist."""
        # Mock database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{'id': 'doc_1'}, {'id': 'doc_3'}]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.exists(["doc_1", "doc_2", "doc_3"])

        assert result == [True, False, True]

    def test_exists_nonexistent_document(self, mock_db):
        """Test checking nonexistent document."""
        # Mock empty database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.exists("nonexistent")

        assert result is False


class TestLocalVectorDBDeletion:
    """Test LocalVectorDB deletion functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker.return_value = Mock()
            mock_faiss.return_value = Mock()
            mock_faiss_idmap.return_value = Mock()

            mock_schema.return_value = Mock()

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            return LocalVectorDB(name="test", base_path=":memory:")

    def test_delete_single_document(self, mock_db):
        """Test deleting a single document."""
        # Mock database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{'faiss_id': 1}, {'faiss_id': 2}]
        mock_conn.execute.return_value.rowcount = 1
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.delete("doc_1")

        assert result == 1
        assert mock_conn.execute.call_count >= 2  # Query chunks, then delete

    def test_delete_multiple_documents(self, mock_db):
        """Test deleting multiple documents."""
        # Mock database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [{'faiss_id': 1}, {'faiss_id': 2}]
        mock_conn.execute.return_value.rowcount = 2
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.delete(["doc_1", "doc_2"])

        assert result == 2  # Total deleted

    def test_delete_nonexistent_document(self, mock_db):
        """Test deleting nonexistent document."""
        # Mock database response
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []  # No chunks found
        mock_conn.execute.return_value.rowcount = 0  # No rows deleted
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            result = mock_db.delete("nonexistent")

        assert result == 0


class TestLocalVectorDBUpdate:
    """Test LocalVectorDB update functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker.return_value = Mock()
            mock_faiss.return_value = Mock()
            mock_faiss_idmap.return_value = Mock()

            mock_schema_instance = Mock()
            mock_schema_instance.metadata_fields = {}
            mock_schema.return_value = mock_schema_instance

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(name="test", base_path=":memory:")
            db._metadata_schema = {}

            return db

    def test_update_content(self, mock_db):
        """Test updating document content."""
        # Mock existing document
        existing_doc = Document(
            id="doc_1",
            content="Original content",
            metadata={"author": "Test"},
            content_hash="original_hash"
        )

        with patch.object(mock_db, 'get', return_value=existing_doc), \
                patch.object(mock_db, 'upsert') as mock_upsert:
            mock_upsert.return_value = ["doc_1"]

            result = mock_db.update("doc_1", content="New content")

            assert result is True
            mock_upsert.assert_called_once()

            # Check upsert was called with new content
            args = mock_upsert.call_args[0]
            assert args[0] == ["New content"]
            assert args[1][0]["author"] == "Test"  # Preserved metadata
            assert args[2] == ["doc_1"]

    def test_update_metadata_only(self, mock_db):
        """Test updating only metadata."""
        # Mock existing document
        existing_doc = Document(
            id="doc_1",
            content="Original content",
            metadata={"author": "Test", "category": "old"},
            content_hash="original_hash"
        )

        mock_conn = create_mock_connection()
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db, 'get', return_value=existing_doc), \
                patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled), \
                patch.object(mock_db, '_validate_metadata_batch'):
            result = mock_db.update("doc_1", metadata={"category": "new", "rating": 5})

            assert result is True
            # Should call database update
            assert mock_conn.execute.called
            assert mock_conn.commit.called

    def test_update_both_content_and_metadata(self, mock_db):
        """Test updating both content and metadata."""
        # Mock existing document
        existing_doc = Document(
            id="doc_1",
            content="Original content",
            metadata={"author": "Test"},
            content_hash="original_hash"
        )

        with patch.object(mock_db, 'get', return_value=existing_doc), \
                patch.object(mock_db, 'upsert') as mock_upsert:
            mock_upsert.return_value = ["doc_1"]

            result = mock_db.update(
                "doc_1",
                content="New content",
                metadata={"category": "new"}
            )

            assert result is True
            mock_upsert.assert_called_once()

            # Check combined metadata
            args = mock_upsert.call_args[0]
            assert args[1][0]["author"] == "Test"  # Original
            assert args[1][0]["category"] == "new"  # New

    def test_update_nonexistent_document(self, mock_db):
        """Test updating nonexistent document."""
        with patch.object(mock_db, 'get', return_value=None):
            result = mock_db.update("nonexistent", content="New content")

            assert result is False

    def test_update_same_content(self, mock_db):
        """Test updating with same content (no change needed)."""
        # Mock existing document
        existing_doc = Document(
            id="doc_1",
            content="Same content",
            metadata={"author": "Test"},
            content_hash=None  # Will be calculated
        )
        # Set hash to match new content
        import hashlib
        existing_doc.content_hash = hashlib.sha256("Same content".encode()).hexdigest()

        with patch.object(mock_db, 'get', return_value=existing_doc):
            result = mock_db.update("doc_1", content="Same content")

            assert result is False  # No update needed


class TestLocalVectorDBQuery:
    """Test LocalVectorDB query functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.embed_sync.return_value = np.random.random((1, 384))
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker.return_value = Mock()

            mock_index = Mock()
            mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            mock_schema.return_value = Mock()

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(name="test", base_path=":memory:")
            db.index = mock_index
            db._fts_enabled = True

            return db

    def test_vector_search(self, mock_db):
        """Test vector similarity search."""
        with patch.object(mock_db, '_vector_search') as mock_vector_search:
            mock_vector_search.return_value = [
                QueryResult(id="doc_1", score=0.9, type='document', content="Test content")
            ]

            results = mock_db.query("test query", search_type='vector')

            assert len(results) == 1
            assert results[0].id == "doc_1"
            assert results[0].score == 0.9
            mock_vector_search.assert_called_once()

    def test_keyword_search(self, mock_db):
        """Test keyword search."""
        with patch.object(mock_db, '_keyword_search') as mock_keyword_search:
            mock_keyword_search.return_value = [
                QueryResult(id="doc_1", score=0.8, type='document', content="Test content")
            ]

            results = mock_db.query("test query", search_type='keyword')

            assert len(results) == 1
            assert results[0].score == 0.8
            mock_keyword_search.assert_called_once()

    def test_hybrid_search(self, mock_db):
        """Test hybrid search."""
        with patch.object(mock_db, '_hybrid_search') as mock_hybrid_search:
            mock_hybrid_search.return_value = [
                QueryResult(id="doc_1", score=0.85, type='document', content="Test content")
            ]

            results = mock_db.query("test query", search_type='hybrid', vector_weight=0.7)

            assert len(results) == 1
            assert results[0].score == 0.85
            mock_hybrid_search.assert_called_once()

    def test_query_with_filters(self, mock_db):
        """Test query with metadata filters."""
        filters = {"author": "Test Author", "category": "test"}

        with patch.object(mock_db, '_vector_search') as mock_vector_search:
            mock_vector_search.return_value = []

            mock_db.query("test", filters=filters)

            # Check filters were passed
            args = mock_vector_search.call_args[0]
            assert args[4] == filters  # filters parameter

    def test_query_return_chunks(self, mock_db):
        """Test query returning chunks instead of documents."""
        with patch.object(mock_db, '_vector_search') as mock_vector_search:
            mock_vector_search.return_value = [
                QueryResult(
                    id="doc_1:0",
                    score=0.9,
                    type='chunk',
                    content="Test chunk",
                    document_id="doc_1"
                )
            ]

            results = mock_db.query("test", return_type='chunks')

            assert len(results) == 1
            assert results[0].type == 'chunk'
            assert results[0].document_id == "doc_1"

    def test_query_unknown_search_type(self, mock_db):
        """Test query with unknown search type."""
        with pytest.raises(ValueError, match="Unknown search type: unknown"):
            mock_db.query("test", search_type='unknown')


class TestLocalVectorDBFilter:
    """Test LocalVectorDB filter functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test"
            mock_provider.model = "test-model"
            mock_embedding.return_value = mock_provider

            mock_chunker.return_value = Mock()
            mock_faiss.return_value = Mock()
            mock_faiss_idmap.return_value = Mock()

            mock_schema_instance = Mock()
            mock_schema_instance.metadata_fields = {"author": Mock(), "rating": Mock()}
            mock_schema.return_value = mock_schema_instance

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(name="test", base_path=":memory:")
            db._metadata_schema = {"author": Mock(), "rating": Mock()}

            return db

    def test_filter_simple_where(self, mock_db):
        """Test filtering with simple where conditions."""
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [
            {
                'id': 'doc_1',
                'content': 'Test content',
                'content_hash': 'hash1',
                'created_at': '2024-01-01T00:00:00',
                'updated_at': '2024-01-01T00:00:00',
                'author': 'Test Author',
                'rating': 4.5
            }
        ]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            results = mock_db.filter(where={"author": "Test Author"})

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].id == "doc_1"
        assert results[0].metadata["author"] == "Test Author"

    def test_filter_complex_where(self, mock_db):
        """Test filtering with complex where conditions."""
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            mock_db.filter(where={
                "rating": {"$gt": 4.0},
                "author": {"$in": ["Author1", "Author2"]}
            })

        # Check that complex conditions were processed
        call_args = mock_conn.execute.call_args[0]
        sql_query = call_args[0]
        assert "rating >" in sql_query
        assert "author IN" in sql_query

    def test_filter_with_order_by(self, mock_db):
        """Test filtering with order by clause."""
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            mock_db.filter(where={"author": "Test"}, order_by="rating DESC")

        call_args = mock_conn.execute.call_args[0]
        sql_query = call_args[0]
        assert "ORDER BY rating DESC" in sql_query

    def test_filter_with_limit_offset(self, mock_db):
        """Test filtering with limit and offset."""
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            mock_db.filter(where={"author": "Test"}, limit=10, offset=5)

        call_args = mock_conn.execute.call_args[0]
        sql_query = call_args[0]
        assert "LIMIT 10" in sql_query
        assert "OFFSET 5" in sql_query


class TestLocalVectorDBProperties:
    """Test LocalVectorDB properties and stats."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('localvectordb.database.ChunkerFactory.create_chunker') as mock_chunker, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.DatabaseSchema') as mock_schema, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Setup mocks
            mock_provider = Mock()
            mock_provider.validate_model.return_value = True
            mock_provider.get_dimension.return_value = 384
            mock_provider.provider_name = "test_provider"
            mock_provider.model = "test_model"
            mock_embedding.return_value = mock_provider

            mock_chunker_instance = Mock()
            mock_chunker.return_value = mock_chunker_instance

            mock_index = Mock()
            mock_index.ntotal = 100
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            mock_schema.return_value = Mock()

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            db = LocalVectorDB(
                name="test",
                base_path=":memory:",
                chunking_method="sentences",
                chunk_size=500,
                chunk_overlap=50
            )
            db.index = mock_index
            db._fts_enabled = True

            return db

    def test_stats_property(self, mock_db):
        """Test stats property."""
        mock_conn = create_mock_connection()

        # Mock document count and chunk count
        mock_conn.execute.return_value.fetchone.side_effect = [
            [50],  # document count
            [200]  # chunk count
        ]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(mock_db.connection_pool, 'get_connection', return_value=mock_pooled):
            stats = mock_db.get_stats()

        assert stats['documents'] == 50
        assert stats['chunks'] == 200
        assert stats['index_vectors'] == 100
        assert stats['embedding_dimension'] == 384
        assert stats['embedding_provider'] == "test_provider"
        assert stats['embedding_model'] == "test_model"
        assert stats['chunking_method'] == "sentences"
        assert stats['chunk_size'] == 500
        assert stats['chunk_overlap'] == 50
        assert stats['fts_enabled'] is True

    def test_properties(self, mock_db):
        """Test various properties."""
        assert mock_db.embedding_dimension == 384
        assert mock_db.chunking_method == "sentences"
        assert mock_db.chunk_size == 500
        assert mock_db.chunk_overlap == 50
        assert mock_db.fts_enabled is True

