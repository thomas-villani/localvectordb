"""
Tests for localvectordb.core module.
"""

import sqlite3
import threading
from datetime import datetime
from unittest.mock import Mock, patch
import pytest

from localvectordb.core import (
    MetadataField, MetadataFieldType, ChunkPosition, Chunk, Document,
    QueryResult
)
from localvectordb._pools import ConnectionPool, ReadWriteLock
from localvectordb._schema import DatabaseSchema


@pytest.mark.unit
class TestMetadataFieldType:
    """Test MetadataFieldType enum."""

    def test_enum_values(self):
        """Test enum values are correct."""
        assert MetadataFieldType.TEXT == "text"
        assert MetadataFieldType.INTEGER == "integer"
        assert MetadataFieldType.REAL == "real"
        assert MetadataFieldType.BOOLEAN == "boolean"
        assert MetadataFieldType.DATE == "date"
        assert MetadataFieldType.JSON == "json"


@pytest.mark.unit
class TestMetadataField:
    """Test MetadataField dataclass."""

    def test_create_with_type_enum(self):
        """Test creating field with MetadataFieldType enum."""
        field = MetadataField(type=MetadataFieldType.TEXT, indexed=True)
        assert field.type == MetadataFieldType.TEXT
        assert field.indexed is True
        assert field.required is False
        assert field.default_value is None

    def test_create_with_string_type(self):
        """Test creating field with string type."""
        field = MetadataField(type="integer", indexed=False, required=True)
        assert field.type == MetadataFieldType.INTEGER
        assert field.indexed is False
        assert field.required is True

    def test_create_with_python_type(self):
        """Test creating field with Python type."""
        field = MetadataField(type=str)
        assert field.type == MetadataFieldType.TEXT

        field = MetadataField(type=int)
        assert field.type == MetadataFieldType.INTEGER

        field = MetadataField(type=float)
        assert field.type == MetadataFieldType.REAL

        field = MetadataField(type=bool)
        assert field.type == MetadataFieldType.BOOLEAN

        field = MetadataField(type=dict)
        assert field.type == MetadataFieldType.JSON

        field = MetadataField(type=list)
        assert field.type == MetadataFieldType.JSON

    def test_with_default_value(self):
        """Test field with default value."""
        field = MetadataField(
            type=MetadataFieldType.TEXT,
            default_value="default_text",
            required=True
        )
        assert field.default_value == "default_text"
        assert field.required is True


@pytest.mark.unit
class TestChunkPosition:
    """Test ChunkPosition dataclass."""

    def test_create_position(self):
        """Test creating chunk position."""
        pos = ChunkPosition(start=10, end=50, line=2, column=5, end_line=2, end_column=40)
        assert pos.start == 10
        assert pos.end == 50
        assert pos.line == 2
        assert pos.column == 5

    def test_to_dict(self):
        """Test converting position to dict."""
        pos = ChunkPosition(start=10, end=50, line=2, column=5, end_line=2, end_column=40)
        expected = {'start': 10, 'end': 50, 'line': 2, 'column': 5, 'end_line': 2, 'end_column': 40}
        assert pos.to_dict() == expected

    def test_from_dict(self):
        """Test creating position from dict."""
        data = {'start': 10, 'end': 50, 'line': 2, 'column': 5, 'end_line': 2, 'end_column': 40}
        pos = ChunkPosition.from_dict(data)
        assert pos.start == 10
        assert pos.end == 50
        assert pos.line == 2
        assert pos.column == 5
        assert pos.end_line == 2
        assert pos.end_column == 40


@pytest.mark.unit
class TestChunk:
    """Test Chunk dataclass."""

    def test_create_chunk(self):
        """Test creating chunk."""
        position = ChunkPosition(start=0, end=10, line=1, column=1, end_line=1, end_column=10)
        chunk = Chunk(
            content="test content",
            position=position,
            tokens=2,
            index=0,
            faiss_id=42
        )
        assert chunk.content == "test content"
        assert chunk.position == position
        assert chunk.tokens == 2
        assert chunk.index == 0
        assert chunk.faiss_id == 42

    def test_get_context(self):
        """Test getting chunk context."""
        position = ChunkPosition(start=10, end=20, line=1, column=11, end_line=1, end_column=20)
        chunk = Chunk(
            content="test chunk",
            position=position,
            tokens=2,
            index=0
        )
        original = "This is a test chunk in a document"
        context = chunk.get_context(original, window=5)
        assert "test chunk" in context
        # Should include context around the chunk
        assert len(context) > len(chunk.content)

    def test_get_context_at_beginning(self):
        """Test getting context for chunk at beginning of document."""
        position = ChunkPosition(start=0, end=4, line=1, column=1, end_line=1, end_column=4)
        chunk = Chunk(
            content="This",
            position=position,
            tokens=1,
            index=0
        )
        original = "This is a test document"
        context = chunk.get_context(original, window=5)
        assert not context.startswith("...")
        assert "This" in context

    def test_get_context_at_end(self):
        """Test getting context for chunk at end of document."""
        original = "This is a test document"
        position = ChunkPosition(start=15, end=23, line=1, column=16, end_line=1, end_column=23)
        chunk = Chunk(
            content="document",
            position=position,
            tokens=1,
            index=0
        )
        context = chunk.get_context(original, window=5)
        assert not context.endswith("...")
        assert "document" in context

    def test_highlight_in_original(self):
        """Test highlighting chunk in original text."""
        position = ChunkPosition(start=10, end=15, line=1, column=11, end_line=1, end_column=15)
        chunk = Chunk(
            content="chunk",
            position=position,
            tokens=1,
            index=0
        )
        original = "This is a chunk in a document"
        highlighted = chunk.highlight_in_original(original)
        assert "This is a <<<chunk>>> in a document" == highlighted


@pytest.mark.unit
class TestDocument:
    """Test Document dataclass."""

    def test_create_document(self):
        """Test creating document."""
        doc = Document(
            id="test_id",
            content="test content",
            metadata={"author": "test"}
        )
        assert doc.id == "test_id"
        assert doc.content == "test content"
        assert doc.metadata == {"author": "test"}
        assert doc.content_hash is not None
        assert doc.created_at is None
        assert doc.updated_at is None

    def test_content_hash_calculation(self):
        """Test content hash is calculated correctly."""
        doc1 = Document(id="1", content="same content")
        doc2 = Document(id="2", content="same content")
        doc3 = Document(id="3", content="different content")

        assert doc1.content_hash == doc2.content_hash
        assert doc1.content_hash != doc3.content_hash

    def test_needs_update(self):
        """Test needs_update method."""
        doc = Document(id="test", content="original content")
        original_hash = doc.content_hash

        # Same content should not need update
        assert not doc.needs_update("original content")

        # Different content should need update
        assert doc.needs_update("new content")

        # Hash should remain unchanged
        assert doc.content_hash == original_hash

    def test_with_datetime_fields(self):
        """Test document with datetime fields."""
        now = datetime.now()
        doc = Document(
            id="test",
            content="test",
            created_at=now,
            updated_at=now
        )
        assert doc.created_at == now
        assert doc.updated_at == now


@pytest.mark.unit
class TestQueryResult:
    """Test QueryResult dataclass."""

    def test_create_document_result(self):
        """Test creating document query result."""
        result = QueryResult(
            id="doc_1",
            score=0.85,
            type='document',
            content="document content",
            metadata={"author": "test"}
        )
        assert result.id == "doc_1"
        assert result.score == 0.85
        assert result.type == 'document'
        assert result.content == "document content"
        assert result.metadata == {"author": "test"}
        assert result.document_id is None
        assert result.position is None

    def test_create_chunk_result(self):
        """Test creating chunk query result."""
        position = ChunkPosition(start=0, end=10, line=1, column=1, end_line=1, end_column=10)
        result = QueryResult(
            id="doc_1:0",
            score=0.75,
            type='chunk',
            content="chunk content",
            metadata={"author": "test"},
            document_id="doc_1",
            position=position,
            # highlights=[{"start": 0, "end": 5}]
        )
        assert result.id == "doc_1:0"
        assert result.type == 'chunk'
        assert result.document_id == "doc_1"
        assert result.position == position

    def test_get_context(self):
        """Test getting context for chunk result."""
        position = ChunkPosition(start=10, end=20, line=1, column=11, end_line=1, end_column=20)
        result = QueryResult(
            id="doc_1:0",
            score=0.75,
            type='chunk',
            content="test chunk",
            position=position
        )
        original = "This is a test chunk in a document"
        context = result.get_context(original, window=5)
        assert "test chunk" in context
        assert len(context) > len(result.content)

    def test_get_context_document_type(self):
        """Test getting context for document result returns None."""
        result = QueryResult(
            id="doc_1",
            score=0.85,
            type='document',
            content="document content"
        )
        assert result.get_context("original text") is None


@pytest.mark.unit
class TestDatabaseSchema:
    """Test DatabaseSchema class."""

    def test_create_schema(self, temp_dir):
        """Test creating database schema."""
        db_path = temp_dir / "test.db"
        read_write_lock = ReadWriteLock()
        schema = DatabaseSchema(db_path, read_write_lock=read_write_lock)
        assert schema.db_path == db_path
        assert schema.metadata_fields == {}

    @patch('sqlite3.connect')
    def test_initialize_base_schema(self, mock_connect, temp_dir):
        """Test initializing base schema."""
        mock_conn = Mock()
        # Correctly set up the context manager to return mock_conn
        mock_connect.return_value.__enter__.return_value = mock_conn

        db_path = temp_dir / "test.db"
        read_write_lock = ReadWriteLock()
        schema = DatabaseSchema(db_path, read_write_lock=read_write_lock)
        schema.initialize()

        # Check that base tables were created
        assert mock_conn.execute.call_count >= len(schema.BASE_SCHEMA)

        # Check that base indexes were created
        call_args = [call[0][0] for call in mock_conn.execute.call_args_list]
        for index_ddl in schema.BASE_INDEXES:
            assert any(index_ddl in arg for arg in call_args)

    @patch('sqlite3.connect')
    def test_initialize_with_metadata_schema(self, mock_connect, temp_dir, sample_metadata_schema):
        """Test initializing with metadata schema."""
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        db_path = temp_dir / "test.db"
        read_write_lock = ReadWriteLock()
        schema = DatabaseSchema(db_path, read_write_lock=read_write_lock)
        schema.initialize(sample_metadata_schema)

        # Should call _setup_metadata_schema
        assert mock_conn.execute.call_count > len(schema.BASE_SCHEMA)

    @patch('sqlite3.connect')
    def test_setup_metadata_schema(self, mock_connect, temp_dir):
        """Test setting up metadata schema."""
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        db_path = temp_dir / "test.db"
        read_write_lock = ReadWriteLock()
        schema = DatabaseSchema(db_path, read_write_lock=read_write_lock)

        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'rating': MetadataField(type=MetadataFieldType.REAL, default_value=0.0)
        }

        schema._setup_metadata_schema(mock_conn, metadata_schema)

        # Should insert schema definitions
        calls = mock_conn.execute.call_args_list
        insert_calls = [call for call in calls if 'INSERT OR REPLACE INTO metadata_schema' in str(call)]
        assert len(insert_calls) == len(metadata_schema)

    @patch('sqlite3.connect')
    def test_add_metadata_column(self, mock_connect, temp_dir):
        """Test adding metadata column."""
        mock_conn = Mock()

        # Mock table_info to return existing columns
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            (0, 'id', 'TEXT', 0, None, 1),
            (1, 'content', 'TEXT', 0, None, 0)
        ]
        mock_conn.execute.return_value = mock_cursor

        db_path = temp_dir / "test.db"
        read_write_lock = ReadWriteLock()
        schema = DatabaseSchema(db_path, read_write_lock=read_write_lock)

        field = MetadataField(type=MetadataFieldType.TEXT, indexed=True)
        schema._add_metadata_column(mock_conn, 'author', field)

        # Should add column and create index
        calls = [str(call) for call in mock_conn.execute.call_args_list]
        assert any('ALTER TABLE documents ADD COLUMN author' in call for call in calls)
        assert any('CREATE INDEX' in call and 'author' in call for call in calls)

    @patch('sqlite3.connect')
    def test_load_metadata_schema(self, mock_connect, temp_dir):
        """Test loading metadata schema from database."""
        mock_conn = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('author', 'text', True, False, '"default_author"'),
            ('rating', 'real', False, True, '0.0')
        ]
        mock_conn.execute.return_value = mock_cursor

        db_path = temp_dir / "test.db"
        read_write_lock = ReadWriteLock()
        schema = DatabaseSchema(db_path, read_write_lock=read_write_lock)
        loaded_schema = schema.load_metadata_schema()

        assert 'author' in loaded_schema
        assert 'rating' in loaded_schema
        assert loaded_schema['author'].type == MetadataFieldType.TEXT
        assert loaded_schema['author'].indexed is True
        assert loaded_schema['rating'].type == MetadataFieldType.REAL
        assert loaded_schema['rating'].required is True



@pytest.mark.unit
class TestConnectionPool:
    """Test ConnectionPool class."""

    def test_create_pool(self, temp_dir):
        """Test creating connection pool."""
        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path, max_connections=5)
        assert pool.db_path == db_path
        assert pool.max_connections == 5
        assert len(pool._pool) == 0

    @patch('sqlite3.connect')
    def test_get_connection_new(self, mock_connect, temp_dir):
        """Test getting connection when pool is empty."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn

        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path)

        with pool.get_connection() as conn:
            assert conn == mock_conn

        mock_connect.assert_called_once_with(db_path, check_same_thread=False, detect_types=1)
        mock_conn.execute.assert_called_with('SELECT 1')
        pool.close_all()


    @patch('sqlite3.connect')
    def test_get_connection_from_pool(self, mock_connect, temp_dir):
        """Test getting connection from pool."""
        mock_conn = Mock()

        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path)

        # Add connection to pool
        pool.return_connection(mock_conn)

        with pool.get_connection() as conn:
            assert conn == mock_conn
            assert len(pool._pool) == 0
            mock_connect.assert_not_called()
        pool.close_all()

    def test_return_connection_to_pool(self, temp_dir):
        """Test returning connection to pool."""
        mock_conn = Mock()

        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path, max_connections=2)

        pool.return_connection(mock_conn)

        assert len(pool._pool) == 1
        assert pool._pool[0] == mock_conn

        pool.close_all()

    def test_return_connection_pool_full(self, temp_dir):
        """Test returning connection when pool is full."""
        mock_conn1 = Mock()
        mock_conn2 = Mock()
        mock_conn3 = Mock()

        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path, max_connections=2)

        # Fill pool
        pool._pool.extend([mock_conn1, mock_conn2])

        # Try to return another connection
        pool.return_connection(mock_conn3)

        assert len(pool._pool) == 2
        mock_conn3.close.assert_called_once()

        pool.close_all()

    def test_close_all(self, temp_dir):
        """Test closing all connections."""
        mock_conn1 = Mock()
        mock_conn2 = Mock()

        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path)
        pool._pool.extend([mock_conn1, mock_conn2])

        pool.close_all()

        assert len(pool._pool) == 0
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()

    def test_thread_safety(self, temp_dir):
        """Test connection pool thread safety."""
        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path)

        results = []


        with patch('sqlite3.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            def get_and_return():
                conn = pool.get_connection()
                results.append(conn)
                pool.return_connection(conn)

            # Run multiple threads
            threads = [threading.Thread(target=get_and_return, daemon=True) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Should have created connections for each thread
        assert len(results) == 5
        pool.close_all()

# TODO: async stuff missing?