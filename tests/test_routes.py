# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/test_routes.py
"""
Tests for localvectordb_server.routes module.
"""

import json

import numpy as np
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from flask import Flask

from localvectordb.core import MetadataField, MetadataFieldType, Document, QueryResult, ChunkPosition
from localvectordb.exceptions import (
    DatabaseNotFoundError, DuplicateDocumentIDError, EmbeddingError
)
from localvectordb_server._error_handlers import ValidationError
from localvectordb_server.routes import (
    api, serialize_document, serialize_query_result, parse_metadata_schema
)
from localvectordb_server._cache import cache


@pytest.fixture(scope="function")
def app():
    """Create Flask app for testing."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config["CACHE_TYPE"] = "NullCache"
    app.config['DB_ROOT_DIR'] = '/tmp/test_dbs'

    cache.init_app(app)

    # Mock database manager
    app.db_manager = Mock()
    app.db_manager.databases = {}
    app.db_manager.list_databases.return_value = ['test_db1', 'test_db2']
    app.db_manager.delete_db.return_value = (True, None)

    # Register blueprint
    app.register_blueprint(api)

    return app


@pytest.fixture(scope="function")
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture(scope="function")
def mock_db():
    """Create mock database instance."""
    db = Mock()
    db.name = "test_db"
    db.embedding_provider = Mock()
    db.embedding_provider.provider_name = "ollama"
    db.embedding_provider.model = "nomic-embed-text"
    db.embedding_dimension = 384
    db.chunking_method = "sentences"
    db.chunk_size = 500
    db.chunk_overlap = 1
    db.metadata_schema = {
        'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
    }
    db.fts_enabled = True
    db.get_stats = lambda: {
        'documents': 10,
        'chunks': 25,
        'index_vectors': 25
    }
    return db


@pytest.fixture(scope="function")
def sample_document():
    """Create sample document for testing."""
    return Document(
        id="doc_1",
        content="This is a test document.",
        metadata={"author": "Test Author", "category": "test"},
        created_at=datetime(2024, 1, 1, 12, 0, 0),
        updated_at=datetime(2024, 1, 1, 12, 0, 0),
        content_hash="abc123"
    )


@pytest.fixture(scope="function")
def sample_query_result():
    """Create sample query result for testing."""
    position = ChunkPosition(start=0, end=24, line=1, column=1, end_line=1, end_column=24)
    return QueryResult(
        id="doc_1:0",
        score=0.85,
        type='chunk',
        content="This is a test document.",
        metadata={"author": "Test Author"},
        document_id="doc_1",
        position=position
    )


@pytest.fixture(scope="function")
def auth_headers():
    """Create authorization headers for testing."""
    return {'Authorization': 'Bearer test_api_key'}


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_serialize_document(self, sample_document):
        """Test document serialization."""
        result = serialize_document(sample_document)

        expected = {
            "id": "doc_1",
            "content": "This is a test document.",
            "metadata": {"author": "Test Author", "category": "test"},
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
            "content_hash": "abc123"
        }

        assert result == expected

    def test_serialize_document_no_dates(self):
        """Test serializing document without dates."""
        doc = Document(
            id="doc_1",
            content="Test content",
            metadata={"author": "Test"},
            content_hash="abc123"
        )

        result = serialize_document(doc)

        assert result["created_at"] is None
        assert result["updated_at"] is None

    def test_serialize_query_result_chunk(self, sample_query_result):
        """Test query result serialization for chunks."""
        result = serialize_query_result(sample_query_result)

        expected = {
            "id": "doc_1:0",
            "score": 0.85,
            "type": "chunk",
            "content": "This is a test document.",
            "metadata": {"author": "Test Author"},
            "document_id": "doc_1",
            "position": {
                "start": 0,
                "end": 24,
                "line": 1,
                "column": 1,
                "end_line": 1,
                "end_column": 24
            }
        }

        assert result == expected

    def test_serialize_query_result_document(self):
        """Test query result serialization for documents."""
        result = QueryResult(
            id="doc_1",
            score=0.90,
            type='document',
            content="Document content",
            metadata={"author": "Test"}
        )

        serialized = serialize_query_result(result)

        assert serialized["type"] == "document"
        assert "document_id" not in serialized
        assert "position" not in serialized

    def test_parse_metadata_schema_simple(self):
        """Test parsing simple metadata schema."""
        schema_data = {
            "author": "text",
            "rating": "real"
        }

        result = parse_metadata_schema(schema_data)

        assert len(result) == 2
        assert result["author"].type == MetadataFieldType.TEXT
        assert result["rating"].type == MetadataFieldType.REAL

    def test_parse_metadata_schema_complex(self):
        """Test parsing complex metadata schema."""
        schema_data = {
            "author": {
                "type": "text",
                "indexed": True,
                "required": True,
                "default_value": "Unknown"
            },
            "tags": {
                "type": "json"
            }
        }

        result = parse_metadata_schema(schema_data)

        assert result["author"].type == MetadataFieldType.TEXT
        assert result["author"].indexed is True
        assert result["author"].required is True
        assert result["author"].default_value == "Unknown"

        assert result["tags"].type == MetadataFieldType.JSON
        assert result["tags"].indexed is False

    def test_parse_metadata_schema_empty(self):
        """Test parsing empty metadata schema."""
        assert parse_metadata_schema({}) == {}
        assert parse_metadata_schema(None) == {}

    def test_parse_metadata_schema_invalid(self, app):
        """Test parsing invalid metadata schema."""
        with app.app_context():
            with pytest.raises(ValidationError):
                parse_metadata_schema({"invalid": 123})


@pytest.mark.integration
@pytest.mark.client
class TestDatabaseManagementRoutes:
    """Test database management routes."""

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_create_database_success(self, client, app):
        """Test successful database creation."""
        # Mock the database manager's create_db method
        with patch.object(app.db_manager, 'create_db') as mock_create_db:
            # Create a mock database object to return
            mock_db = Mock()
            mock_db.configure_mock(**{
                'name': "test_db",
                'embedding_provider.provider_name': "ollama",
                'embedding_provider.model': "nomic-embed-text",
                'embedding_dimension': 384,
                'chunking_method': "sentences",
                'chunk_size': 500,
                'chunk_overlap': 1,
                'metadata_schema': {},  # Empty dict that can be iterated
                'fts_enabled': True
            })

            # Configure the mock to return our mock database
            mock_create_db.return_value = mock_db

            data = {
                "name": "test_db",
                "embedding": {"model": "nomic-embed-text"},
                "database": {"chunk_size": 500}
            }

            response = client.post('/api/v1/databases',
                                   data=json.dumps(data),
                                   content_type='application/json')

            # Verify the response
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "success"
            assert "test_db" in result["message"]

            # Verify create_db was called with correct parameters
            mock_create_db.assert_called_once()
            call_args = mock_create_db.call_args

            # Check that the first argument is the database name
            assert call_args[0][0] == "test_db"

            # Check that metadata_schema is provided (second argument)
            metadata_schema = call_args[0][1]
            assert metadata_schema is not None or metadata_schema == {}

            # Check that db_config (third argument) has the right chunk_size
            db_config = call_args[0][2]
            assert db_config.chunk_size == 500

            # Check that embedding_config (fourth argument) has the right model
            embedding_config = call_args[0][3]
            assert embedding_config.model == "nomic-embed-text"

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_create_database_with_metadata_schema(self, client, app):
        """Test database creation with custom metadata schema."""
        with patch.object(app.db_manager, 'create_db') as mock_create_db:
            mock_db = Mock()
            mock_db.configure_mock(**{
                'name': "test_db_with_schema",
                'embedding_provider.provider_name': "ollama",
                'embedding_provider.model': "nomic-embed-text",
                'embedding_dimension': 384,
                'chunking_method': "sentences",
                'chunk_size': 500,
                'chunk_overlap': 1,
                'metadata_schema': {
                    'title': Mock(type=Mock(value='text'), indexed=True, required=False, default_value=None),
                    'author': Mock(type=Mock(value='text'), indexed=False, required=True, default_value=None)
                },
                'fts_enabled': True
            })
            mock_create_db.return_value = mock_db

            data = {
                "name": "test_db_with_schema",
                "metadata_schema": {
                    "title": {"type": "text", "indexed": True},
                    "author": {"type": "text", "required": True}
                },
                "embedding": {"model": "nomic-embed-text"}
            }

            response = client.post('/api/v1/databases',
                                   data=json.dumps(data),
                                   content_type='application/json')

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "success"

            # Verify create_db was called
            mock_create_db.assert_called_once()
            call_args = mock_create_db.call_args

            # Verify metadata schema was parsed and passed correctly
            metadata_schema = call_args[0][1]
            assert metadata_schema is not None
            assert len(metadata_schema) == 2  # title and author fields

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_create_database_already_exists(self, client, app):
        """Test database creation when database already exists."""
        with patch.object(app.db_manager, 'create_db') as mock_create_db:
            # Mock the database manager to raise BadRequest for existing database
            from werkzeug.exceptions import BadRequest
            mock_create_db.side_effect = BadRequest("Database 'existing_db' already exists")

            data = {
                "name": "existing_db",
                "embedding": {"model": "nomic-embed-text"}
            }

            response = client.post('/api/v1/databases',
                                   data=json.dumps(data),
                                   content_type='application/json')

            assert response.status_code == 400
            result = json.loads(response.data)
            assert "already exists" in result["error"]["message"]

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_create_database_no_name(self, client):
        """Test database creation without name."""
        data = {"embedding_model": "test-model"}

        response = client.post('/api/v1/databases',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_create_database_no_data(self, client):
        """Test database creation without data."""
        response = client.post('/api/v1/databases')
        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_list_databases(self, client, app):
        """Test listing databases."""
        response = client.get('/api/v1/databases')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "databases" in result
        assert result["count"] == 2

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_get_database_info(self, client, app, mock_db):
        """Test getting database info."""
        app.db_manager.get_db.return_value = mock_db

        response = client.get('/api/v1/test_db/info')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["name"] == "test_db"
        assert "stats" in result
        assert "config" in result

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_delete_database(self, client, app, temp_dir):
        """Test database deletion."""
        # Create mock database files
        db_file = temp_dir / "test_db.sqlite"
        faiss_file = temp_dir / "test_db.faiss"
        db_file.touch()
        faiss_file.touch()

        app.config['DB_ROOT_DIR'] = str(temp_dir)

        # Mock database in manager
        mock_db = Mock()
        app.db_manager.databases = {"test_db": (mock_db, datetime.now())}

        response = client.delete('/api/v1/test_db')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestDocumentManagementRoutes:
    """Test document management routes."""

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_upsert_documents_single(self, client, app, mock_db):
        """Test upserting single document."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.upsert.return_value = ["doc_1"]

        data = {
            "documents": "Test document content",
            "metadata": {"author": "Test Author"}
        }

        response = client.post('/api/v1/test_db/documents',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert result["ids"] == ["doc_1"]

        mock_db.upsert.assert_called_once()

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_upsert_documents_multiple(self, client, app, mock_db):
        """Test upserting multiple documents."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.upsert.return_value = ["doc_1", "doc_2"]

        data = {
            "documents": ["Doc 1", "Doc 2"],
            "metadata": [{"author": "Author 1"}, {"author": "Author 2"}],
            "ids": ["doc_1", "doc_2"]
        }

        response = client.post('/api/v1/test_db/documents',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert len(result["ids"]) == 2

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_insert_documents(self, client, app, mock_db):
        """Test inserting documents."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.insert.return_value = ["doc_1"]

        data = {
            "documents": ["Test document"],
            "errors": "ignore",
            "similarity_threshold": 0.95
        }

        response = client.post('/api/v1/test_db/documents/insert',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        mock_db.insert.assert_called_once()

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_get_document_success(self, client, app, mock_db, sample_document):
        """Test getting document successfully."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.get.return_value = sample_document

        response = client.get('/api/v1/test_db/documents/doc_1')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["id"] == "doc_1"
        assert result["content"] == "This is a test document."

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_get_document_not_found(self, client, app, mock_db):
        """Test getting non-existent document."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.get.return_value = None

        response = client.get('/api/v1/test_db/documents/nonexistent')

        assert response.status_code == 404

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_update_document_success(self, client, app, mock_db):
        """Test updating document successfully."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.update.return_value = True

        data = {
            "content": "Updated content",
            "metadata": {"author": "Updated Author"}
        }

        response = client.put('/api/v1/test_db/documents/doc_1',
                              data=json.dumps(data),
                              content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert result["updated"] is True

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_update_document_not_found(self, client, app, mock_db):
        """Test updating non-existent document."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.update.return_value = False

        data = {"content": "Updated content"}

        response = client.put('/api/v1/test_db/documents/nonexistent',
                              data=json.dumps(data),
                              content_type='application/json')

        assert response.status_code == 404

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_delete_document_success(self, client, app, mock_db):
        """Test deleting document successfully."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.exists.return_value = True
        mock_db.delete.return_value = 1

        response = client.delete('/api/v1/test_db/documents/doc_1')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["status"] == "success"
        assert result["deleted_count"] == 1

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_delete_document_not_found(self, client, app, mock_db):
        """Test deleting non-existent document."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.exists.return_value = False

        response = client.delete('/api/v1/test_db/documents/nonexistent')

        assert response.status_code == 404

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_check_documents_exist(self, client, app, mock_db):
        """Test checking document existence."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.exists.return_value = [True, False]

        data = {"ids": ["doc_1", "doc_2"]}

        response = client.post('/api/v1/test_db/documents/exists',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["exists"] == [True, False]

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_list_documents(self, client, app, mock_db, sample_document):
        """Test listing documents with pagination."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.filter.return_value = [sample_document]

        response = client.get('/api/v1/test_db/documents?page=1&limit=10')

        print(response.text)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert "documents" in result
        assert "pagination" in result
        assert len(result["documents"]) == 1


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestSearchRoutes:
    """Test search routes."""

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_query_documents_vector(self, client, app, mock_db, sample_query_result):
        """Test vector query."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.query.return_value = [sample_query_result]

        data = {
            "query": "test query",
            "search_type": "vector",
            "k": 5
        }

        response = client.post('/api/v1/test_db/query',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["search_type"] == "vector"
        assert len(result["results"]) == 1

        mock_db.query.assert_called_once()

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_query_documents_hybrid(self, client, app, mock_db):
        """Test hybrid query."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.query.return_value = []

        data = {
            "query": "test query",
            "search_type": "hybrid",
            "vector_weight": 0.8,
            "filters": {"author": "Test Author"}
        }

        response = client.post('/api/v1/test_db/query',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["search_type"] == "hybrid"

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_vector_search_convenience(self, client, app, mock_db):
        """Test vector search convenience endpoint."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.query.return_value = []

        data = {"query": "test query", "k": 10}

        response = client.post('/api/v1/test_db/search/vector',
                               data=json.dumps(data),
                               content_type='application/json')
        assert response.status_code == 200

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_keyword_search_convenience(self, client, app, mock_db):
        """Test keyword search convenience endpoint."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.query.return_value = []

        data = {"query": "test query"}

        response = client.post('/api/v1/test_db/search/keyword',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_hybrid_search_convenience(self, client, app, mock_db):
        """Test hybrid search convenience endpoint."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.query.return_value = []

        data = {"query": "test query"}

        response = client.post('/api/v1/test_db/search/hybrid',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestFilterRoutes:
    """Test filter routes."""

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_filter_documents_where(self, client, app, mock_db, sample_document):
        """Test filtering documents with where clause."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.filter.return_value = [sample_document]

        data = {
            "where": {"author": "Test Author"},
            "limit": 10,
            "offset": 0
        }

        response = client.post('/api/v1/test_db/filter',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert len(result["documents"]) == 1

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_filter_documents_sql(self, client, app, mock_db):
        """Test filtering documents with SQL clause."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.filter.return_value = []

        data = {
            "sql": "author = 'Test Author'",
            "order_by": "created_at DESC"
        }

        response = client.post('/api/v1/test_db/filter',
                               data=json.dumps(data),
                               content_type='application/json')
        print(response.text)

        assert response.status_code == 200

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_filter_documents_no_criteria(self, client, app, mock_db):
        """Test filtering without criteria."""
        data = {}

        response = client.post('/api/v1/test_db/filter',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 400


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestGlobalSearchRoutes:
    """Test global search routes."""

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_global_search_all_dbs(self, client, app, sample_query_result):
        """Test global search across all databases."""
        # Setup mock databases
        mock_db1 = Mock()
        mock_db2 = Mock()
        mock_db1.query.return_value = []
        mock_db2.query.return_value = []

        app.db_manager.list_databases.return_value = ['db1', 'db2']
        app.db_manager.get_db.side_effect = lambda name: mock_db1 if name == 'db1' else mock_db2
        app.db_manager.search_databases.return_value = {
            "db1": [sample_query_result],
            "db2": [sample_query_result]
        }

        data = {
            "query": "test query",
            "search_type": "vector"
        }

        response = client.post('/api/v1/search',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "results" in result
        assert "db1" in result["results"]
        assert "db2" in result["results"]

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_global_search_specific_dbs(self, client, app, sample_query_result):
        """Test global search on specific databases."""
        mock_db = Mock()
        mock_db.query.return_value = []
        app.db_manager.get_db.return_value = mock_db
        app.db_manager.search_databases.return_value = {
            "db1": [sample_query_result]
        }

        data = {
            "query": "test query",
            "databases": ["db1"]
        }

        response = client.post('/api/v1/search',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "db1" in result["results"]


@pytest.mark.integration
@pytest.mark.client
class TestHealthRoutes:
    """Test health and system routes."""

    def test_health_check(self, client, app):
        """Test health check endpoint."""
        with patch('localvectordb_server.routes.check_ollama_service', return_value=True):
            response = client.get('/api/v1/health')

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "healthy"
            assert "version" in result
            assert result["ollama_available"] is True


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.embedding
class TestEmbeddingRoutes:
    """Test embedding routes."""

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_get_embeddings_for_db(self, client, app, mock_db):
        """Test getting embeddings using database provider."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.embedding_provider.embed_sync.return_value = np.array([[0.1, 0.2, 0.3]])

        data = {"texts": ["test text"]}

        response = client.post('/api/v1/test_db/embeddings',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "embeddings" in result

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_get_embeddings_generic(self, client):
        """Test getting embeddings from specific provider."""
        with patch('localvectordb.embeddings.EmbeddingRegistry.create_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.embed_sync.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_create.return_value = mock_provider

            data = {
                "texts": ["test text"],
                "provider": "mock",
                "model": "test-model"
            }

            response = client.post('/api/v1/embeddings',
                                   data=json.dumps(data),
                                   content_type='application/json')

            assert response.status_code == 200
            result = json.loads(response.data)
            assert "embeddings" in result


@pytest.mark.integration
@pytest.mark.client
class TestErrorHandling:
    """Test error handling."""

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_database_not_found_error(self, client, app):
        """Test DatabaseNotFoundError handling."""
        app.db_manager.get_db.side_effect = DatabaseNotFoundError("Database not found")

        response = client.get('/api/v1/nonexistent/info')

        assert response.status_code == 404
        result = json.loads(response.data)
        assert result["error"]["code"] == "DATABASE_NOT_FOUND"

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_duplicate_document_id_error(self, client, app, mock_db):
        """Test DuplicateDocumentIDError handling."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.insert.side_effect = DuplicateDocumentIDError("Duplicate ID")

        data = {"documents": ["test"]}

        response = client.post('/api/v1/test_db/documents/insert',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 409
        result = json.loads(response.data)
        assert result["error"]["code"] == "DUPLICATE_DOCUMENT_ID"

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_embedding_error(self, client, app, mock_db):
        """Test EmbeddingError handling."""
        app.db_manager.get_db.return_value = mock_db
        mock_db.embedding_provider.embed_sync.side_effect = EmbeddingError("Embedding failed")

        data = {"texts": ["test"]}

        response = client.post('/api/v1/test_db/embeddings',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 503
        result = json.loads(response.data)
        assert result["error"]["code"] == "EMBEDDING_ERROR"

    def test_bad_request_error(self, client):
        """Test BadRequest error handling."""
        response = client.post('/api/v1/databases')  # No data

        assert response.status_code == 400
        result = json.loads(response.data)
        assert "error" in result


@pytest.mark.integration
@pytest.mark.client
class TestRequestValidation:
    """Test request validation."""

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_query_no_data(self, client):
        """Test query endpoint without data."""
        response = client.post('/api/v1/test_db/query', data=json.dumps({}))
        assert response.status_code == 400

        response = client.post('/api/v1/test_db/query', data=json.dumps({}), content_type='application/json')
        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_query_no_query_text(self, client):
        """Test query endpoint without query text."""
        data = {"search_type": "vector"}

        response = client.post('/api/v1/test_db/query',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_update_document_no_data(self, client):
        """Test update endpoint without data."""
        response = client.put('/api/v1/test_db/documents/doc_1', content_type='application/json')
        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_write_permission', lambda f: f)
    def test_update_document_empty_data(self, client):
        """Test update endpoint with empty data."""
        data = {}

        response = client.put('/api/v1/test_db/documents/doc_1',
                              data=json.dumps(data),
                              content_type='application/json')

        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_embeddings_no_texts(self, client):
        """Test embeddings endpoint without texts."""
        data = {"provider": "mock", "model": "test"}

        response = client.post('/api/v1/embeddings',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 400

    @patch('localvectordb_server._auth.require_read_permission', lambda f: f)
    def test_embeddings_no_provider(self, client):
        """Test embeddings endpoint without provider."""
        data = {"texts": ["test"], "model": "test"}

        response = client.post('/api/v1/embeddings',
                               data=json.dumps(data),
                               content_type='application/json')

        assert response.status_code == 400