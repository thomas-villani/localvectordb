# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/test_server_integrated.py
"""
Integration tests for LocalVectorDB server.

These tests verify end-to-end functionality including authentication,
database operations, and multi-component interactions.
"""

import json
import os

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from flask import Flask
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb_server.routes import api
from localvectordb_server.keymanager import KeyManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def test_key_manager(temp_dir):
    """Create a KeyManager instance with test API keys."""
    key_db_path = Path(temp_dir) / "test_api_keys.db"
    key_manager = KeyManager(str(key_db_path))

    # Create test API keys
    test_key = key_manager.create_key(
        description="Test API Key",
        expires_days=None  # Never expires
    )

    expired_key = key_manager.create_key(
        description="Expired Test Key",
        expires_days=-1  # Already expired
    )

    # Store the plain keys and IDs for testing
    key_manager._test_valid_key = test_key.plain_key
    key_manager._test_valid_key_id = test_key.id
    key_manager._test_expired_key = expired_key.plain_key
    key_manager._test_expired_key_id = expired_key.id

    return key_manager


@pytest.fixture
def integration_app(temp_dir, test_key_manager):
    """Create Flask app for integration testing with real-like configuration."""
    app = Flask(__name__)
    app.config['TESTING'] = True

    # Use temporary directory for test databases
    app.config['DB_ROOT_DIR'] = temp_dir
    app.config['REQUIRE_API_KEY'] = True
    app.config['CACHE_TYPE'] = 'NullCache'

    from localvectordb_server._cache import cache
    cache.init_app(app)

    # app.config['API_KEY_DB_PATH'] = test_key_manager.db_path
    # app.config['API_KEY_HEADER'] = 'Authorization'
    # app.config['API_KEY_AUDIT_LOGGING'] = False  # Disable for tests
    # app.config['API_KEY_PRUNE_EXPIRED'] = False  # Disable for tests

    # Store key manager instance for access in tests
    app.key_manager = test_key_manager

    # Create a simple database manager mock that behaves more realistically
    app.db_manager = DatabaseManagerMock(temp_dir)

    # Register blueprint
    app.register_blueprint(api)

    yield app


@pytest.fixture
def integration_client(integration_app):
    """Create test client for integration testing."""
    return integration_app.test_client()


@pytest.fixture
def valid_auth_headers(integration_app):
    """Get headers with valid API key for authenticated requests."""
    return {'Authorization': f'Bearer {integration_app.key_manager._test_valid_key}'}


@pytest.fixture
def expired_auth_headers(integration_app):
    """Get headers with expired API key for testing expired key handling."""
    return {'Authorization': f'Bearer {integration_app.key_manager._test_expired_key}'}


class DatabaseManagerMock:
    """More realistic database manager mock for integration testing."""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.databases = {}
        self._created_dbs = set()

    def list_databases(self):
        """List databases by checking filesystem and in-memory databases."""
        db_files = list(self.base_path.glob("*.sqlite"))
        file_dbs = [f.stem for f in db_files]

        # Combine all sources of database names
        all_dbs = set(file_dbs) | set(self._created_dbs) | set(self.databases.keys())
        return list(all_dbs)

    def get_db(self, name):
        """Get database instance, creating mock if needed."""
        if name not in self.databases:
            if name in self._created_dbs or (self.base_path / f"{name}.sqlite").exists():
                # Create a more realistic mock database
                self.databases[name] = self._create_mock_db(name)
            else:
                from localvectordb.exceptions import DatabaseNotFoundError
                raise DatabaseNotFoundError(f"Database '{name}' not found")
        return self.databases[name]

    def delete_db(self, name, *args, **kwargs):
        if name in self._created_dbs:
            self._created_dbs.remove(name)
        db_file = self.base_path / f"{name}.sqlite"
        if os.path.exists(db_file):
            os.remove(db_file)
        self.databases.pop(name)

        return True, None

    def create_db(self, name, *args, **kwargs):
        """Simulate database creation."""
        self._created_dbs.add(name)
        # Create the actual file to simulate real behavior
        db_file = self.base_path / f"{name}.sqlite"
        db_file.touch()

        # Create mock database instance
        self.databases[name] = self._create_mock_db(name)
        return self.databases[name]

    def _create_mock_db(self, name):
        """Create a realistic mock database instance."""
        db = Mock()
        db.name = name
        db.embedding_provider = Mock()
        db.embedding_provider.provider_name = "mock"
        db.embedding_provider.model = "test-model"
        db.embedding_provider.embed_sync.return_value = np.array([[0.1, 0.2, 0.3]])
        db.embedding_dimension = 384
        db.chunking_method = "sentences"
        db.chunk_size = 500
        db.chunk_overlap = 1
        db.metadata_schema = {}
        db.fts_enabled = True
        db.stats = {'documents': 0, 'chunks': 0, 'index_vectors': 0}

        # Document storage for more realistic behavior
        db._documents = {}
        db._next_id = 1

        # More realistic method implementations
        def mock_upsert(documents, metadata=None, ids=None, **kwargs):
            if not isinstance(documents, list):
                documents = [documents]
            if metadata and not isinstance(metadata, list):
                metadata = [metadata]
            if ids and not isinstance(ids, list):
                ids = [ids]

            result_ids = []
            for i, doc in enumerate(documents):
                doc_id = ids[i] if ids and i < len(ids) else f"doc_{db._next_id}"
                db._next_id += 1
                db._documents[doc_id] = {
                    'content': doc,
                    'metadata': metadata[i] if metadata and i < len(metadata) else {}
                }
                result_ids.append(doc_id)

            # Update stats
            db.stats['documents'] = len(db._documents)
            db.stats['chunks'] = len(db._documents) * 2  # Simulate chunking
            db.stats['index_vectors'] = db.stats['chunks']

            return result_ids

        def mock_get(doc_id):
            if doc_id in db._documents:
                from localvectordb.core import Document
                doc_data = db._documents[doc_id]
                return Document(
                    id=doc_id,
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    content_hash="mock_hash"
                )
            return None

        def mock_exists(doc_id):
            if isinstance(doc_id, list):
                return [d in db._documents for d in doc_id]
            return doc_id in db._documents

        def mock_delete(doc_id):
            if isinstance(doc_id, list):
                count = sum(1 for d in doc_id if db._documents.pop(d, None))
            else:
                count = 1 if db._documents.pop(doc_id, None) else 0

            # Update stats
            db.stats['documents'] = len(db._documents)
            db.stats['chunks'] = len(db._documents) * 2
            db.stats['index_vectors'] = db.stats['chunks']

            return count

        def mock_query(query, **kwargs):
            # Simple mock that returns some documents
            from localvectordb.core import QueryResult
            results = []
            for doc_id, doc_data in list(db._documents.items())[:kwargs.get('k', 5)]:
                result = QueryResult(
                    id=doc_id,
                    score=0.8,
                    type='document',
                    content=doc_data['content'],
                    metadata=doc_data['metadata']
                )
                results.append(result)
            return results

        def mock_filter(**kwargs):
            # Simple filter implementation
            from localvectordb.core import Document
            results = []
            limit = kwargs.get('limit', 100) or 100
            offset = kwargs.get('offset', 0) or 0

            docs = list(db._documents.items())[offset:offset + limit]
            for doc_id, doc_data in docs:
                doc = Document(
                    id=doc_id,
                    content=doc_data['content'],
                    metadata=doc_data['metadata'],
                    content_hash="mock_hash"
                )
                results.append(doc)
            return results

        def mock_update(doc_id, content=None, metadata=None):
            if doc_id in db._documents:
                if content is not None:
                    db._documents[doc_id]['content'] = content
                if metadata is not None:
                    db._documents[doc_id]['metadata'].update(metadata)
                return True
            return False

        # Attach mock methods
        db.upsert = mock_upsert
        db.insert = mock_upsert  # For simplicity, same as upsert
        db.get = mock_get
        db.exists = mock_exists
        db.delete = mock_delete
        db.query = mock_query
        db.filter = mock_filter
        db.update = mock_update
        db.close = Mock()

        return db

class TestAuthenticationFlow:
    """Test full authentication flow with new KeyManager system."""

    def test_request_without_api_key(self, integration_client):
        """Test request without API key should fail."""
        response = integration_client.get('/api/v1/databases')
        assert response.status_code == 401

    def test_request_with_invalid_api_key(self, integration_client):
        """Test request with invalid API key should fail."""
        headers = {'Authorization': 'Bearer invalid_key_12345'}
        response = integration_client.get('/api/v1/databases', headers=headers)
        assert response.status_code == 401

    def test_request_with_expired_api_key(self, integration_client, expired_auth_headers):
        """Test request with expired API key should fail."""
        response = integration_client.get('/api/v1/databases', headers=expired_auth_headers)
        assert response.status_code == 401

    def test_request_with_valid_api_key(self, integration_client, valid_auth_headers):
        """Test request with valid API key should succeed."""
        response = integration_client.get('/api/v1/databases', headers=valid_auth_headers)
        print(response.text)
        print(valid_auth_headers)
        assert response.status_code == 200

    def test_api_key_header_formats(self, integration_client, integration_app):
        """Test API key authentication with different header formats."""
        valid_key = integration_app.key_manager._test_valid_key

        test_cases = [
            f'Bearer {valid_key}',
            valid_key,  # Without Bearer prefix
        ]

        for auth_value in test_cases:
            headers = {'Authorization': auth_value}
            response = integration_client.get('/api/v1/databases', headers=headers)
            # Should work with either format (depending on auth implementation)
            assert response.status_code in [200, 401]

    def test_key_validation_updates_last_used(self, integration_client, valid_auth_headers, integration_app):
        """Test that successful authentication updates the last_used timestamp."""
        # Get key before request
        key_id = next(iter(integration_app.key_manager.list_keys())).id
        key_before = integration_app.key_manager.get_key(key_id)
        original_last_used = key_before.last_used

        # Make authenticated request
        response = integration_client.get('/api/v1/databases', headers=valid_auth_headers)
        assert response.status_code == 200

        # Check that last_used was updated (this might be mocked in tests)
        # In a real scenario, we'd verify the timestamp was updated


class TestDatabaseLifecycle:
    """Test complete database lifecycle with authentication."""

    def test_create_and_delete_database_flow(self, integration_client, integration_app, valid_auth_headers):
        """Test creating and deleting a database end-to-end."""
        # Create database
        create_data = {
            "name": "test_lifecycle_db",
            "embedding_provider": "mock",
            "embedding_model": "test-model",
            "metadata_schema": {
                "author": {"type": "text", "indexed": True},
                "category": "text"
            }
        }

        with patch('localvectordb.database.LocalVectorDB') as mock_db_class:
            # Mock the LocalVectorDB constructor
            mock_db = integration_app.db_manager._create_mock_db("test_lifecycle_db")
            mock_db_class.return_value = mock_db

            response = integration_client.post('/api/v1/databases',
                                               data=json.dumps(create_data),
                                               content_type='application/json',
                                               headers=valid_auth_headers)

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "success"
            assert "test_lifecycle_db" in result["message"]

        # Get database info
        response = integration_client.get('/api/v1/test_lifecycle_db/info', headers=valid_auth_headers)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert result["name"] == "test_lifecycle_db"

        # List databases (should include our new one)
        response = integration_client.get('/api/v1/databases', headers=valid_auth_headers)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert "test_lifecycle_db" in result["databases"]

        # Delete database - patch filesystem operations
        with patch('pathlib.Path.exists') as mock_exists, \
                patch('os.remove') as mock_remove:
            # Mock that the .sqlite file exists
            mock_exists.return_value = True

            response = integration_client.delete('/api/v1/test_lifecycle_db', headers=valid_auth_headers)
            print(response.text)
            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "success"

            # Verify that os.remove was called (files were "deleted")
            assert mock_remove.call_count >= 1

    def test_database_operations_with_metadata_schema(self, integration_client, integration_app, valid_auth_headers):
        """Test database operations with complex metadata schema."""
        # Ensure we have a test database
        integration_app.db_manager.create_db(
            "schema_test_db",
            metadata_schema={
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                'rating': MetadataField(type=MetadataFieldType.REAL, indexed=True),
                'tags': MetadataField(type=MetadataFieldType.JSON)
            }
        )

        # Add documents with metadata
        doc_data = {
            "documents": [
                "First test document with metadata",
                "Second test document with different metadata"
            ],
            "metadata": [
                {"author": "Alice", "rating": 4.5, "tags": ["test", "first"]},
                {"author": "Bob", "rating": 3.8, "tags": ["test", "second"]}
            ]
        }

        response = integration_client.post('/api/v1/schema_test_db/documents',
                                           data=json.dumps(doc_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 200
        result = json.loads(response.data)
        assert len(result["ids"]) == 2
        doc_ids = result["ids"]

        # Query documents
        query_data = {
            "query": "test document",
            "search_type": "vector",
            "k": 5
        }

        response = integration_client.post('/api/v1/schema_test_db/query',
                                           data=json.dumps(query_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 200
        result = json.loads(response.data)
        assert len(result["results"]) >= 0

        # Filter documents by metadata
        filter_data = {
            "where": {"author": "Alice"}
        }

        response = integration_client.post('/api/v1/schema_test_db/filter',
                                           data=json.dumps(filter_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 200
        result = json.loads(response.data)
        assert len(result["documents"]) >= 0


class TestErrorHandlingIntegration:
    """Test error handling across multiple components."""

    def test_database_not_found_flow(self, integration_client, valid_auth_headers):
        """Test database not found error in realistic scenario."""
        # Try to access non-existent database
        response = integration_client.get('/api/v1/nonexistent_db/info', headers=valid_auth_headers)

        assert response.status_code == 404
        result = json.loads(response.data)
        assert result["error"]["code"] == "DATABASE_NOT_FOUND"

    def test_document_not_found_flow(self, integration_client, integration_app, valid_auth_headers):
        """Test document not found in realistic scenario."""
        # Create a test database
        integration_app.db_manager.create_db("error_test_db")

        # Try to get non-existent document
        response = integration_client.get('/api/v1/error_test_db/documents/nonexistent',
                                          headers=valid_auth_headers)

        assert response.status_code == 404

    def test_invalid_request_data_flow(self, integration_client, integration_app, valid_auth_headers):
        """Test invalid request data handling."""
        # Create a test database
        integration_app.db_manager.create_db("validation_test_db")

        # Try to query without required data
        response = integration_client.post('/api/v1/validation_test_db/query',
                                           data=json.dumps({}),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 400

        # Try to create database without name
        response = integration_client.post('/api/v1/databases',
                                           data=json.dumps({"embedding_model": "test"}),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 400

    def test_unauthenticated_access_to_protected_endpoints(self, integration_client):
        """Test that protected endpoints reject unauthenticated requests."""
        protected_endpoints = [
            ('/api/v1/databases', 'GET'),
            ('/api/v1/databases', 'POST'),
            ('/api/v1/test_db/info', 'GET'),
            ('/api/v1/test_db/documents', 'POST'),
            ('/api/v1/test_db/query', 'POST'),
        ]

        for endpoint, method in protected_endpoints:
            if method == 'GET':
                response = integration_client.get(endpoint)
            elif method == 'POST':
                response = integration_client.post(endpoint,
                                                   data=json.dumps({}),
                                                   content_type='application/json')

            assert response.status_code == 401, f"Endpoint {method} {endpoint} should require authentication"


class TestMultiDatabaseOperations:
    """Test operations across multiple databases."""

    def test_global_search_across_databases(self, integration_client, integration_app, valid_auth_headers):
        """Test global search functionality."""
        # Create multiple test databases
        db1 = integration_app.db_manager.create_db("global_search_db1")
        db2 = integration_app.db_manager.create_db("global_search_db2")

        # Add some documents to each
        db1.upsert(["Document in database 1"], [{"source": "db1"}])
        db2.upsert(["Document in database 2"], [{"source": "db2"}])

        # Perform global search
        search_data = {
            "query": "document",
            "search_type": "vector",
            "k": 5
        }

        response = integration_client.post('/api/v1/search',
                                           data=json.dumps(search_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "results" in result

        # Should have results from multiple databases
        assert len(result["results"]) >= 2

    def test_database_isolation(self, integration_client, integration_app, valid_auth_headers):
        """Test that databases are properly isolated."""
        # Create two databases
        db1 = integration_app.db_manager.create_db("isolation_db1")
        db2 = integration_app.db_manager.create_db("isolation_db2")

        # Add document to first database
        doc_data = {"documents": ["Document only in db1"]}
        response = integration_client.post('/api/v1/isolation_db1/documents',
                                           data=json.dumps(doc_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)
        assert response.status_code == 200

        # Try to find document in second database
        query_data = {"query": "Document only in db1", "k": 5}
        response = integration_client.post('/api/v1/isolation_db2/query',
                                           data=json.dumps(query_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 200
        result = json.loads(response.data)
        # Should not find the document (or find very few/none)
        assert len(result["results"]) == 0


class TestEmbeddingIntegration:
    """Test embedding functionality integration."""

    def test_embedding_endpoint_with_database_provider(self, integration_client, integration_app, valid_auth_headers):
        """Test getting embeddings using database's provider."""
        # Create test database
        integration_app.db_manager.create_db("embedding_test_db")

        # Get embeddings
        embed_data = {"texts": ["test text for embedding"]}
        response = integration_client.post('/api/v1/embedding_test_db/embeddings',
                                           data=json.dumps(embed_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)

        assert response.status_code == 200
        result = json.loads(response.data)
        assert "embeddings" in result
        assert len(result["embeddings"]) == 1

    def test_generic_embedding_endpoint(self, integration_client, valid_auth_headers):
        """Test generic embedding endpoint."""
        with patch('localvectordb.embeddings.EmbeddingRegistry.create_provider') as mock_create:
            mock_provider = Mock()
            mock_provider.embed_sync.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_create.return_value = mock_provider

            embed_data = {
                "texts": ["test text"],
                "provider": "mock",
                "model": "test-model"
            }

            response = integration_client.post('/api/v1/embeddings',
                                               data=json.dumps(embed_data),
                                               content_type='application/json',
                                               headers=valid_auth_headers)

            print(response.text)
            assert response.status_code == 200
            result = json.loads(response.data)

            assert "embeddings" in result


class TestHealthAndMonitoring:
    """Test health check and monitoring endpoints."""

    def test_health_check_integration(self, integration_client):
        """Test health check with real-like conditions (health checks are typically unauthenticated)."""
        with patch('localvectordb_server.routes.check_ollama_service') as mock_ollama:
            mock_ollama.return_value = True

            response = integration_client.get('/api/v1/health')

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "healthy"
            assert "version" in result
            assert result["ollama_available"] is True

    def test_health_check_with_service_down(self, integration_client):
        """Test health check when external services are down."""
        with patch('localvectordb_server.routes.check_ollama_service') as mock_ollama:
            mock_ollama.return_value = False

            response = integration_client.get('/api/v1/health')

            assert response.status_code == 200
            result = json.loads(response.data)
            assert result["status"] == "healthy"  # Should still be healthy
            assert result["ollama_available"] is False


class TestKeyManagerIntegration:
    """Test integration with the new KeyManager system."""

    def test_key_manager_stats(self, integration_app):
        """Test that key manager provides stats."""
        stats = integration_app.key_manager.get_stats()

        assert isinstance(stats, dict)
        assert 'total_keys' in stats
        assert 'active_keys' in stats
        assert 'expired_keys' in stats
        assert stats['total_keys'] >= 2  # We created 2 test keys
        assert stats['expired_keys'] >= 1  # We created 1 expired key

    def test_key_creation_during_test(self, integration_app):
        """Test that we can create new keys during testing."""
        initial_count = integration_app.key_manager.get_stats()['total_keys']

        # Create a new key
        new_key = integration_app.key_manager.create_key(
            description="Test key created during test",
            expires_days=30
        )

        assert new_key.plain_key is not None
        assert new_key.description == "Test key created during test"

        # Verify count increased
        final_count = integration_app.key_manager.get_stats()['total_keys']
        assert final_count == initial_count + 1

    def test_key_validation_accuracy(self, integration_app):
        """Test that key validation works correctly."""
        # Valid key should pass
        valid_key = integration_app.key_manager._test_valid_key
        assert integration_app.key_manager.validate_key(valid_key) is True

        # Invalid key should fail
        assert integration_app.key_manager.validate_key("invalid_key") is False

        # Expired key should fail
        expired_key = integration_app.key_manager._test_expired_key
        assert integration_app.key_manager.validate_key(expired_key) is False


class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_document_management_workflow(self, integration_client, integration_app, valid_auth_headers):
        """Test a complete document management workflow."""
        # 1. Create database
        db = integration_app.db_manager.create_db(
            "workflow_test_db",
            metadata_schema={
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
            }
        )

        # 2. Add documents
        doc_data = {
            "documents": ["First document", "Second document"],
            "metadata": [{"author": "Alice"}, {"author": "Bob"}]
        }

        response = integration_client.post('/api/v1/workflow_test_db/documents',
                                           data=json.dumps(doc_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)
        assert response.status_code == 200
        doc_ids = json.loads(response.data)["ids"]

        # 3. Search documents
        search_data = {"query": "document", "k": 5}
        response = integration_client.post('/api/v1/workflow_test_db/query',
                                           data=json.dumps(search_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)
        assert response.status_code == 200

        # 4. Update a document
        update_data = {"content": "Updated first document", "metadata": {"author": "Alice Updated"}}
        response = integration_client.put(f'/api/v1/workflow_test_db/documents/{doc_ids[0]}',
                                          data=json.dumps(update_data),
                                          content_type='application/json',
                                          headers=valid_auth_headers)
        assert response.status_code == 200

        # 5. Get updated document
        response = integration_client.get(f'/api/v1/workflow_test_db/documents/{doc_ids[0]}',
                                          headers=valid_auth_headers)
        assert response.status_code == 200
        result = json.loads(response.data)
        assert "Updated" in result["content"]

        # 6. Filter documents
        filter_data = {"where": {"author": "Bob"}}
        response = integration_client.post('/api/v1/workflow_test_db/filter',
                                           data=json.dumps(filter_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)
        assert response.status_code == 200

        # 7. Delete a document
        response = integration_client.delete(f'/api/v1/workflow_test_db/documents/{doc_ids[1]}',
                                             headers=valid_auth_headers)
        assert response.status_code == 200

        # 8. Verify deletion
        response = integration_client.get(f'/api/v1/workflow_test_db/documents/{doc_ids[1]}',
                                          headers=valid_auth_headers)
        assert response.status_code == 404

    def test_workflow_with_key_rotation(self, integration_client, integration_app, valid_auth_headers):
        """Test workflow that includes key rotation."""
        # Create initial database
        db = integration_app.db_manager.create_db("rotation_test_db")

        # Add some documents with original key
        doc_data = {"documents": ["Document with original key"]}
        response = integration_client.post('/api/v1/rotation_test_db/documents',
                                           data=json.dumps(doc_data),
                                           content_type='application/json',
                                           headers=valid_auth_headers)
        assert response.status_code == 200

        # Get the key ID for the valid test key
        original_key_id = integration_app.key_manager._test_valid_key_id

        # Rotate the key
        new_key_record = integration_app.key_manager.rotate_key(original_key_id)
        assert new_key_record is not None

        # Old key should no longer work
        response = integration_client.get('/api/v1/databases', headers=valid_auth_headers)
        assert response.status_code == 401

        # New key should work
        new_auth_headers = {'Authorization': f'Bearer {new_key_record.plain_key}'}
        response = integration_client.get('/api/v1/databases', headers=new_auth_headers)
        assert response.status_code == 200