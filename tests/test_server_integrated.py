"""
Integration tests for LocalVectorDB server (FastAPI).

These tests verify end-to-end functionality including authentication,
database operations, and multi-component interactions.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.exceptions import DocumentNotFoundError
from localvectordb_server.config import Config
from localvectordb_server.keymanager import KeyManager


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture(scope="function")
def test_key_manager(temp_dir):
    """Create a KeyManager instance with test API keys."""
    key_db_path = Path(temp_dir) / "test_api_keys.db"
    key_manager = KeyManager(str(key_db_path))

    test_key = key_manager.create_key(description="Test API Key", expires_days=None)
    expired_key = key_manager.create_key(description="Expired Test Key", expires_days=-1)

    key_manager._test_valid_key = test_key.plain_key
    key_manager._test_valid_key_id = test_key.id
    key_manager._test_expired_key = expired_key.plain_key
    key_manager._test_expired_key_id = expired_key.id

    return key_manager


class DatabaseManagerMock:
    """More realistic database manager mock for integration testing."""

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.databases = {}
        self._created_dbs = set()

    def list_databases(self):
        db_files = list(self.base_path.glob("*.sqlite"))
        file_dbs = [f.stem for f in db_files]
        all_dbs = set(file_dbs) | set(self._created_dbs) | set(self.databases.keys())
        return list(all_dbs)

    def get_db(self, name):
        if name not in self.databases:
            if name in self._created_dbs or (self.base_path / f"{name}.sqlite").exists():
                self.databases[name] = self._create_mock_db(name)
            else:
                from localvectordb.exceptions import DatabaseNotFoundError

                raise DatabaseNotFoundError(f"Database '{name}' not found")
        return self.databases[name]

    def search_databases(
        self,
        query: str,
        database_names=None,
        search_type="vector",
        return_type="documents",
        k: int = 10,
        score_threshold: float = 0.0,
        filters=None,
        vector_weight: float = 0.7,
        context_window: int = 2,
        context_unit: str = "chunks",
        context_truncate: bool = False,
    ):
        names = database_names or list(self._created_dbs)
        all_results = {}
        from localvectordb.core import QueryResult

        for name in names:
            results = []
            for i in range(min(k, 5)):
                result = QueryResult(id=f"doc{i}", score=0.8, type=return_type, content=f"doc{i} content")
                results.append(result)
            all_results[name] = results
        return all_results

    def get_embeddings_for_model(self, query_texts, provider: str, model: str):
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        return [np.array([0.1, 0.2, 0.3]).tolist() for _ in query_texts]

    def delete_database(self, name: str) -> bool:
        if name in self._created_dbs:
            self._created_dbs.remove(name)
        db_file = self.base_path / f"{name}.sqlite"
        if os.path.exists(db_file):
            os.remove(db_file)
        if name in self.databases:
            self.databases.pop(name)
        return True

    def create_db(self, new_db_name: str, metadata_schema=None, db_config=None, embedding_config=None):
        self._created_dbs.add(new_db_name)
        db_file = self.base_path / f"{new_db_name}.sqlite"
        db_file.touch()
        self.databases[new_db_name] = self._create_mock_db(new_db_name)
        return self.databases[new_db_name]

    def close_all(self):
        pass

    def _create_mock_db(self, name):
        db = Mock()
        db.name = name
        db.embedding_provider = Mock()
        db.embedding_provider.provider_name = "mock"
        db.embedding_provider.model = "test-model"
        db.embedding_provider.embed_sync = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        db.embedding_provider.embed_documents = Mock(return_value=np.array([[0.1, 0.2, 0.3]]))
        db.embedding_provider.embed_query = Mock(return_value=np.array([0.1, 0.2, 0.3]))

        async def mock_embed_batch(texts, batch_size=None):
            return np.array([[0.1, 0.2, 0.3] for _ in texts])

        db.embedding_provider.embed_batch = mock_embed_batch
        db.embedding_dimension = 384
        db.chunking_method = "sentences"
        db.chunk_size = 500
        db.chunk_overlap = 1
        db.metadata_schema = {}
        db.fts_enabled = True
        db._stats = {"documents": 0, "chunks": 0, "index_vectors": 0}
        db.get_stats = lambda: db._stats

        db._documents = {}
        db._next_id = 1

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
                    "content": doc,
                    "metadata": metadata[i] if metadata and i < len(metadata) else {},
                }
                result_ids.append(doc_id)

            db._stats["documents"] = len(db._documents)
            db._stats["chunks"] = len(db._documents) * 2
            db._stats["index_vectors"] = db._stats["chunks"]
            return result_ids

        def mock_get(doc_id):
            if doc_id in db._documents:
                from localvectordb.core import Document

                doc_data = db._documents[doc_id]
                return Document(
                    id=doc_id, content=doc_data["content"], metadata=doc_data["metadata"], content_hash="mock_hash"
                )
            raise DocumentNotFoundError(f"Document '{doc_id}' cannot be found!", doc_id)

        def mock_exists(doc_id):
            if isinstance(doc_id, list):
                return [d in db._documents for d in doc_id]
            return doc_id in db._documents

        def mock_delete(doc_id):
            if isinstance(doc_id, list):
                count = sum(1 for d in doc_id if db._documents.pop(d, None))
            else:
                count = 1 if db._documents.pop(doc_id, None) else 0
            db._stats["documents"] = len(db._documents)
            db._stats["chunks"] = len(db._documents) * 2
            db._stats["index_vectors"] = db._stats["chunks"]
            return count

        def mock_query(query, **kwargs):
            from localvectordb.core import QueryResult

            results = []
            for doc_id, doc_data in list(db._documents.items())[: kwargs.get("k", 5)]:
                result = QueryResult(
                    id=doc_id, score=0.8, type="document", content=doc_data["content"], metadata=doc_data["metadata"]
                )
                results.append(result)
            return results

        # Signatures deliberately mirror the real LocalVectorDB methods so a
        # router calling with a wrong keyword (e.g. count(where=...)) fails
        # here the same way it fails in production.
        def mock_filter(where=None, order_by=None, limit=None, offset=0):
            from localvectordb.core import Document

            results = []
            limit = limit or 100
            docs = list(db._documents.items())[offset : offset + limit]
            for doc_id, doc_data in docs:
                doc = Document(
                    id=doc_id, content=doc_data["content"], metadata=doc_data["metadata"], content_hash="mock_hash"
                )
                results.append(doc)
            return results

        def mock_count(filters=None):
            return len(db._documents)

        def mock_update(doc_id, content=None, metadata=None):
            if doc_id in db._documents:
                if content is not None:
                    db._documents[doc_id]["content"] = content
                if metadata is not None:
                    db._documents[doc_id]["metadata"].update(metadata)
                return True
            return False

        # Server search endpoints call the async query API.
        async def mock_query_async(query, **kwargs):
            return mock_query(query, **kwargs)

        db.upsert = mock_upsert
        db.insert = mock_upsert
        db.get = mock_get
        db.exists = mock_exists
        db.delete = mock_delete
        db.query = mock_query
        db.query_async = mock_query_async
        db.filter = mock_filter
        db.count = mock_count
        db.update = mock_update
        db.close = Mock()

        return db


@pytest.fixture(scope="function")
def integration_app(temp_dir, test_key_manager):
    """Create FastAPI app for integration testing with real-like configuration."""
    from localvectordb_server.routers import register_routers

    app = FastAPI()

    # Set up Config
    config = Config()
    config.database.root_dir = temp_dir
    config.server.cache_enabled = False
    config.server.security.require_api_key = True
    config.server.security.api_key_header = "Authorization"
    config.server.security.key_audit_logging = False
    config.server.security.auto_prune_expired_keys = False

    app.state.config = config
    app.state.key_manager = test_key_manager
    app.state.db_manager = DatabaseManagerMock(temp_dir)

    # Use the same exception-handler registration as create_app so the test app
    # can't drift from production (it previously omitted the HTTPException handler).
    from localvectordb_server.app import register_exception_handlers

    register_exception_handlers(app)

    register_routers(app)
    yield app


@pytest.fixture(scope="function")
def integration_client(integration_app):
    """Create test client for integration testing."""
    return TestClient(integration_app)


@pytest.fixture(scope="function")
def valid_auth_headers(integration_app):
    """Get headers with valid API key for authenticated requests."""
    return {"Authorization": f"Bearer {integration_app.state.key_manager._test_valid_key}"}


@pytest.fixture(scope="function")
def expired_auth_headers(integration_app):
    """Get headers with expired API key for testing expired key handling."""
    return {"Authorization": f"Bearer {integration_app.state.key_manager._test_expired_key}"}


@pytest.mark.integration
@pytest.mark.client
class TestAuthenticationFlow:
    """Test full authentication flow with new KeyManager system."""

    def test_request_without_api_key(self, integration_client):
        response = integration_client.get("/api/v1/databases")
        assert response.status_code == 401
        # Auth failures must use the same {"error": {...}} envelope as every other
        # error (not Starlette's {"detail": ...}).
        body = response.json()
        assert "error" in body and "detail" not in body
        assert body["error"]["code"] == "UNAUTHORIZED"
        assert "message" in body["error"]

    def test_request_with_invalid_api_key(self, integration_client):
        headers = {"Authorization": "Bearer invalid_key_12345"}
        response = integration_client.get("/api/v1/databases", headers=headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_request_with_expired_api_key(self, integration_client, expired_auth_headers):
        response = integration_client.get("/api/v1/databases", headers=expired_auth_headers)
        assert response.status_code == 401

    def test_request_with_valid_api_key(self, integration_client, valid_auth_headers):
        response = integration_client.get("/api/v1/databases", headers=valid_auth_headers)
        assert response.status_code == 200

    def test_api_key_header_formats(self, integration_client, integration_app):
        valid_key = integration_app.state.key_manager._test_valid_key

        headers = {"Authorization": f"Bearer {valid_key}"}
        response = integration_client.get("/api/v1/databases", headers=headers)
        assert response.status_code == 200

        headers = {"Authorization": valid_key}
        response = integration_client.get("/api/v1/databases", headers=headers)
        assert response.status_code == 401

        headers = {"Authorization": f"InvalidType {valid_key}"}
        response = integration_client.get("/api/v1/databases", headers=headers)
        assert response.status_code == 401

    def test_key_validation_updates_last_used(self, integration_client, valid_auth_headers, integration_app):
        # Use the key that valid_auth_headers actually authenticates with.
        key_id = integration_app.state.key_manager._test_valid_key_id
        before = integration_app.state.key_manager.get_key(key_id).last_used

        response = integration_client.get("/api/v1/databases", headers=valid_auth_headers)
        assert response.status_code == 200

        after = integration_app.state.key_manager.get_key(key_id).last_used
        assert after is not None, "last_used should be recorded after an authenticated request"
        if before is not None:
            assert after >= before


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestDatabaseLifecycle:
    """Test complete database lifecycle with authentication."""

    def test_create_and_delete_database_flow(self, integration_client, integration_app, valid_auth_headers):
        create_data = {
            "name": "test_lifecycle_db",
            "embedding_provider": "mock",
            "embedding_model": "test-model",
            "metadata_schema": {"author": {"type": "text", "indexed": True}, "category": "text"},
        }

        with patch("localvectordb.database.LocalVectorDB") as mock_db_class:
            mock_db = integration_app.state.db_manager._create_mock_db("test_lifecycle_db")
            mock_db_class.return_value = mock_db

            response = integration_client.post(
                "/api/v1/databases",
                json=create_data,
                headers=valid_auth_headers,
            )

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            assert "test_lifecycle_db" in result["message"]

        response = integration_client.get("/api/v1/test_lifecycle_db/info", headers=valid_auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "test_lifecycle_db"

        response = integration_client.get("/api/v1/databases", headers=valid_auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert "test_lifecycle_db" in result["databases"]

        with patch("pathlib.Path.exists") as mock_exists, patch("os.remove"):
            mock_exists.return_value = True
            response = integration_client.delete("/api/v1/test_lifecycle_db", headers=valid_auth_headers)
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"

    def test_database_operations_with_metadata_schema(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db(
            "schema_test_db",
            metadata_schema={
                "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                "rating": MetadataField(type=MetadataFieldType.REAL, indexed=True),
                "tags": MetadataField(type=MetadataFieldType.JSON),
            },
        )

        doc_data = {
            "documents": ["First test document with metadata", "Second test document with different metadata"],
            "metadata": [
                {"author": "Alice", "rating": 4.5, "tags": ["test", "first"]},
                {"author": "Bob", "rating": 3.8, "tags": ["test", "second"]},
            ],
        }

        response = integration_client.post(
            "/api/v1/schema_test_db/documents", json=doc_data, headers=valid_auth_headers
        )
        assert response.status_code == 200
        result = response.json()
        assert len(result["ids"]) == 2

        query_data = {"query": "test document", "search_type": "vector", "k": 5}
        response = integration_client.post("/api/v1/schema_test_db/query", json=query_data, headers=valid_auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) >= 0

        filter_data = {"filters": {"author": "Alice"}}
        response = integration_client.post(
            "/api/v1/schema_test_db/filter", json=filter_data, headers=valid_auth_headers
        )
        assert response.status_code == 200
        result = response.json()
        assert len(result["documents"]) >= 0


@pytest.mark.integration
@pytest.mark.client
class TestErrorHandlingIntegration:
    """Test error handling across multiple components."""

    def test_database_not_found_flow(self, integration_client, valid_auth_headers):
        response = integration_client.get("/api/v1/nonexistent_db/info", headers=valid_auth_headers)
        assert response.status_code == 404

    def test_document_not_found_flow(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db("error_test_db")
        response = integration_client.get("/api/v1/error_test_db/documents/nonexistent", headers=valid_auth_headers)
        assert response.status_code == 404

    def test_invalid_request_data_flow(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db("validation_test_db")

        response = integration_client.post("/api/v1/validation_test_db/query", json={}, headers=valid_auth_headers)
        assert response.status_code == 400

        response = integration_client.post(
            "/api/v1/databases", json={"embedding_model": "test"}, headers=valid_auth_headers
        )
        assert response.status_code == 400

    def test_unauthenticated_access_to_protected_endpoints(self, integration_client):
        protected_endpoints = [
            ("/api/v1/databases", "GET"),
            ("/api/v1/databases", "POST"),
            ("/api/v1/test_db/info", "GET"),
            ("/api/v1/test_db/documents", "POST"),
            ("/api/v1/test_db/query", "POST"),
        ]

        for endpoint, method in protected_endpoints:
            if method == "GET":
                response = integration_client.get(endpoint)
            elif method == "POST":
                response = integration_client.post(endpoint, json={})
            assert response.status_code == 401, f"Endpoint {method} {endpoint} should require authentication"


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestMultiDatabaseOperations:
    """Test operations across multiple databases."""

    def test_global_search_across_databases(self, integration_client, integration_app, valid_auth_headers):
        db1 = integration_app.state.db_manager.create_db("global_search_db1")
        db2 = integration_app.state.db_manager.create_db("global_search_db2")

        db1.upsert(["Document in database 1"], [{"source": "db1"}])
        db2.upsert(["Document in database 2"], [{"source": "db2"}])

        search_data = {"query": "document", "search_type": "vector", "k": 5}
        response = integration_client.post("/api/v1/search", json=search_data, headers=valid_auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert len(result["results"]) >= 2

    def test_database_isolation(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db("isolation_db1")
        integration_app.state.db_manager.create_db("isolation_db2")

        doc_data = {"documents": ["Document only in db1"]}
        response = integration_client.post("/api/v1/isolation_db1/documents", json=doc_data, headers=valid_auth_headers)
        assert response.status_code == 200

        query_data = {"query": "Document only in db1", "k": 5}
        response = integration_client.post("/api/v1/isolation_db2/query", json=query_data, headers=valid_auth_headers)
        assert response.status_code == 200
        result = response.json()
        assert len(result["results"]) == 0


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.embedding
class TestEmbeddingIntegration:
    """Test embedding functionality integration."""

    def test_embedding_endpoint_with_database_provider(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db("embedding_test_db")
        embed_data = {"texts": ["test text for embedding"]}
        response = integration_client.post(
            "/api/v1/embedding_test_db/embeddings", json=embed_data, headers=valid_auth_headers
        )
        assert response.status_code == 200
        result = response.json()
        assert "embeddings" in result
        assert len(result["embeddings"]) == 1

    def test_generic_embedding_endpoint(self, integration_client, valid_auth_headers):
        with patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_create:
            mock_provider = Mock()
            mock_provider.embed_batch = AsyncMock(return_value=np.array([[0.1, 0.2, 0.3]]))
            mock_create.return_value = mock_provider

            embed_data = {"texts": ["test text"], "provider": "mock", "model": "test-model"}
            response = integration_client.post("/api/v1/embeddings", json=embed_data, headers=valid_auth_headers)
            assert response.status_code == 200
            result = response.json()
            assert "embeddings" in result


@pytest.mark.integration
@pytest.mark.client
class TestHealthAndMonitoring:
    """Test health check and monitoring endpoints."""

    def test_health_check_integration(self, integration_client):
        with patch("localvectordb_server.routers.health.check_ollama_service") as mock_ollama:
            mock_ollama.return_value = True
            response = integration_client.get("/api/v1/health")
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "healthy"
            assert "version" in result
            assert result["ollama_available"] is True

    def test_health_check_with_service_down(self, integration_client):
        with patch("localvectordb_server.routers.health.check_ollama_service") as mock_ollama:
            mock_ollama.return_value = False
            response = integration_client.get("/api/v1/health")
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "healthy"
            assert result["ollama_available"] is False


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
class TestKeyManagerIntegration:
    """Test integration with the new KeyManager system."""

    def test_key_manager_stats(self, integration_app):
        stats = integration_app.state.key_manager.get_stats()
        assert isinstance(stats, dict)
        assert "total_keys" in stats
        assert "active_keys" in stats
        assert "expired_keys" in stats
        assert stats["total_keys"] >= 2
        assert stats["expired_keys"] >= 1

    def test_key_creation_during_test(self, integration_app):
        initial_count = integration_app.state.key_manager.get_stats()["total_keys"]
        new_key = integration_app.state.key_manager.create_key(
            description="Test key created during test", expires_days=30
        )
        assert new_key.plain_key is not None
        assert new_key.description == "Test key created during test"
        final_count = integration_app.state.key_manager.get_stats()["total_keys"]
        assert final_count == initial_count + 1

    def test_key_validation_accuracy(self, integration_app):
        valid_key = integration_app.state.key_manager._test_valid_key
        assert integration_app.state.key_manager.validate_key(valid_key) is True
        assert integration_app.state.key_manager.validate_key("invalid_key") is False
        expired_key = integration_app.state.key_manager._test_expired_key
        assert integration_app.state.key_manager.validate_key(expired_key) is False


@pytest.mark.integration
@pytest.mark.client
@pytest.mark.database
@pytest.mark.embedding
@pytest.mark.slow
class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_document_management_workflow(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db(
            "workflow_test_db",
            metadata_schema={"author": MetadataField(type=MetadataFieldType.TEXT, indexed=True)},
        )

        doc_data = {
            "documents": ["First document", "Second document"],
            "metadata": [{"author": "Alice"}, {"author": "Bob"}],
        }
        response = integration_client.post(
            "/api/v1/workflow_test_db/documents", json=doc_data, headers=valid_auth_headers
        )
        assert response.status_code == 200
        doc_ids = response.json()["ids"]

        search_data = {"query": "document", "k": 5}
        response = integration_client.post(
            "/api/v1/workflow_test_db/query", json=search_data, headers=valid_auth_headers
        )
        assert response.status_code == 200

        update_data = {"content": "Updated first document", "metadata": {"author": "Alice Updated"}}
        # Partial document update is PATCH (was PUT) in the model-driven API.
        response = integration_client.patch(
            f"/api/v1/workflow_test_db/documents/{doc_ids[0]}", json=update_data, headers=valid_auth_headers
        )
        assert response.status_code == 200

        response = integration_client.get(
            f"/api/v1/workflow_test_db/documents/{doc_ids[0]}", headers=valid_auth_headers
        )
        assert response.status_code == 200
        result = response.json()
        assert "Updated" in result["content"]

        filter_data = {"filters": {"author": "Bob"}}
        response = integration_client.post(
            "/api/v1/workflow_test_db/filter", json=filter_data, headers=valid_auth_headers
        )
        assert response.status_code == 200

        response = integration_client.delete(
            f"/api/v1/workflow_test_db/documents/{doc_ids[1]}", headers=valid_auth_headers
        )
        assert response.status_code == 200

        response = integration_client.get(
            f"/api/v1/workflow_test_db/documents/{doc_ids[1]}", headers=valid_auth_headers
        )
        assert response.status_code == 404

    def test_workflow_with_key_rotation(self, integration_client, integration_app, valid_auth_headers):
        integration_app.state.db_manager.create_db("rotation_test_db")

        doc_data = {"documents": ["Document with original key"]}
        response = integration_client.post(
            "/api/v1/rotation_test_db/documents", json=doc_data, headers=valid_auth_headers
        )
        assert response.status_code == 200

        original_key_id = integration_app.state.key_manager._test_valid_key_id
        new_key_record = integration_app.state.key_manager.rotate_key(original_key_id)
        assert new_key_record is not None

        response = integration_client.get("/api/v1/databases", headers=valid_auth_headers)
        assert response.status_code == 401

        new_auth_headers = {"Authorization": f"Bearer {new_key_record.plain_key}"}
        response = integration_client.get("/api/v1/databases", headers=new_auth_headers)
        assert response.status_code == 200


@pytest.fixture(scope="function")
def read_only_headers(integration_app):
    """Auth headers for a READ_ONLY API key."""
    from localvectordb_server.keymanager import PermissionLevel

    km = integration_app.state.key_manager
    ro_key = km.create_key(description="Read-only key", permission_level=PermissionLevel.READ_ONLY)
    return {"Authorization": f"Bearer {ro_key.plain_key}"}


@pytest.mark.integration
@pytest.mark.client
class TestPermissionEnforcement:
    """A READ_ONLY key may read but must be rejected (403) on every write endpoint."""

    def test_read_only_key_allows_reads(self, integration_client, read_only_headers):
        response = integration_client.get("/api/v1/databases", headers=read_only_headers)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "method, path, json_body",
        [
            ("post", "/api/v1/databases", {"name": "perm_new_db"}),
            ("post", "/api/v1/perm_db/documents", {"documents": ["hello"]}),
            ("post", "/api/v1/perm_db/documents/insert", {"documents": ["hello"]}),
            ("patch", "/api/v1/perm_db/documents/doc_1", {"content": "changed"}),
            ("delete", "/api/v1/perm_db/documents/doc_1", None),
            ("delete", "/api/v1/perm_db", None),
        ],
    )
    def test_read_only_key_blocked_on_writes(self, integration_client, read_only_headers, method, path, json_body):
        request = getattr(integration_client, method)
        response = (
            request(path, headers=read_only_headers, json=json_body)
            if json_body is not None
            else request(path, headers=read_only_headers)
        )
        assert response.status_code == 403, f"{method.upper()} {path} should require write permission"
        body = response.json()
        # Errors use the shared {"error": {...}} envelope, never Starlette's {"detail": ...}.
        assert "error" in body and "detail" not in body

    def test_read_write_key_allows_writes(self, integration_client, valid_auth_headers):
        """Sanity check: the same write endpoint succeeds for a READ_WRITE key."""
        response = integration_client.post("/api/v1/databases", headers=valid_auth_headers, json={"name": "perm_rw_db"})
        assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.client
class TestAuthHeaderNegatives:
    """Malformed/missing Authorization headers are rejected with 401."""

    @pytest.mark.parametrize(
        "header",
        [
            None,  # missing entirely
            "",  # empty
            "lvdb_sometoken",  # no scheme
            "Basic lvdb_sometoken",  # wrong scheme
            "Bearer",  # scheme with no token
            "Bearer not_an_lvdb_token",  # token missing lvdb_ prefix
            "Bearer lvdb_totally_invalid_key",  # well-formed prefix but not a real key
        ],
    )
    def test_bad_auth_header_is_401(self, integration_client, header):
        headers = {"Authorization": header} if header is not None else {}
        response = integration_client.get("/api/v1/databases", headers=headers)
        assert response.status_code == 401
        assert "error" in response.json()


@pytest.fixture(scope="function")
def transport_patch(integration_app):
    """Route RemoteVectorDB's real sync httpx layer through the ASGI app.

    The TestClient is built from the client's own ``_get_headers()`` so the real
    Authorization-header injection and URL/payload construction are exercised
    end-to-end against a real FastAPI server (backed by the mock DB manager).
    """
    from localvectordb.client import RemoteVectorDB

    def fake_ensure_sync_client(self):
        if getattr(self, "_patched_client", None) is None:
            self._patched_client = TestClient(
                integration_app, base_url="http://testserver", headers=self._get_headers()
            )
        return self._patched_client

    with patch.object(RemoteVectorDB, "_ensure_sync_client", fake_ensure_sync_client):
        yield


@pytest.mark.integration
@pytest.mark.client
class TestRemoteTransportRoundTrip:
    """End-to-end RemoteVectorDB <-> FastAPI over a real (ASGI) transport."""

    def test_create_upsert_query_get_delete_over_http(self, integration_app, transport_patch):
        from localvectordb.client import RemoteVectorDB

        key = integration_app.state.key_manager._test_valid_key
        db = RemoteVectorDB(name="remote_rt", base_url="http://testserver", api_key=key)

        ids = db.upsert(["hello world", "second document"])
        assert isinstance(ids, list) and len(ids) == 2

        results = db.query("hello", k=5)
        assert isinstance(results, list)

        doc = db.get(ids[0])
        assert doc.id == ids[0]

        # Regression: the count endpoint called db.count(where=...) and 500'd.
        assert db.count() == 2

        listed = db.filter(where=None, limit=10)
        assert {d.id for d in listed} == set(ids)

        deleted = db.delete(ids[0])
        assert deleted >= 1
        assert db.count() == 1

    def test_bad_api_key_is_rejected_over_http(self, integration_app, transport_patch):
        from localvectordb.client import RemoteVectorDB

        # A bogus key must fail authentication at the server (401 -> PermissionError).
        with pytest.raises(PermissionError):
            RemoteVectorDB(
                name="remote_rt",
                base_url="http://testserver",
                api_key="lvdb_totally_bogus_key",
                create_if_not_exists=False,
            )


@pytest.mark.integration
@pytest.mark.client
class TestUncoveredRoutersAuth:
    """Every previously-uncovered router endpoint is registered and auth-protected."""

    @pytest.mark.parametrize(
        "method, path",
        [
            ("get", "/api/v1/any_db/schema"),
            ("post", "/api/v1/any_db/query/stream"),
            ("post", "/api/v1/any_db/upload"),
            ("post", "/api/v1/any_db/compare"),
            ("post", "/api/v1/any_db/factcheck"),
            ("get", "/api/v1/any_db/tuning"),
        ],
    )
    def test_endpoint_requires_auth(self, integration_client, method, path):
        response = getattr(integration_client, method)(path)
        assert response.status_code == 401


@pytest.mark.integration
@pytest.mark.client
class TestUncoveredRoutersHappyPath:
    """Happy-path e2e coverage for routers that previously had zero tests."""

    def _prepare_db(self, app, name):
        app.state.db_manager.create_db(name)
        return app.state.db_manager.get_db(name)

    def test_get_schema_info(self, integration_app, integration_client, valid_auth_headers):
        mock_db = self._prepare_db(integration_app, "schema_db")
        mock_db.get_metadata_schema_info.return_value = {"fields": {}, "field_count": 0}
        response = integration_client.get("/api/v1/schema_db/schema", headers=valid_auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["database"] == "schema_db"
        assert body["status"] == "success"

    def test_compare_documents(self, integration_app, integration_client, valid_auth_headers):
        mock_db = self._prepare_db(integration_app, "cmp_db")
        mock_db.compare_documents.return_value = 0.87
        response = integration_client.post(
            "/api/v1/cmp_db/compare",
            json={"doc_id_1": "a", "doc_id_2": "b"},
            headers=valid_auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["similarity"] == 0.87

    def test_get_tuning(self, integration_app, integration_client, valid_auth_headers):
        mock_db = self._prepare_db(integration_app, "tune_db")
        mock_db.get_sqlite_tuning.return_value = {"profile": "balanced", "pragmas": {}}
        response = integration_client.get("/api/v1/tune_db/tuning", headers=valid_auth_headers)
        assert response.status_code == 200
        assert response.json()["tuning"]["profile"] == "balanced"

    def test_upload_text_file(self, integration_app, integration_client, valid_auth_headers):
        self._prepare_db(integration_app, "upload_db")
        integration_app.state.config.server.file_upload_enabled = True
        files = {"files": ("note.txt", b"hello world from an uploaded file", "text/plain")}
        response = integration_client.post(
            "/api/v1/upload_db/upload",
            files=files,
            data={"use_filename_as_id": "true"},
            headers=valid_auth_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["files_processed"] == 1
        assert body["status"] in ("success", "partial")
        assert body["document_ids"]  # a real id was assigned

    def test_streaming_query_returns_sse(self, integration_app, integration_client, valid_auth_headers):
        mock_db = self._prepare_db(integration_app, "stream_db")
        # Remove the auto-created cursor/stream attributes so the router falls through
        # to the plain db.query() path (the mock provides a real query()).
        del mock_db.query_cursor_async
        del mock_db.query_stream
        integration_client.post(
            "/api/v1/stream_db/documents", json={"documents": ["hello world"]}, headers=valid_auth_headers
        )
        with integration_client.stream(
            "POST",
            "/api/v1/stream_db/query/stream",
            json={"query": "hello", "search_type": "vector", "k": 5},
            headers=valid_auth_headers,
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert "done" in body

    def test_missing_db_returns_404(self, integration_client, valid_auth_headers):
        response = integration_client.get("/api/v1/does_not_exist/schema", headers=valid_auth_headers)
        assert response.status_code == 404
