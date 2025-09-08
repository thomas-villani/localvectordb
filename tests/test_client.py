"""
Tests for localvectordb.client module.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from localvectordb.client import RemoteVectorDB, Document, QueryResult
from localvectordb.core import ChunkPosition
from localvectordb.exceptions import (
    DatabaseNotFoundError, DuplicateDocumentIDError,
    EmbeddingError, BaseLocalVectorDBException, DocumentNotFoundError
)

@pytest.fixture(autouse=True)
def mock_httpx_client():
    """
    Module-level fixture that automatically mocks httpx.Client for all tests.
    Provides sensible defaults that work for most tests, can be customized per test.
    """
    with patch('httpx.Client') as mock_client_class:
        mock_client = Mock()

        # Default successful response that works for most tests
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "config": {
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "embedding_dimension": 384,
                "chunking_method": "sentences",
                "chunk_size": 500,
                "chunk_overlap": 1,
                "fts_enabled": True,
                "metadata_schema": {}
            },
            "stats": {
                "documents": 0,
                "chunks": 0,
                "index_vectors": 0
            }
        }
        mock_response.text = "OK"

        # Configure all HTTP methods to return the default response
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response
        mock_client.put.return_value = mock_response
        mock_client.delete.return_value = mock_response
        mock_client.patch.return_value = mock_response
        mock_client.request.return_value = mock_response

        # Handle both context manager and direct usage
        mock_client.__enter__ = lambda self: mock_client
        mock_client.__exit__ = lambda self, *args: None
        mock_client_class.return_value = mock_client
        mock_client_class.return_value.__enter__.return_value = mock_client

        yield mock_client


@pytest.mark.client
class TestRemoteVectorDBInitialization:
    """Test RemoteVectorDB initialization."""

    def test_create_new_database(self, mock_httpx_client, sample_metadata_schema):
        """Test creating a new remote database."""
        # Customize the response for this specific test
        mock_httpx_client.request.return_value.json.return_value = {
            "config": {
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
                "embedding_dimension": 384,
                "chunking_method": "sentences",
                "chunk_size": 500,
                "chunk_overlap": 1,
                "fts_enabled": True,
                "metadata_schema": {
                    "author": {"type": "text", "indexed": True},
                    "rating": {"type": "real", "indexed": False}
                }
            }
        }
        mock_httpx_client.get.return_value = mock_httpx_client.request.return_value  # For database info check

        db = RemoteVectorDB(
            name="test_db",
            base_url="http://localhost:5000",
            api_key="test-key",
            metadata_schema=sample_metadata_schema,
            create_if_not_exists=True
        )

        assert db.name == "test_db"
        assert db.base_url == "http://localhost:5000"
        assert db.api_key == "test-key"
        assert db.embedding_dimension == 384

    def test_connect_existing_database(self, mock_httpx_client):
        """Test connecting to existing database."""
        db = RemoteVectorDB(
            name="existing_db",
            base_url="http://localhost:5000",
            create_if_not_exists=False
        )

        assert db.name == "existing_db"
        # Should call get to load database info
        mock_httpx_client.request.assert_called()

    def test_database_not_found(self, mock_httpx_client):
        """Test error when database doesn't exist."""
        # Override default response for 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"code": "database_not_found", "message": "Database not found"}}
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(DatabaseNotFoundError):
            RemoteVectorDB(
                name="nonexistent",
                base_url="http://localhost:5000",
                create_if_not_exists=False
            )

    def test_default_parameters(self, mock_httpx_client):
        """Test default parameters."""
        db = RemoteVectorDB("test_db")

        assert db.base_url == "http://127.0.0.1:5000"
        assert db.api_key is None
        assert db.request_timeout is None

    def test_url_normalization(self, mock_httpx_client):
        """Test URL normalization."""
        db = RemoteVectorDB("test_db", base_url="http://localhost:5000/")
        assert db.base_url == "http://localhost:5000"


@pytest.mark.client
class TestRemoteVectorDBDocumentOperations:
    """Test RemoteVectorDB document operations."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        """Create a mock remote database for testing."""
        return RemoteVectorDB("test_db", api_key="test-key")

    def test_upsert_single_document(self, mock_httpx_client, mock_db):
        """Test upserting a single document."""
        # Customize response for this test
        mock_httpx_client.request.return_value.json.return_value = {"ids": ["doc_1"]}

        result = mock_db.upsert("Test document")

        assert result == ["doc_1"]
        mock_httpx_client.request.assert_called()

        # Check request payload
        call_args = mock_httpx_client.request.call_args
        assert call_args[0][1] == "http://127.0.0.1:5000/api/v1/test_db/documents"
        payload = call_args[1]["json"]
        assert payload["documents"] == ["Test document"]

    def test_upsert_multiple_documents(self, mock_httpx_client, mock_db):
        """Test upserting multiple documents."""
        mock_httpx_client.request.return_value.json.return_value = {"ids": ["doc_1", "doc_2"]}

        documents = ["Doc 1", "Doc 2"]
        metadata = [{"author": "A"}, {"author": "B"}]
        ids = ["custom_1", "custom_2"]

        result = mock_db.upsert(documents, metadata=metadata, ids=ids)

        assert result == ["doc_1", "doc_2"]

        # Check request payload
        payload = mock_httpx_client.request.call_args[1]["json"]
        assert payload["documents"] == documents
        assert payload["metadata"] == metadata
        assert payload["ids"] == ids

    def test_insert_new_documents(self, mock_httpx_client, mock_db):
        """Test inserting new documents."""
        mock_httpx_client.request.return_value.json.return_value = {"ids": ["doc_1", "doc_2"]}

        result = mock_db.insert(
            ["Doc 1", "Doc 2"],
            errors="ignore",
            similarity_threshold=0.95
        )

        assert result == ["doc_1", "doc_2"]

        # Check endpoint and payload
        call_args = mock_httpx_client.request.call_args
        assert "/documents/insert" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["errors"] == "ignore"
        assert payload["similarity_threshold"] == 0.95

    def test_insert_duplicate_error(self, mock_httpx_client, mock_db):
        """Test insert with duplicate document error."""
        # Override response for 409 error
        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.json.return_value = {
            "error": {
                "code": "duplicate_document_id",
                "message": "Document already exists"
            }
        }
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(DuplicateDocumentIDError):
            mock_db.insert("Test doc", ids="existing_id")

    def test_get_single_document(self, mock_httpx_client, mock_db):
        """Test getting a single document."""
        mock_httpx_client.request.return_value.json.return_value = {
            "id": "doc_1",
            "content": "Test content",
            "metadata": {"author": "Test"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "content_hash": "hash123"
        }

        result = mock_db.get("doc_1")

        assert isinstance(result, Document)
        assert result.id == "doc_1"
        assert result.content == "Test content"
        assert result.metadata == {"author": "Test"}
        assert isinstance(result.created_at, datetime)

        # Check endpoint
        mock_httpx_client.request.assert_called_with("GET",
            "http://127.0.0.1:5000/api/v1/test_db/documents/doc_1"
        )

    def test_get_multiple_documents(self, mock_httpx_client, mock_db):
        """Test getting multiple documents."""
        # Mock multiple individual requests
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "documents": [
                {
                    "id": "doc_1",
                    "content": "Content 1",
                    "metadata": {},
                    "content_hash": "hash1"
                },
                {
                    "id": "doc_2",
                    "content": "Content 2",
                    "metadata": {},
                    "content_hash": "hash2"
                }
            ],
            "returned_ids": ["doc_1", "doc_2"],
            "missing_ids": []
        }
        # mock_response2 = Mock()
        # mock_response2.status_code = 200
        # mock_response2.json.return_value = {
        #     "id": "doc_2",
        #     "content": "Content 2",
        #     "metadata": {},
        #     "content_hash": "hash2"
        # }
        # mock_httpx_client.request.side_effect = [mock_response1, mock_response2]
        mock_httpx_client.request.return_value = mock_response1

        result = mock_db.get(["doc_1", "doc_2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(doc, Document) for doc in result)
        assert result[0].id == "doc_1"
        assert result[1].id == "doc_2"

        # Should make two separate requests (and others made from trying to get db info)
        assert mock_httpx_client.request.call_count >= 2

    def test_get_nonexistent_document(self, mock_httpx_client, mock_db):
        """Test getting nonexistent document."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            'error': {
                "message": f"Document 'nonexistent' not found in database '{mock_db.name}'",
                "code": "DOCUMENT_NOT_FOUND"
            }
        }
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(DocumentNotFoundError):
            result = mock_db.get("nonexistent")


    def test_exists_documents(self, mock_httpx_client, mock_db):
        """Test checking document existence."""
        mock_httpx_client.request.return_value.json.return_value = {"exists": [True, False, True]}

        result = mock_db.exists(["doc_1", "doc_2", "doc_3"])

        assert result == [True, False, True]

        # Check request
        call_args = mock_httpx_client.request.call_args
        assert "/documents/exists" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["ids"] == ["doc_1", "doc_2", "doc_3"]

    def test_exists_single_document(self, mock_httpx_client, mock_db):
        """Test checking single document existence."""
        mock_httpx_client.request.return_value.json.return_value = {"exists": [True]}

        result = mock_db.exists("doc_1")

        assert result is True

    def test_delete_documents(self, mock_httpx_client, mock_db):
        """Test deleting documents."""
        mock_httpx_client.request.return_value.json.return_value = {"deleted_count": 1}

        result = mock_db.delete(["doc_1", "doc_2"])

        assert result == 2  # Two requests, each deleting 1.
        # httpx.Client.request is called at least 2 times.
        assert mock_httpx_client.request.call_count >= 2

    def test_update_document(self, mock_httpx_client, mock_db):
        """Test updating a document."""
        mock_httpx_client.request.return_value.json.return_value = {"updated": True}

        result = mock_db.update(
            "doc_1",
            content="New content",
            metadata={"author": "New Author"}
        )

        assert result is True

        # Check request
        call_args = mock_httpx_client.request.call_args
        assert "/documents/doc_1" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["content"] == "New content"
        assert payload["metadata"] == {"author": "New Author"}

    def test_update_nonexistent_document(self, mock_httpx_client, mock_db):
        """Test updating nonexistent document."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"code": "database_not_found", "message": "Not found"}}
        mock_httpx_client.request.return_value = mock_response

        result = mock_db.update("nonexistent", content="New content")

        assert result is False


@pytest.mark.client
class TestRemoteVectorDBQuery:
    """Test RemoteVectorDB query functionality."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        """Create a mock remote database for testing."""
        return RemoteVectorDB("test_db", api_key="test-key")

    def test_vector_query(self, mock_httpx_client, mock_db):
        """Test vector similarity query."""
        mock_httpx_client.request.return_value.json.return_value = {
            "results": [
                {
                    "id": "doc_1",
                    "score": 0.95,
                    "type": "document",
                    "content": "Test content",
                    "metadata": {"author": "Test"}
                }
            ]
        }

        results = mock_db.query("test query", search_type="vector", k=5)

        assert len(results) == 1
        assert isinstance(results[0], QueryResult)
        assert results[0].id == "doc_1"
        assert results[0].score == 0.95
        assert results[0].type == "document"

        # Check request
        call_args = mock_httpx_client.request.call_args
        assert "/query" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["query"] == "test query"
        assert payload["search_type"] == "vector"
        assert payload["k"] == 5

    def test_keyword_query(self, mock_httpx_client, mock_db):
        """Test keyword search query."""
        mock_httpx_client.request.return_value.json.return_value = {
            "results": [
                {
                    "id": "doc_1:0",
                    "score": 0.85,
                    "type": "chunk",
                    "content": "Test chunk",
                    "metadata": {"author": "Test"},
                    "document_id": "doc_1",
                    "position": {"start": 0, "end": 10, "line": 1, "column": 1, "end_line": 1, "end_column": 10},
                }
            ]
        }

        results = mock_db.query(
            "test",
            search_type="keyword",
            return_type="chunks"
        )

        assert len(results) == 1
        result = results[0]
        assert result.type == "chunk"
        assert result.document_id == "doc_1"
        assert isinstance(result.position, ChunkPosition)

    def test_hybrid_query(self, mock_httpx_client, mock_db):
        """Test hybrid search query."""
        mock_httpx_client.request.return_value.json.return_value = {"results": []}

        mock_db.query(
            "test query",
            search_type="hybrid",
            vector_weight=0.7,
            score_threshold=0.5,
            filters={"author": "Test"}
        )

        # Check request payload
        payload = mock_httpx_client.request.call_args[1]["json"]
        assert payload["search_type"] == "hybrid"
        assert payload["vector_weight"] == 0.7
        assert payload["score_threshold"] == 0.5
        assert payload["filters"] == {"author": "Test"}

    def test_filter_documents(self, mock_httpx_client, mock_db):
        """Test filtering documents."""
        mock_httpx_client.request.return_value.json.return_value = {
            "documents": [
                {
                    "id": "doc_1",
                    "content": "Test content",
                    "metadata": {"author": "Test"},
                    "content_hash": "hash1"
                }
            ]
        }

        results = mock_db.filter(
            where={"author": "Test"},
            order_by="created_at DESC",
            limit=10,
            offset=5
        )

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].id == "doc_1"

        # Check request
        call_args = mock_httpx_client.request.call_args
        assert "/filter" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["where"] == {"author": "Test"}
        assert payload["order_by"] == "created_at DESC"
        assert payload["limit"] == 10
        assert payload["offset"] == 5


@pytest.mark.client
class TestRemoteVectorDBProperties:
    """Test RemoteVectorDB properties and utility methods."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        """Create a mock remote database for testing."""
        db = RemoteVectorDB("test_db")
        # Set mock properties
        db._embedding_model = "test-model"
        db._embedding_provider = "test-provider"
        db._embedding_dimension = 384
        db._chunk_size = 500
        db._chunk_overlap = 50
        db._chunking_method = "sentences"
        db._enable_fts = True
        return db

    def test_properties(self, mock_httpx_client, mock_db):
        """Test various properties."""
        assert mock_db.embedding_model == "test-model"
        assert mock_db.embedding_dimension == 384
        assert mock_db.chunk_size == 500
        assert mock_db.chunk_overlap == 50
        assert mock_db.chunking_method == "sentences"
        assert mock_db.fts_enabled is True

    def test_stats_property(self, mock_httpx_client, mock_db):
        """Test stats property."""
        mock_httpx_client.get.return_value.json.return_value = {
            "stats": {
                "documents": 100,
                "chunks": 500,
                "index_vectors": 500
            }
        }

        stats = mock_db.get_stats()

        assert stats["documents"] == 100
        assert stats["chunks"] == 500
        assert stats["index_vectors"] == 500

    def test_save_method(self, mock_httpx_client, mock_db):
        """Test save method (no-op for remote client)."""
        # Should not raise any errors
        mock_db.save()


@pytest.mark.client
class TestRemoteVectorDBLegacyMethods:
    """Test legacy method compatibility."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        """Create a mock remote database for testing."""
        return RemoteVectorDB("test_db")

    def test_hybrid_query_method(self, mock_httpx_client, mock_db):
        """Test legacy hybrid_query method."""
        with patch.object(mock_db, 'query') as mock_query:
            mock_query.return_value = []

            mock_db.hybrid_query(
                "test query",
                k=10,
                vector_weight=0.8,
                metadata_filters={"author": "Test"}
            )

            mock_query.assert_called_once_with(
                "test query",
                search_type="hybrid",
                k=10,
                filters={"author": "Test"},
                vector_weight=0.8
            )

    def test_keyword_search_method(self, mock_httpx_client, mock_db):
        """Test legacy keyword_search method."""
        with patch.object(mock_db, 'query') as mock_query:
            mock_query.return_value = []

            mock_db.keyword_search(
                "test query",
                k=5,
                metadata_filters={"category": "test"}
            )

            mock_query.assert_called_once_with(
                "test query",
                search_type="keyword",
                k=5,
                filters={"category": "test"}
            )


@pytest.mark.client
class TestRemoteVectorDBErrorHandling:
    """Test RemoteVectorDB error handling."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        """Create a mock remote database for testing."""
        return RemoteVectorDB("test_db", api_key="test-key")

    def test_authentication_error(self, mock_httpx_client, mock_db):
        """Test authentication error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(PermissionError, match="Authentication failed"):
            mock_db.upsert("Test document")

    def test_embedding_error(self, mock_httpx_client, mock_db):
        """Test embedding error handling."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": {
            "code": "embedding_error",
            "message": "Embedding model not available"
        }}
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(EmbeddingError):
            mock_db.upsert("Test document")

    def test_generic_error(self, mock_httpx_client, mock_db):
        """Test generic error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.json.return_value = {"error": {"message": "Internal server error", "code": "unknown"}}
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(BaseLocalVectorDBException):
            mock_db.upsert("Test document")

    def test_malformed_response(self, mock_httpx_client, mock_db):
        """Test handling of malformed JSON response."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Malformed response"
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(ValueError):
            mock_db.upsert("Test document")


class TestRemoteVectorDBUtilityMethods:
    """Test RemoteVectorDB utility methods."""

    def test_database_exists_true(self, mock_httpx_client):
        """Test database_exists method when database exists."""
        mock_httpx_client.get.return_value.json.return_value = {"databases": ["test_db", "other_db"]}

        result = RemoteVectorDB.database_exists("test_db", api_key="test-key")

        assert result is True
        mock_httpx_client.get.assert_called_with(
            "http://127.0.0.1:5000/api/v1/databases",
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"}
        )

    def test_database_exists_false(self, mock_httpx_client):
        """Test database_exists method when database doesn't exist."""
        mock_httpx_client.get.return_value.json.return_value = {"databases": ["other_db"]}

        result = RemoteVectorDB.database_exists("test_db")

        assert result is False

    def test_database_exists_error(self, mock_httpx_client):
        """Test database_exists method with connection error."""
        mock_httpx_client.get.side_effect = Exception("Connection failed")

        with pytest.raises(Exception) as exc:
            RemoteVectorDB.database_exists("test_db")

        assert exc.value.args[0] == "Connection failed"

    def test_get_headers_with_api_key(self, mock_httpx_client):
        """Test header generation with API key."""
        db = RemoteVectorDB("test_db", api_key="test-key")
        headers = db._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"

    def test_get_headers_without_api_key(self, mock_httpx_client):
        """Test header generation without API key."""
        db = RemoteVectorDB("test_db")
        headers = db._get_headers()

        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers

    def test_build_url(self, mock_httpx_client):
        """Test URL building."""
        db = RemoteVectorDB("test_db", base_url="http://localhost:5000")

        url = db._build_url("/api/v1/test")
        assert url == "http://localhost:5000/api/v1/test"


@pytest.mark.client
class TestDocumentClass:
    """Test Document dataclass for remote client."""

    def test_from_dict_complete(self):
        """Test creating Document from complete dict."""
        data = {
            "id": "doc_1",
            "content": "Test content",
            "metadata": {"author": "Test"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "content_hash": "hash123"
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc_1"
        assert doc.content == "Test content"
        assert doc.metadata == {"author": "Test"}
        assert isinstance(doc.created_at, datetime)
        assert isinstance(doc.updated_at, datetime)
        assert doc.content_hash == "hash123"

    def test_from_dict_minimal(self):
        """Test creating Document from minimal dict."""
        data = {
            "id": "doc_1",
            "content": "Test content"
        }

        doc = Document.from_dict(data)

        assert doc.id == "doc_1"
        assert doc.content == "Test content"
        assert doc.metadata == {}
        assert doc.created_at is None
        assert doc.updated_at is None
        assert doc.content_hash is not None

    def test_from_dict_none(self):
        """Test creating Document from None."""
        doc = Document.from_dict(None)
        assert doc is None

    def test_from_dict_empty(self):
        """Test creating Document from empty dict."""
        doc = Document.from_dict({})
        assert doc is None


@pytest.mark.client
class TestQueryResultClass:
    """Test QueryResult dataclass for remote client."""

    def test_from_dict_document_result(self):
        """Test creating document QueryResult from dict."""
        data = {
            "id": "doc_1",
            "score": 0.95,
            "type": "document",
            "content": "Test content",
            "metadata": {"author": "Test"}
        }

        result = QueryResult.from_dict(data)

        assert result.id == "doc_1"
        assert result.score == 0.95
        assert result.type == "document"
        assert result.content == "Test content"
        assert result.metadata == {"author": "Test"}
        assert result.document_id is None
        assert result.position is None

    def test_from_dict_chunk_result(self):
        """Test creating chunk QueryResult from dict."""
        data = {
            "id": "doc_1:0",
            "score": 0.85,
            "type": "chunk",
            "content": "Test chunk",
            "metadata": {"author": "Test"},
            "document_id": "doc_1",
            "position": {"start": 0, "end": 10, "line": 1, "column": 1, "end_line": 1, "end_column": 10},
        }

        result = QueryResult.from_dict(data)

        assert result.id == "doc_1:0"
        assert result.type == "chunk"
        assert result.document_id == "doc_1"
        assert isinstance(result.position, ChunkPosition)
        assert result.position.start == 0
        assert result.position.end == 10

    def test_from_dict_none(self):
        """Test creating QueryResult from None."""
        result = QueryResult.from_dict(None)
        assert result is None

    def test_from_dict_minimal(self):
        """Test creating QueryResult from minimal dict."""
        data = {
            "id": "doc_1",
            "content": "Test content"
        }

        result = QueryResult.from_dict(data)

        assert result.id == "doc_1"
        assert result.score == 0.0  # Default
        assert result.type == "document"  # Default
        assert result.content == "Test content"
        assert result.metadata == {}