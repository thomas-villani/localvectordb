"""
Tests for localvectordb.client module.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from localvectordb.client import Document, QueryResult, RemoteVectorDB
from localvectordb.core import ChunkPosition
from localvectordb.exceptions import (
    BaseLocalVectorDBException,
    DatabaseNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentIDError,
    EmbeddingError,
)


@pytest.fixture(autouse=True)
def mock_httpx_client():
    """
    Module-level fixture that automatically mocks httpx.Client for all tests.
    Provides sensible defaults that work for most tests, can be customized per test.
    """
    with patch("httpx.Client") as mock_client_class:
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
                "metadata_schema": {},
            },
            "stats": {"documents": 0, "chunks": 0, "index_vectors": 0},
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
                    "rating": {"type": "real", "indexed": False},
                },
            }
        }
        mock_httpx_client.get.return_value = mock_httpx_client.request.return_value  # For database info check

        db = RemoteVectorDB(
            name="test_db",
            base_url="http://localhost:5000",
            api_key="test-key",
            metadata_schema=sample_metadata_schema,
            create_if_not_exists=True,
        )

        assert db.name == "test_db"
        assert db.base_url == "http://localhost:5000"
        assert db.api_key == "test-key"
        assert db.embedding_dimension == 384

    def test_connect_existing_database(self, mock_httpx_client):
        """Test connecting to existing database."""
        db = RemoteVectorDB(name="existing_db", base_url="http://localhost:5000", create_if_not_exists=False)

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
            RemoteVectorDB(name="nonexistent", base_url="http://localhost:5000", create_if_not_exists=False)

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
        assert call_args[0][1] == "http://127.0.0.1:5000/api/v1/databases/test_db/documents"
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

        result = mock_db.insert(["Doc 1", "Doc 2"], errors="ignore", similarity_threshold=0.95)

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
            "error": {"code": "duplicate_document_id", "message": "Document already exists"}
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
            "content_hash": "hash123",
        }

        result = mock_db.get("doc_1")

        assert isinstance(result, Document)
        assert result.id == "doc_1"
        assert result.content == "Test content"
        assert result.metadata == {"author": "Test"}
        assert isinstance(result.created_at, datetime)

        # Check endpoint
        mock_httpx_client.request.assert_called_with(
            "GET", "http://127.0.0.1:5000/api/v1/databases/test_db/documents/doc_1"
        )

    def test_get_multiple_documents(self, mock_httpx_client, mock_db):
        """Test getting multiple documents."""
        # Mock multiple individual requests
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "documents": [
                {"id": "doc_1", "content": "Content 1", "metadata": {}, "content_hash": "hash1"},
                {"id": "doc_2", "content": "Content 2", "metadata": {}, "content_hash": "hash2"},
            ],
            "returned_ids": ["doc_1", "doc_2"],
            "missing_ids": [],
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
            "error": {
                "message": f"Document 'nonexistent' not found in database '{mock_db.name}'",
                "code": "DOCUMENT_NOT_FOUND",
            }
        }
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(DocumentNotFoundError):
            mock_db.get("nonexistent")

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
        mock_httpx_client.request.return_value.json.return_value = {"deleted_count": 2}

        result = mock_db.delete(["doc_1", "doc_2"])

        assert result == 2  # Two requests, each deleting 1.
        # httpx.Client.request is called at least 2 times.
        assert mock_httpx_client.request.call_count >= 2

    def test_update_document(self, mock_httpx_client, mock_db):
        """Test updating a document."""
        mock_httpx_client.request.return_value.json.return_value = {"updated": True}

        result = mock_db.update("doc_1", content="New content", metadata={"author": "New Author"})

        assert result is True

        # Check request
        call_args = mock_httpx_client.request.call_args
        assert "/documents/doc_1" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["content"] == "New content"
        assert payload["metadata"] == {"author": "New Author"}

    def test_update_nonexistent_document_raises(self, mock_httpx_client, mock_db):
        """A missing document must raise, matching LocalVectorDB.update().

        It used to be swallowed into a False return, which conflated "not found"
        with "no updates needed" and diverged from the local backend.
        """
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {"code": "document_not_found", "message": "Document not found: nonexistent"}
        }
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(DocumentNotFoundError):
            mock_db.update("nonexistent", content="New content")

    def test_update_nonexistent_database_raises(self, mock_httpx_client, mock_db):
        """A missing database must raise too; it used to be swallowed into False."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": {"code": "database_not_found", "message": "Not found"}}
        mock_httpx_client.request.return_value = mock_response

        with pytest.raises(DatabaseNotFoundError):
            mock_db.update("doc_1", content="New content")

    def test_update_noop_returns_false(self, mock_httpx_client, mock_db):
        """False means 'no updates needed', never 'not found'."""
        mock_httpx_client.request.return_value.json.return_value = {"updated": False}

        assert mock_db.update("doc_1", content="identical content") is False

    def test_update_with_empty_content_reaches_the_wire(self, mock_httpx_client, mock_db):
        # content="" clears the document. Short-circuiting on falsiness (rather than
        # on None) used to swallow it client-side, so the edit never happened.
        mock_httpx_client.request.return_value.json.return_value = {"updated": True}

        result = mock_db.update("doc_1", content="")

        assert result is True
        payload = mock_httpx_client.request.call_args[1]["json"]
        assert payload["content"] == ""

    def test_update_with_empty_metadata_reaches_the_wire(self, mock_httpx_client, mock_db):
        mock_httpx_client.request.return_value.json.return_value = {"updated": False}

        mock_db.update("doc_1", metadata={})

        payload = mock_httpx_client.request.call_args[1]["json"]
        assert payload["metadata"] == {}

    def test_update_with_nothing_to_do_makes_no_request(self, mock_httpx_client, mock_db):
        # `mock_db` construction already issues a GET /info, so reset before asserting.
        mock_httpx_client.request.reset_mock()

        result = mock_db.update("doc_1")

        assert result is False
        mock_httpx_client.request.assert_not_called()


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
                    "metadata": {"author": "Test"},
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

        results = mock_db.query("test", search_type="keyword", return_type="chunks")

        assert len(results) == 1
        result = results[0]
        assert result.type == "chunk"
        assert result.document_id == "doc_1"
        assert isinstance(result.position, ChunkPosition)

    def test_hybrid_query(self, mock_httpx_client, mock_db):
        """Test hybrid search query."""
        mock_httpx_client.request.return_value.json.return_value = {"results": []}

        mock_db.query(
            "test query", search_type="hybrid", vector_weight=0.7, score_threshold=0.5, filters={"author": "Test"}
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
                {"id": "doc_1", "content": "Test content", "metadata": {"author": "Test"}, "content_hash": "hash1"}
            ]
        }

        results = mock_db.filter(where={"author": "Test"}, order_by="created_at DESC", limit=10, offset=5)

        assert len(results) == 1
        assert isinstance(results[0], Document)
        assert results[0].id == "doc_1"

        # Check request
        call_args = mock_httpx_client.request.call_args
        assert "/filter" in call_args[0][1]
        payload = call_args[1]["json"]
        assert payload["filters"] == {"author": "Test"}
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
            "stats": {"documents": 100, "chunks": 500, "index_vectors": 500}
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
        mock_response.json.return_value = {
            "error": {"code": "embedding_error", "message": "Embedding model not available"}
        }
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
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
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
            "content_hash": "hash123",
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
        data = {"id": "doc_1", "content": "Test content"}

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
            "metadata": {"author": "Test"},
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
        data = {"id": "doc_1", "content": "Test content"}

        result = QueryResult.from_dict(data)

        assert result.id == "doc_1"
        assert result.score == 0.0  # Default
        assert result.type == "document"  # Default
        assert result.content == "Test content"


@pytest.mark.client
class TestRemoteVectorDBFileOperations:
    """Test RemoteVectorDB file upload operations."""

    def test_upsert_from_file(self, mock_httpx_client, tmp_path):
        """Test upserting documents from files."""
        # Create temporary test files
        test_file1 = tmp_path / "test1.txt"
        test_file1.write_text("Test content 1")
        test_file2 = tmp_path / "test2.txt"
        test_file2.write_text("Test content 2")

        # Mock the response for request method (used when files are involved)
        mock_httpx_client.request.return_value.json.return_value = {
            "document_ids": ["doc1", "doc2"],
            "status": "success",
        }

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        # Test single file
        result = db.upsert_from_file(test_file1)
        assert result == ["doc1", "doc2"]

        # Verify the request was made
        assert mock_httpx_client.request.called

        # Test multiple files with metadata
        result = db.upsert_from_file(
            [test_file1, test_file2], metadata=[{"author": "user1"}, {"author": "user2"}], ids=["custom1", "custom2"]
        )
        assert result == ["doc1", "doc2"]

    def test_insert_from_file(self, mock_httpx_client, tmp_path):
        """Test inserting documents from files."""
        # Create temporary test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Mock the response for request method (used when files are involved)
        mock_httpx_client.request.return_value.json.return_value = {"document_ids": ["new_doc"], "status": "success"}

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        result = db.insert_from_file(test_file, metadata={"author": "test"}, errors="raise")
        assert result == ["new_doc"]

        # Verify the request was made
        assert mock_httpx_client.request.called

    def test_file_not_found_error(self, mock_httpx_client):
        """Test error when file doesn't exist."""
        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        with pytest.raises(FileNotFoundError, match="File not found"):
            db.upsert_from_file("/nonexistent/file.txt")

    def test_file_validation_errors(self, mock_httpx_client, tmp_path):
        """Test validation errors for file operations."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test")

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        # Test metadata count mismatch
        with pytest.raises(ValueError, match="Number of metadata entries must match"):
            db.upsert_from_file([test_file], metadata=[{"a": 1}, {"b": 2}])  # 2 metadata for 1 file

        # Test ID count mismatch
        with pytest.raises(ValueError, match="Number of IDs must match"):
            db.insert_from_file([test_file], ids=["id1", "id2"])  # 2 IDs for 1 file


@pytest.mark.client
class TestRemoteVectorDBChunkOperations:
    """Test RemoteVectorDB chunk operations."""

    def test_upsert_from_chunks(self, mock_httpx_client):
        """Test upserting documents from chunks."""
        from localvectordb.core import Chunk, ChunkPosition

        # Mock the response
        mock_httpx_client.post.return_value.json.return_value = {"ids": ["doc1", "doc2"], "status": "success"}

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        # Test with Chunk objects
        chunks_with_objects = {
            "doc1": [
                Chunk(
                    content="Chunk 1",
                    position=ChunkPosition(start=0, end=7, line=1, column=1, end_line=1, end_column=8),
                    tokens=2,
                    index=0,
                ),
                Chunk(
                    content="Chunk 2",
                    position=ChunkPosition(start=8, end=15, line=1, column=9, end_line=1, end_column=16),
                    tokens=2,
                    index=1,
                ),
            ],
            "doc2": [
                Chunk(
                    content="Chunk A",
                    position=ChunkPosition(start=0, end=7, line=1, column=1, end_line=1, end_column=8),
                    tokens=2,
                    index=0,
                )
            ],
        }

        result = db.upsert_from_chunks(
            chunks_with_objects, metadata={"doc1": {"author": "user1"}, "doc2": {"author": "user2"}}
        )
        assert result == ["doc1", "doc2"]

        # Verify the request payload (the client issues client.request(method, url, ...)).
        call_args = mock_httpx_client.request.call_args
        assert call_args is not None
        assert call_args[0][0] == "POST"
        assert call_args[0][1].endswith("/documents/chunks")
        payload = call_args[1]["json"]
        assert "chunks_by_document" in payload
        # Chunk objects are serialized with their content under "text".
        assert payload["chunks_by_document"]["doc1"][0]["text"] == "Chunk 1"

    def test_upsert_from_chunks_with_strings(self, mock_httpx_client):
        """Test upserting documents from string chunks."""
        # Mock the response
        mock_httpx_client.post.return_value.json.return_value = {"ids": ["doc1"], "status": "success"}

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        # Test with plain strings
        chunks_with_strings = {"doc1": ["String chunk 1", "String chunk 2"]}

        result = db.upsert_from_chunks(chunks_with_strings)
        assert result == ["doc1"]

        # Verify the string chunks were passed as-is
        call_args = mock_httpx_client.request.call_args
        assert call_args is not None
        payload = call_args[1]["json"]
        assert payload["chunks_by_document"]["doc1"] == ["String chunk 1", "String chunk 2"]

    def test_insert_from_chunks(self, mock_httpx_client):
        """Test inserting documents from chunks with conflict handling."""
        from localvectordb.core import Chunk, ChunkPosition

        # Mock the response
        mock_httpx_client.post.return_value.json.return_value = {"ids": ["new_doc"], "status": "success"}

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        chunks = {
            "new_doc": [
                Chunk(
                    content="New chunk",
                    position=ChunkPosition(start=0, end=9, line=1, column=1, end_line=1, end_column=10),
                    tokens=2,
                    index=0,
                )
            ]
        }

        result = db.insert_from_chunks(chunks, errors="raise", similarity_threshold=0.9)
        assert result == ["new_doc"]

        # Verify the request
        call_args = mock_httpx_client.request.call_args
        assert call_args is not None
        assert call_args[0][1].endswith("/documents/chunks/insert")
        payload = call_args[1]["json"]
        assert payload["errors"] == "raise"
        assert payload["similarity_threshold"] == 0.9

    def test_insert_from_chunks_ignore_errors(self, mock_httpx_client):
        """Test inserting chunks with ignore errors mode."""
        # Mock the response
        mock_httpx_client.post.return_value.json.return_value = {
            "ids": [],  # No documents inserted due to conflicts
            "status": "success",
        }

        db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

        chunks = {"existing_doc": ["chunk1", "chunk2"]}

        result = db.insert_from_chunks(chunks, errors="ignore")
        assert result == []

        # Verify errors parameter was sent
        call_args = mock_httpx_client.request.call_args
        assert call_args is not None
        assert call_args[1]["json"]["errors"] == "ignore"


@pytest.mark.asyncio
@pytest.mark.client
class TestRemoteVectorDBAsyncFileOperations:
    """Test async file operations."""

    async def test_upsert_from_file_async(self, tmp_path):
        """Test async file upsert."""
        from unittest.mock import AsyncMock, patch

        # Create test file
        test_file = tmp_path / "async_test.txt"
        test_file.write_text("Async test content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"document_ids": ["async_doc"], "status": "success"}

            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

            result = await db.upsert_from_file_async(test_file)
            assert result == ["async_doc"]

    async def test_insert_from_file_async(self, tmp_path):
        """Test async file insert."""
        from unittest.mock import AsyncMock, patch

        # Create test file
        test_file = tmp_path / "async_test.txt"
        test_file.write_text("Async test content")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"document_ids": ["new_async_doc"], "status": "success"}

            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

            result = await db.insert_from_file_async(test_file, metadata={"async": True}, errors="ignore")
            assert result == ["new_async_doc"]


@pytest.mark.asyncio
@pytest.mark.client
class TestRemoteVectorDBAsyncChunkOperations:
    """Test async chunk operations."""

    async def test_upsert_from_chunks_async(self):
        """Test async chunk upsert."""
        from unittest.mock import AsyncMock

        from localvectordb.core import Chunk, ChunkPosition

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()

            # Create a proper mock response that won't interfere with status_code checks
            mock_response = Mock()
            mock_response.status_code = 200  # This needs to be a plain int
            mock_response.json.return_value = {"ids": ["async_chunk_doc"], "status": "success"}

            # Make sure the async client methods return our mock response
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

            chunks = {
                "async_chunk_doc": [
                    Chunk(
                        content="Async chunk",
                        position=ChunkPosition(start=0, end=11, line=1, column=1, end_line=1, end_column=12),
                        tokens=2,
                        index=0,
                    )
                ]
            }

            result = await db.upsert_from_chunks_async(chunks)
            assert result == ["async_chunk_doc"]

    async def test_insert_from_chunks_async(self):
        """Test async chunk insert."""
        from unittest.mock import AsyncMock

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()

            # Create a proper mock response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ids": ["new_async_chunk"], "status": "success"}

            # Make sure the async client methods return our mock response
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")

            chunks = {"new_async_chunk": ["Simple string chunk"]}

            result = await db.insert_from_chunks_async(chunks, errors="raise", batch_size=50)
            assert result == ["new_async_chunk"]


@pytest.mark.client
class TestRemoteRerankerWiring:
    """B4: remote query must forward reranker_config (and search_level), and must
    reject a non-serializable reranker instance instead of silently dropping it."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        return RemoteVectorDB("test_db", api_key="test-key")

    def test_query_forwards_reranker_config_and_search_level(self, mock_httpx_client, mock_db):
        mock_httpx_client.request.return_value.json.return_value = {"results": []}

        mock_db.query(
            "test query",
            search_level="sections",
            reranker_config={"provider": "jina", "model": "jina-reranker-v2-base-multilingual"},
        )

        payload = mock_httpx_client.request.call_args[1]["json"]
        assert payload["search_level"] == "sections"
        assert payload["reranker_config"] == {
            "provider": "jina",
            "model": "jina-reranker-v2-base-multilingual",
        }

    def test_query_rejects_reranker_instance(self, mock_db):
        class _FakeReranker:
            pass

        with pytest.raises(ValueError, match="reranker_config"):
            mock_db.query("test query", reranker=_FakeReranker())

    async def test_query_async_forwards_reranker_config(self):
        from unittest.mock import AsyncMock

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_client.request.return_value = mock_response
            mock_client_class.return_value = mock_client

            db = RemoteVectorDB(name="test_db", base_url="http://localhost:5000")
            await db.query_async("test query", reranker_config={"provider": "jina", "model": "m"})

            payload = mock_client.request.call_args[1]["json"]
            assert payload["reranker_config"] == {"provider": "jina", "model": "m"}

    async def test_query_async_rejects_reranker_instance(self, mock_db):
        with pytest.raises(ValueError, match="reranker_config"):
            await mock_db.query_async("test query", reranker=object())


class TestRemoteContractRegressions:
    """Regression tests for RemoteVectorDB<->server HTTP-contract bugs found in the
    pre-release audit. Each asserts the exact verb/path/payload/response handling so a
    future contract drift fails in CI -- these endpoints are otherwise unexercised."""

    @pytest.fixture
    def mock_db(self, mock_httpx_client):
        return RemoteVectorDB("test_db", api_key="test-key")

    def test_analyze_system_resources_path_and_unwrap(self, mock_httpx_client, mock_db):
        # Was GET /api/system/resources (missing /v1 -> 404) and returned the envelope.
        mock_httpx_client.request.return_value.json.return_value = {
            "system_resources": {"cpu_count": 8, "memory_gb": 16},
            "status": "success",
        }
        result = mock_db.analyze_system_resources()
        method, url = mock_httpx_client.request.call_args[0][:2]
        assert method == "GET"
        assert url.endswith("/api/v1/system/resources")
        assert result == {"cpu_count": 8, "memory_gb": 16}

    def test_get_sqlite_tuning_unwraps_tuning(self, mock_httpx_client, mock_db):
        # Was returning the {database, tuning, status} envelope instead of the inner config.
        mock_httpx_client.request.return_value.json.return_value = {
            "database": "test_db",
            "tuning": {"profile": "balanced", "overrides": {}, "pragmas": {}},
            "status": "success",
        }
        result = mock_db.get_sqlite_tuning()
        assert result == {"profile": "balanced", "overrides": {}, "pragmas": {}}
        assert "database" not in result and "status" not in result

    def test_auto_tune_calls_server_and_unwraps(self, mock_httpx_client, mock_db):
        # Was inherited from TuningMixin -> profiled the CLIENT machine, never hit the server.
        mock_httpx_client.request.return_value.json.return_value = {
            "database": "test_db",
            "recommendation": {"profile_name": "fast_ingest", "applied": False},
            "status": "success",
        }
        result = mock_db.auto_tune(workload={"workload_type": "write_heavy"}, apply=False)
        method, url = mock_httpx_client.request.call_args[0][:2]
        payload = mock_httpx_client.request.call_args[1]["json"]
        assert method == "POST"
        assert url.endswith("/api/v1/databases/test_db/auto-tune")
        assert payload == {"workload": {"workload_type": "write_heavy"}, "apply": False}
        assert result == {"profile_name": "fast_ingest", "applied": False}

    def test_auto_tune_interactive_rejected(self, mock_db):
        with pytest.raises(ValueError, match="interactive"):
            mock_db.auto_tune(interactive=True)

    def test_update_metadata_schema_resolves_common_name(self, mock_httpx_client, mock_db):
        # A common-schema NAME must be resolved to a field dict client-side; the server
        # /schema endpoint types metadata_schema as an object and 422s on a bare string.
        mock_httpx_client.request.return_value.json.return_value = {
            "message": "ok",
            "status": "success",
            "changes": {},
            "new_schema": {},
        }
        mock_db.update_metadata_schema("research_papers")
        method, url = mock_httpx_client.request.call_args[0][:2]
        payload = mock_httpx_client.request.call_args[1]["json"]
        assert method == "PUT"
        assert url.endswith("/api/v1/databases/test_db/schema")
        assert isinstance(payload["metadata_schema"], dict) and payload["metadata_schema"]
        assert all(isinstance(v, dict) and "type" in v for v in payload["metadata_schema"].values())

    def test_update_metadata_schema_unknown_name_raises(self, mock_db):
        with pytest.raises(ValueError, match="Unknown common schema"):
            mock_db.update_metadata_schema("not_a_real_schema")

    def test_upload_extractor_kwargs_rejected(self, tmp_path, mock_db):
        # extractor_kwargs cannot cross HTTP (server governs extraction); must fail loudly
        # rather than silently drop the user's options.
        sample = tmp_path / "sample.txt"
        sample.write_text("hello world")
        with pytest.raises(ValueError, match="extractor_kwargs"):
            mock_db.upsert_from_file(str(sample), extractor_kwargs={"foo": "bar"})
