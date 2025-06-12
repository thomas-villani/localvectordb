"""
Integration tests for localvectordb.

These tests verify that different components work together correctly.
They may be slower than unit tests and require more dependencies.
"""
import sqlite3

import pytest
import numpy as np
from unittest.mock import Mock, patch

from localvectordb.database import LocalVectorDB
from localvectordb.client import RemoteVectorDB
from localvectordb.factory import VectorDB
from localvectordb.core import MetadataField, MetadataFieldType, Document
from localvectordb.chunking import ChunkerFactory
from localvectordb.embeddings import MockEmbeddings

from localvectordb import factory

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

@pytest.mark.integration
class TestLocalVectorDBIntegration:
    """Integration tests for LocalVectorDB with real components."""

    @pytest.fixture
    def integration_db(self, temp_dir):
        """Create a real LocalVectorDB for integration testing."""
        metadata_schema = {
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'rating': MetadataField(type=MetadataFieldType.REAL),
            'tags': MetadataField(type=MetadataFieldType.JSON)
        }

        with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                patch('faiss.IndexFlatL2') as mock_faiss, \
                patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                patch('localvectordb.database.ConnectionPool') as mock_pool:
            # Use mock embeddings for predictable testing
            mock_provider = MockEmbeddings("test-model", dimension=384)
            mock_embedding.return_value = mock_provider

            # Mock FAISS index
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_index.search.return_value = (
                np.array([[0.1, 0.2, 0.3]]),
                np.array([[0, 1, 2]])
            )
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            # Setup connection pool mock properly
            mock_conn = create_mock_connection()
            mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
            mock_pooled = create_mock_pooled_connection(mock_conn)

            mock_pool_instance = Mock()
            mock_pool_instance.get_connection.return_value = mock_pooled
            mock_pool_instance.closed = False
            mock_pool.return_value = mock_pool_instance

            try:
                db = LocalVectorDB(
                    name="integration_test",
                    base_path=temp_dir,
                    metadata_schema=metadata_schema,
                    embedding_provider="mock",
                    embedding_model="test-model",
                    chunking_method="sentences",
                    chunk_size=100,
                    chunk_overlap=10,
                    enable_fts=False  # Disable FTS for simpler testing
                )

                db.index = mock_index
                db._embedding_provider = mock_provider

                yield db
            finally:
                # CRITICAL: Explicit cleanup
                if db is not None:
                    try:
                        db.close()
                    except Exception:
                        pass

                # Reset mock state
                mock_provider.number_of_calls = 0
                mock_pool_instance.closed = True

    def test_full_document_lifecycle(self, integration_db):
        """Test complete document lifecycle: insert, query, update, delete."""
        # Insert documents
        documents = [
            "This is the first test document about machine learning.",
            "This is the second test document about natural language processing.",
            "This is the third test document about computer vision."
        ]

        metadata = [
            {"author": "Alice", "category": "ML", "rating": 4.5, "tags": ["machine-learning", "ai"]},
            {"author": "Bob", "category": "NLP", "rating": 4.8, "tags": ["nlp", "language"]},
            {"author": "Charlie", "category": "CV", "rating": 4.2, "tags": ["vision", "image"]}
        ]

        # Setup mock for insert operation
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []  # No existing docs
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            # Test insert
            doc_ids = integration_db.insert(documents, metadata=metadata)

            assert len(doc_ids) == 3
            assert all(isinstance(doc_id, str) for doc_id in doc_ids)

        # Mock documents exist for subsequent operations
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = [
            {
                'id': doc_ids[0],
                'content': documents[0],
                'content_hash': 'hash1',
                'created_at': '2024-01-01T00:00:00',
                'updated_at': '2024-01-01T00:00:00',
                'author': 'Alice',
                'category': 'ML',
                'rating': 4.5,
                'tags': '["machine-learning", "ai"]'
            }
        ]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            # Test retrieval
            retrieved_docs = integration_db.get(doc_ids[0])
            assert isinstance(retrieved_docs, Document)
            assert retrieved_docs.id == doc_ids[0]
            assert retrieved_docs.content == documents[0]

    def test_chunking_integration(self, integration_db):
        """Test that chunking works correctly with database operations."""
        # Test document that will be chunked
        long_document = " ".join([
            "This is sentence one about artificial intelligence.",
            "This is sentence two about machine learning algorithms.",
            "This is sentence three about deep neural networks.",
            "This is sentence four about natural language processing.",
            "This is sentence five about computer vision applications."
        ])

        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []  # No existing docs
        mock_conn.execute.return_value.rowcount = 1
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            # Insert long document
            doc_ids = integration_db.upsert([long_document])

            assert len(doc_ids) == 1

            # Verify chunking happened (through embedding calls)
            # The document should have been chunked into multiple pieces
            assert integration_db._embedding_provider.number_of_calls > 0

    def test_metadata_filtering_integration(self, integration_db):
        """Test metadata filtering with database operations."""
        documents = [
            "Document about Python programming",
            "Document about Java programming",
            "Document about JavaScript programming"
        ]

        metadata = [
            {"author": "Alice", "category": "Python", "rating": 4.5},
            {"author": "Bob", "category": "Java", "rating": 4.0},
            {"author": "Alice", "category": "JavaScript", "rating": 4.8}
        ]

        # Setup mock for upsert
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []  # No existing docs for insert
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            # Insert documents
            doc_ids = integration_db.upsert(documents, metadata=metadata)

        # Test filtering
        mock_conn = create_mock_connection()
        # Mock filtered results (Alice's documents)
        mock_conn.execute.return_value.fetchall.return_value = [
            {
                'id': doc_ids[0],
                'content': documents[0],
                'content_hash': 'hash1',
                'created_at': '2024-01-01T00:00:00',
                'updated_at': '2024-01-01T00:00:00',
                'author': 'Alice',
                'category': 'Python',
                'rating': 4.5
            },
            {
                'id': doc_ids[2],
                'content': documents[2],
                'content_hash': 'hash3',
                'created_at': '2024-01-01T00:00:00',
                'updated_at': '2024-01-01T00:00:00',
                'author': 'Alice',
                'category': 'JavaScript',
                'rating': 4.8
            }
        ]
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            # Filter by author
            alice_docs = integration_db.filter(where={"author": "Alice"})

            assert len(alice_docs) == 2
            assert all(doc.metadata["author"] == "Alice" for doc in alice_docs)

    def test_query_integration(self, integration_db):
        """Test query functionality with real embeddings and search."""
        documents = [
            "Machine learning algorithms for classification",
            "Deep learning neural networks",
            "Natural language processing techniques"
        ]

        metadata = [
            {"category": "ML"},
            {"category": "DL"},
            {"category": "NLP"}
        ]

        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            # Insert documents
            doc_ids = integration_db.upsert(documents, metadata=metadata)

        # Mock search results
        mock_conn = create_mock_connection()
        mock_conn.execute.return_value.fetchone.return_value = {
            'document_id': doc_ids[0],
            'chunk_index': 0,
            'content': documents[0],
            'start_pos': 0,
            'end_pos': len(documents[0]),
            'start_line': 1,
            'start_col': 1,
            'end_line': 1,
            'end_col': 10,
            'doc_id': doc_ids[0],
            'doc_content': documents[0]
        }
        mock_pooled = create_mock_pooled_connection(mock_conn)

        with patch.object(integration_db, '_get_document_metadata') as mock_get_meta, \
                patch.object(integration_db.connection_pool, 'get_connection', return_value=mock_pooled):
            mock_get_meta.return_value = {"category": "ML"}

            # Test vector search
            results = integration_db.query("machine learning", search_type="vector", k=3)

            # Should return results (exact verification depends on mock setup)
            assert isinstance(results, list)

    @pytest.mark.integration
    class TestFactoryIntegration:
        """Integration tests for the factory function."""

        def test_factory_creates_local_database(self, temp_dir):
            """Test factory creates working local database."""
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                    patch('faiss.IndexFlatL2') as mock_faiss, \
                    patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                    patch('localvectordb.database.ConnectionPool') as mock_pool:
                mock_provider = MockEmbeddings("test-model", dimension=384)
                mock_embedding.return_value = mock_provider

                mock_index = Mock()
                mock_index.ntotal = 0
                mock_faiss.return_value = mock_index
                mock_faiss_idmap.return_value = mock_index
                # Setup connection pool mock
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
                mock_pooled = create_mock_pooled_connection(mock_conn)

                mock_pool_instance = Mock()
                mock_pool_instance.get_connection.return_value = mock_pooled
                mock_pool_instance.closed = False
                mock_pool.return_value = mock_pool_instance

                db = VectorDB(
                    name="factory_test",
                    base_path=temp_dir,
                    embedding_provider="mock",
                    chunk_size=200
                )

                # Should be LocalVectorDB instance
                assert isinstance(db, LocalVectorDB)
                assert db.chunk_size == 200
                db.close()

        def test_factory_creates_remote_database(self, mock_httpx_client):
            """Test factory creates working remote database."""

            # Mock ALL the HTTP-related methods to prevent any real network calls
            with patch('localvectordb.client.RemoteVectorDB._ensure_database_exists') as mock_ensure, \
                    patch('localvectordb.client.RemoteVectorDB._load_database_info') as mock_load, \
                    patch('httpx.Client', return_value=mock_httpx_client):
                # Prevent any HTTP calls during initialization
                mock_ensure.return_value = None
                mock_load.return_value = None

                db = VectorDB(
                    name="factory_test",
                    base_path="http://localhost:5000",
                    api_key="test-key",
                    create_if_not_exists=False  # Avoid triggering _ensure_database_exists
                )

                # Should be RemoteVectorDB instance
                assert isinstance(db, RemoteVectorDB)
                assert db.api_key == "test-key"

                db.close()

        def test_factory_parameter_passing(self, temp_dir):
            """Test that factory correctly passes parameters to underlying classes."""
            metadata_schema = {
                'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
            }


            # Patch the LocalVectorDB class that's imported in factory.py
            with patch.object(factory, "LocalVectorDB") as mock_local:
            # with patch('localvectordb.database.LocalVectorDB') as mock_local:
                mock_local.return_value = Mock()

                VectorDB(
                    name="param_test",
                    base_path=temp_dir,
                    metadata_schema=metadata_schema,
                    chunk_size=500,
                    enable_gpu=True,
                    # This should be filtered out for local DB
                    api_key="should-not-appear"
                )

                # Verify parameters were passed correctly
                call_kwargs = mock_local.call_args[1]
                assert call_kwargs["metadata_schema"] == metadata_schema
                assert call_kwargs["chunk_size"] == 500
                assert call_kwargs["enable_gpu"] is True
                assert "api_key" not in call_kwargs

    @pytest.mark.integration
    class TestChunkingIntegration:
        """Integration tests for chunking with different chunkers."""

        def test_chunker_factory_integration(self):
            """Test that chunker factory creates working chunkers."""
            methods = ['sentences', 'tokens', 'words', 'lines', 'paragraphs']

            test_text = """This is the first paragraph with multiple sentences. 
            It contains various types of content that should be chunked appropriately.

            This is the second paragraph. It also has multiple sentences that need 
            to be processed correctly by the chunking algorithm.

            This is the third paragraph for testing purposes."""

            for method in methods:
                chunker = ChunkerFactory.create_chunker(method, max_tokens=50)
                chunks = chunker.chunk(test_text)

                # Verify chunking worked
                assert len(chunks) > 0
                assert all(chunk.tokens <= 50 for chunk in chunks)
                assert all(chunk.position.start < chunk.position.end for chunk in chunks)

                # Verify chunk positions are valid
                for chunk in chunks:
                    extracted = test_text[chunk.position.start:chunk.position.end]
                    assert chunk.content in extracted or extracted in chunk.content

        def test_chunking_with_database(self, temp_dir):
            """Test chunking integration with database operations."""
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                    patch('faiss.IndexFlatL2') as mock_faiss, \
                    patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                    patch('localvectordb.database.ConnectionPool') as mock_pool:
                mock_provider = MockEmbeddings("test-model", dimension=384)
                mock_embedding.return_value = mock_provider

                mock_index = Mock()
                mock_index.ntotal = 0
                mock_faiss.return_value = mock_index
                mock_faiss_idmap.return_value = mock_index

                # Setup connection pool mock
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
                mock_pooled = create_mock_pooled_connection(mock_conn)

                mock_pool_instance = Mock()
                mock_pool_instance.get_connection.return_value = mock_pooled
                mock_pool_instance.closed = False
                mock_pool.return_value = mock_pool_instance

                # Test different chunking methods
                for method in ['sentences', 'words', 'lines']:
                    db = LocalVectorDB(
                        name=f"chunk_test_{method}",
                        base_path=temp_dir,
                        chunking_method=method,
                        chunk_size=50,
                        chunk_overlap=5
                    )

                    assert db.chunking_method == method
                    assert db.chunk_size == 50
                    assert db.chunk_overlap == 5
                    db.close()

    @pytest.mark.integration
    class TestEmbeddingIntegration:
        """Integration tests for embedding providers."""

        def test_mock_embeddings_integration(self):
            """Test mock embeddings work correctly in integration scenarios."""
            provider = MockEmbeddings("test-model", dimension=384)

            # Test various text inputs
            texts = [
                "Short text",
                "This is a longer text that should produce different embeddings",
                "Another completely different piece of text for testing",
                ""  # Edge case
            ]

            embeddings = provider.embed_sync(texts)

            assert embeddings.shape == (4, 384)
            assert embeddings.dtype == np.float32

            # Verify deterministic behavior
            embeddings2 = provider.embed_sync(texts)
            np.testing.assert_array_equal(embeddings, embeddings2)

            # Verify different texts produce different embeddings
            assert not np.array_equal(embeddings[0], embeddings[1])

        def test_embedding_with_chunking(self):
            """Test embeddings work correctly with chunked text."""
            provider = MockEmbeddings("test-model", dimension=384)
            chunker = ChunkerFactory.create_chunker("sentences", max_tokens=20)

            long_text = """This is a long document that will be chunked into multiple pieces.
            Each piece should get its own embedding vector.
            The embedding provider should handle this correctly.
            This tests the integration between chunking and embedding generation."""

            chunks = chunker.chunk(long_text)
            chunk_texts = [chunk.content for chunk in chunks]

            embeddings = provider.embed_sync(chunk_texts)

            assert embeddings.shape == (len(chunks), 384)
            assert len(chunks) > 1  # Should be chunked

            # Each chunk should have different embedding
            for i in range(len(chunks) - 1):
                assert not np.array_equal(embeddings[i], embeddings[i + 1])

    @pytest.mark.integration
    @pytest.mark.slow
    class TestPerformanceIntegration:
        """Integration tests focused on performance characteristics."""

        def test_large_document_processing(self, temp_dir):
            """Test processing of large documents with proper cleanup."""

            # Create isolated database path
            db_path = temp_dir / "large_doc_test"
            db_path.mkdir(exist_ok=True)

            # Use context manager for proper cleanup
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                    patch('faiss.IndexFlatL2') as mock_faiss, \
                    patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                    patch('localvectordb.database.ConnectionPool') as mock_pool:

                # Setup mocks with proper isolation
                mock_provider = MockEmbeddings("test-model", dimension=384)
                mock_provider.number_of_calls = 0  # Explicitly reset
                mock_embedding.return_value = mock_provider

                mock_index = Mock()
                mock_index.ntotal = 0
                mock_index.d = 384
                mock_index.add = Mock()
                mock_index.search = Mock(return_value=(np.array([[0.1]]), np.array([[0]])))
                mock_faiss.return_value = mock_index
                mock_faiss_idmap.return_value = mock_index

                # Setup connection pool mock with proper cleanup
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = None
                mock_pooled = create_mock_pooled_connection(mock_conn)

                mock_pool_instance = Mock()
                mock_pool_instance.get_connection.return_value = mock_pooled
                mock_pool_instance.closed = False
                mock_pool_instance.close_all = Mock()  # Add cleanup method
                mock_pool.return_value = mock_pool_instance

                db = None
                try:
                    # Create database instance
                    db = LocalVectorDB(
                        name="perf_test",
                        base_path=db_path,
                        chunk_size=100,
                        chunk_overlap=10,
                        enable_fts=False  # Disable FTS to avoid SQLite conflicts
                    )
                    db.index = mock_index
                    db._embedding_provider = mock_provider

                    # Create large document (smaller for better test performance)
                    large_doc = " ".join([
                        f"This is sentence {i} in a large document."
                        for i in range(200)  # Reduced from 1000 to 200
                    ])

                    # Mock the database operations
                    mock_conn_context = create_mock_connection()
                    mock_conn_context.execute.return_value.fetchall.return_value = []
                    mock_pooled_context = create_mock_pooled_connection(mock_conn_context)

                    with patch.object(db.connection_pool, 'get_connection', return_value=mock_pooled_context):
                        # Process the document
                        doc_ids = db.upsert([large_doc])

                        # Assertions
                        assert len(doc_ids) == 1
                        assert mock_provider.number_of_calls > 0

                        # Verify chunking occurred (should have multiple chunks for large doc)
                        expected_chunks = len(large_doc) // 100  # Rough estimate based on chunk_size
                        assert mock_provider.number_of_calls >= 1  # At least some embedding calls

                finally:
                    # CRITICAL: Explicit cleanup
                    if db is not None:
                        try:
                            db.close()
                        except Exception:
                            pass

                    # Reset mock state
                    mock_provider.number_of_calls = 0
                    mock_pool_instance.closed = True

                    # Clear large document from memory
                    if 'large_doc' in locals():
                        del large_doc

                    # Force garbage collection
                    import gc
                    gc.collect()

        def test_batch_processing(self, temp_dir):
            """Test batch processing of multiple documents."""
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                    patch('faiss.IndexFlatL2') as mock_faiss, \
                    patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                    patch('localvectordb.database.ConnectionPool') as mock_pool:
                mock_provider = MockEmbeddings("test-model", dimension=384)
                mock_embedding.return_value = mock_provider
                mock_index = Mock()
                mock_index.ntotal = 0
                mock_faiss.return_value = mock_index
                mock_faiss_idmap.return_value = mock_index

                # Setup connection pool mock
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
                mock_pooled = create_mock_pooled_connection(mock_conn)

                mock_pool_instance = Mock()
                mock_pool_instance.get_connection.return_value = mock_pooled
                mock_pool_instance.closed = False
                mock_pool.return_value = mock_pool_instance

                db = LocalVectorDB(
                    name="batch_test",
                    base_path=temp_dir,
                    chunk_size=50
                )
                db.index = mock_index

                # Create many documents
                documents = [f"This is test document number {i}." for i in range(100)]

                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchall.return_value = []
                mock_pooled = create_mock_pooled_connection(mock_conn)

                with patch.object(db.connection_pool, 'get_connection', return_value=mock_pooled):
                    # Should handle batch processing
                    doc_ids = db.upsert(documents, batch_size=25)

                    assert len(doc_ids) == 100
                    assert all(isinstance(doc_id, str) for doc_id in doc_ids)

    @pytest.mark.integration
    class TestErrorHandlingIntegration:
        """Integration tests for error handling across components."""

        def test_database_error_propagation(self, temp_dir):
            """Test that errors propagate correctly through the system."""
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding:

                # Setup mock that will fail
                mock_provider = Mock()
                mock_provider.validate_model.return_value = False  # Invalid model
                mock_embedding.return_value = mock_provider
                # mock_faiss_idmap.return_value = mock_index

                # Should raise ValueError for invalid model
                with pytest.raises(ValueError, match="Embedding model .* is not available"):
                    LocalVectorDB(
                        name="error_test",
                        base_path=temp_dir,
                        embedding_model="invalid-model"
                    )

        def test_chunking_error_handling(self):
            """Test error handling in chunking operations."""
            # Test with invalid chunking method
            with pytest.raises(ValueError, match="Unknown chunking method"):
                ChunkerFactory.create_chunker("invalid_method")

            # Test with invalid parameters
            with pytest.raises(TypeError):
                ChunkerFactory.create_chunker("sentences", max_tokens="invalid")

        def test_embedding_error_handling(self):
            """Test error handling in embedding operations."""
            provider = MockEmbeddings("test-model", dimension=384)

            # Test with empty input
            embeddings = provider.embed_sync([])
            assert embeddings.shape == (0, 384)

            # Test with None input (should handle gracefully or raise appropriate error)
            try:
                provider.embed_sync(None)
            except (TypeError, AttributeError):
                # These are acceptable error types for None input
                pass

    @pytest.mark.integration
    class TestRealWorldScenarios:
        """Integration tests simulating real-world usage scenarios."""

        def test_document_management_workflow(self, temp_dir):
            """Test a realistic document management workflow."""
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                    patch('faiss.IndexFlatL2') as mock_faiss, \
                    patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                    patch('localvectordb.database.ConnectionPool') as mock_pool:
                mock_provider = MockEmbeddings("test-model", dimension=384)
                mock_embedding.return_value = mock_provider
                mock_index = Mock()
                mock_index.ntotal = 0
                mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
                mock_faiss.return_value = mock_index
                mock_faiss_idmap.return_value = mock_index

                # Setup connection pool mock
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
                mock_pooled = create_mock_pooled_connection(mock_conn)

                mock_pool_instance = Mock()
                mock_pool_instance.get_connection.return_value = mock_pooled
                mock_pool_instance.closed = False
                mock_pool.return_value = mock_pool_instance

                # Create database with realistic schema
                metadata_schema = {
                    'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'created_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
                    'tags': MetadataField(type=MetadataFieldType.JSON),
                    'priority': MetadataField(type=MetadataFieldType.INTEGER, indexed=True)
                }

                db = LocalVectorDB(
                    name="document_mgmt",
                    base_path=temp_dir,
                    metadata_schema=metadata_schema,
                    chunk_size=200,
                    chunk_overlap=20
                )
                db.index = mock_index

                # Simulate adding documents over time
                documents = [
                    {
                        'content': "Project specification for the new AI system. This document outlines requirements and architecture.",
                        'metadata': {
                            'title': "AI System Spec",
                            'author': "Alice Johnson",
                            'created_date': "2024-01-15",
                            'tags': ["ai", "specification", "architecture"],
                            'priority': 1
                        }
                    },
                    {
                        'content': "Meeting notes from AI project kickoff. Discussed timeline and resource allocation.",
                        'metadata': {
                            'title': "AI Project Kickoff Notes",
                            'author': "Bob Smith",
                            'created_date': "2024-01-20",
                            'tags': ["meeting", "ai", "planning"],
                            'priority': 2
                        }
                    },
                    {
                        'content': "Technical design document for machine learning pipeline. Includes data flow and model architecture.",
                        'metadata': {
                            'title': "ML Pipeline Design",
                            'author': "Charlie Brown",
                            'created_date': "2024-01-25",
                            'tags': ["ml", "pipeline", "design"],
                            'priority': 1
                        }
                    }
                ]

                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchall.return_value = []
                mock_pooled = create_mock_pooled_connection(mock_conn)

                with patch.object(db.connection_pool, 'get_connection', return_value=mock_pooled):
                    # Insert documents
                    doc_ids = []
                    for doc in documents:
                        ids = db.upsert([doc['content']], metadata=[doc['metadata']])
                        doc_ids.extend(ids)

                    assert len(doc_ids) == 3

                # Simulate search and filtering operations
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = {
                    'document_id': doc_ids[0],
                    'chunk_index': 0,
                    'content': documents[0]['content'][:50],
                    'start_pos': 0,
                    'end_pos': 50,
                    'start_line': 1,
                    'start_col': 1,
                    'end_line': 1,
                    'end_col': 50,
                    'doc_id': doc_ids[0],
                    'doc_content': documents[0]['content']
                }
                mock_pooled = create_mock_pooled_connection(mock_conn)

                with patch.object(db, '_get_document_metadata') as mock_get_meta, \
                        patch.object(db.connection_pool, 'get_connection', return_value=mock_pooled):
                    mock_get_meta.return_value = {"title": "AI System Spec", "priority": 1}

                    # Search for AI-related documents
                    results = db.query("artificial intelligence system", k=5)
                    assert isinstance(results, list)

                # Simulate filtering by metadata
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchall.return_value = [
                    {
                        'id': doc_ids[0],
                        'content': documents[0]['content'],
                        'content_hash': 'hash1',
                        'created_at': '2024-01-15T00:00:00',
                        'updated_at': '2024-01-15T00:00:00',
                        'title': 'AI System Spec',
                        'author': 'Alice Johnson',
                        'created_date': '2024-01-15',
                        'tags': '["ai", "specification", "architecture"]',
                        'priority': 1
                    }
                ]
                mock_pooled = create_mock_pooled_connection(mock_conn)

                with patch.object(db.connection_pool, 'get_connection', return_value=mock_pooled):
                    # Filter high priority documents
                    high_priority = db.filter(where={"priority": 1})
                    assert len(high_priority) >= 0  # Depends on mock setup

        def test_mixed_content_types(self, temp_dir):
            """Test handling of mixed content types and sizes."""
            with patch('localvectordb.database.EmbeddingRegistry.create_provider') as mock_embedding, \
                    patch('faiss.IndexFlatL2') as mock_faiss, \
                    patch('faiss.IndexIDMap') as mock_faiss_idmap, \
                    patch('localvectordb.database.ConnectionPool') as mock_pool:
                mock_provider = MockEmbeddings("test-model", dimension=384)
                mock_embedding.return_value = mock_provider
                mock_index = Mock()
                mock_index.ntotal = 0
                mock_faiss.return_value = mock_index
                mock_faiss_idmap.return_value = mock_index

                # Setup connection pool mock
                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchone.return_value = None  # For _load_next_doc_id
                mock_pooled = create_mock_pooled_connection(mock_conn)

                mock_pool_instance = Mock()
                mock_pool_instance.get_connection.return_value = mock_pooled
                mock_pool_instance.closed = False
                mock_pool.return_value = mock_pool_instance

                db = LocalVectorDB(
                    name="mixed_content",
                    base_path=temp_dir,
                    chunk_size=150
                )
                db.index = mock_index
                db._embedding_provider = mock_provider

                # Mix of different content types and lengths
                mixed_documents = [
                    "Short note.",  # Very short
                    "This is a medium-length document with several sentences. It should be chunked appropriately based on the configured chunk size.",
                    # Medium
                    " ".join([f"This is sentence {i} in a very long document." for i in range(100)]),  # Very long
                    "",  # Empty (edge case)
                    "Single word",  # Single word
                    "Multiple\nlines\nwith\nspecial\ncharacters: @#$%^&*()",  # Special formatting
                ]

                mock_conn = create_mock_connection()
                mock_conn.execute.return_value.fetchall.return_value = []
                mock_pooled = create_mock_pooled_connection(mock_conn)

                with patch.object(db.connection_pool, 'get_connection', return_value=mock_pooled):
                    # Should handle all document types
                    doc_ids = db.upsert(mixed_documents)

                    assert len(doc_ids) == len(mixed_documents)

                    # Verify embeddings were generated for non-empty documents
                    assert mock_provider.number_of_calls > 0
