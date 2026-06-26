"""
Performance tests for localvectordb.

These tests measure performance characteristics and ensure the system
scales appropriately with different workloads.
"""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import numpy as np
import pytest

from localvectordb.chunking import ChunkerFactory
from localvectordb.core import Document
from localvectordb.database import LocalVectorDB
from localvectordb.embeddings import MockEmbeddings


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
    mock_conn.close = Mock()

    # Add context manager support
    mock_conn.__enter__ = Mock(return_value=mock_conn)
    mock_conn.__exit__ = Mock(return_value=None)
    # Add row_factory for compatibility
    mock_conn.row_factory = sqlite3.Row
    return mock_conn


def create_mock_pooled_connection(mock_conn):
    """Create a mock PooledConnection that properly implements context manager."""
    mock_pooled = Mock()
    mock_pooled.__enter__ = Mock(return_value=mock_conn)
    mock_pooled.__exit__ = Mock(return_value=None)
    mock_pooled.connection = mock_conn
    mock_pooled.close = Mock()
    return mock_pooled


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.database
class TestDatabasePerformance:
    """Test database performance characteristics."""

    @pytest.fixture(scope="function")
    def perf_db(self, temp_dir):
        """Create a database optimized for performance testing."""
        with (
            patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_embedding,
            patch("faiss.IndexFlatL2") as mock_faiss,
            patch("faiss.IndexIDMap2") as mock_faiss_idmap,
        ):
            mock_provider = MockEmbeddings("test-model", dimension=128)  # Smaller for speed
            mock_embedding.return_value = mock_provider

            mock_index = Mock()
            mock_index.ntotal = 0
            mock_index.search.return_value = (np.random.random((10, 1)), np.arange(10).reshape(10, 1))
            mock_index.add = Mock()
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            # Create mock connection and pooled connection
            mock_conn = create_mock_connection()
            mock_pooled_conn = create_mock_pooled_connection(mock_conn)

            with patch("localvectordb._pools.ConnectionPool.get_connection") as mock_get_conn:
                mock_get_conn.return_value = mock_pooled_conn

                db = LocalVectorDB(
                    name="perf_test",
                    base_path=temp_dir,
                    chunk_size=200,
                    chunk_overlap=20,
                    enable_fts=False,  # Disable for simpler testing
                    connection_pool_size=5,
                )
                db.index = mock_index
                db._embedding_provider = mock_provider
                db._embedding_dimension = mock_provider.get_dimension()

                yield db

    def test_single_document_insert_performance(self, perf_db):
        """Test performance of inserting single documents."""
        documents = [f"Test document {i} with some content." for i in range(100)]

        start_time = time.time()

        for doc in documents:
            perf_db.upsert([doc])

        end_time = time.time()
        total_time = end_time - start_time

        # Should be able to insert 100 docs in reasonable time
        assert total_time < 10.0, f"Single document inserts took {total_time:.2f}s"

        # Calculate throughput
        throughput = len(documents) / total_time
        assert throughput > 10, f"Throughput too low: {throughput:.1f} docs/sec"

    def test_batch_insert_performance(self, perf_db):
        """Test performance of batch insertions."""
        batch_sizes = [1, 10, 50, 100]
        num_docs = 200

        results = {}

        for batch_size in batch_sizes:
            documents = [f"Batch test document {i}" for i in range(num_docs)]

            start_time = time.time()

            # Insert in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                perf_db.upsert(batch)

            end_time = time.time()
            total_time = end_time - start_time
            throughput = num_docs / total_time

            results[batch_size] = throughput

            # Should complete in reasonable time
            assert total_time < 20.0, f"Batch size {batch_size} took {total_time:.2f}s"

        # Larger batches should generally be faster
        assert results[100] > results[1], "Batch processing should be more efficient"

    def test_query_performance(self, perf_db):
        """Test query performance with various result sizes."""
        # First insert some documents
        documents = [f"Query test document {i} about topic {i % 10}" for i in range(500)]
        perf_db.upsert(documents, batch_size=50)

        # Test query performance
        k_values = [1, 10, 50, 100]

        for k in k_values:
            start_time = time.time()

            # Perform multiple queries
            for _ in range(20):
                perf_db.query("test query", k=k)

            end_time = time.time()
            avg_query_time = (end_time - start_time) / 20

            # Queries should be fast
            assert avg_query_time < 0.5, f"Average query time for k={k}: {avg_query_time:.3f}s"

            # Query time shouldn't increase dramatically with k
            if k <= 50:
                assert avg_query_time < 0.1, f"Query time too high for k={k}: {avg_query_time:.3f}s"

    def test_memory_usage_scaling(self, perf_db):
        """Test that memory usage scales reasonably."""
        try:
            import os

            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Insert progressively more documents
        doc_counts = [100, 200, 500, 1000]
        memory_usage = [initial_memory]

        for doc_count in doc_counts:
            documents = [f"Memory test doc {i}" for i in range(doc_count)]
            perf_db.upsert(documents, batch_size=100)

            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)

        # Memory shouldn't grow excessively
        final_memory = memory_usage[-1]
        memory_growth = final_memory - initial_memory

        # Allow reasonable memory growth (this is very approximate)
        assert memory_growth < 500, f"Memory grew by {memory_growth:.1f}MB"

    def test_concurrent_operations_performance(self, perf_db):
        """Test performance under concurrent operations."""
        num_threads = 4
        docs_per_thread = 50

        def insert_documents(thread_id):
            """Insert documents in a thread."""
            documents = [f"Thread {thread_id} doc {i}" for i in range(docs_per_thread)]
            start_time = time.time()
            perf_db.upsert(documents, batch_size=10)
            end_time = time.time()
            return end_time - start_time

        # Test concurrent inserts
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(insert_documents, i) for i in range(num_threads)]
            thread_times = [future.result() for future in as_completed(futures)]

        total_time = time.time() - start_time

        # Should handle concurrent operations
        assert total_time < 15.0, f"Concurrent operations took {total_time:.2f}s"

        # All threads should complete successfully
        assert len(thread_times) == num_threads
        assert all(t < 10.0 for t in thread_times), "Some threads took too long"


@pytest.mark.performance
@pytest.mark.chunking
class TestChunkingPerformance:
    """Test chunking performance with different methods and sizes."""

    def test_chunking_methods_performance(self):
        """Test performance of different chunking methods."""
        # Create a large document
        large_doc = " ".join([f"This is sentence {i} in a large document." for i in range(1000)])

        methods = ["sentences", "words", "lines", "tokens"]
        results = {}

        for method in methods:
            chunker = ChunkerFactory.create_chunker(method, max_tokens=100)

            start_time = time.time()
            chunks = chunker.chunk(large_doc)
            end_time = time.time()

            chunk_time = end_time - start_time
            results[method] = {"time": chunk_time, "chunks": len(chunks), "throughput": len(large_doc) / chunk_time}

            # Should complete quickly
            assert chunk_time < 8.0, f"{method} chunking took {chunk_time:.2f}s"

            # Should produce reasonable number of chunks
            assert len(chunks) > 0, f"{method} produced no chunks"

        # All methods should be reasonably fast
        for method, result in results.items():
            assert result["throughput"] > 1000, f"{method} throughput too low: {result['throughput']:.1f} chars/sec"

    @pytest.mark.slow
    def test_chunking_scaling(self):
        """Test how chunking performance scales with document size.

        Marked slow: asserts on timing ratios that are flaky on shared CI hardware.
        """
        chunker = ChunkerFactory.create_chunker("sentences", max_tokens=100)

        doc_sizes = [100, 500, 1000, 5000]  # Number of sentences
        times = []

        for size in doc_sizes:
            doc = " ".join([f"This is sentence {i}." for i in range(size)])

            start_time = time.time()
            chunker.chunk(doc)
            end_time = time.time()

            chunk_time = end_time - start_time
            times.append(chunk_time)

            # Should scale reasonably
            assert chunk_time < size * 0.01, f"Chunking {size} sentences took {chunk_time:.2f}s"

        # Time should scale roughly linearly (allowing some overhead)
        for i in range(1, len(times)):
            ratio = times[i] / times[0]
            size_ratio = doc_sizes[i] / doc_sizes[0]
            # Allow up to 2x overhead for larger documents
            assert ratio < size_ratio * 3, f"Chunking doesn't scale linearly: {ratio:.2f}x vs {size_ratio:.2f}x"

    @pytest.mark.slow
    def test_chunk_overlap_performance(self):
        """Test performance impact of chunk overlap.

        Marked slow: asserts on timing ratios that are flaky on shared CI hardware.
        """
        doc = " ".join([f"Sentence {i} for overlap testing." for i in range(500)])

        overlap_values = [0, 1, 5, 10, 20]
        times = []

        for overlap in overlap_values:
            chunker = ChunkerFactory.create_chunker("sentences", max_tokens=50, overlap=overlap)

            start_time = time.time()
            chunker.chunk(doc)
            end_time = time.time()

            chunk_time = end_time - start_time
            times.append(chunk_time)

            # Higher overlap should still be reasonable
            assert chunk_time < 2.0, f"Overlap {overlap} took {chunk_time:.2f}s"

        # Overlap shouldn't dramatically increase processing time
        max_time = times[-1]
        min_time = times[1]
        assert max_time < min_time * 10, f"Overlap impact too high: {max_time:.3f}s vs {min_time:.3f}s"


@pytest.mark.performance
@pytest.mark.embedding
class TestEmbeddingPerformance:
    """Test embedding generation performance."""

    @pytest.mark.slow
    def test_mock_embedding_performance(self):
        """Test MockEmbeddings performance.

        Marked slow: asserts a throughput floor on a sub-millisecond mock operation,
        which is dominated by fixed overhead and flaky on shared CI hardware.
        """
        provider = MockEmbeddings("test-model", dimension=384)

        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100, 500]

        for batch_size in batch_sizes:
            texts = [f"Test text {i}" for i in range(batch_size)]

            start_time = time.time()
            embeddings = provider.embed_sync(texts)
            end_time = time.time()

            embed_time = end_time - start_time
            throughput = batch_size / (embed_time or 1e-10)

            # Should be very fast for mock embeddings
            assert embed_time < 1.0, f"Mock embedding batch {batch_size} took {embed_time:.2f}s"
            assert throughput > 50, f"Mock embedding throughput too low: {throughput:.1f} texts/sec"

            # Verify output shape
            assert embeddings.shape == (batch_size, 384)

    @pytest.mark.slow
    def test_embedding_dimension_impact(self):
        """Test how embedding dimension affects performance.

        Marked slow: asserts on timing ratios that are flaky on shared CI hardware.
        """
        dimensions = [128, 256, 384, 512, 768]
        texts = [f"Test text {i}" for i in range(100)]

        times = []

        for dim in dimensions:
            provider = MockEmbeddings("test-model", dimension=dim)

            start_time = time.time()
            embeddings = provider.embed_sync(texts)
            end_time = time.time()

            embed_time = end_time - start_time
            times.append(embed_time)

            # Should complete quickly regardless of dimension
            assert embed_time < 2.0, f"Dimension {dim} took {embed_time:.2f}s"
            assert embeddings.shape == (100, dim)

        max_time = max(times)
        min_time = min(times)
        assert max_time < min_time * 100, f"Dimension impact too high: {max_time:.3f}s vs {min_time:.3f}s"

    @pytest.mark.slow
    def test_repeated_embedding_performance(self):
        """Test performance of repeated embedding calls.

        Marked slow: asserts on timing variance that is flaky on shared CI hardware.
        """
        provider = MockEmbeddings("test-model", dimension=384)
        texts = [f"Repeated test {i}" for i in range(50)]

        num_iterations = 100
        times = []

        for _ in range(num_iterations):
            start_time = time.time()
            provider.embed_sync(texts)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)

        # Should be consistently fast
        assert avg_time < 0.1, f"Average embedding time too high: {avg_time:.3f}s"
        assert max_time < 0.5, f"Maximum embedding time too high: {max_time:.3f}s"

        # Should be relatively consistent
        time_variance = max_time - min_time
        assert time_variance < avg_time * 5, "Too much variance in embedding times"


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.database
class TestMemoryPerformance:
    """Test memory usage patterns and performance."""

    def test_large_document_memory_usage(self, temp_dir):
        """Test memory usage with large documents."""
        import os

        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")

        with (
            patch("faiss.IndexFlatL2") as mock_faiss,
            patch("faiss.IndexIDMap2") as mock_faiss_idmap,
            patch("localvectordb._pools.ConnectionPool.get_connection") as mock_get_conn,
        ):

            # mock_provider = MockEmbeddings("test-model", dimension=128)
            # mock_embedding.return_value = mock_provider
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            # Create mock connection
            mock_conn = create_mock_connection()
            mock_pooled_conn = create_mock_pooled_connection(mock_conn)
            mock_get_conn.return_value = mock_pooled_conn

            db = LocalVectorDB(
                name="memory_test",
                base_path=temp_dir,
                chunk_size=200,
                chunk_overlap=0,
                embedding_provider="mock",
                embedding_model="test-model",
            )
            db.index = mock_index

            time.sleep(2.0)  # pause for startup time
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(initial_memory)
            time.sleep(2.0)

            # Create very large document
            large_doc = " ".join([f"Memory test sentence {i}." for i in range(10000)])

            # Process large document
            db.upsert([large_doc])
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Memory increase should be reasonable (very approximate)
            doc_size_mb = len(large_doc) * 2 / 1024 / 1024

            # Allow for chunking overhead, embeddings, etc.
            max_expected_increase = doc_size_mb * 20

            assert (
                memory_increase < max_expected_increase
            ), f"Memory increased by {memory_increase:.1f}MB for {doc_size_mb:.1f}MB document"

            db.close()

    # Something is wrong with this test and it hangs.
    # def test_memory_cleanup_after_operations(self, temp_dir):
    #     """Test that memory is cleaned up after operations."""
    #     import gc
    #     import os
    #     try:
    #         import psutil
    #     except ImportError:
    #         pytest.skip("psutil not available for memory testing")
    #
    #     process = psutil.Process(os.getpid())
    #     print("running now")
    #     with patch('faiss.IndexFlatL2') as mock_faiss, \
    #             patch('localvectordb._pools.ConnectionPool.get_connection') as mock_get_conn:
    #
    #         # mock_provider = MockEmbeddings("test-model", dimension=128)
    #         # mock_embedding.return_value = mock_provider
    #         mock_faiss.return_value = Mock()
    #
    #         # Create mock connection
    #         mock_conn = create_mock_connection()
    #         mock_pooled_conn = create_mock_pooled_connection(mock_conn)
    #         mock_get_conn.return_value = mock_pooled_conn
    #
    #         initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    #
    #         # Create and destroy database multiple times
    #         for i in range(3):
    #             db = LocalVectorDB(
    #                 name=f"cleanup_test_{i}",
    #                 base_path=temp_dir,
    #                 chunk_size=100,
    #                 embedding_model="test-model",
    #                 embedding_provider="mock"
    #             )
    #
    #             documents = [f"Cleanup test doc {j}" for j in range(100)]
    #             db.upsert(documents)
    #
    #             # Close database
    #             db.close()
    #             del db
    #
    #             # Force garbage collection
    #             gc.collect()
    #
    #         final_memory = process.memory_info().rss / 1024 / 1024  # MB
    #         memory_growth = final_memory - initial_memory
    #
    #         # Memory shouldn't grow significantly after cleanup
    #         # (This is approximate due to Python's memory management)
    #         assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB after cleanup"


@pytest.mark.performance
class TestScalabilityBenchmarks:
    """Comprehensive scalability benchmarks."""

    @pytest.mark.slow
    def test_document_count_scaling(self, temp_dir):
        """Test how performance scales with document count.

        Marked slow: asserts on timing/throughput ratios that are flaky on shared CI hardware.
        """
        with (
            patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_embedding,
            patch("faiss.IndexFlatL2") as mock_faiss,
            patch("faiss.IndexIDMap2") as mock_faiss_idmap,
            patch("localvectordb._pools.ConnectionPool.get_connection") as mock_get_conn,
        ):

            mock_provider = MockEmbeddings("test-model", dimension=128)
            mock_embedding.return_value = mock_provider
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_index.add = Mock()
            mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            # Create mock connection
            mock_conn = create_mock_connection()
            mock_pooled_conn = create_mock_pooled_connection(mock_conn)
            mock_get_conn.return_value = mock_pooled_conn

            db = LocalVectorDB(name="scale_test", base_path=temp_dir, chunk_size=100)
            db.index = mock_index
            db._embedding_provider = mock_provider
            db._embedding_dimension = mock_provider.get_dimension()

            document_counts = [100, 500, 1000, 2000]
            insert_times = []
            query_times = []

            for count in document_counts:
                documents = [f"Scale test document {i}" for i in range(count)]

                # Test insert time
                start_time = time.time()
                db.upsert(documents, batch_size=100)
                insert_time = time.time() - start_time
                insert_times.append(insert_time)

                # Test query time
                start_time = time.time()
                for _ in range(10):  # Multiple queries for average
                    db.query("test query", k=10)
                query_time = (time.time() - start_time) / 10
                query_times.append(query_time)

            # Analyze scaling characteristics
            for i, count in enumerate(document_counts):
                insert_rate = count / insert_times[i]

                # Should maintain reasonable throughput
                assert insert_rate > 50, f"Insert rate too low for {count} docs: {insert_rate:.1f} docs/sec"

                # Query time shouldn't increase dramatically
                assert query_times[i] < 0.5, f"Query time too high for {count} docs: {query_times[i]:.3f}s"

            # Insert time should scale roughly linearly
            for i in range(1, len(document_counts)):
                time_ratio = insert_times[i] / insert_times[0]
                count_ratio = document_counts[i] / document_counts[0]

                # Allow some overhead, but should be roughly linear
                assert time_ratio < count_ratio * 2.0, f"Insert scaling issue: {time_ratio:.2f}x vs {count_ratio:.2f}x"

    def test_concurrent_user_simulation(self, temp_dir):
        """Simulate multiple concurrent users."""
        with (
            patch("localvectordb.embeddings.EmbeddingRegistry.create_provider") as mock_embedding,
            patch("faiss.IndexFlatL2") as mock_faiss,
            patch("faiss.IndexIDMap2") as mock_faiss_idmap,
            patch("localvectordb._pools.ConnectionPool.get_connection") as mock_get_conn,
        ):

            mock_provider = MockEmbeddings("test-model", dimension=128)
            mock_embedding.return_value = mock_provider
            mock_index = Mock()
            mock_index.ntotal = 0
            mock_index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))
            mock_faiss.return_value = mock_index
            mock_faiss_idmap.return_value = mock_index

            # Create mock connection
            mock_conn = create_mock_connection()
            mock_pooled_conn = create_mock_pooled_connection(mock_conn)
            mock_get_conn.return_value = mock_pooled_conn

            db = LocalVectorDB(name="concurrent_test", base_path=temp_dir, connection_pool_size=10)
            db._embedding_provider = mock_provider
            db._embedding_dimension = mock_provider.get_dimension()

            def simulate_user(user_id, num_operations=20):
                """Simulate a user performing various operations."""
                operations = []

                for i in range(num_operations):
                    start_time = time.time()

                    if i % 3 == 0:
                        # Insert operation
                        docs = [f"User {user_id} doc {i}"]
                        db.upsert(docs)
                        op_type = "insert"
                    elif i % 3 == 1:
                        # Query operation
                        db.query(f"user {user_id} query", k=5)
                        op_type = "query"
                    else:
                        # Update operation - mock the get method to return a document
                        with patch.object(db, "get") as mock_get:
                            mock_get.return_value = Document(id=f"user_{user_id}_doc_{i}", content="existing content")
                            db.update(f"user_{user_id}_doc_{i}", content=f"Updated by user {user_id}")
                        op_type = "update"

                    end_time = time.time()
                    operations.append((op_type, end_time - start_time))

                return operations

            num_users = 5
            operations_per_user = 10

            start_time = time.time()

            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(simulate_user, user_id, operations_per_user) for user_id in range(num_users)]

                all_operations = []
                for future in as_completed(futures):
                    user_operations = future.result()
                    all_operations.extend(user_operations)

            total_time = time.time() - start_time
            total_operations = len(all_operations)

            # Should handle concurrent users efficiently
            assert total_time < 30.0, f"Concurrent simulation took {total_time:.2f}s"

            throughput = total_operations / total_time
            assert throughput > 5, f"Overall throughput too low: {throughput:.1f} ops/sec"

            # Analyze operation times
            op_times = [op_time for _, op_time in all_operations]
            avg_op_time = sum(op_times) / len(op_times)
            max_op_time = max(op_times)

            assert avg_op_time < 1.0, f"Average operation time too high: {avg_op_time:.3f}s"
            assert max_op_time < 3.0, f"Maximum operation time too high: {max_op_time:.3f}s"
