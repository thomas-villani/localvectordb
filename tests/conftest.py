# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Enhanced pytest fixtures with better test isolation for LocalVectorDB.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from localvectordb.core import Chunk, ChunkPosition, Document, MetadataField, MetadataFieldType
from localvectordb.embeddings import MockEmbeddings


@pytest.fixture(autouse=True)
def global_cleanup():
    """Global cleanup fixture to prevent test interference."""
    # Store initial state
    initial_modules = set(sys.modules.keys())

    # Import registry here to avoid circular imports
    from localvectordb.embeddings import EmbeddingRegistry

    initial_providers = EmbeddingRegistry._providers.copy()

    yield

    # Cleanup after test
    # 1. Clean up EmbeddingRegistry state
    EmbeddingRegistry._providers = initial_providers

    # 2. Remove dynamically loaded migration test modules
    modules_to_remove = []
    for module_name in sys.modules:
        # Only remove migration modules that look like dynamically loaded test migrations
        if (
            "migration_" in module_name
            and module_name not in initial_modules
            and any(pattern in module_name for pattern in ["_1_1_0", "_1_2_0", "_1_3_0", "_1_4_0", "_2_0_0"])
        ):
            modules_to_remove.append(module_name)
    for module_name in modules_to_remove:
        del sys.modules[module_name]


@pytest.fixture(scope="function", autouse=False)
def cleanup_resources():
    """Automatically cleanup resources after each test."""
    # Before test
    yield

    # After test - cleanup
    try:
        # Close any remaining asyncio loops
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
        except RuntimeError:
            pass

        # Cleanup threading
        for thread in threading.enumerate():
            if thread != threading.current_thread() and hasattr(thread, "_stop"):
                thread._stop()
    except Exception:
        pass


@pytest.fixture(scope="function")
def temp_dir():
    """Create isolated temporary directory for each test."""
    with tempfile.TemporaryDirectory(prefix="lvdb_test_") as tmpdir:
        temp_path = Path(tmpdir)

        # Ensure the directory is writable
        os.chmod(temp_path, 0o755)

        yield temp_path

        # Force cleanup
        try:
            if temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)
        except Exception:
            pass


@pytest.fixture(scope="function")
def isolated_db_path(temp_dir):
    """Create unique database path for each test."""
    import uuid

    unique_id = str(uuid.uuid4())[:8]
    db_path = temp_dir / f"test_db_{unique_id}"
    db_path.mkdir(exist_ok=True)
    return db_path


@pytest.fixture(scope="function")
def mock_embeddings():
    """Mock embedding provider with proper cleanup."""
    mock_provider = MockEmbeddings("test-model", dimension=384)
    yield mock_provider

    # Cleanup
    mock_provider.number_of_calls = 0


@pytest.fixture(scope="function")
def mock_httpx_client():
    """Mock httpx client that properly handles async context."""

    class MockResponse:
        def __init__(self, json_data=None, status_code=200):
            self.status_code = status_code
            self._json_data = json_data or {"embeddings": [[0.1, 0.2, 0.3]]}

        def json(self):
            return self._json_data

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    class MockAsyncClient:
        def __init__(self):
            self.closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.closed = True

        async def post(self, url, **kwargs):
            return MockResponse()

        async def get(self, url, **kwargs):
            return MockResponse({"models": [{"name": "test-model"}]})

        def close(self):
            self.closed = True

    class MockSyncClient:
        def __init__(self):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.closed = True

        def post(self, url, **kwargs):
            return MockResponse()

        def get(self, url, **kwargs):
            return MockResponse({"models": [{"name": "test-model"}]})

        def request(self, method, url, **kwargs):
            return MockResponse()

        def close(self):
            self.closed = True

    # Patch both sync and async clients
    with patch("httpx.AsyncClient", MockAsyncClient), patch("httpx.Client", MockSyncClient):
        yield MockSyncClient()


@pytest.fixture(scope="function")
def mock_faiss_index():
    """Mock FAISS index with proper state management."""
    mock_index = Mock()
    mock_index.ntotal = 0
    mock_index.d = 384

    # Track added vectors
    added_vectors = []

    def mock_add(vectors):
        mock_index.ntotal += len(vectors)
        added_vectors.extend(vectors)

    def mock_search(query_vectors, k=10):
        n_queries = len(query_vectors) if hasattr(query_vectors, "__len__") else 1
        distances = np.random.random((n_queries, min(k, mock_index.ntotal))) * 0.5
        indices = np.random.randint(0, max(1, mock_index.ntotal), (n_queries, min(k, mock_index.ntotal)))
        return distances, indices

    mock_index.add = Mock(side_effect=mock_add)
    mock_index.search = Mock(side_effect=mock_search)
    mock_index.reset = Mock(side_effect=lambda: setattr(mock_index, "ntotal", 0))

    yield mock_index

    # Cleanup
    mock_index.reset()


@pytest.fixture(scope="function", autouse=False)
def patch_asyncio():
    """Patch asyncio to prevent event loop conflicts."""
    original_get_event_loop = asyncio.get_event_loop
    original_new_event_loop = asyncio.new_event_loop
    original_set_event_loop = asyncio.set_event_loop

    def safe_get_event_loop():
        try:
            return original_get_event_loop()
        except RuntimeError:
            # Create new loop if none exists
            loop = original_new_event_loop()
            original_set_event_loop(loop)
            return loop

    with patch("asyncio.get_event_loop", side_effect=safe_get_event_loop):
        yield


@pytest.fixture(scope="function")
def mock_tiktoken():
    """Mock tiktoken encoding to avoid loading real tokenizer."""
    mock_encoding = Mock()

    def mock_encode(text):
        # Simple word-based tokenization for testing
        return list(range(len(text.split())))

    def mock_decode(tokens):
        return " ".join(f"token{i}" for i in tokens)

    mock_encoding.encode = Mock(side_effect=mock_encode)
    mock_encoding.decode = Mock(side_effect=mock_decode)

    with patch("tiktoken.get_encoding", return_value=mock_encoding):
        yield mock_encoding


@pytest.fixture
def sample_metadata_schema():
    """Sample metadata schema for testing."""
    return {
        "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "date": MetadataField(type=MetadataFieldType.DATE, indexed=True),
        "tags": MetadataField(type=MetadataFieldType.JSON),
        "rating": MetadataField(type=MetadataFieldType.REAL, indexed=True),
        "active": MetadataField(type=MetadataFieldType.BOOLEAN),
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "The quick brown fox jumps over the lazy dog. This is a test document.",
        (
            "Machine learning is a subset of artificial intelligence."
            " It enables computers to learn without being explicitly programmed."
        ),
        "Python is a high-level programming language. It's known for its simplicity and readability.",
        "Vector databases store and retrieve high-dimensional vectors efficiently. They enable semantic search.",
        (
            "Natural language processing involves computational linguistics"
            " and machine learning to understand human language."
        ),
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata matching sample_documents."""
    return [
        {"author": "John Doe", "category": "test", "rating": 4.5, "active": True, "tags": ["sample", "test"]},
        {"author": "Jane Smith", "category": "ai", "rating": 5.0, "active": True, "tags": ["ml", "ai"]},
        {
            "author": "Bob Johnson",
            "category": "programming",
            "rating": 4.0,
            "active": True,
            "tags": ["python", "code"],
        },
        {"author": "Alice Brown", "category": "database", "rating": 4.8, "active": True, "tags": ["vector", "search"]},
        {"author": "Charlie Wilson", "category": "ai", "rating": 4.2, "active": False, "tags": ["nlp", "linguistics"]},
    ]


# Test isolation helpers
def create_test_document(doc_id: str = "test_doc", content: str = "Test content") -> Document:
    """Create a test document."""
    return Document(id=doc_id, content=content, metadata={"author": "Test Author"}, content_hash="test_hash")


def create_test_chunk(content: str = "Test chunk", index: int = 0, start: int = 0) -> Chunk:
    """Create a test chunk."""
    end = start + len(content)
    return Chunk(
        content=content,
        position=ChunkPosition(start=start, end=end, line=1, column=start + 1, end_line=1, end_column=len(content) + 1),
        tokens=len(content.split()),
        index=index,
        faiss_id=index,
    )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    markers = [
        "unit: Unit tests",
        "integration: Integration tests",
        "slow: Slow tests that may take more time",
        "network: Tests that require network access",
        "database: Tests that involve database operations",
        "embedding: Tests that involve embedding operations",
        "chunking: Tests that involve text chunking",
        "client: Tests for remote client functionality",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_runtest_teardown(item, nextitem):
    """Clean up after each test."""
    try:
        # Force garbage collection
        import gc

        gc.collect()

        # Close any remaining database connections
        import sqlite3

        # This is a bit hacky but helps with cleanup
        for obj in gc.get_objects():
            if isinstance(obj, sqlite3.Connection):
                try:
                    obj.close()
                except Exception:
                    pass
    except Exception:
        pass
