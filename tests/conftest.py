# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# tests/conftest.py
"""
Common pytest fixtures and utilities for LocalVectorDB tests.
"""
import pytest
import tempfile
import sqlite3
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from localvectordb.core import MetadataField, MetadataFieldType, Document, Chunk, ChunkPosition
from localvectordb.embeddings import MockEmbeddings


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_metadata_schema():
    """Sample metadata schema for testing."""
    return {
        'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        'date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
        'tags': MetadataField(type=MetadataFieldType.JSON),
        'rating': MetadataField(type=MetadataFieldType.REAL, indexed=True),
        'active': MetadataField(type=MetadataFieldType.BOOLEAN),
    }


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        "The quick brown fox jumps over the lazy dog. This is a test document.",
        "Machine learning is a subset of artificial intelligence. It enables computers to learn without being explicitly programmed.",
        "Python is a high-level programming language. It's known for its simplicity and readability.",
        "Vector databases store and retrieve high-dimensional vectors efficiently. They enable semantic search.",
        "Natural language processing involves computational linguistics and machine learning to understand human language."
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata matching sample_documents."""
    return [
        {"author": "John Doe", "category": "test", "rating": 4.5, "active": True, "tags": ["sample", "test"]},
        {"author": "Jane Smith", "category": "ai", "rating": 5.0, "active": True, "tags": ["ml", "ai"]},
        {"author": "Bob Johnson", "category": "programming", "rating": 4.0, "active": True, "tags": ["python", "code"]},
        {"author": "Alice Brown", "category": "database", "rating": 4.8, "active": True, "tags": ["vector", "search"]},
        {"author": "Charlie Wilson", "category": "ai", "rating": 4.2, "active": False, "tags": ["nlp", "linguistics"]}
    ]


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        Chunk(
            content="The quick brown fox jumps over the lazy dog.",
            position=ChunkPosition(start=0, end=43, line=1, column=1),
            tokens=9,
            index=0,
            faiss_id=0
        ),
        Chunk(
            content="This is a test document.",
            position=ChunkPosition(start=44, end=68, line=1, column=45),
            tokens=5,
            index=1,
            faiss_id=1
        )
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embedding provider for testing."""
    return MockEmbeddings("test-model", dimension=384)


@pytest.fixture
def mock_faiss_index():
    """Mock FAISS index for testing."""
    mock_index = Mock()
    mock_index.ntotal = 0
    mock_index.d = 384
    mock_index.add = Mock()
    mock_index.search = Mock(return_value=(np.array([[0.1, 0.2]]), np.array([[0, 1]])))
    return mock_index


@pytest.fixture
def mock_sqlite_connection():
    """Mock SQLite connection for testing."""
    mock_conn = Mock(spec=sqlite3.Connection)
    mock_cursor = Mock()
    mock_conn.execute = Mock(return_value=mock_cursor)
    mock_conn.commit = Mock()
    mock_conn.rollback = Mock()
    mock_conn.close = Mock()
    mock_cursor.fetchone = Mock(return_value=None)
    mock_cursor.fetchall = Mock(return_value=[])
    mock_cursor.rowcount = 0
    return mock_conn


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for testing HTTP requests."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={"embeddings": [[0.1, 0.2, 0.3]]})
    mock_response.raise_for_status = Mock()
    mock_client.post = Mock(return_value=mock_response)
    mock_client.get = Mock(return_value=mock_response)
    mock_client.put = Mock(return_value=mock_response)
    mock_client.delete = Mock(return_value=mock_response)
    return mock_client


@pytest.fixture
def mock_tiktoken_encoding():
    """Mock tiktoken encoding for testing."""
    mock_encoding = Mock()
    mock_encoding.encode = Mock(side_effect=lambda text: list(range(len(text.split()))))
    mock_encoding.decode = Mock(side_effect=lambda tokens: " ".join(f"token{i}" for i in tokens))
    return mock_encoding


@pytest.fixture(autouse=True)
def patch_tiktoken():
    """Automatically patch tiktoken for all tests."""
    with patch('tiktoken.get_encoding') as mock_get_encoding:
        mock_encoding = Mock()
        mock_encoding.encode = Mock(side_effect=lambda text: list(range(len(text.split()))))
        mock_encoding.decode = Mock(side_effect=lambda tokens: " ".join(f"token{i}" for i in tokens))
        mock_get_encoding.return_value = mock_encoding
        yield mock_encoding


class MockAsyncClient:
    """Mock async HTTP client for testing."""

    def __init__(self, response_data=None):
        self.response_data = response_data or {"embeddings": [[0.1, 0.2, 0.3]]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def post(self, url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value=self.response_data)
        mock_response.raise_for_status = Mock()
        return mock_response

    async def get(self, url, **kwargs):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json = Mock(return_value={"models": [{"name": "test-model"}]})
        mock_response.raise_for_status = Mock()
        return mock_response


@pytest.fixture
def mock_async_client():
    """Mock async HTTP client fixture."""
    return MockAsyncClient()


# Common test data
TEST_DOCUMENTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning enables computers to learn without explicit programming.",
    "Python is a versatile programming language.",
    "Vector databases enable semantic search capabilities.",
    "Natural language processing combines linguistics and machine learning."
]

TEST_METADATA = [
    {"author": "Test Author 1", "category": "test"},
    {"author": "Test Author 2", "category": "ai"},
    {"author": "Test Author 3", "category": "programming"},
    {"author": "Test Author 4", "category": "database"},
    {"author": "Test Author 5", "category": "nlp"}
]


# Utility functions for tests
def create_test_document(doc_id: str = "test_doc", content: str = "Test content") -> Document:
    """Create a test document."""
    return Document(
        id=doc_id,
        content=content,
        metadata={"author": "Test Author"},
        content_hash="test_hash"
    )


def create_test_chunk(content: str = "Test chunk", index: int = 0, start: int = 0) -> Chunk:
    """Create a test chunk."""
    end = start + len(content)
    return Chunk(
        content=content,
        position=ChunkPosition(start=start, end=end, line=1, column=start + 1),
        tokens=len(content.split()),
        index=index,
        faiss_id=index
    )


# Pytest markers
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow tests that may take more time",
    "network: Tests that require network access",
    "database: Tests that involve database operations",
    "embedding: Tests that involve embedding operations",
    "chunking: Tests that involve text chunking",
    "client: Tests for remote client functionality"
]


def pytest_configure(config):
    """Configure pytest markers."""
    for marker in pytest_markers:
        config.addinivalue_line("markers", marker)