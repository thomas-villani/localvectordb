"""
LocalVectorDB v1.0 - Document-First Vector Database with SQLite + FAISS

LocalVectorDB is a Python library that provides a document-focused vector database
built on SQLite, FAISS, and pluggable embedding providers. It offers both local
and remote (client-server) usage with a unified API that hides chunking complexity
from users.

Key Features
------------
- **Document-First API**: Work with complete documents rather than chunks
- **Unified Search Interface**: Vector similarity, keyword (FTS5), and hybrid search
- **Position-Tracking Chunking**: Perfect document reconstruction with multiple chunking strategies
- **Structured Metadata**: Indexed metadata fields with type validation
- **Pluggable Embeddings**: Support for Ollama, OpenAI, and custom embedding providers
- **Local & Remote**: Use locally or connect to a LocalVectorDB server
- **Full-Text Search**: SQLite FTS5 integration for keyword search
- **CLI Interface**: Complete command-line tool for database management

Architecture
------------
- **Storage**: SQLite for documents/metadata, FAISS for vector indices
- **Chunking**: Position-aware chunking with overlap support (sentences, tokens, etc.)
- **Embeddings**: Plugin architecture supporting multiple providers
- **Search**: Normalized scoring across vector, keyword, and hybrid modes
- **Server**: Optional HTTP server for remote access with authentication

Quick Start
-----------

Local Usage::

    from localvectordb import LocalVectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    # Create/open a database
    db = LocalVectorDB(
        name="my_docs",
        base_path="./vector_db",
        metadata_schema={
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
            'tags': MetadataField(type=MetadataFieldType.JSON)
        },
        embedding_model="nomic-embed-text",
        chunk_size=500
    )

    # Add documents
    doc_ids = db.upsert(
        documents=["Python is great", "Machine learning tutorial"],
        metadata=[
            {"author": "Alice", "date": "2024-01-01", "tags": ["python"]},
            {"author": "Bob", "date": "2024-02-01", "tags": ["ml", "ai"]}
        ]
    )

    # Search with unified interface
    results = db.query("python programming", search_type="vector", k=5)
    results = db.query("machine learning", search_type="hybrid", k=5)

    # Filter by metadata
    docs = db.filter(where={"author": "Alice"})

Remote Usage::

    from localvectordb import RemoteVectorDB

    # Connect to server
    db = RemoteVectorDB(
        name="my_docs",
        base_url="http://localhost:5000",
        api_key="your_api_key"
    )

    # Same API as local database
    results = db.query("search query", k=10)

Factory Pattern::

    from localvectordb import VectorDB

    # Automatically chooses LocalVectorDB or RemoteVectorDB based on path
    local_db = VectorDB("docs", "./local_path")
    remote_db = VectorDB("docs", "http://localhost:5000", api_key="key")

Command Line Interface
----------------------

Start a server::

    $ lvdb serve --host 0.0.0.0 --port 5000

Create a database::

    $ lvdb create mydatabase --embedding-model nomic-embed-text

Add documents::

    $ lvdb db mydatabase add document.txt
    $ cat file.txt | lvdb db mydatabase add -

Search documents::

    $ lvdb db mydatabase search "query text" --search-type hybrid --limit 5

Manage API keys::

    $ lvdb auth create-key --description "My App" --expires-days 30
    $ lvdb auth list-keys

Main Classes
------------
- `LocalVectorDB`: Main local database implementation
- `RemoteVectorDB`: Client for remote database access
- `VectorDB`: Factory function for automatic local/remote selection
- `Document`: Document representation with metadata
- `QueryResult`: Search result with normalized scoring
- `MetadataField`: Schema definition for metadata fields

Embedding Providers
-------------------
- **Ollama**: Local embedding models via Ollama
- **OpenAI**: OpenAI's embedding API
- **Mock**: Testing provider with deterministic embeddings
- **Plugin System**: Extensible via entry points

Chunking Strategies
-------------------
- **Sentences**: Split by sentence boundaries (default)
- **Tokens**: Split by token count using tiktoken
- **Words**: Split by word boundaries
- **Lines**: Split by line breaks
- **Characters**: Split by character count
- **Paragraphs**: Split by paragraph breaks
- **Sections**: Split by markdown-style headers
- **Code Blocks**: Language-aware code chunking

Search Types
------------
- **Vector**: Semantic similarity using embeddings
- **Keyword**: Full-text search using SQLite FTS5
- **Hybrid**: Weighted combination of vector and keyword search

Examples
--------

Advanced local database with custom schema::

    from localvectordb import LocalVectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    db = LocalVectorDB(
        name="research_papers",
        metadata_schema={
            'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'authors': MetadataField(type=MetadataFieldType.JSON),
            'publication_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
            'journal': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'keywords': MetadataField(type=MetadataFieldType.JSON),
            'citation_count': MetadataField(type=MetadataFieldType.INTEGER)
        },
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        chunking_method="paragraphs",
        chunk_size=800,
        chunk_overlap=100
    )

Server with authentication::

    from localvectordb_server import create_app

    app = create_app(
        database_directory="./databases",
        require_api_key=True,
        cors_enabled=True
    )
    app.run(host="0.0.0.0", port=5000)

Dependencies
------------
- **Core**: sqlite3, faiss-cpu, numpy, tiktoken
- **Embeddings**: httpx (for API providers)
- **Server**: flask, flask-cors (optional)
- **CLI**: click

License
-------
MIT License. See LICENSE file for details.
"""

from localvectordb.backup import BackupManager, IncrementalBackupManager, PointInTimeRecoveryManager
from localvectordb.chunking import ChunkerFactory
from localvectordb.client import RemoteVectorDB
from localvectordb.core import MetadataField
from localvectordb.database import LocalVectorDB
from localvectordb.embeddings import EmbeddingRegistry
from localvectordb.extractors import ExtractorRegistry, get_extractor_registry
from localvectordb.factory import VectorDB
from localvectordb.migration import Migration, MigrationEngine
from localvectordb.query_builder import QueryBuilder
from localvectordb._schema import get_common_metadata_schemas
from localvectordb.versioning import VersionManager
from localvectordb.sqlite_tuning import (SqliteProfile, get_sqlite_pragma_profile,
                                         get_profile_description, is_valid_sqlite_pragma_profile)

__all__ = ["LocalVectorDB", "ChunkerFactory", "EmbeddingRegistry", "RemoteVectorDB", "VectorDB", "MetadataField",
           "factory", "utils", "chunking", "core", "embeddings", "client", "database", "exceptions", "backup",
           "BackupManager", "IncrementalBackupManager", "PointInTimeRecoveryManager",
           "migration", "Migration", "MigrationEngine",
           "versioning", "VersionManager", "QueryBuilder", "query_builder", "extractors", "get_extractor_registry",
           "ExtractorRegistry", "sqlite_tuning", "SqliteProfile", "get_profile_description",
           "is_valid_sqlite_pragma_profile", "get_sqlite_pragma_profile",
           "get_common_metadata_schemas"]
