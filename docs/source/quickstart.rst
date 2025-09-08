Quickstart
==========

Overview
--------

LocalVectorDB is a **document-first vector database** that combines the simplicity of SQLite with the power of FAISS for vector similarity search. Version 1.0 introduces a completely redesigned architecture focused on documents rather than chunks, making it easier to work with while providing more powerful features.

**Key Features:**

- **Document-First API**: Work with documents, not chunks - chunking is handled automatically
- **Unified Search Interface**: Vector, keyword, and hybrid search with normalized scoring
- **Position-Tracking Chunking**: Perfect document reconstruction with precise highlighting
- **Structured Metadata**: SQLite-backed metadata with indexed columns and schema validation
- **Plugin-Based Embeddings**: Support for Ollama, OpenAI, and custom embedding providers
- **Full-Text Search**: Built-in FTS5 support for keyword search
- **Remote Client**: HTTP API client for distributed deployments
- **Production Ready**: Comprehensive CLI, configuration management, and server deployment

Quick Installation
------------------

.. code-block:: bash

    # Basic installation
    pip install localvectordb

    # With server dependencies
    pip install localvectordb[server]

    # Development installation
    pip install localvectordb[dev]

    # The whole kit and kaboodle:
    pip install localvectordb[all]


5-Minute Example
----------------

.. code-block:: python

    from localvectordb import VectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    # Create a document database with metadata schema
    db = VectorDB(
        name="my_documents",
        base_path="./my_vectordb",
        metadata_schema={
            'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True, 
                                 embedding_enabled=True),  # Enable embeddings for title
            'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'abstract': MetadataField(type=MetadataFieldType.TEXT, 
                                    embedding_enabled=True),  # Enable embeddings for abstract
            'date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
            'tags': MetadataField(type=MetadataFieldType.JSON)
        },
        embedding_model="nomic-embed-text",
        chunk_size=500
    )

    # You can also define schema with shorthand:
    #   metadata_schema = {
    #       "title": ("text", True),   # type, indexed
    #       "author": ("text", True, True),  # type, indexed, required
    #       # ... etc.
    #   }

    # Add documents with metadata
    documents = [
        "LocalVectorDB is a document-first vector database...",
        "Python is a powerful programming language...",
        "Machine learning enables computers to learn..."
    ]

    metadata = [
        {"title": "LocalVectorDB Guide", "author": "AI Assistant", "date": "2024-01-01"},
        {"title": "Python Basics", "author": "Developer", "date": "2024-01-02"},
        {"title": "ML Introduction", "author": "Data Scientist", "date": "2024-01-03"}
    ]

    # Insert documents
    doc_ids = db.upsert(documents, metadata=metadata)
    print(f"Added documents: {doc_ids}")

    # Search documents
    results = db.query("vector database", search_type="vector", k=3)
    for result in results:
        print(f"Score: {result.score:.3f} | {result.content[:100]}...")

    # Hybrid search combining vector and keyword search
    results = db.query("python programming", search_type="hybrid", k=2)

    # Multi-column search - searches document content AND metadata fields
    # This searches across content, title, and abstract simultaneously
    multi_results = db.query_multi_column("machine learning", k=5)
    for result in multi_results:
        column = result.metadata.get('_search_column', 'unknown')
        print(f"Found in {column}: Score {result.score:.3f}")

    # Search specific columns only
    title_results = db.query_multi_column(
        "database", 
        columns=['title', 'abstract'],  # Only search these fields
        k=3
    )

    # Filter by metadata
    python_docs = db.filter(where={"author": "Developer"})

    # Close database
    db.close()


Server Example
--------------

.. code-block:: bash

    # Start LocalVectorDB server
    lvdb serve --host 127.0.0.1 --port 5000

    # Create database via CLI
    lvdb create my_database --embedding-model nomic-embed-text

    # Add documents
    lvdb db my_database add document.txt

    # Search documents
    lvdb db my_database search "query text" --limit 5


Remote Database Example
-----------------------

You can connect to the LocalVectorDB server using the ``VectorDB`` class, which allows for connecting to local or remote
databases.

.. code-block:: python

    from localvectordb import VectorDB

    # Connect to remote LocalVectorDB server
    db = VectorDB(
        name="my_remote_db",
        base_path="http://localhost:5000",
        # Include api_key if you configured the server with api keys.
        # api_key="your_api_key"
    )

    # Same API as local database!
    doc_ids = db.upsert(["Remote document content"])
    results = db.query("search query")

