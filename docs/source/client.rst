RemoteVectorDB Client
=====================

The ``RemoteVectorDB`` client provides a powerful HTTP interface to LocalVectorDB servers with near-perfect API parity to the local ``LocalVectorDB`` implementation. This enables a seamless development-to-production workflow: rapidly prototype with LocalVectorDB locally, then deploy to production using RemoteVectorDB without changing your code.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

RemoteVectorDB is the HTTP client for LocalVectorDB that maintains complete API compatibility with the local implementation. This design philosophy enables developers to:

1. **Develop Locally**: Start with ``LocalVectorDB`` for rapid prototyping with no server setup
2. **Test Remotely**: Switch to ``RemoteVectorDB`` for integration testing
3. **Deploy to Production**: Use the same code in production with distributed servers
4. **Scale Seamlessly**: Add servers and load balancing without code changes

The client communicates with LocalVectorDB servers via a RESTful HTTP API, supporting:

- Full document lifecycle management (CRUD operations)
- Vector, keyword, and hybrid search
- Metadata filtering and schema management
- SQLite tuning and optimization
- Async operations for high-performance applications
- Automatic retry logic and connection pooling

Key Features
------------

**Near-Perfect API Parity**
   Every method available in ``LocalVectorDB`` is also available in ``RemoteVectorDB``, including:

   - All document operations (upsert, insert, get, update, delete)
   - All query methods (query, filter, query_multi_column)
   - Metadata schema management
   - Database tuning operations
   - Async variants of all methods

**Drop-in Replacement**
   Switch between local and remote with a single line change::

       # Development: Local database
       db = LocalVectorDB("mydb", "./data")

       # Production: Remote database (same API!)
       db = RemoteVectorDB("mydb", "https://vectordb.company.com")

**VectorDB Factory Pattern**
   The ``VectorDB()`` factory automatically selects the right implementation::

       from localvectordb import VectorDB

       # Automatically creates LocalVectorDB
       db = VectorDB("mydb", "./data")

       # Automatically creates RemoteVectorDB
       db = VectorDB("mydb", "https://vectordb.company.com")

**Built-in Resilience**
   - Automatic retry with exponential backoff
   - Connection pooling for efficiency
   - Timeout configuration
   - Comprehensive error handling

**Authentication Support**
   - API key authentication
   - Environment variable configuration
   - Custom authorization headers

Development Workflow
--------------------

The LocalVectorDB ecosystem is designed for a natural progression from development to production:

1. Start with LocalVectorDB
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Begin development using ``LocalVectorDB`` for immediate productivity::

    from localvectordb import LocalVectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    # Local development database
    db = LocalVectorDB(
        name="products",
        base_path="./dev_data",
        metadata_schema={
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'price': MetadataField(type=MetadataFieldType.REAL, indexed=True),
            'in_stock': MetadataField(type=MetadataFieldType.BOOLEAN)
        }
    )

    # Develop your application
    db.upsert(documents, metadata=metadata)
    results = db.query("search term", filters={"category": "electronics"})

2. Use VectorDB Factory for Flexibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Refactor to use the ``VectorDB`` factory for environment-based configuration::

    from localvectordb import VectorDB
    import os

    # Configuration from environment
    db_path = os.getenv("VECTOR_DB_PATH", "./dev_data")

    # Works with both local paths and URLs!
    db = VectorDB(
        name="products",
        base_path=db_path,  # "./dev_data" or "https://api.company.com"
        api_key=os.getenv("LVDB_API_KEY"),  # Ignored for local, used for remote
        metadata_schema={
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'price': MetadataField(type=MetadataFieldType.REAL, indexed=True),
            'in_stock': MetadataField(type=MetadataFieldType.BOOLEAN)
        }
    )

3. Deploy with RemoteVectorDB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In production, simply change the configuration::

    # Production: Set environment variables
    # VECTOR_DB_PATH=https://vectordb.company.com
    # LVDB_API_KEY=your-secure-api-key

    # Same code works in production!
    db = VectorDB(
        name="products",
        base_path=os.getenv("VECTOR_DB_PATH"),
        api_key=os.getenv("LVDB_API_KEY")
    )

Connection and Authentication
-----------------------------

Basic Connection
~~~~~~~~~~~~~~~~

Connect to a LocalVectorDB server::

    from localvectordb.client import RemoteVectorDB

    # Basic connection (no authentication)
    db = RemoteVectorDB(
        name="my_database",
        base_url="http://localhost:8000"
    )

    # HTTPS connection with authentication
    db = RemoteVectorDB(
        name="my_database",
        base_url="https://vectordb.company.com",
        api_key="your-api-key"
    )

Authentication Methods
~~~~~~~~~~~~~~~~~~~~~~

RemoteVectorDB supports multiple authentication configurations:

**Direct API Key**::

    db = RemoteVectorDB(
        name="mydb",
        base_url="https://api.example.com",
        api_key="sk-abc123def456"
    )

**Environment Variable** (default: ``LVDB_API_KEY``)::

    # Automatically reads from LVDB_API_KEY
    db = RemoteVectorDB(
        name="mydb",
        base_url="https://api.example.com"
    )

**Custom Environment Variable**::

    # Use a custom environment variable
    db = RemoteVectorDB(
        name="mydb",
        base_url="https://api.example.com",
        api_key="$MY_CUSTOM_API_KEY"  # Reads from MY_CUSTOM_API_KEY env var
    )

Connection Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Fine-tune connection behavior::

    from httpx import Limits

    db = RemoteVectorDB(
        name="mydb",
        base_url="https://api.example.com",
        api_key="your-api-key",

        # Timeout configuration
        request_timeout=30,  # Request timeout in seconds

        # Retry configuration
        max_retries=3,
        retry_delay=1.0,  # Base delay for exponential backoff

        # Connection pooling
        connection_pool_limits=Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30.0
        ),

        # Concurrent request limits
        max_concurrent_requests=5,

        # Custom auth header (for non-standard servers)
        authorization_header="X-API-Key"
    )

Complete API Reference
----------------------

RemoteVectorDB implements all methods from LocalVectorDB with both synchronous and asynchronous variants.

Document Operations
~~~~~~~~~~~~~~~~~~~

**Upsert Documents**::

    # Sync: Add or update documents
    doc_ids = db.upsert(
        documents=["Document 1", "Document 2"],
        metadata=[
            {"author": "Alice", "year": 2024},
            {"author": "Bob", "year": 2023}
        ],
        ids=["doc1", "doc2"]  # Optional custom IDs
    )

    # Async variant
    doc_ids = await db.upsert_async(documents, metadata, ids)

**Insert Documents**::

    # Sync: Insert new documents (fails if IDs exist)
    doc_ids = db.insert(documents, metadata, ids)

    # Async variant
    doc_ids = await db.insert_async(documents, metadata, ids)

**Get Documents**::

    # Get single document
    doc = db.get("doc1")

    # Get multiple documents
    docs = db.get(["doc1", "doc2", "doc3"])

    # Async variants
    doc = await db.get_async("doc1")
    docs = await db.get_async(["doc1", "doc2", "doc3"])

**Update Documents**::

    # Update document content and/or metadata
    db.update(
        id="doc1",
        content="Updated content",  # Optional
        metadata={"author": "Alice Smith", "updated": True}  # Optional
    )

    # Async variant
    await db.update_async(id, content, metadata)

**Delete Documents**::

    # Delete single or multiple documents
    deleted_count = db.delete("doc1")
    deleted_count = db.delete(["doc1", "doc2"])

    # Async variants
    deleted_count = await db.delete_async(ids)

**Check Existence**::

    # Check if documents exist
    exists = db.exists("doc1")  # Returns: bool
    exist_list = db.exists(["doc1", "doc2"])  # Returns: List[bool]

    # Async variants
    exists = await db.exists_async(ids)

**Count Documents**::

    # Count all documents
    total = db.count()

    # Count with filters
    count = db.count(filters={"author": "Alice", "year": {"$gte": 2023}})

    # Async variant
    count = await db.count_async(filters)

File Operations
~~~~~~~~~~~~~~~

**Upsert from Files**::

    # Process files and add to database
    doc_ids = db.upsert_from_file(
        file_paths=["document.pdf", "report.docx"],
        metadata=[
            {"source": "research", "confidential": False},
            {"source": "internal", "confidential": True}
        ],
        ids=["pdf1", "docx1"],  # Optional
        # Optional security options passed through to the extractor
        extractor_kwargs={"strip_dangerous_elements": True}
    )

    # Async variant
    doc_ids = await db.upsert_from_file_async(
        file_paths, metadata, ids
    )

**Insert from Files**::

    # Insert new files (fails if IDs exist)
    doc_ids = db.insert_from_file(
        file_paths, metadata, ids, errors="raise"
    )

    # Async variant
    doc_ids = await db.insert_from_file_async(
        file_paths, metadata, ids, errors="raise"
    )

Chunk Operations
~~~~~~~~~~~~~~~~

**Upsert Chunks Directly**::

    from localvectordb.core import Chunk, ChunkPosition

    # upsert_from_chunks takes a dict mapping each document ID to its chunks.
    # Chunks may be plain strings or Chunk objects. A Chunk's fields are:
    # content, position (a ChunkPosition), tokens, index, and the optional
    # faiss_id / content_hash.
    chunks_by_document = {
        "doc1": [
            Chunk(
                content="First chunk content",
                position=ChunkPosition(
                    start=0, end=100, line=1, column=1, end_line=5, end_column=20
                ),
                tokens=18,
                index=0,
            ),
            Chunk(
                content="Second chunk content",
                position=ChunkPosition(
                    start=100, end=200, line=5, column=21, end_line=9, end_column=15
                ),
                tokens=20,
                index=1,
            ),
        ]
    }

    # Upsert chunks (per-document metadata is optional)
    db.upsert_from_chunks(chunks_by_document, metadata={"doc1": {"section": "intro"}})

    # A list of plain strings per document also works:
    db.upsert_from_chunks({"doc2": ["chunk one text", "chunk two text"]})

    # Async variant
    await db.upsert_from_chunks_async(chunks_by_document)

**Get Chunk Embeddings**::

    # Retrieve embeddings for specific chunks
    embeddings = db.get_chunk_embeddings(["chunk1", "chunk2"])

    # Returns numpy array of shape (n_chunks, embedding_dim)

Search and Query Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Unified Query Interface**::

    # Vector search (semantic)
    results = db.query(
        query="machine learning algorithms",
        search_type="vector",
        k=10,
        filters={"year": {"$gte": 2020}}
    )

    # Keyword search (FTS)
    results = db.query(
        query="exact phrase match",
        search_type="keyword",
        k=10
    )

    # Hybrid search (combined)
    results = db.query(
        query="neural networks",
        search_type="hybrid",
        k=10,
        vector_weight=0.7  # 70% vector, 30% keyword (keyword weight = 1 - vector_weight)
    )

    # Async variants
    results = await db.query_async(query, **kwargs)

    # Hierarchical search (server database created with hierarchical_embeddings=True)
    results = db.query(
        query="how do I rotate the API key?",
        search_level="sections",   # "chunks" (default), "sections", or "documents"
    )
    # ...or section-grouped results from the chunk index:
    results = db.query(query="billing", return_type="sections")

``RemoteVectorDB.query`` accepts the same ``search_level`` and
``return_type="sections"`` options as the local database; see
:doc:`hierarchical`. The options are forwarded to the server, which must host a
database created with ``hierarchical_embeddings=True``.

**Multi-Column Search**::

    # Search the main content plus embedding-enabled metadata columns.
    # Pass a single query string and the columns to search (use "content"
    # for the main document text). If columns is omitted, all
    # embedding-enabled fields plus the main content are searched.
    results = db.query_multi_column(
        "neural networks",
        columns=["content", "title", "abstract"],
        search_type="vector",
        k=10
    )

    # Async variant
    results = await db.query_multi_column_async(
        "neural networks", columns=["content", "title", "abstract"]
    )

**MongoDB-style Filtering**::

    # Filter documents with complex conditions
    docs = db.filter(
        where={
            "author": "Alice",
            "year": {"$gte": 2020, "$lt": 2025},
            "tags": {"$in": ["AI", "ML"]},
            "$or": [
                {"published": True},
                {"internal_review": "approved"}
            ]
        },
        order_by="year DESC",
        limit=20,
        offset=0
    )

    # Async variant
    docs = await db.filter_async(where, **kwargs)

Metadata Operations
~~~~~~~~~~~~~~~~~~~

**Update Metadata Schema**::

    from localvectordb.core import MetadataField, MetadataFieldType

    # Add new metadata fields
    new_schema = {
        'new_field': MetadataField(
            type=MetadataFieldType.TEXT,
            indexed=True
        ),
        'score': MetadataField(
            type=MetadataFieldType.REAL,
            indexed=True,
            default_value=0.0
        )
    }
    db.update_metadata_schema(
        new_schema,
        drop_columns=False  # Keep columns removed from the schema (safe default)
    )

    # Async variant
    await db.update_metadata_schema_async(new_schema, drop_columns=False)

**Get Schema Information**::

    # Retrieve current metadata schema
    schema_info = db.get_metadata_schema_info()
    # Returns: {
    #     "fields": {...},
    #     "field_count": 5,
    #     "indexed_fields": [...],
    #     "required_fields": [...],
    #     "field_types": {...}
    # }

    # Async variant
    schema_info = await db.get_metadata_schema_info_async()

Database Tuning Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

RemoteVectorDB inherits from ``TuningMixin``, providing remote access to SQLite tuning::

    # Get current tuning settings
    settings = db.get_sqlite_tuning()

    # Apply tuning profile
    db.set_sqlite_tuning(
        profile="fast_ingest",  # or "balanced", "read_optimized", "durable", "memory_saver"
        overrides={"cache_size": -262144}  # 256MB cache
    )

    # Maintenance operations
    db.sqlite_checkpoint(mode="FULL")  # WAL checkpoint
    db.sqlite_optimize()  # Optimize query planner
    db.sqlite_vacuum()  # Reclaim space
    db.sqlite_incremental_vacuum(pages=1000)  # Incremental cleanup

    # System analysis
    resources = db.analyze_system_resources()

    # Conditional checkpoint
    was_checkpointed = db.checkpoint_if_wal_large(wal_mb_threshold=128)

Database Management
~~~~~~~~~~~~~~~~~~~

**Get Statistics**::

    stats = db.get_stats()
    # Returns: {
    #     "total_documents": 10000,
    #     "total_chunks": 50000,
    #     "vector_dim": 768,
    #     "db_size_mb": 1024.5,
    #     "indexes": [...],
    #     ...
    # }

    # Async variant
    stats = await db.get_stats_async()

**Health Checks**::

    # Check if server is healthy (property, not a method)
    is_healthy = db.healthy

    # Ping server (with caching)
    is_alive = db.ping(force=False)  # Uses cached result if recent
    is_alive = db.ping(force=True)   # Forces new ping

**Connection Management**::

    # Check if connection is closed (property, not a method)
    is_closed = db.closed

    # Save any pending changes
    db.save()
    await db.save_async()

    # Close connection and cleanup
    db.close()
    await db.close_async()

**Context Manager Support**::

    # Automatic cleanup with context manager
    with RemoteVectorDB("mydb", "http://localhost:8000") as db:
        db.upsert(documents)
        results = db.query("search")
    # Connection automatically closed

    # Async context manager
    async with RemoteVectorDB("mydb", "http://localhost:8000") as db:
        await db.upsert_async(documents)
        results = await db.query_async("search")

Async Operations
----------------

Every method in RemoteVectorDB has an async variant for high-performance applications:

Basic Async Pattern
~~~~~~~~~~~~~~~~~~~

All async methods follow the naming convention of adding ``_async`` suffix::

    import asyncio
    from localvectordb.client import RemoteVectorDB

    async def main():
        db = RemoteVectorDB("mydb", "http://localhost:8000")

        # Async document operations
        doc_ids = await db.upsert_async(["Doc 1", "Doc 2"])
        doc = await db.get_async(doc_ids[0])
        await db.update_async(doc_ids[0], content="Updated")
        await db.delete_async(doc_ids)

        # Async search
        results = await db.query_async("search term", k=5)

        # Async schema management
        await db.update_metadata_schema_async(updates)

        # Cleanup
        await db.close_async()

    asyncio.run(main())

Concurrent Operations
~~~~~~~~~~~~~~~~~~~~~

Leverage async for concurrent operations::

    async def process_batch(db, batch_data):
        tasks = []

        # Create concurrent upsert tasks
        for content, metadata in batch_data:
            task = db.upsert_async([content], [metadata])
            tasks.append(task)

        # Execute all upserts concurrently
        doc_ids_lists = await asyncio.gather(*tasks)
        return [ids[0] for ids in doc_ids_lists]

Async Context Manager
~~~~~~~~~~~~~~~~~~~~~

Use async context manager for automatic cleanup::

    async def search_documents():
        async with RemoteVectorDB("mydb", "http://localhost:8000") as db:
            # Concurrent searches
            results = await asyncio.gather(
                db.query_async("machine learning", search_type="vector"),
                db.query_async("machine learning", search_type="keyword"),
                db.filter_async({"author": "Alice"})
            )
            return results
        # Connection automatically closed

Code Examples
-------------

Migration from Local to Remote
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to migrate existing LocalVectorDB code to RemoteVectorDB:

**Original Local Code**::

    from localvectordb import LocalVectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    # Local database
    db = LocalVectorDB(
        name="products",
        base_path="./data",
        metadata_schema={
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'price': MetadataField(type=MetadataFieldType.REAL, indexed=True)
        }
    )

    # Operations
    db.upsert(["Product A", "Product B"], metadata=[...])
    results = db.query("electronics", filters={"price": {"$lt": 1000}})

**Migrated to Remote (Option 1: Direct)**::

    from localvectordb.client import RemoteVectorDB
    from localvectordb.core import MetadataField, MetadataFieldType

    # Remote database - same API!
    db = RemoteVectorDB(
        name="products",
        base_url="https://vectordb.company.com",
        api_key="your-api-key",
        metadata_schema={
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'price': MetadataField(type=MetadataFieldType.REAL, indexed=True)
        }
    )

    # Exact same operations work!
    db.upsert(["Product A", "Product B"], metadata=[...])
    results = db.query("electronics", filters={"price": {"$lt": 1000}})

**Migrated to Remote (Option 2: Using Factory)**::

    from localvectordb import VectorDB
    from localvectordb.core import MetadataField, MetadataFieldType
    import os

    # Environment-based configuration
    # Development: DB_PATH="./data"
    # Production: DB_PATH="https://vectordb.company.com"

    db = VectorDB(
        name="products",
        base_path=os.getenv("DB_PATH", "./data"),
        api_key=os.getenv("LVDB_API_KEY"),  # Only used if remote
        metadata_schema={
            'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            'price': MetadataField(type=MetadataFieldType.REAL, indexed=True)
        }
    )

    # Same operations for both local and remote!
    db.upsert(["Product A", "Product B"], metadata=[...])
    results = db.query("electronics", filters={"price": {"$lt": 1000}})

RAG Application Example
~~~~~~~~~~~~~~~~~~~~~~~

Build a Retrieval-Augmented Generation system::

    from localvectordb import VectorDB
    from localvectordb.core import MetadataField, MetadataFieldType
    import os

    class RAGSystem:
        def __init__(self, db_path: str):
            # Works with both local and remote databases
            self.db = VectorDB(
                name="knowledge_base",
                base_path=db_path,
                api_key=os.getenv("LVDB_API_KEY"),
                metadata_schema={
                    'source': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'section': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                    'last_updated': MetadataField(type=MetadataFieldType.DATE, indexed=True)
                },
                embedding_model="nomic-embed-text",
                chunk_size=512,
                chunk_overlap=50
            )

        def add_documents(self, file_paths: list, source: str):
            """Add documents from files"""
            metadata = [{"source": source} for _ in file_paths]
            return self.db.upsert_from_file(file_paths, metadata=metadata)

        def retrieve_context(self, query: str, max_docs: int = 5):
            """Retrieve relevant context for a query"""
            results = self.db.query(
                query=query,
                search_type="hybrid",
                k=max_docs,
                vector_weight=0.7
            )

            contexts = []
            for result in results:
                contexts.append({
                    'content': result.content,
                    'source': result.metadata.get('source', 'Unknown'),
                    'score': result.score
                })

            return contexts

        async def retrieve_context_async(self, query: str, max_docs: int = 5):
            """Async version for high-performance scenarios"""
            results = await self.db.query_async(
                query=query,
                search_type="hybrid",
                k=max_docs,
                vector_weight=0.7
            )
            # Process results...
            return contexts

    # Usage - works identically for local and remote!

    # Development
    rag = RAGSystem("./local_data")

    # Production
    rag = RAGSystem("https://vectordb.company.com")

    # Add knowledge base
    rag.add_documents(["doc1.pdf", "doc2.md"], source="technical_docs")

    # Retrieve context
    contexts = rag.retrieve_context("How does the system handle errors?")

Document Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Process documents with metadata extraction::

    import asyncio
    from datetime import datetime
    from localvectordb import VectorDB

    class DocumentProcessor:
        def __init__(self, db_url: str):
            self.db = VectorDB("documents", db_url)

        async def process_document_batch(self, documents: list):
            """Process documents in parallel"""
            tasks = []

            for doc in documents:
                task = self.process_single_document(doc)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful = [r for r in results if not isinstance(r, Exception)]
            failed = [r for r in results if isinstance(r, Exception)]

            return {
                'successful': len(successful),
                'failed': len(failed),
                'errors': failed
            }

        async def process_single_document(self, doc: dict):
            """Process a single document"""
            # Extract metadata
            metadata = {
                'title': doc.get('title', 'Untitled'),
                'author': doc.get('author', 'Unknown'),
                'processed_at': datetime.now().isoformat(),
                'word_count': len(doc['content'].split()),
                'language': doc.get('language', 'en')
            }

            # Upsert to database
            doc_id = await self.db.upsert_async(
                documents=[doc['content']],
                metadata=[metadata],
                ids=[doc.get('id')]
            )

            return doc_id[0]

        async def search_recent(self, query: str, days: int = 7):
            """Search recently processed documents"""
            from datetime import datetime, timedelta

            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            results = await self.db.query_async(
                query=query,
                search_type="hybrid",
                filters={
                    "processed_at": {"$gte": cutoff_date}
                },
                k=20
            )

            return results

Error Handling
--------------

RemoteVectorDB provides comprehensive error handling:

Exception Types
~~~~~~~~~~~~~~~

::

    from localvectordb.exceptions import (
        DatabaseNotFoundError,
        DocumentNotFoundError,
        DuplicateDocumentIDError,
        EmbeddingError,
        DatabaseError
    )

    try:
        db = RemoteVectorDB("nonexistent", "http://localhost:8000")
    except DatabaseNotFoundError as e:
        print(f"Database not found: {e}")

    try:
        doc = db.get("missing_id")
    except DocumentNotFoundError as e:
        print(f"Document not found: {e}")

    try:
        db.insert(["content"], ids=["existing_id"])
    except DuplicateDocumentIDError as e:
        print(f"Document already exists: {e}")

Retry Logic
~~~~~~~~~~~

RemoteVectorDB includes automatic retry with exponential backoff::

    db = RemoteVectorDB(
        name="mydb",
        base_url="http://localhost:8000",
        max_retries=3,  # Retry failed requests up to 3 times
        retry_delay=1.0  # Base delay, exponentially increases
    )

    # Requests will automatically retry on:
    # - Network errors
    # - 5xx server errors
    # - Timeout errors

Connection Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor and handle connection health::

    import time

    def ensure_healthy_connection(db: RemoteVectorDB):
        """Ensure database connection is healthy"""
        if db.closed:  # property, not a method
            raise RuntimeError("Database connection is closed")

        if not db.ping():
            # Try to wait and retry
            time.sleep(1)
            if not db.ping(force=True):
                raise RuntimeError("Database server is not responding")

        if not db.healthy:  # property, not a method
            raise RuntimeError("Database is not healthy")

        return True

Performance Optimization
------------------------

Connection Pooling
~~~~~~~~~~~~~~~~~~

Optimize connection reuse::

    from httpx import Limits

    # Configure connection pooling for high-throughput
    db = RemoteVectorDB(
        name="mydb",
        base_url="http://localhost:8000",
        connection_pool_limits=Limits(
            max_keepalive_connections=50,  # Keep more connections alive
            max_connections=200,  # Allow more total connections
            keepalive_expiry=60.0  # Keep connections alive longer
        ),
        max_concurrent_requests=20  # More concurrent embedding requests
    )

Batch Operations
~~~~~~~~~~~~~~~~

Use batch operations for efficiency::

    # Inefficient: Multiple round trips
    for doc in documents:
        db.upsert([doc['content']], [doc['metadata']])

    # Efficient: Single batch operation
    contents = [doc['content'] for doc in documents]
    metadata = [doc['metadata'] for doc in documents]
    db.upsert(contents, metadata)

Async Concurrency
~~~~~~~~~~~~~~~~~

Leverage async for parallel operations::

    async def parallel_search(db, queries):
        """Execute multiple searches in parallel"""
        tasks = [
            db.query_async(query, k=10)
            for query in queries
        ]

        results = await asyncio.gather(*tasks)
        return results

Use Appropriate Search Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose the right search type for your use case::

    # Fast: Keyword search for exact matches
    results = db.query("ERROR: Connection failed", search_type="keyword")

    # Accurate: Vector search for semantic similarity
    results = db.query("network connectivity issues", search_type="vector")

    # Balanced: Hybrid for best of both
    results = db.query("connection problems", search_type="hybrid")

Database URI Support
--------------------

RemoteVectorDB supports URI-based connections::

    from localvectordb.factory import from_uri

    # Local database
    db = from_uri("lvdb:///path/to/databases/mydb")

    # Remote HTTP
    db = from_uri("lvdb+http://localhost:8000/mydb")

    # Remote HTTPS with auth
    db = from_uri("lvdb+https://api_key@vectordb.company.com/mydb")

    # With query parameters
    db = from_uri(
        "lvdb+https://vectordb.company.com/mydb?"
        "chunk_size=1000&embedding_model=nomic-embed-text"
    )

Best Practices
--------------

1. **Use Environment Variables for Configuration**

   Store sensitive configuration in environment variables::

       # .env file
       VECTOR_DB_URL=https://vectordb.company.com
       LVDB_API_KEY=sk-abc123def456

       # Application code
       db = VectorDB(
           "mydb",
           os.getenv("VECTOR_DB_URL"),
           api_key=os.getenv("LVDB_API_KEY")
       )

2. **Implement Health Checks**

   Monitor database health in production::

       async def health_check():
           try:
               if not db.healthy:  # property, not a method
                   alert_ops_team("Database unhealthy")
                   return False
               return True
           except Exception as e:
               alert_ops_team(f"Health check failed: {e}")
               return False

3. **Use Context Managers**

   Ensure proper cleanup with context managers::

       async def process_documents():
           async with RemoteVectorDB("mydb", url) as db:
               # Process documents
               await db.upsert_async(documents)
           # Connection automatically closed

4. **Handle Errors Gracefully**

   Implement comprehensive error handling::

       from localvectordb.exceptions import BaseLocalVectorDBException

       try:
           results = db.query("search term")
       except BaseLocalVectorDBException as e:
           logger.error(f"Database operation failed: {e}")
           # Implement fallback logic
       except Exception as e:
           logger.critical(f"Unexpected error: {e}")
           # Alert and fail safely

5. **Optimize for Your Use Case**

   - **High Write Throughput**: Use batch operations and async upserts
   - **Low Latency Queries**: Keep connection pools warm, use caching
   - **Large Documents**: Use file operations with chunking
   - **Complex Filtering**: Create appropriate indexes on metadata fields

Migration Guide
---------------

Migrating from the legacy API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If migrating from an older, pre-release API::

    # Old (legacy) code
    from localvectordb import LocalVectorDB

    db = LocalVectorDB(path="./data/mydb.db")
    db.add_texts(["text1", "text2"])
    results = db.similarity_search("query")

    # v0.1.0 code (local)
    from localvectordb import LocalVectorDB

    db = LocalVectorDB("mydb", "./data")
    db.upsert(["text1", "text2"])
    results = db.query("query", search_type="vector")

    # v0.1.0 code (remote)
    from localvectordb.client import RemoteVectorDB

    db = RemoteVectorDB("mydb", "http://server:8000")
    db.upsert(["text1", "text2"])
    results = db.query("query", search_type="vector")

See Also
--------

- :doc:`quickstart` - Get started with LocalVectorDB
- :doc:`server/index` - Deploy LocalVectorDB server
- :doc:`query` - Query methods and options
- :doc:`metadata.filtering` - Metadata schema and filtering
- :doc:`embeddings` - Embedding providers configuration
- :doc:`performance` - Performance tuning guide