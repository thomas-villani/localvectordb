Overview
========

LocalVectorDB provides a document-first API that abstracts away the complexity of chunking while providing powerful
search and metadata capabilities. The library supports both local databases and remote connections to LocalVectorDB servers,
with comprehensive async/await support and a SQL-like query interface.

Core Concepts
-------------

Documents vs Chunks
^^^^^^^^^^^^^^^^^^^

**Documents** are the primary unit of storage—complete texts like articles, papers, or files. **Chunks** are
automatically created internal representations that enable efficient vector search. Users work with documents;
the system handles chunking transparently.

.. code-block:: python

   # You work with documents
   doc_content = "This is a complete document with multiple sentences..."
   doc_id = db.upsert([doc_content])

   # System automatically creates chunks for vector search
   # Chunks maintain position tracking for perfect reconstruction

Metadata Schema
^^^^^^^^^^^^^^^

Define structured metadata with type validation and indexing:

.. code-block:: python

   from localvectordb.core import MetadataField, MetadataFieldType

   schema = {
       'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'publish_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
       'tags': MetadataField(type=MetadataFieldType.JSON),
       'word_count': MetadataField(type=MetadataFieldType.INTEGER),
       'is_published': MetadataField(type=MetadataFieldType.BOOLEAN, default_value=False)
   }

Search Types
^^^^^^^^^^^^

- **Vector Search**: Semantic similarity using embeddings
- **Keyword Search**: Traditional full-text search with FTS5
- **Hybrid Search**: Combines vector and keyword with weighted scoring

Basic Usage
-----------

Creating a Database
^^^^^^^^^^^^^^^^^^^

Synchronous Local Database
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from localvectordb import VectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Local database
   db = VectorDB(
       name="my_database",
       base_path="./vector_data",
       metadata_schema={
           'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'priority': MetadataField(type=MetadataFieldType.INTEGER, default_value=1)
       },
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       chunk_size=500,
       chunking_method="sentences"
   )

Asynchronous Local Database
~~~~~~~~~~~~~~~~~~~~~~~~~~~

LocalVectorDB includes full async/await support for improved performance in I/O-intensive applications
and web servers. Async operations are particularly beneficial when working with remote embedding providers
or handling multiple concurrent operations.

.. code-block:: python

   # Async local database using factory
   async_db = VectorDB(
       name="my_database",
       base_path="./vector_data",
       async_mode=True,  # Enable async mode
       metadata_schema=schema,
       embedding_provider="ollama",
       embedding_model="nomic-embed-text"
   )

   # Or use async context manager (recommended)
   async with VectorDB("my_db", "./data", async_mode=True) as db:
       doc_ids = await db.upsert(["Document 1", "Document 2"])
       results = await db.query("search term", k=5)

Alternative Async Factory
~~~~~~~~~~~~~~~~~~~~~~~~~

For convenience, you can also use the direct async factory function:

.. code-block:: python

   from localvectordb.async_database import AsyncVectorDB

   # Direct async factory function - automatically initializes the database
   async_db = await AsyncVectorDB("my_db", "./local_path")
   try:
       results = await async_db.query("search term")
   finally:
       await async_db.close()

Remote Database Connection
^^^^^^^^^^^^^^^^^^^^^^^^^^

Understanding the LocalVectorDB Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LocalVectorDB can operate in two modes: **local** (embedded database) and **remote** (client-server).
The remote mode allows you to:

- **Centralize databases**: Multiple applications can access the same vector databases
- **Scale horizontally**: Run the server on powerful hardware while keeping clients lightweight
- **Share resources**: Multiple users can collaborate on the same document collections
- **Deploy in production**: Separate your application logic from database management

The LocalVectorDB server provides a REST API that mirrors the local database interface, so you can
switch between local and remote modes with minimal code changes.

Setting Up the LocalVectorDB Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before connecting remotely, you need to start a LocalVectorDB server. The easiest way is using the
built-in CLI:

**Basic Server Setup:**

.. code-block:: bash

   # Install the server component (if not already installed)
   pip install localvectordb[server]

   # Start a basic server
   lvdb serve --host 0.0.0.0 --port 5000

   # Start with a specific database directory
   lvdb serve --host 0.0.0.0 --port 5000 --db-folder ./my_databases

This starts a server that will:

- Listen on all interfaces (``0.0.0.0``) at port 5000
- Store databases in the specified folder (or default ``.lvdb`` directory)
- Accept connections without authentication (suitable for development)

**Production Server Setup:**

For production deployments, you'll want additional security and configuration:

.. code-block:: bash

   # Initialize a configuration file with production settings
   lvdb config init --interactive

   # Or create a production config directly
   lvdb config init --enable-auth --enable-cors --cors-origins "https://myapp.com" \
                    --enable-rate-limiting --rate-limit "1000 per hour" \
                    --output ./production-config.toml

   # Start server with configuration
   lvdb serve --config ./production-config.toml

**Server Configuration Example:**

.. code-block:: toml

   [server]
   host = "0.0.0.0"
   port = 5000
   debug = false
   cors_enabled = true
   cors_origins = ["https://myapp.com", "https://admin.myapp.com"]

   [database]
   root_directory = "./databases"
   default_embedding_provider = "ollama"
   default_embedding_model = "nomic-embed-text"
   default_chunk_size = 500

   [auth]
   enabled = true
   api_key_required = true

   [rate_limiting]
   enabled = true
   default_limit = "1000 per hour"

**Managing API Keys:**

If you enable authentication, create API keys for your applications:

.. code-block:: bash

   # Create an API key
   lvdb auth create-key --description "My Application" --expires-days 90

   # List existing keys
   lvdb auth list-keys

   # Revoke a key
   lvdb auth revoke-key <key-id>

Connecting to Remote Databases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once your server is running, you can connect from any Python application:

**Synchronous Remote Connection:**

.. code-block:: python

   # Connect to remote server
   db = VectorDB(
       name="my_database",
       base_path="http://localhost:5000",  # Server URL
       api_key="your_api_key",              # If authentication is enabled
       create_if_not_exists=True,           # Create database if it doesn't exist
       metadata_schema=schema
   )

   # Use exactly like a local database
   doc_ids = db.upsert(["Document 1", "Document 2"])
   results = db.query("search term", k=5)

**Asynchronous Remote Connection:**

.. code-block:: python

   # Async remote database - ideal for web applications
   async with VectorDB(
       name="my_database",
       base_path="http://localhost:5000",
       async_mode=True,                    # Enable async mode
       api_key="your_api_key",
       max_retries=3,                      # Handle network issues gracefully
       timeout=60.0                       # Request timeout
   ) as db:
       doc_ids = await db.upsert(["Document 1", "Document 2"])
       results = await db.query("search term", k=5)

**Connection Configuration:**

You can configure various aspects of the remote connection:

.. code-block:: python

   # Advanced remote configuration
   db = VectorDB(
       name="production_db",
       base_path="https://vectordb.mycompany.com",
       api_key=os.getenv("LVDB_API_KEY"),    # Use environment variable
       async_mode=True,

       # Connection settings
       timeout=120.0,                        # 2 minute timeout
       max_retries=5,                        # Retry failed requests
       retry_delay=1.0,                      # Base delay between retries

       # Database settings (applied when creating)
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       chunk_size=800,
       chunking_method="paragraphs"
   )

**Environment Variables:**

For security and convenience, you can use environment variables:

.. code-block:: bash

   # Set API key environment variable
   export LVDB_API_KEY="your_api_key_here"

.. code-block:: python

   # No need to specify api_key in code
   db = VectorDB(
       name="my_database",
       base_path="http://localhost:5000"
       # api_key automatically loaded from LVDB_API_KEY
   )

Explicit Database Type Specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you may want to explicitly control whether to use local or remote databases,
regardless of how the ``base_path`` looks:

.. code-block:: python

   from localvectordb.factory import create_vectordb

   # Force local database even with URL-like path
   local_db = create_vectordb(
       "my_db",
       "http_server_backup",              # This looks like URL but will be treated as local path
       database_type="local"
   )

   # Force remote database with local-looking path
   remote_db = create_vectordb(
       "my_db",
       "localhost:5000",                  # Will be treated as http://localhost:5000
       database_type="remote",
       api_key="your_api_key"
   )

   # Async variants
   async_local = create_vectordb(
       "my_db",
       "./local_path",
       database_type="local",
       async_mode=True
   )

.. note::

    The ``VectorDB`` class is a factory that returns different implementations based on the parameters:

    - **LocalVectorDB**: When ``base_path`` is a local file path
    - **RemoteVectorDB**: When ``base_path`` starts with ``http://`` or ``https://``
    - **AsyncLocalVectorDB**: Local with ``async_mode=True``
    - **AsyncRemoteVectorDB**: Remote with ``async_mode=True``

    This allows you to use the same codebase for local development and remote production deployments.

Document Operations
-------------------

Adding Documents
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Single document
   doc_id = await db.upsert("This is my document content")

   # Multiple documents with metadata
   documents = [
       "First document content...",
       "Second document content...",
       "Third document content..."
   ]

   metadata = [
       {"category": "tech", "priority": 5},
       {"category": "business", "priority": 3},
       {"category": "personal", "priority": 1}
   ]

   doc_ids = await db.upsert(
       documents=documents,
       metadata=metadata,
       ids=["doc_1", "doc_2", "doc_3"]  # Optional custom IDs
   )

Searching Documents
^^^^^^^^^^^^^^^^^^^

.. todo: should explain the search types in more detail

Basic Search Operations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Vector search (semantic similarity)
   results = await db.query(
       "machine learning algorithms",
       search_type="vector",
       k=5,
       score_threshold=0.7
   )

   # Keyword search (exact matches using FTS5)
   results = await db.query(
       "python programming",
       search_type="keyword",
       k=10
   )

   # Hybrid search (combines vector and keyword)
   results = await db.query(
       "neural networks",
       search_type="hybrid",
       k=5,
       vector_weight=0.7  # 70% vector, 30% keyword
   )

Search with Metadata Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Search with metadata filters
   results = await db.query(
       "artificial intelligence",
       search_type="vector",
       k=3,
       filters={"category": "tech", "priority": {">=": 3}}
   )

   # Process results
   for result in results:
       print(f"Score: {result.score:.3f}")
       print(f"Document: {result.id}")
       print(f"Content: {result.content[:200]}...")
       print(f"Metadata: {result.metadata}")
       print("---")

Return Types and Context Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to increasing context windows in modern LLMs, by default, search results return the entire document
where there is a match. You can also retrieve only the matched chunks or a context window surrounding
the matched chunks.

.. code-block:: python

   # Return only the relevant chunks
   results = await db.query(
       "machine learning algorithms",
       search_type="vector",
       return_type="chunks",
       k=5
   )

   # Return a context-window surrounding the relevant chunks
   results = await db.query(
       "machine learning algorithms",
       search_type="vector",
       return_type="context",
       k=5,
       context_window=2  # 2 chunks before and after the relevant chunk
   )

Advanced Query Builder
----------------------

The Query Builder provides a SQL-like interface for complex queries with advanced filtering, grouping,
and aggregation capabilities. This is particularly powerful for analytical workloads and complex
document discovery scenarios.

Basic Query Builder Usage
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simple semantic search
   results = (db.query_builder()
       .search("machine learning algorithms")
       .limit(10)
       .execute())

   # For async databases, execution is awaitable
   results = await (db.query_builder()
       .search("machine learning algorithms")
       .limit(10)
       .execute())

Advanced Filtering and Search Combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Complex multi-field query with exact and semantic filters
   results = await (db.query_builder()
       .search("deep learning frameworks")
       .filter("year", gte_=2020, lt_=2024)          # Year range
       .filter("category", "AI")                     # Exact match
       .filter("tags", contains_="pytorch")          # Tag filtering
       .semantic_filter("methodology", "supervised learning", threshold=0.75)
       .order_by("year", "desc")
       .limit(50)
       .execute())

   # Multi-term search with different types
   results = await (db.query_builder()
       .vector("machine learning", score_threshold=0.8)
       .keyword("python OR pytorch")
       .filter("category", in_=["AI", "research"])
       .execute())

Aggregations and Grouping
^^^^^^^^^^^^^^^^^^^^^^^^^^

The Query Builder supports SQL-like aggregations for analytical queries:

.. code-block:: python

   # Group by category and count documents
   results = await (db.query_builder()
       .search("artificial intelligence")
       .group_by("category")
       .count_by("*", "doc_count")
       .having("doc_count", "gt", 5)
       .order_by("doc_count", "desc")
       .execute())

   # Multiple aggregations with complex grouping
   results = await (db.query_builder()
       .search("research papers")
       .group_by("year", "category")
       .count_by("*", "paper_count")
       .avg_by("citation_count", "avg_citations")
       .max_by("impact_factor", "max_impact")
       .sum_by("page_count", "total_pages")
       .having("paper_count", "gte", 10)
       .order_by("avg_citations", "desc")
       .execute())

Result Reranking and Post-Processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Rerank by recency (newer documents ranked higher)
   results = await (db.query_builder()
       .search("news")
       .rerank("recency", date_field="published_date", weight=0.3)
       .limit(20)
       .execute())

   # Rerank by diversity (promote variety in results)
   results = await (db.query_builder()
       .search("scientific articles")
       .rerank("diversity", field="category", weight=0.4)
       .execute())

   # Custom reranking with multiple factors
   results = await (db.query_builder()
       .search("technical documentation")
       .rerank("custom",
               relevance_weight=0.6,
               recency_weight=0.2,
               diversity_weight=0.2,
               date_field="updated_at")
       .execute())

Query Explanation and Performance Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Enable query explanation for debugging and optimization
   results = await (db.query_builder()
       .search("complex query")
       .filter("category", "research")
       .group_by("author")
       .count_by("*", "doc_count")
       .explain(detailed=True)
       .execute())

   # Results include execution metadata
   if results and hasattr(results[0], 'execution_stats'):
       print(f"Query took: {results[0].execution_stats['total_time']:.3f}s")
       print(f"Search time: {results[0].execution_stats['search_time']:.3f}s")
       print(f"Filter time: {results[0].execution_stats['filter_time']:.3f}s")
       print(f"Steps: {results[0].execution_plan}")

Document Management
-------------------

Retrieving Documents
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get single document
   doc = await db.get("doc_1")
   if doc:
       print(f"Content: {doc.content}")
       print(f"Metadata: {doc.metadata}")
       print(f"Created: {doc.created_at}")

   # Get multiple documents
   docs = await db.get(["doc_1", "doc_2", "doc_3"])

   # Check if documents exist without retrieving content
   exists = await db.exists(["doc_1", "doc_2"])  # Returns [True, False]

Updating Documents
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Update content only
   success = await db.update("doc_1", content="Updated document content")

   # Update metadata only
   success = await db.update("doc_1", metadata={"priority": 5})

   # Update both content and metadata
   success = await db.update(
       "doc_1",
       content="New content",
       metadata={"category": "updated", "priority": 10}
   )

Filtering and Querying Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simple filters
   docs = await db.filter(where={"category": "tech"})

   # Complex filters with operators
   docs = await db.filter(where={
       "priority": {"$gte": 3},
       "category": {"$in": ["tech", "business"]},
       "publish_date": {"$between": ["2024-01-01", "2024-12-31"]}
   })

   # Pagination support
   docs = await db.filter(
       where={"category": "tech"},
       order_by="created_at DESC",
       limit=20,
       offset=40  # Page 3 of 20 items per page
   )

Advanced Configuration
----------------------

Custom Chunking Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Different chunking methods for different content types
   db_sentences = VectorDB("docs", chunking_method="sentences", chunk_size=300)
   db_paragraphs = VectorDB("docs", chunking_method="paragraphs", chunk_size=800)
   db_sections = VectorDB("docs", chunking_method="sections", chunk_size=1000)

   # Code-specific chunking for programming content
   db_code = VectorDB("code", chunking_method="code-blocks", chunk_size=500)

   # Token-based chunking for precise control
   db_tokens = VectorDB("precise", chunking_method="tokens", chunk_size=512)

Multiple Embedding Providers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Ollama (local, free)
   db_ollama = VectorDB(
       "db1",
       embedding_provider="ollama",
       embedding_model="nomic-embed-text"
   )

   # OpenAI (cloud, requires API key)
   db_openai = VectorDB(
       "db2",
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       embedding_config={"api_key": "your_key"}  # Or set OPENAI_API_KEY environment variable
   )

.. important::
   Don't leave your API keys in code committed to version control. Use environment variables
   or configuration files that are excluded from version control.

Batch Operations
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Large document insertion with batching for performance
   large_docs = ["doc content " + str(i) for i in range(10000)]
   doc_ids = await db.upsert(documents=large_docs, batch_size=100)

   # Similarity threshold to avoid duplicates
   doc_ids = await db.insert(
       documents=new_docs,
       similarity_threshold=0.95,  # Skip if 95%+ similar to existing
       errors="ignore"  # Don't fail on duplicates
   )

Performance Tuning
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # High-performance configuration
   db = VectorDB(
       name="high_perf_db",
       base_path="./data",

       # Embedding settings
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",

       # Chunking optimization
       chunking_method="sentences",
       chunk_size=400,           # Smaller chunks = better precision
       chunk_overlap=2,          # Overlap sentences for context

       # Performance settings
       enable_gpu=True,          # Use GPU for FAISS if available
       enable_fts=True,          # Enable full-text search
       connection_pool_size=20,  # More connections for concurrent access

       # Async-specific settings (for AsyncLocalVectorDB)
       async_mode=True,
       max_workers=8             # More workers for CPU-heavy workloads
   )

Memory-Only Database
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # In-memory database for testing/temporary use
   temp_db = VectorDB("temp", ":memory:", async_mode=True)

Command Line Interface
----------------------

LocalVectorDB includes a comprehensive CLI for server management and database operations.

Server Management
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start a basic development server
   lvdb serve --host 0.0.0.0 --port 5000

   # Start with configuration file
   lvdb serve --config ./my-config.toml --db-folder ./databases

   # Start with debug mode and custom logging
   lvdb serve --debug --log-level DEBUG

   # Production server with specific settings
   lvdb serve --host 0.0.0.0 --port 5000 --disable-ollama-check

Configuration Management
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Initialize configuration interactively
   lvdb config init --interactive

   # Create production configuration with common settings
   lvdb config init --redis-registry redis://localhost:6379/1 \
                    --enable-cache --cache-type redis \
                    --enable-rate-limiting --rate-limit "1000 per hour" \
                    --enable-cors --cors-origins "https://myapp.com" \
                    --enable-auth

   # View current configuration
   lvdb config show
   lvdb config show --format json --section database

   # Get and set specific configuration values
   lvdb config get server.host
   lvdb config set database.chunk_size 1000

Database Operations
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # List available databases with details
   lvdb list --details

   # Create a new database with specific settings
   lvdb create mydatabase --embedding-model nomic-embed-text --chunk-size 500

   # Add documents to a database
   lvdb db mydatabase add document.txt
   lvdb db mydatabase add "documents/*.py"        # Glob patterns
   cat document.txt | lvdb db mydatabase add -   # From stdin

   # Search documents with various options
   lvdb db mydatabase search "query text" --limit 5
   lvdb db mydatabase search "query text" --search-type hybrid \
       --metadata-filter '{"author":"Smith"}'

   # Get specific documents
   lvdb db mydatabase get doc_1

   # Find similar documents using k-nearest neighbors
   lvdb db mydatabase knn doc_1 --k 5

   # Interactive database shell
   lvdb db mydatabase shell

Authentication Management
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Create API keys for applications
   lvdb auth create-key --description "My Application" --expires-days 30
   lvdb auth create-key --description "Admin Access" --no-expiry

   # List and manage existing keys
   lvdb auth list-keys
   lvdb auth revoke-key <key-id>

Database Statistics and Management
-----------------------------------

.. code-block:: python

   # Get comprehensive database statistics
   stats = await db.get_stats()
   print(f"Documents: {stats['documents']}")
   print(f"Chunks: {stats['chunks']}")
   print(f"Embedding model: {stats['embedding_model']}")
   print(f"Index vectors: {stats['index_vectors']}")
   print(f"Database size: {stats['size_mb']:.1f} MB")

   # Save database state (for local databases)
   await db.save()

   # Close database and cleanup resources
   await db.close()

   # Context manager ensures automatic cleanup
   async with VectorDB("temp_db", ":memory:", async_mode=True) as db:
       await db.upsert(["temporary document"])
       results = await db.query("search")
   # Database automatically closed and resources cleaned up

Updating Metadata Schema
^^^^^^^^^^^^^^^^^^^^^^^^^

You can update a database's metadata schema after creation to add new fields, modify existing ones,
or change indexing:

.. code-block:: python

   # Add new metadata fields to existing database
   new_schema = {
       # Keep existing fields
       'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),

       # Add new fields
       'category': MetadataField(
           type=MetadataFieldType.TEXT,
           indexed=True,
           required=True,
           default_value="general"
       ),
       'rating': MetadataField(type=MetadataFieldType.REAL, indexed=True),
       'tags': MetadataField(type=MetadataFieldType.JSON, default_value=[])
   }

   # Apply schema update
   changes = await db.update_metadata_schema(new_schema)

   # Review what changed
   print(f"Added fields: {changes['added_fields']}")
   print(f"Populated defaults: {len(changes['populated_defaults'])} documents updated")

.. note::
   - Existing document data is preserved when updating schema
   - New required fields get populated with default values automatically
   - Removed fields are kept in the database for safety (use ``drop_columns=True`` to actually remove)
   - Schema changes are applied in a transaction and rolled back on error

Error Handling
--------------

.. code-block:: python

   from localvectordb.exceptions import (
       DatabaseNotFoundError,
       DuplicateDocumentIDError,
       EmbeddingError,
       ConfigurationError
   )

   try:
       db = VectorDB("nonexistent", create_if_not_exists=False)
   except DatabaseNotFoundError:
       print("Database does not exist")

   try:
       await db.insert(["content"], ids=["existing_id"])
   except DuplicateDocumentIDError:
       print("Document ID already exists")

   try:
       results = await db.query("search query")
   except EmbeddingError as e:
       print(f"Embedding generation failed: {e}")

   try:
       # Server connection issues
       remote_db = VectorDB("test", "http://nonexistent-server:5000")
   except ConfigurationError as e:
       print(f"Configuration error: {e}")

Migration Guide
---------------

Migrating from Synchronous to Asynchronous
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Old synchronous code
   db = VectorDB("my_db", "./data")
   doc_ids = db.upsert(["Document 1", "Document 2"])
   results = db.query("search term", k=5)

   # New asynchronous code
   async with VectorDB("my_db", "./data", async_mode=True) as db:
       doc_ids = await db.upsert(["Document 1", "Document 2"])
       results = await db.query("search term", k=5)

Migrating to Query Builder
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Old direct query method
   results = await db.query(
       "search term",
       search_type="hybrid",
       k=10,
       filters={"category": "tech"}
   )

   # New query builder approach (more powerful and flexible)
   results = await (db.query_builder()
       .hybrid("search term")
       .filter("category", "tech")
       .limit(10)
       .execute())

Performance Considerations
--------------------------

**Async vs Sync Selection**
   Use async variants for I/O-intensive applications, web servers, and when handling multiple
   concurrent operations. Sync variants are simpler for scripts and single-threaded applications.

**Batch Operations**
   Use the ``batch_size`` parameter for large document insertions to optimize memory usage and performance.

**Thread Pool Configuration**
   For ``AsyncLocalVectorDB``, adjust ``max_workers`` based on CPU cores. More workers help with
   CPU-intensive operations like chunking and embedding generation.

**Connection Pooling**
   Increase ``connection_pool_size`` for high-concurrency scenarios with multiple simultaneous database operations.

**GPU Acceleration**
   Enable ``enable_gpu=True`` if FAISS GPU support is available for faster vector similarity search.

**Chunking Strategy Selection**
   - Smaller chunks provide better precision but increase storage overhead
   - Larger chunks are more efficient but may reduce search accuracy
   - Choose chunking method based on content type (sentences for prose, code-blocks for programming content)

Architecture Overview
---------------------

**Local Databases**
   - **Storage**: SQLite for documents and metadata, FAISS for vector indices
   - **Chunking**: Position-aware chunking with multiple strategies and overlap support
   - **Embeddings**: Plugin architecture supporting multiple providers
   - **Search**: Normalized scoring across vector, keyword, and hybrid modes

**Remote Databases**
   - **Client**: HTTP client with connection pooling, retry logic, and async support
   - **Server**: Flask-based REST API with authentication and rate limiting
   - **Scaling**: Multi-worker deployment with Redis for coordination

**Async Implementation**
   - **Thread Pool**: CPU and I/O bound operations executed in thread pools
   - **Native Async**: Direct async calls for embedding APIs when available
   - **Resource Management**: Proper cleanup with context managers and connection pooling

**Factory Pattern**
   The ``VectorDB`` factory automatically selects the appropriate implementation based on parameters,
   enabling seamless switching between local/remote and sync/async variants with minimal code changes.