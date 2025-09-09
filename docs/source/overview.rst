Overview
========

LocalVectorDB is a **document-first vector database** that combines the simplicity of SQLite with the power of FAISS for vector similarity search. The library provides a unified API that works seamlessly with both local embedded databases and remote server deployments, featuring comprehensive async/await support and advanced multi-column search capabilities.

Core Concepts
-------------

Documents vs Chunks
^^^^^^^^^^^^^^^^^^^

**Documents** are the primary unit of storage—complete texts like articles, papers, or files that you work with directly. **Chunks** are automatically created internal representations that enable efficient vector search, maintaining position tracking for perfect document reconstruction.

.. code-block:: python

   # You work with documents
   doc_content = "This is a complete document with multiple sentences..."
   doc_id = db.upsert([doc_content])

   # System automatically creates chunks for vector search
   # Chunks maintain position tracking for perfect reconstruction

Local vs Remote Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^

LocalVectorDB operates in two modes:

- **Local**: Embedded database stored as files on your filesystem
- **Remote**: Client-server architecture with HTTP API communication

The same API works for both modes, allowing easy migration from local development to remote production deployments.

Synchronous vs Asynchronous APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every operation is available in both sync and async variants on the same database instance, allowing you to choose the best approach for each operation:

.. code-block:: python

   # Create database instance (works for both sync and async)
   db = VectorDB("my_db", "./data")
   
   # Synchronous API - simpler for scripts and single-threaded apps
   doc_ids = db.upsert(["Document 1", "Document 2"])
   results = db.query("search term", k=5)

   # Asynchronous API - same instance, just use async methods
   doc_ids = await db.upsert_async(["Document 1", "Document 2"])
   results = await db.query_async("search term", k=5)
   
   # Can use async context manager for automatic cleanup
   async with db:
       results = await db.query_async("search term", k=5)

Metadata Schema
^^^^^^^^^^^^^^^

Define structured metadata with type validation, indexing, and multi-column embedding support:

.. code-block:: python

   from localvectordb.core import MetadataField, MetadataFieldType

   schema = {
       'title': MetadataField(
           type=MetadataFieldType.TEXT, 
           indexed=True, 
           embedding_enabled=True,  # Generate embeddings for this field
           fts_enabled=True        # Enable full-text search
       ),
       'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'abstract': MetadataField(type=MetadataFieldType.TEXT, embedding_enabled=True),
       'publish_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
       'tags': MetadataField(type=MetadataFieldType.JSON),
       'word_count': MetadataField(type=MetadataFieldType.INTEGER),
       'is_published': MetadataField(type=MetadataFieldType.BOOLEAN, default_value=False)
   }

**Multi-Column Embeddings**: Enable ``embedding_enabled`` on TEXT or JSON fields to make them searchable using vector similarity, not just the main document content.

Search Types
^^^^^^^^^^^^

LocalVectorDB supports four types of search operations:

- **Vector Search**: Semantic similarity using embeddings
- **Keyword Search**: Traditional full-text search with SQLite FTS5
- **Hybrid Search**: Combines vector and keyword with weighted scoring
- **Multi-Column Search**: Search across document content AND metadata fields simultaneously

Getting Started
---------------

Installation
^^^^^^^^^^^^

.. code-block:: bash

   # Basic installation
   pip install localvectordb

   # With server capabilities
   pip install localvectordb[server]

   # Development installation with all features
   pip install localvectordb[all]

Quick Example
^^^^^^^^^^^^^

Here's a complete example showing both sync and async usage:

.. code-block:: python

   from localvectordb import VectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Define schema with embedding-enabled fields
   schema = {
       'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True, embedding_enabled=True),
       'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
       'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
   }

   # Synchronous usage
   db = VectorDB("my_docs", "./data", metadata_schema=schema)
   
   # Add documents with metadata
   doc_ids = db.upsert(
       documents=["Artificial intelligence is transforming healthcare...", 
                  "Machine learning algorithms can detect patterns..."],
       metadata=[
           {"title": "AI in Healthcare", "author": "Dr. Smith", "category": "medical"},
           {"title": "ML Pattern Detection", "author": "Prof. Jones", "category": "technical"}
       ]
   )

   # Search across all columns
   results = db.query_multi_column("healthcare AI", k=5)
   for result in results:
       print(f"Found in {result.metadata.get('_search_column')}: {result.content[:50]}...")

   # Asynchronous usage (same database instance, use async methods)
   async def async_example():
       db = VectorDB("my_docs", "./data", metadata_schema=schema)
       async with db:
           doc_ids = await db.upsert_async(
               documents=["Neural networks are powerful tools...", 
                          "Deep learning models require large datasets..."],
               metadata=[
                   {"title": "Neural Network Basics", "author": "Dr. Brown", "category": "education"},
                   {"title": "Deep Learning Data", "author": "Prof. Wilson", "category": "research"}
               ]
           )
           
           results = await db.query_multi_column_async("neural networks", k=5)
           return results

Local Database Usage
--------------------

Creating Local Databases
^^^^^^^^^^^^^^^^^^^^^^^^^

Local databases are stored as files on your filesystem and provide the fastest performance for single-application use cases:

.. code-block:: python

   # Synchronous local database
   db = VectorDB(
       name="my_database",
       base_path="./vector_data",           # Directory to store database files
       metadata_schema=schema,
       embedding_provider="ollama",         # Local embedding provider
       embedding_model="nomic-embed-text",
       chunk_size=500,
       chunking_method="sentences"
   )

   # Same database for async operations (recommended for web applications)
   db = VectorDB(
       name="my_database", 
       base_path="./vector_data",
       metadata_schema=schema
   )
   
   # Use async context manager and async methods
   async with db:
       doc_ids = await db.upsert_async(["Document content..."])
       results = await db.query_async("search query", k=10)

Local Database Operations
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Document operations (same API for sync/async)
   
   # Adding documents (sync)
   doc_ids = db.upsert(
       documents=["First document...", "Second document..."],
       metadata=[{"category": "tech"}, {"category": "business"}],
       ids=["doc1", "doc2"]  # Optional custom IDs
   )

   # Adding documents (async)
   doc_ids = await db.upsert_async(
       documents=["First document...", "Second document..."],
       metadata=[{"category": "tech"}, {"category": "business"}],
       ids=["doc1", "doc2"]
   )

   # Searching with different types (async examples)
   vector_results = await db.query_async("machine learning", search_type="vector", k=5)
   keyword_results = await db.query_async("python programming", search_type="keyword", k=5)
   hybrid_results = await db.query_async("AI research", search_type="hybrid", k=5)

   # Multi-column search across content and metadata
   multi_results = await db.query_multi_column_async(
       "deep learning",
       columns=["content", "title", "abstract"],  # Search specific fields
       k=10
   )

   # Document management (async examples)
   doc = await db.get_async("doc1")                    # Retrieve document
   success = await db.update_async("doc1", metadata={"priority": 5})  # Update metadata
   count = await db.delete_async(["doc1", "doc2"])     # Delete documents

Remote Database Usage
---------------------

Setting Up the Server
^^^^^^^^^^^^^^^^^^^^^^

Before using remote databases, start a LocalVectorDB server:

.. code-block:: bash

   # Basic development server
   lvdb serve --host 0.0.0.0 --port 5000

   # Production server with authentication
   lvdb config init --enable-auth --enable-cors
   lvdb serve --config ./production-config.toml

   # Create API key for clients (with permission level)
   lvdb auth create-key --description "My App" --permission-level read_write --expires-days 90

Connecting to Remote Databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remote databases provide the same API as local databases but communicate over HTTP:

.. code-block:: python

   # Synchronous remote connection
   remote_db = VectorDB(
       name="my_database",
       base_path="http://localhost:5000",   # Server URL
       api_key="your_api_key",             # If authentication enabled
       metadata_schema=schema
   )

   # Use exactly like a local database
   doc_ids = remote_db.upsert(["Remote document content..."])
   results = remote_db.query("search term", k=5)

   # Asynchronous remote connection (recommended)
   remote_db = VectorDB(
       name="my_database",
       base_path="http://localhost:5000",
       api_key="your_api_key",
       timeout=60.0,                       # Request timeout
       max_retries=3                       # Handle network issues
   )
   
   async with remote_db:
       doc_ids = await remote_db.upsert_async(["Remote document..."])
       results = await remote_db.query_async("search term", k=5)

Production Remote Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For production deployments, configure authentication, rate limiting, and monitoring:

.. code-block:: python

   # Production remote database configuration
   production_db = VectorDB(
       name="production_db",
       base_path="https://vectordb.company.com",
       api_key=os.getenv("LVDB_API_KEY"),  # Use environment variable
       
       # Connection settings
       timeout=120.0,                      # 2 minute timeout
       max_retries=5,                      # Retry failed requests
       retry_delay=1.0,                    # Base delay between retries
       
       # Database settings (applied when creating)
       metadata_schema=production_schema,
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       chunk_size=800
   )

Multi-Column Search (Advanced)
------------------------------

Multi-column search is LocalVectorDB's most powerful feature, allowing you to search across both document content and metadata fields simultaneously.

Enabling Multi-Column Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, enable embeddings on metadata fields in your schema:

.. code-block:: python

   schema = {
       'title': MetadataField(
           type=MetadataFieldType.TEXT, 
           indexed=True,
           embedding_enabled=True      # This field will be searchable
       ),
       'abstract': MetadataField(
           type=MetadataFieldType.TEXT,
           embedding_enabled=True      # This field will be searchable
       ),
       'author': MetadataField(
           type=MetadataFieldType.TEXT,
           indexed=True                # Only indexed, not embedding-enabled
       )
   }

Using Multi-Column Search
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Search all embedding-enabled fields plus main content
   results = await db.query_multi_column_async("machine learning", k=10)
   
   # Search specific columns only
   results = await db.query_multi_column_async(
       "neural networks",
       columns=["content", "title", "abstract"],  # Only search these fields
       search_type="vector",
       k=5
   )
   
   # Results include column attribution
   for result in results:
       column = result.metadata.get('_search_column')
       score = result.score
       print(f"Found in {column} (score: {score:.3f}): {result.content[:100]}...")

Advanced Search Examples
^^^^^^^^^^^^^^^^^^^^^^^^
There are many options to give fine-tuned control over search results. In particular, there are a number of options
for the ``document_scoring_method`` that can be employed to control how the overall score of a document is calculated
from the scores of the chunks. :doc:`Learn more about document_scoring_method options<document-scoring>`.

.. code-block:: python

   # Scientific paper search across multiple fields
   paper_results = await db.query_async(
       "deep learning transformers",
       search_type="hybrid",           # Combine vector and keyword search
       k=20,
       filters={                       # Filter by metadata
           "year": {"$gte": 2020},
           "category": {"$in": ["AI", "ML", "NLP"]}
       },
       document_scoring_method="frequency_boost"  # Boost documents with multiple matches
   )

   # Customer support search across tickets and knowledge base
   support_results = await db.query_multi_column_async(
       "password reset issue",
       columns=["content", "title", "tags", "solution"],
       search_type="hybrid",
       k=10,
       filters={"status": "resolved"},
       score_threshold=0.7
   )

Document Operations
-------------------

The same comprehensive document operations work for both local and remote databases, in both sync and async modes.

Adding Documents
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Single document (sync)
   doc_id = db.upsert("This is my document content")
   
   # Single document (async)
   doc_id = await db.upsert_async("This is my document content")

   # Multiple documents with metadata
   documents = [
       "First document about machine learning...",
       "Second document about data science...",
       "Third document about artificial intelligence..."
   ]
   
   metadata = [
       {"title": "ML Basics", "category": "education", "tags": ["ml", "basics"]},
       {"title": "Data Science", "category": "technical", "tags": ["data", "analysis"]},
       {"title": "AI Overview", "category": "overview", "tags": ["ai", "general"]}
   ]

   # Async version
   doc_ids = await db.upsert_async(
       documents=documents,
       metadata=metadata,
       ids=["ml_basics", "data_sci", "ai_overview"],  # Custom IDs
       batch_size=100,                                # For large batches
       similarity_threshold=0.95                      # Skip near-duplicates
   )

Searching Documents
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Vector search (semantic similarity) - async
   vector_results = await db.query_async(
       "machine learning algorithms",
       search_type="vector",
       k=5,
       score_threshold=0.7,
       return_type="documents"        # Return full documents (default)
   )

   # Keyword search (exact text matching) - async
   keyword_results = await db.query_async(
       "python programming",
       search_type="keyword", 
       k=10,
       return_type="chunks"           # Return matching chunks only
   )

   # Hybrid search (combines vector + keyword) - async
   hybrid_results = await db.query_async(
       "neural networks",
       search_type="hybrid",
       k=5,
       vector_weight=0.7,             # 70% vector, 30% keyword
       return_type="context",         # Return chunks with surrounding context
       context_window=2               # 2 chunks before and after match
   )

   # Search with metadata filters - async
   filtered_results = await db.query_async(
       "artificial intelligence",
       search_type="vector",
       k=10,
       filters={                      # MongoDB-style filters
           "category": "technical",
           "tags": {"$contains": "ai"},
           "created_at": {"$gte": "2024-01-01"}
       }
   )

Managing Documents
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Retrieve documents (async)
   doc = await db.get_async("ml_basics")                    # Single document
   docs = await db.get_async(["ml_basics", "data_sci"])     # Multiple documents
   
   if doc:
       print(f"Title: {doc.metadata.get('title')}")
       print(f"Content: {doc.content[:200]}...")
       print(f"Created: {doc.created_at}")

   # Check existence without retrieving content (async)
   exists = await db.exists_async(["ml_basics", "nonexistent"])  # Returns [True, False]

   # Update documents (async)
   success = await db.update_async("ml_basics", 
                                   content="Updated content about machine learning...",
                                   metadata={"category": "advanced", "last_updated": "2024-12-01"})

   # Delete documents (async)
   deleted_count = await db.delete_async(["old_doc1", "old_doc2"])

   # Filter documents by metadata (async)
   tech_docs = await db.filter_async(
       where={"category": "technical", "tags": {"$contains": "python"}},
       order_by="created_at DESC",
       limit=20,
       offset=0  # Pagination support
   )

Advanced Configuration
----------------------

Embedding Providers
^^^^^^^^^^^^^^^^^^^^

LocalVectorDB supports multiple embedding providers through a plugin architecture:

.. code-block:: python

   # Ollama (local, free)
   db_ollama = VectorDB(
       "my_db", "./data",
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       embedding_config={"base_url": "http://localhost:11434"}
   )

   # OpenAI (cloud, requires API key)
   db_openai = VectorDB(
       "my_db", "./data", 
       embedding_provider="openai",
       embedding_model="text-embedding-3-small",
       embedding_config={"api_key": os.getenv("OPENAI_API_KEY")}
   )

Chunking Strategies
^^^^^^^^^^^^^^^^^^^

Choose chunking methods based on your content type:

.. code-block:: python

   # Sentence-based chunking for general text
   db_sentences = VectorDB("docs", chunking_method="sentences", chunk_size=300)
   
   # Paragraph-based for structured documents
   db_paragraphs = VectorDB("docs", chunking_method="paragraphs", chunk_size=800)
   
   # Token-based for precise control
   db_tokens = VectorDB("docs", chunking_method="tokens", chunk_size=512)
   
   # Code-specific chunking for programming content
   db_code = VectorDB("code", chunking_method="code-blocks", chunk_size=500)

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

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
       chunk_size=400,                # Balance between precision and efficiency
       chunk_overlap=2,               # Overlap for better context

       # Performance settings
       enable_gpu=True,               # Use GPU for FAISS if available
       enable_fts=True,               # Enable full-text search
       connection_pool_size=20,       # More connections for concurrency
       
       # FAISS index optimization
       faiss_index_type="IndexHNSWFlat",  # Faster approximate search
       faiss_index_hnsw_flat_neighbors=32  # Trade memory for speed
   )
   
   # Use async methods for better concurrency in web applications
   async with db:
       await db.upsert_async(large_document_batch, batch_size=100)
       results = await db.query_async("search query", k=10)

Command Line Interface
----------------------

LocalVectorDB includes a powerful CLI for server management and database operations:

Server Management
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Start development server
   lvdb serve --host 0.0.0.0 --port 5000

   # Production server with configuration
   lvdb config init --enable-auth --enable-cors
   lvdb serve --config ./config.toml

   # Manage API keys with permission levels
   lvdb auth create-key --description "My App" --permission-level read_write --expires-days 90
   lvdb auth create-key --description "Analytics" --permission-level read_only --expires-days 365
   lvdb auth list-keys
   lvdb auth revoke-key <key-id>

Database Operations
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # List and create databases
   lvdb list --details
   lvdb create mydatabase --embedding-model nomic-embed-text

   # Add and search documents  
   lvdb db mydatabase add document.txt
   lvdb db mydatabase add "docs/*.md"
   lvdb db mydatabase search "query text" --limit 5 --search-type hybrid

   # Interactive shell
   lvdb db mydatabase shell

Migration and Best Practices
----------------------------

Choosing Sync vs Async
^^^^^^^^^^^^^^^^^^^^^^^

**Use Async when:**
- Building web applications or APIs
- Handling multiple concurrent operations
- Working with remote databases over networks
- Processing large batches of documents

**Use Sync when:**
- Writing simple scripts or data processing pipelines  
- Single-threaded applications
- Quick prototypes and experiments

**Migration Example:**

.. code-block:: python

   # Synchronous code
   def process_documents():
       db = VectorDB("my_db", "./data")
       for doc in documents:
           db.upsert([doc])
       return db.query("search term", k=5)

   # Asynchronous code (better performance)
   async def process_documents_async():
       db = VectorDB("my_db", "./data")
       async with db:
           # Batch operations are more efficient
           await db.upsert_async(documents, batch_size=100)
           return await db.query_async("search term", k=5)

Local to Remote Migration
^^^^^^^^^^^^^^^^^^^^^^^^^

The same code works for both local and remote databases:

.. code-block:: python

   # Development with local database
   db = VectorDB("my_app_db", "./local_data")

   # Production with remote database (just change the base_path!)
   db = VectorDB("my_app_db", "https://vectordb.company.com", 
                api_key=os.getenv("LVDB_API_KEY"))
   
   # Same async methods work for both local and remote
   async with db:
       results = await db.query_async("search query", k=10)

Architecture Overview
---------------------

**Factory Pattern**: The `VectorDB` function automatically selects the appropriate implementation:
- `LocalVectorDB`: File-based storage with SQLite + FAISS
- `RemoteVectorDB`: HTTP client for server communication  
- Async variants available for both with full async/await support

**Storage Architecture**:
- **Documents & Metadata**: SQLite with full ACID transactions
- **Vector Indices**: FAISS for fast similarity search
- **Full-Text Search**: SQLite FTS5 for keyword search  
- **Multi-Column Embeddings**: Additional FAISS indices for metadata fields

**Performance Features**:
- Connection pooling for database operations
- Batch processing for large document sets
- GPU acceleration support for FAISS
- Intelligent caching and similarity-based deduplication

This unified architecture provides a seamless experience whether you're building a local prototype or deploying a production system with remote databases and async operations.