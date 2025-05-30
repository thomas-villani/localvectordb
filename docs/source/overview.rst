Overview
========

LocalVectorDB provides a document-first API that abstracts away the complexity of chunking while providing powerful
search and metadata capabilities. The library supports both local databases and remote connections to LocalVectorDB servers.

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


Remote Database Connection
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Connect to remote server
   db = VectorDB(
       name="my_database",
       base_path="http://localhost:5000",
       api_key="your_api_key",
       create_if_not_exists=True,
       metadata_schema=schema
   )

.. note::

    The ``VectorDB`` class is a factory that returns a ``localvectordb.database.LocalVectorDB`` object if loading
    a database from the local filesystem (i.e. ``base_path`` is a local file), and a
    ``localvectordb.client.RemoteVectorDB`` object if loading from remote (i.e. ``base_path`` is a url). This
    allows for using the same codebase for local testing and remote for production, just by swapping a single variable.

Adding Documents
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Single document
   doc_id = db.upsert("This is my document content")

   # Multiple documents
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

   doc_ids = db.upsert(
       documents=documents,
       metadata=metadata,
       ids=["doc_1", "doc_2", "doc_3"]  # Optional custom IDs
   )

Searching Documents
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Vector search (semantic similarity)
   results = db.query(
       "machine learning algorithms",
       search_type="vector",
       k=5,
       score_threshold=0.7
   )

   # Keyword search (exact matches)
   results = db.query(
       "python programming",
       search_type="keyword",
       k=10
   )

   # Hybrid search (best of both)
   results = db.query(
       "neural networks",
       search_type="hybrid",
       k=5,
       vector_weight=0.7  # 70% vector, 30% keyword
   )

   # Search with metadata filters
   results = db.query(
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

Retrieving Documents
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get single document
   doc = db.get("doc_1")
   if doc:
       print(f"Content: {doc.content}")
       print(f"Metadata: {doc.metadata}")
       print(f"Created: {doc.created_at}")

   # Get multiple documents
   docs = db.get(["doc_1", "doc_2", "doc_3"])

   # Check if documents exist
   exists = db.exists(["doc_1", "doc_2"])  # Returns [True, False]

Updating Documents
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Update content only
   success = db.update("doc_1", content="Updated document content")

   # Update metadata only
   success = db.update("doc_1", metadata={"priority": 5})

   # Update both
   success = db.update(
       "doc_1",
       content="New content",
       metadata={"category": "updated", "priority": 10}
   )

Filtering and Querying Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Simple filters
   docs = db.filter(where={"category": "tech"})

   # Complex filters
   docs = db.filter(where={
       "priority": {">=": 3},
       "category": {"in": ["tech", "business"]},
       "publish_date": {"between": ["2024-01-01", "2024-12-31"]}
   })

   # SQL-like queries
   docs = db.filter(
       sql="priority > 3 AND category LIKE '%tech%'",
       order_by="priority DESC",
       limit=10
   )

   # Pagination
   docs = db.filter(
       where={"category": "tech"},
       order_by="created_at DESC",
       limit=20,
       offset=40  # Page 3 of 20 items per page
   )

Advanced Usage
--------------

Custom Chunking Strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Different chunking methods
   db_sentences = VectorDB("docs", chunking_method="sentences", chunk_size=300)
   db_paragraphs = VectorDB("docs", chunking_method="paragraphs", chunk_size=800)
   db_sections = VectorDB("docs", chunking_method="sections", chunk_size=1000)

   # Code-specific chunking
   db_code = VectorDB("code", chunking_method="code-blocks", chunk_size=500)

Multiple Embedding Providers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
       embedding_config={"api_key": "your_key"}
   )

Batch Operations
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Large document insertion with batching
   large_docs = ["doc content " + str(i) for i in range(10000)]
   doc_ids = db.upsert(documents=large_docs, batch_size=100)

   # Similarity threshold to avoid duplicates
   doc_ids = db.insert(
       documents=new_docs,
       similarity_threshold=0.95,  # Skip if 95%+ similar to existing
       errors="ignore"  # Don't fail on duplicates
   )

Working with Chunks
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Search returning chunks instead of documents
   results = db.query(
       "specific technical detail",
       search_type="vector",
       return_type="chunks",  # Get individual chunks
       k=10
   )

   for result in results:
       print(f"Chunk from document: {result.document_id}")
       print(f"Position: line {result.position.line}")
       print(f"Content: {result.content}")

Database Statistics and Management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get database statistics
   stats = db.stats
   print(f"Documents: {stats['documents']}")
   print(f"Chunks: {stats['chunks']}")
   print(f"Embedding model: {stats['embedding_model']}")
   print(f"Index vectors: {stats['index_vectors']}")

   # Save database (for local databases)
   db.save()

   # Close database
   db.close()

   # Context manager (automatic cleanup)
   with VectorDB("temp_db", ":memory:") as db:
       db.upsert(["temporary document"])
       results = db.query("search")
   # Database automatically closed

Database Configuration
----------------------

Performance Tuning
^^^^^^^^^^^^^^^^^^

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
       connection_pool_size=20   # More connections for concurrent access
   )

Memory-Only Database
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # In-memory database for testing/temporary use
   temp_db = VectorDB("temp", ":memory:")

Backup and Migration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Create backup-friendly configuration
   backup_db = VectorDB(
       "production_db",
       "./backups/vector_data",

       # Consistent settings for migration
       embedding_provider="ollama",
       embedding_model="nomic-embed-text",
       chunk_size=500,
       chunking_method="sentences"
   )

   # Export all documents for backup
   all_docs = backup_db.filter()  # Get all documents
   backup_data = [
       {
           "id": doc.id,
           "content": doc.content,
           "metadata": doc.metadata,
           "created_at": doc.created_at.isoformat() if doc.created_at else None
       }
       for doc in all_docs
   ]

   # Save backup
   import json
   with open("database_backup.json", "w") as f:
       json.dump(backup_data, f, indent=2)

Error Handling
--------------

.. code-block:: python

   from localvectordb.exceptions import (
       DatabaseNotFoundError,
       DuplicateDocumentIDError,
       EmbeddingError
   )

   try:
       db = VectorDB("nonexistent", create_if_not_exists=False)
   except DatabaseNotFoundError:
       print("Database does not exist")

   try:
       db.insert(["content"], ids=["existing_id"])
   except DuplicateDocumentIDError:
       print("Document ID already exists")

   try:
       results = db.query("search query")
   except EmbeddingError as e:
       print(f"Embedding generation failed: {e}")
