Async Database Operations
=========================

LocalVectorDB provides an async wrapper for high-performance concurrent operations through the ``AsyncLocalVectorDB`` class.
This guide covers how to use async operations effectively for improved performance in async applications.

Overview
--------

The ``AsyncLocalVectorDB`` class provides the same document-focused interface as ``LocalVectorDB`` but with async/await support. It uses intelligent optimization strategies:

- **Async embedding generation**: Direct async calls to HTTP-based embedding providers
- **Thread pool execution**: CPU-bound operations (chunking, FAISS) run in background threads
- **Optimized batching**: Large operations are automatically optimized for performance
- **Resource management**: Proper async context manager support


Basic Usage with Context Manager
--------------------------------

The recommended way to use ``AsyncLocalVectorDB`` is with an async context manager:

.. code-block:: python

   from localvectordb import AsyncLocalVectorDB

   async def main():
       async with AsyncLocalVectorDB("my_database") as db:
           # Insert documents
           doc_ids = await db.upsert([
               "This is a test document",
               "Another document for testing"
           ])

           # Query documents
           results = await db.query("test document", k=5)

           # Print results
           for result in results:
               print(f"{result.id}: {result.score:.3f}")

Factory Function
----------------

For more control over initialization:

.. code-block:: python

   from localvectordb import create_async_vectordb

   async def main():
       db = await create_async_vectordb(
           "my_database",
           embedding_model="nomic-embed-text",
           chunk_size=500,
           max_workers=8  # More workers for CPU-heavy workloads
       )

       try:
           stats = await db.get_stats()
           print(f"Database has {stats['documents']} documents")
       finally:
           await db.close()


Document Management
-------------------

**Adding Documents**

.. code-block:: python

   async with AsyncLocalVectorDB("my_db") as db:
       # Single document
       doc_id = await db.upsert("My document content")

       # Multiple documents with metadata
       doc_ids = await db.upsert(
           documents=["Doc 1", "Doc 2", "Doc 3"],
           metadata=[
               {"author": "Alice", "category": "tech"},
               {"author": "Bob", "category": "science"},
               {"author": "Charlie", "category": "tech"}
           ]
       )

       # Large batch with optimization
       large_docs = [f"Document {i}" for i in range(1000)]
       await db.upsert(large_docs, batch_size=100)

**Retrieving Documents**

.. code-block:: python

   async with AsyncLocalVectorDB("my_db") as db:
       # Single document
       doc = await db.get("doc_1")
       if doc:
           print(f"Content: {doc.content}")
           print(f"Metadata: {doc.metadata}")

       # Multiple documents
       docs = await db.get(["doc_1", "doc_2", "doc_3"])

       # Check existence
       exists = await db.exists(["doc_1", "doc_2"])
       print(f"Documents exist: {exists}")

**Updating and Deleting**

.. code-block:: python

   async with AsyncLocalVectorDB("my_db") as db:
       # Update content and metadata
       updated = await db.update(
           "doc_1",
           content="Updated content",
           metadata={"updated_at": "2024-12-01"}
       )

       # Delete documents
       deleted_count = await db.delete(["doc_1", "doc_2"])
       print(f"Deleted {deleted_count} documents")

Search Operations
-----------------

**Vector Search**

.. code-block:: python

   async with AsyncLocalVectorDB("my_db") as db:
       # Basic vector search
       results = await db.query("machine learning", k=10)

       # With score threshold
       results = await db.query(
           "python programming",
           score_threshold=0.7,
           k=5
       )

       # With metadata filters
       results = await db.query(
           "database optimization",
           filters={"category": "tech", "author": "Alice"},
           k=5
       )

**Keyword and Hybrid Search**

.. code-block:: python

   async with AsyncLocalVectorDB("my_db") as db:
       # Keyword search (requires FTS)
       results = await db.query(
           "machine learning",
           search_type="keyword",
           k=10
       )

       # Hybrid search (combines vector + keyword)
       results = await db.query(
           "machine learning algorithms",
           search_type="hybrid",
           vector_weight=0.7,  # 70% vector, 30% keyword
           k=10
       )

       # Return chunks instead of documents
       chunk_results = await db.query(
           "neural networks",
           return_type="chunks",
           k=20
       )

**Advanced Filtering**

.. code-block:: python

   async with AsyncLocalVectorDB("my_db") as db:
       # Simple filters
       docs = await db.filter(
           where={"author": "Alice"},
           limit=10
       )

       # Complex filters with ordering
       docs = await db.filter(
           where={"category": "tech"},
           order_by="updated_at DESC",
           limit=20,
           offset=10
       )

       # SQL-like filtering
       docs = await db.filter(
           sql="author = 'Alice' AND category IN ('tech', 'science')",
           limit=50
       )


Batch Operations
----------------

For large operations, use appropriate batch sizes:

.. code-block:: python

   async with AsyncLocalVectorDB("my_db", max_workers=8) as db:
       # Large document insertion
       large_document_list = [...]  # 1000+ documents

       # Optimize batch size based on document size
       if avg_doc_size < 1000:
           batch_size = 200
       else:
           batch_size = 50

       doc_ids = await db.upsert(
           large_document_list,
           batch_size=batch_size
       )

Embedding Provider Optimization
-------------------------------

The async wrapper automatically optimizes embedding generation:

.. code-block:: python

   # HTTP-based providers (OpenAI, remote Ollama) use direct async calls
   db = await create_async_vectordb(
       "my_db",
       embedding_provider="openai",
       embedding_model="text-embedding-3-small"
   )

   # Local providers fall back to thread pool execution
   db = await create_async_vectordb(
       "my_db",
       embedding_provider="ollama",
       embedding_model="nomic-embed-text"
   )

Thread Pool Tuning
-------------------

Adjust thread pool size based on your workload:

.. code-block:: python

   # CPU-intensive workloads (large documents, complex chunking)
   db = AsyncLocalVectorDB("my_db", max_workers=16)

   # I/O-intensive workloads (many small operations)
   db = AsyncLocalVectorDB("my_db", max_workers=8)

   # Custom thread pool executor
   import concurrent.futures

   executor = concurrent.futures.ThreadPoolExecutor(
       max_workers=12,
       thread_name_prefix="VectorDB"
   )

   db = AsyncLocalVectorDB("my_db", executor=executor)

Practical Examples
------------------

Document Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from pathlib import Path
   from localvectordb import AsyncLocalVectorDB

   async def process_documents(file_paths, db_name):
       """Process multiple documents concurrently"""

       async with AsyncLocalVectorDB(db_name) as db:
           # Read files concurrently
           async def read_file(path):
               with open(path, 'r') as f:
                   return f.read(), {"source": str(path), "filename": Path(path).name}

           # Process files in batches
           batch_size = 20
           for i in range(0, len(file_paths), batch_size):
               batch_paths = file_paths[i:i + batch_size]

               # Read files concurrently
               tasks = [read_file(path) for path in batch_paths]
               results = await asyncio.gather(*tasks)

               # Extract documents and metadata
               documents = [content for content, _ in results]
               metadata = [meta for _, meta in results]

               # Upsert to database
               doc_ids = await db.upsert(documents, metadata=metadata)
               print(f"Processed batch {i//batch_size + 1}: {len(doc_ids)} documents")

Search Service
~~~~~~~~~~~~~~

.. code-block:: python

   from fastapi import FastAPI
   from localvectordb import AsyncLocalVectorDB

   app = FastAPI()

   # Initialize database on startup
   @app.on_event("startup")
   async def startup():
       app.db = await create_async_vectordb("search_index")

   @app.on_event("shutdown")
   async def shutdown():
       await app.db.close()

   @app.post("/search")
   async def search(query: str, limit: int = 10):
       results = await app.db.query(query, k=limit)
       return {
           "results": [
               {
                   "id": r.id,
                   "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                   "score": r.score,
                   "metadata": r.metadata
               }
               for r in results
           ]
       }

   @app.post("/add-document")
   async def add_document(content: str, metadata: dict = None):
       doc_ids = await app.db.upsert([content], metadata=[metadata or {}])
       return {"doc_id": doc_ids[0]}

Concurrent Query Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def concurrent_search_example():
       """Demonstrate concurrent query processing"""

       async with AsyncLocalVectorDB("my_db") as db:
           queries = [
               "machine learning algorithms",
               "database optimization techniques",
               "web development frameworks",
               "data visualization methods",
               "cloud computing platforms"
           ]

           # Process queries concurrently
           tasks = [
               db.query(query, k=5)
               for query in queries
           ]

           results = await asyncio.gather(*tasks)

           # Process results
           for query, query_results in zip(queries, results):
               print(f"\nQuery: {query}")
               for result in query_results:
                   print(f"  {result.id}: {result.score:.3f}")


Proper Error Handling
----------------------

.. code-block:: python

   from localvectordb.exceptions import DatabaseError, DatabaseNotFoundError

   async def robust_database_operations():
       try:
           async with AsyncLocalVectorDB("my_db") as db:
               # Database operations
               results = await db.query("test query")

       except DatabaseNotFoundError:
           print("Database not found, creating new one...")
           async with AsyncLocalVectorDB("my_db", create_if_not_exists=True) as db:
               # Initialize with some data
               await db.upsert(["Initial document"])

       except DatabaseError as e:
           print(f"Database error: {e}")
           # Handle database-specific errors

       except Exception as e:
           print(f"Unexpected error: {e}")
           # Handle other errors

Resource Management
-------------------

.. code-block:: python

   async def resource_management_example():
       """Proper resource management patterns"""

       # Pattern 1: Context manager (recommended)
       async with AsyncLocalVectorDB("my_db") as db:
           await db.upsert(["Document content"])
           # Database automatically closed

       # Pattern 2: Manual management
       db = await create_async_vectordb("my_db")
       try:
           await db.upsert(["Document content"])
       finally:
           await db.close()  # Always close manually created instances

       # Pattern 3: Custom executor
       from concurrent.futures import ThreadPoolExecutor

       with ThreadPoolExecutor(max_workers=4) as executor:
           db = AsyncLocalVectorDB("my_db", executor=executor)
           async with db:
               await db.upsert(["Document content"])
           # Executor managed externally

Performance Monitoring
----------------------

.. code-block:: python

   import time

   async def performance_monitoring():
       """Monitor async database performance"""

       async with AsyncLocalVectorDB("my_db") as db:
           # Time operations
           start_time = time.time()

           results = await db.query("test query", k=10)

           query_time = time.time() - start_time
           print(f"Query completed in {query_time:.3f}s")

           # Get database statistics
           stats = await db.get_stats()
           print(f"Database contains {stats['documents']} documents")
           print(f"Index has {stats['index_vectors']} vectors")


Converting Sync Code
---------------------

Here's how to migrate from synchronous to asynchronous code:

**Before (Sync)**

.. code-block:: python

   from localvectordb import LocalVectorDB

   def sync_example():
       with LocalVectorDB("my_db") as db:
           doc_ids = db.upsert(["Document 1", "Document 2"])
           results = db.query("search query", k=5)
           return results

**After (Async)**

.. code-block:: python

   from localvectordb import AsyncLocalVectorDB

   async def async_example():
       async with AsyncLocalVectorDB("my_db") as db:
           doc_ids = await db.upsert(["Document 1", "Document 2"])
           results = await db.query("search query", k=5)
           return results

Common Migration Patterns
-------------------------

.. code-block:: python

   # Sync pattern
   db = LocalVectorDB("my_db")
   try:
       results = db.query("query")
   finally:
       db.close()

   # Async equivalent
   db = await create_async_vectordb("my_db")
   try:
       results = await db.query("query")
   finally:
       await db.close()

   # Or better with context manager
   async with AsyncLocalVectorDB("my_db") as db:
       results = await db.query("query")


FastAPI Integration
-------------------

.. code-block:: python

   from fastapi import FastAPI, BackgroundTasks
   from localvectordb import AsyncLocalVectorDB

   app = FastAPI()

   class DatabaseManager:
       def __init__(self):
           self.db = None

       async def initialize(self):
           self.db = await create_async_vectordb("api_db")

       async def close(self):
           if self.db:
               await self.db.close()

   db_manager = DatabaseManager()

   @app.on_event("startup")
   async def startup():
       await db_manager.initialize()

   @app.on_event("shutdown")
   async def shutdown():
       await db_manager.close()

   @app.get("/search/{query}")
   async def search_endpoint(query: str, limit: int = 10):
       results = await db_manager.db.query(query, k=limit)
       return {"results": [{"id": r.id, "score": r.score} for r in results]}

Aiohttp Integration
-------------------

.. code-block:: python

   from aiohttp import web
   from localvectordb import AsyncLocalVectorDB

   async def init_db(app):
       app['db'] = await create_async_vectordb("web_db")

   async def cleanup_db(app):
       await app['db'].close()

   async def search_handler(request):
       db = request.app['db']
       query = request.query.get('q', '')
       limit = int(request.query.get('limit', 10))

       results = await db.query(query, k=limit)

       return web.json_response({
           'results': [{'id': r.id, 'score': r.score} for r in results]
       })

   app = web.Application()
   app.on_startup.append(init_db)
   app.on_cleanup.append(cleanup_db)
   app.router.add_get('/search', search_handler)

This guide covers the essential patterns for using ``AsyncLocalVectorDB`` effectively.
The async interface provides significant performance benefits for concurrent applications while maintaining the same intuitive API as the synchronous version.