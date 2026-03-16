Streaming Query Results
=======================

LocalVectorDB supports cursor-based streaming for efficient, memory-friendly iteration over large result sets. Instead
of loading all results into memory at once, a ``QueryCursor`` performs the expensive FAISS/FTS search once, caches
lightweight candidate IDs and scores, and lazily loads content and metadata from SQLite in batches as you iterate.

Why Streaming?
--------------

The standard ``query()`` method returns all results at once as a ``List[QueryResult]``. For small result sets this is
fine, but for large-scale retrieval it has two drawbacks:

1. **Memory**: All result content and metadata must fit in memory simultaneously.
2. **Latency**: The caller must wait for all results to be fully hydrated before processing the first one.

The cursor-based streaming API solves both problems by separating the search phase (fast, returns only IDs and scores)
from the hydration phase (loads content per batch, on demand).

Quick Start
-----------

Using ``query_stream``
^^^^^^^^^^^^^^^^^^^^^^

The simplest way to stream results:

.. code-block:: python

   # Stream results in batches of 10
   for batch in db.query_stream(
       "machine learning",
       search_type="hybrid",
       return_type="chunks",
       k=100,
       batch_size=10,
   ):
       for result in batch:
           process(result)

Each ``batch`` is a ``List[QueryResult]``. FAISS and FTS are queried once; each iteration loads only the next batch of
content from SQLite.

Using ``QueryCursor`` Directly
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more control over the iteration lifecycle:

.. code-block:: python

   # Create a cursor
   cursor = db.query_cursor(
       "machine learning",
       search_type="vector",
       return_type="documents",
       k=50,
       batch_size=10,
   )

   # Use as a context manager for automatic cleanup
   with cursor:
       print(f"Total candidates: {cursor.total_candidates}")
       print(f"Remaining: {cursor.remaining}")

       # Fetch one batch at a time
       first_batch = cursor.fetch_batch()

       # Or iterate all remaining batches
       for batch in cursor.stream():
           for result in batch:
               print(result.id, result.score)

   # Cursor is automatically closed after the with block

Using the QueryBuilder
^^^^^^^^^^^^^^^^^^^^^^

The ``QueryBuilder`` also supports cursor-based streaming:

.. code-block:: python

   # Create a cursor from a query builder chain
   cursor = (
       db.query_builder()
       .search("neural networks")
       .filter("year", gte_=2023)
       .limit(50)
       .cursor(batch_size=10)
   )

   with cursor:
       results = cursor.fetch_all()

   # Or use the stream shorthand
   for batch in (
       db.query_builder()
       .hybrid("deep learning", vector_weight=0.7)
       .limit(100)
       .stream(batch_size=20)
   ):
       process_batch(batch)

Async Streaming
---------------

All streaming APIs have async counterparts that work with ``async for`` and naturally support backpressure (the next
batch is not fetched until the consumer is ready):

.. code-block:: python

   # Async stream via convenience method
   async for batch in db.query_stream_async(
       "machine learning",
       search_type="vector",
       return_type="chunks",
       k=100,
       batch_size=10,
   ):
       await process_batch(batch)

   # Async cursor with explicit lifecycle
   cursor = await db.query_cursor_async(
       "deep learning",
       search_type="hybrid",
       return_type="documents",
       k=50,
   )

   async with cursor:
       # Fetch individual results one at a time
       async for result in cursor.stream_individual_async():
           await handle(result)

   # Async QueryBuilder
   async for batch in (
       db.query_builder()
       .search("transformers")
       .limit(100)
       .stream_async(batch_size=25)
   ):
       await process_batch(batch)

QueryCursor API
---------------

Creation
^^^^^^^^

Cursors are created via ``query_cursor()`` / ``query_cursor_async()`` on the database, or ``cursor()`` /
``cursor_async()`` on the QueryBuilder. Parameters match the standard ``query()`` method, with two additions:

* ``batch_size`` (int, default=50): Default number of results per batch.
* ``cursor_ttl`` (float, default=300.0): Time-to-live in seconds. The cursor expires after this duration of
  inactivity, raising ``CursorExpiredError`` on subsequent access.

Fetching Results
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Description
   * - ``fetch_batch(batch_size=None)``
     - Fetch the next batch (sync). Returns ``[]`` when exhausted.
   * - ``fetch_all()``
     - Fetch all remaining results at once (sync).
   * - ``stream(batch_size=None)``
     - Generator yielding ``List[QueryResult]`` batches (sync).
   * - ``stream_individual(batch_size=None)``
     - Generator yielding individual ``QueryResult`` objects (sync).
   * - ``fetch_batch_async(batch_size=None)``
     - Async version of ``fetch_batch``.
   * - ``fetch_all_async()``
     - Async version of ``fetch_all``.
   * - ``stream_async(batch_size=None)``
     - Async generator yielding batches.
   * - ``stream_individual_async(batch_size=None)``
     - Async generator yielding individual results.

Lifecycle and Properties
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Property / Method
     - Description
   * - ``total_candidates``
     - Total number of scored candidates found by the search.
   * - ``remaining``
     - Number of unfetched candidates.
   * - ``is_exhausted``
     - ``True`` when all candidates have been fetched.
   * - ``closed``
     - ``True`` when the cursor has been closed.
   * - ``close()``
     - Explicitly close the cursor and release the database reference.

The cursor supports both sync (``with``) and async (``async with``) context managers for automatic cleanup.

Supported Search and Return Types
---------------------------------

The cursor supports all combinations of search types and return types:

**Search types:** ``vector``, ``keyword``, ``hybrid``

**Return types:**

* ``chunks`` -- Individual chunks, lazily hydrated per batch.
* ``documents`` -- Document scores are pre-aggregated eagerly from chunk scores during cursor creation; document
  content is loaded lazily per batch.
* ``context`` -- Chunks with surrounding context, assembled per batch.
* ``enriched`` -- Chunks with intra-document semantic enrichment, computed per batch.

How It Works
------------

The streaming architecture separates the query into two phases:

1. **Search phase** (runs once during cursor creation):

   - Embeds the query text (for vector/hybrid search).
   - Runs FAISS search and/or FTS5 search to collect candidate IDs and scores.
   - Applies semantic deduplication if configured (requires FAISS embeddings).
   - For hybrid search, merges vector and keyword candidates using a lightweight ID lookup.
   - For ``return_type="documents"``, pre-aggregates chunk scores into document scores.
   - Releases all FAISS locks before returning the cursor.

2. **Hydration phase** (runs per batch during iteration):

   - Loads chunk/document content and metadata from SQLite for only the current batch.
   - Applies metadata filters per batch.
   - Applies context window or enrichment post-processing per batch.

This means FAISS is only queried once regardless of how many batches you consume, and SQLite I/O is proportional to
the batch size rather than the total result set.

Performance Considerations
--------------------------

* **Batch size tuning**: Smaller batches reduce peak memory but increase the number of SQLite round-trips. A batch
  size of 20--100 is typical. Start with the default (50) and adjust based on your content size.

* **TTL management**: The default 5-minute TTL prevents resource leaks from abandoned cursors. For long-running
  pipelines, increase ``cursor_ttl``.

* **Metadata filters with cursors**: When metadata filters have a high rejection rate, some batches may return
  fewer results than ``batch_size``. The cursor automatically fetches additional candidates to fill the batch.

* **Reranking limitation**: Cross-encoder reranking requires the full text of all candidates to re-score them.
  When reranking is configured, use the standard ``query()`` method instead of cursors. This will be addressed in a
  future release.

Exceptions
----------

* ``CursorExpiredError`` -- Raised when accessing a cursor that has been closed or whose TTL has elapsed.
* ``CursorExhaustedError`` -- Available for application-level use when detecting exhaustion (the cursor itself
  returns an empty list rather than raising).

See Also
--------

* :doc:`query` -- Standard query types and return modes
* :doc:`querybuilder` -- QueryBuilder fluent API
* :doc:`document-scoring` -- Document scoring methods used with ``return_type="documents"``
