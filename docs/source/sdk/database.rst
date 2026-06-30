DatabaseHandle
==============

``DatabaseHandle`` provides all operations scoped to a single database. Obtain an instance via
:doc:`client`:

.. code-block:: typescript

   const db = client.database("my_database");

.. contents:: Table of Contents
   :local:
   :depth: 2

Database Info
-------------

.. code-block:: typescript

   const info = await db.info();
   console.log(info.name, info.stats, info.config);

Document Operations
-------------------

Upsert Documents
~~~~~~~~~~~~~~~~

Insert or update documents. If a document with the same ID exists, it is replaced.

.. code-block:: typescript

   // Single document
   const { ids } = await db.upsert("Hello world");

   // Multiple documents with metadata
   const result = await db.upsert(
     ["Document one", "Document two"],
     {
       metadata: [
         { author: "Alice", year: 2024 },
         { author: "Bob", year: 2023 },
       ],
       ids: ["doc-1", "doc-2"],
       batch_size: 100,
     }
   );

Insert Documents
~~~~~~~~~~~~~~~~

Like upsert, but fails if a document with the same ID already exists:

.. code-block:: typescript

   // Strict insert — throws DuplicateDocumentError on conflict
   await db.insert(["New document"], { ids: ["unique-id"] });

   // Ignore duplicates instead of throwing
   await db.insert(["Maybe new"], { ids: ["maybe-exists"], errors: "ignore" });

Get Documents
~~~~~~~~~~~~~

.. code-block:: typescript

   // Single document — returns Document
   const doc = await db.get("doc-1");
   console.log(doc.id, doc.content, doc.metadata);

   // Multiple documents — returns Document[]
   const docs = await db.get(["doc-1", "doc-2"]);

Update a Document
~~~~~~~~~~~~~~~~~

.. code-block:: typescript

   await db.update("doc-1", {
     content: "Updated content",
     metadata: { author: "Alice Smith", revised: true },
   });

Delete Documents
~~~~~~~~~~~~~~~~

.. code-block:: typescript

   // Single document — uses HTTP DELETE
   await db.delete("doc-1");

   // Batch delete — uses POST
   const { deleted_count, failed_ids } = await db.delete(["doc-1", "doc-2", "doc-3"]);

Count Documents
~~~~~~~~~~~~~~~

.. code-block:: typescript

   // Count all documents
   const total = await db.count();

   // Count with metadata filter
   const filtered = await db.count({ filters: { author: "Alice" } });

Check Existence
~~~~~~~~~~~~~~~

.. code-block:: typescript

   const { exists } = await db.exists(["doc-1", "doc-2"]);
   // exists: { "doc-1": true, "doc-2": false }

List Documents
~~~~~~~~~~~~~~

.. code-block:: typescript

   const page = await db.list({ page: 1, limit: 20 });
   console.log(page.documents.length, page.pagination);

Chunked Documents
~~~~~~~~~~~~~~~~~

If you have already chunked your documents externally, you can insert them directly:

.. code-block:: typescript

   await db.upsertChunks({
     "doc-1": ["chunk 1 of doc 1", "chunk 2 of doc 1"],
     "doc-2": ["chunk 1 of doc 2"],
   });

   await db.insertChunks(
     { "doc-3": ["chunk a", "chunk b"] },
     { errors: "ignore" }
   );

Search Operations
-----------------

Unified Query
~~~~~~~~~~~~~

The ``query()`` method supports vector, keyword, and hybrid search:

.. code-block:: typescript

   // Vector search (default)
   const results = await db.query("semantic meaning");

   // Keyword search
   const kw = await db.query("exact terms", { search_type: "keyword" });

   // Hybrid search with tuned weighting
   const hybrid = await db.query("machine learning", {
     search_type: "hybrid",
     vector_weight: 0.7,      // 70% vector, 30% keyword
     k: 10,
     score_threshold: 0.3,
   });

   // With metadata filters
   const filtered = await db.query("neural networks", {
     search_type: "hybrid",
     filters: { author: "Alice", year: { "$gte": 2023 } },
     return_type: "chunks",
   });

Multi-Column Query
~~~~~~~~~~~~~~~~~~

Search across specific metadata columns that have embeddings enabled:

.. code-block:: typescript

   const results = await db.queryMultiColumn("python data science", {
     columns: ["content", "summary"],
     search_type: "hybrid",
   });

Filter by Metadata
~~~~~~~~~~~~~~~~~~

MongoDB-style metadata filtering without a search query:

.. code-block:: typescript

   // Simple equality
   const docs = await db.filter({ author: "Alice" });

   // Advanced operators
   const recent = await db.filter(
     {
       year: { "$gte": 2023 },
       category: { "$in": ["ml", "ai"] },
     },
     {
       order_by: "year DESC",
       limit: 10,
       offset: 0,
     }
   );

Streaming
~~~~~~~~~

See :doc:`streaming` for the full streaming guide. Quick example:

.. code-block:: typescript

   for await (const result of db.queryStream("search text", { k: 50 })) {
     console.log(result.id, result.score);
   }

Schema Management
-----------------

.. code-block:: typescript

   // Get current schema
   const schema = await db.getSchema();

   // Add or modify fields
   await db.updateSchema(
     {
       category: { type: "text", indexed: true },
       rating: { type: "real", indexed: true },
     },
     { drop_columns: false }
   );

Embeddings
----------

Retrieve embeddings for existing chunk IDs or generate them for custom text:

.. code-block:: typescript

   // By chunk IDs
   const byId = await db.getEmbeddings({ ids: ["chunk-id-1", "chunk-id-2"] });

   // By text
   const byText = await db.getEmbeddings({ texts: ["custom text to embed"] });

   // byId.embeddings / byText.embeddings: number[][]

Document Comparison
-------------------

Compare Documents
~~~~~~~~~~~~~~~~~

.. code-block:: typescript

   // Quick similarity score
   const { similarity } = await db.compare("doc-1", "doc-2");
   console.log(`Similarity: ${(similarity * 100).toFixed(1)}%`);

   // Detailed chunk-level comparison
   const detailed = await db.compareDetailed("doc-1", "doc-2", {
     chunk_threshold: 0.7,
   });

Nearest Neighbors
~~~~~~~~~~~~~~~~~

.. code-block:: typescript

   const { results } = await db.nearestNeighbors("doc-1", 5);
   for (const r of results) {
     console.log(`${r.id}: ${r.score.toFixed(3)}`);
   }

Similarity Matrix
~~~~~~~~~~~~~~~~~

.. code-block:: typescript

   // All documents
   const matrix = await db.similarityMatrix();

   // Specific subset
   const subset = await db.similarityMatrix(["doc-1", "doc-2", "doc-3"]);
   // subset.matrix: number[][] (3x3 pairwise similarities)

Tuning & Maintenance
--------------------

.. code-block:: typescript

   // Get current SQLite tuning
   const tuning = await db.getTuning();

   // Apply a tuning profile
   // (valid profiles: balanced, fast_ingest, read_optimized, durable, memory_saver)
   await db.setTuning("fast_ingest", { persist: true });

   // Maintenance operations
   await db.checkpoint("PASSIVE");
   await db.optimize();
   await db.vacuum();
   await db.incrementalVacuum(2000);

   // Checkpoint only if WAL is large
   await db.checkpointIfWalLarge(128);  // threshold in MB

   // Auto-tuning recommendations
   const recs = await db.autoTune({ workload: "read_heavy" });

   // Apply recommendations automatically
   await db.autoTune({ workload: "read_heavy", apply: true });

Fact-Checking
-------------

Check a text claim against documents in the database:

.. code-block:: typescript

   const result = await db.factCheck("The earth orbits the sun", {
     llm_provider: "anthropic",
     search_type: "hybrid",
     top_k: 10,
   });

QueryOptions Reference
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 15 40

   * - Property
     - Type
     - Default
     - Description
   * - ``search_type``
     - ``"vector" | "keyword" | "hybrid"``
     - ``"vector"``
     - Search strategy
   * - ``return_type``
     - ``"documents" | "chunks" | "sections" | "context" | "enriched"``
     - ``"documents"``
     - Result granularity
   * - ``k``
     - ``number``
     - ``10``
     - Maximum results to return
   * - ``score_threshold``
     - ``number``
     - ``0.0``
     - Minimum score (0–1)
   * - ``filters``
     - ``object``
     -
     - MongoDB-style metadata filters
   * - ``vector_weight``
     - ``number``
     - ``0.7``
     - Weight of vector score in hybrid search (0–1)
   * - ``context_window``
     - ``number``
     - ``2``
     - Surrounding chunks to include in context mode
   * - ``document_scoring_method``
     - ``DocumentScoringMethod``
     - ``"frequency_boost"``
     - How to aggregate chunk scores to document scores
