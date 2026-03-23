Quick Start
===========

.. contents:: Table of Contents
   :local:
   :depth: 2

Installation
------------

Install with npm (or your preferred package manager):

.. code-block:: bash

   npm install @localvectordb/sdk

.. code-block:: bash

   # or with yarn / pnpm
   yarn add @localvectordb/sdk
   pnpm add @localvectordb/sdk

Requirements
~~~~~~~~~~~~

- **Node.js 18+** (for built-in ``fetch`` and ``FormData``)
- **TypeScript 5.4+** (if using TypeScript)
- A running LocalVectorDB server (see :doc:`/server/quickstart`)

Basic Usage
-----------

Create a client, connect to a database, add documents, and search:

.. code-block:: typescript

   import { LocalVectorDBClient } from "@localvectordb/sdk";

   // 1. Create a client pointing at your server
   const client = new LocalVectorDBClient({
     baseUrl: "http://localhost:5000",
     apiKey: "lvdb_your_api_key",  // optional
   });

   // 2. Create a database
   await client.createDatabase("my_docs", {
     embedding: { provider: "ollama", model: "nomic-embed-text" },
     database: { chunk_size: 512, chunking_method: "sentences" },
   });

   // 3. Get a handle for the database
   const db = client.database("my_docs");

   // 4. Add documents
   await db.upsert(
     ["Introduction to machine learning", "Advanced neural networks"],
     {
       metadata: [
         { author: "Alice", topic: "ml" },
         { author: "Bob", topic: "deep-learning" },
       ],
     }
   );

   // 5. Search
   const results = await db.query("neural network fundamentals", {
     search_type: "hybrid",
     k: 5,
   });

   for (const r of results.results) {
     console.log(`${r.id} (score: ${r.score.toFixed(3)}): ${r.content.slice(0, 80)}`);
   }

Configuration
-------------

The client constructor accepts the following options:

.. code-block:: typescript

   const client = new LocalVectorDBClient({
     baseUrl: "http://localhost:5000",  // Server URL (required)
     apiKey: "lvdb_...",                // Bearer token (optional)
     timeout: 30000,                    // Request timeout in ms (default: 30 000)
     maxRetries: 3,                     // Retries for 5xx / network errors (default: 3)
     retryDelay: 1000,                  // Base delay in ms, doubles each retry (default: 1 000)
   });

The API key is sent as ``Authorization: Bearer <apiKey>`` on every request. If your server
does not require authentication, simply omit the ``apiKey`` field.

Python Comparison
-----------------

If you are already familiar with the Python ``RemoteVectorDB`` client, the JS/TS SDK will feel
natural. The two-level API (client + database handle) maps directly:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Python (RemoteVectorDB)
     - TypeScript (@localvectordb/sdk)
   * - .. code-block:: python

          db = RemoteVectorDB(
              name="mydb",
              base_url="http://localhost:5000",
              api_key="lvdb_..."
          )

     - .. code-block:: typescript

          const client = new LocalVectorDBClient({
            baseUrl: "http://localhost:5000",
            apiKey: "lvdb_...",
          });
          const db = client.database("mydb");

   * - .. code-block:: python

          db.upsert(["doc1", "doc2"])

     - .. code-block:: typescript

          await db.upsert(["doc1", "doc2"]);

   * - .. code-block:: python

          results = db.query("search",
              search_type="hybrid", k=5)

     - .. code-block:: typescript

          const results = await db.query("search", {
            search_type: "hybrid", k: 5,
          });

.. seealso::

   :doc:`/client` — Python RemoteVectorDB client documentation
