LocalVectorDBClient
===================

``LocalVectorDBClient`` is the top-level entry point for the SDK. It manages the server connection
and provides methods for database management and cross-database operations.

.. contents:: Table of Contents
   :local:
   :depth: 2

Creating a Client
-----------------

.. code-block:: typescript

   import { LocalVectorDBClient } from "@localvectordb/sdk";

   const client = new LocalVectorDBClient({
     baseUrl: "http://localhost:5000",
     apiKey: "lvdb_your_key",
   });

The constructor is synchronous — no network calls are made until you invoke a method.

Database Handles
----------------

Use ``database()`` to get a lightweight handle for a specific database:

.. code-block:: typescript

   const db = client.database("my_database");

This is synchronous and makes no server call. If the database does not exist, the first API call
against the handle will throw a ``DatabaseNotFoundError``.

You can hold multiple handles simultaneously:

.. code-block:: typescript

   const docs = client.database("documents");
   const logs = client.database("logs");

   await docs.upsert(["Important document"]);
   await logs.upsert(["User performed search"]);

Database Management
-------------------

Create Database
~~~~~~~~~~~~~~~

.. code-block:: typescript

   await client.createDatabase("products", {
     embedding: {
       provider: "ollama",
       model: "nomic-embed-text",
     },
     database: {
       chunk_size: 512,
       chunking_method: "sentences",
       chunk_overlap: 1,
       enable_fts: true,
     },
     metadata_schema: {
       category: { type: "text", indexed: true },
       price: { type: "real", indexed: true },
       in_stock: { type: "boolean" },
       tags: { type: "json" },
     },
   });

List Databases
~~~~~~~~~~~~~~

.. code-block:: typescript

   const { databases, count } = await client.listDatabases();
   console.log(`${count} databases: ${databases.join(", ")}`);

Delete Database
~~~~~~~~~~~~~~~

.. code-block:: typescript

   await client.deleteDatabase("old_database");

Health & System Info
--------------------

.. code-block:: typescript

   // Server health check
   const health = await client.health();
   console.log(health.status, health.version, health.ollama_available);

   // System resource information
   const resources = await client.systemResources();

Cross-Database Operations
-------------------------

Global Search
~~~~~~~~~~~~~

Search across multiple databases simultaneously:

.. code-block:: typescript

   const results = await client.globalSearch("machine learning", {
     databases: ["papers", "articles"],
     search_type: "hybrid",
     k: 10,
   });

   // results.results is keyed by database name
   for (const [dbName, hits] of Object.entries(results.results)) {
     console.log(`--- ${dbName} ---`);
     for (const hit of hits) {
       console.log(`  ${hit.id}: ${hit.score.toFixed(3)}`);
     }
   }

Generate Embeddings
~~~~~~~~~~~~~~~~~~~

Generate embeddings without inserting documents:

.. code-block:: typescript

   const { embeddings } = await client.embeddings(
     ["text to embed", "another text"],
     "ollama",
     "nomic-embed-text"
   );
   // embeddings: number[][] (one vector per input text)

Global Fact-Check
~~~~~~~~~~~~~~~~~

Check a claim against multiple databases:

.. code-block:: typescript

   const result = await client.factCheck(
     "The speed of light is approximately 300,000 km/s",
     {
       databases: ["physics", "reference"],
       llm_provider: "anthropic",
     }
   );

ClientConfig Reference
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Property
     - Type
     - Default
     - Description
   * - ``baseUrl``
     - ``string``
     - *(required)*
     - Server URL (e.g. ``http://localhost:5000``)
   * - ``apiKey``
     - ``string``
     - ``undefined``
     - Bearer token for authentication
   * - ``timeout``
     - ``number``
     - ``30000``
     - Request timeout in milliseconds
   * - ``maxRetries``
     - ``number``
     - ``3``
     - Retry count for 5xx and network errors
   * - ``retryDelay``
     - ``number``
     - ``1000``
     - Base retry delay in ms (doubles each attempt)
