Streaming Queries
=================

The SDK supports streaming query results via Server-Sent Events (SSE). Instead of waiting for all
results to arrive, you can process them as they stream in.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Usage
-----------

``queryStream()`` returns an ``AsyncGenerator<QueryResult>`` — consume it with ``for await``:

.. code-block:: typescript

   const db = client.database("my_docs");

   for await (const result of db.queryStream("search text")) {
     console.log(result.id, result.score, result.content.slice(0, 80));
   }

The loop ends automatically when the server signals completion.

With Options
------------

``queryStream()`` accepts the same options as ``query()``:

.. code-block:: typescript

   for await (const result of db.queryStream("machine learning", {
     search_type: "hybrid",
     k: 50,
     score_threshold: 0.3,
     filters: { year: { "$gte": 2023 } },
   })) {
     process(result);
   }

Cancellation
------------

Break out of the loop to cancel the stream. The underlying HTTP connection is cleaned up
automatically:

.. code-block:: typescript

   let count = 0;
   for await (const result of db.queryStream("search text", { k: 100 })) {
     process(result);
     count++;
     if (count >= 10) break;  // cancel after 10 results
   }

Collecting Results
------------------

If you want all results as an array but still need the streaming endpoint (e.g., for progress
updates), collect them manually:

.. code-block:: typescript

   const results: QueryResult[] = [];
   for await (const result of db.queryStream("search text")) {
     results.push(result);
     updateProgressBar(results.length);
   }
   console.log(`Received ${results.length} results`);

.. tip::

   If you don't need streaming behavior, use ``db.query()`` instead — it returns all results
   in a single response and is simpler.

Error Handling
--------------

If the server encounters an error mid-stream, a ``ServerError`` is thrown:

.. code-block:: typescript

   import { ServerError } from "@localvectordb/sdk";

   try {
     for await (const result of db.queryStream("search text")) {
       process(result);
     }
   } catch (err) {
     if (err instanceof ServerError) {
       console.error("Stream error:", err.message);
     }
   }

How It Works
------------

Under the hood, ``queryStream()`` sends a ``POST`` request to the server's SSE endpoint
(``/api/v1/databases/{db}/query/stream``) and parses the event stream. The SDK uses a custom SSE parser
rather than the browser's native ``EventSource`` because ``EventSource`` only supports ``GET``
requests.

The server emits three event types:

- ``event: result`` — a single ``QueryResult`` as JSON
- ``event: done`` — signals the stream is complete
- ``event: error`` — an error occurred server-side
