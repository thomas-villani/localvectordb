Error Handling
==============

The SDK throws typed error classes that mirror the server's error codes. All errors extend
``LocalVectorDBError``, so you can catch broadly or narrowly.

.. contents:: Table of Contents
   :local:
   :depth: 2

Error Hierarchy
---------------

::

   LocalVectorDBError (base)
   ├── ValidationError              (400)
   ├── AuthenticationError           (401)
   ├── PermissionError               (403)
   ├── NotFoundError                 (404)
   │   ├── DatabaseNotFoundError
   │   └── DocumentNotFoundError
   ├── ConflictError                 (409)
   │   ├── DuplicateDocumentError
   │   └── DatabaseAlreadyExistsError
   ├── ServiceUnavailableError       (503)
   │   ├── EmbeddingError
   │   ├── OllamaNotAvailableError
   │   └── DatabaseConnectionError
   ├── ServerError                   (500)
   │   └── ConfigurationError
   ├── ConnectionError               (network failures)
   └── TimeoutError                  (request timeout)

Catching Errors
---------------

Use ``instanceof`` to catch specific error types:

.. code-block:: typescript

   import {
     DatabaseNotFoundError,
     DocumentNotFoundError,
     AuthenticationError,
     DuplicateDocumentError,
     ConnectionError,
     TimeoutError,
     LocalVectorDBError,
   } from "@localvectordb/sdk";

   try {
     await db.get("nonexistent-doc");
   } catch (err) {
     if (err instanceof DocumentNotFoundError) {
       console.log("Document does not exist");
     } else if (err instanceof AuthenticationError) {
       console.log("Invalid API key");
     } else if (err instanceof ConnectionError) {
       console.log("Cannot reach server:", err.message);
     } else if (err instanceof TimeoutError) {
       console.log("Request timed out");
     } else if (err instanceof LocalVectorDBError) {
       // Catch-all for any server error
       console.log(`Error [${err.code}]: ${err.message}`);
     }
   }

Catch a category with the parent class:

.. code-block:: typescript

   import { NotFoundError } from "@localvectordb/sdk";

   try {
     await db.get("doc-id");
   } catch (err) {
     if (err instanceof NotFoundError) {
       // Catches both DatabaseNotFoundError and DocumentNotFoundError
     }
   }

Error Properties
----------------

Every ``LocalVectorDBError`` carries these properties:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Property
     - Type
     - Description
   * - ``message``
     - ``string``
     - Human-readable error description
   * - ``code``
     - ``string``
     - Machine-readable error code (e.g. ``"DATABASE_NOT_FOUND"``)
   * - ``statusCode``
     - ``number``
     - HTTP status code (``0`` for network errors)
   * - ``details``
     - ``object``
     - Additional context from the server (e.g. ``{ field: "query" }``)
   * - ``recoverable``
     - ``boolean``
     - Whether retrying might succeed
   * - ``requestId``
     - ``string | undefined``
     - Server request ID for debugging
   * - ``timestamp``
     - ``string | undefined``
     - ISO 8601 timestamp of the error

Retry Behavior
--------------

The SDK's built-in retry logic (configured via ``maxRetries`` and ``retryDelay``) only applies to:

- **5xx server errors** — retried with exponential backoff
- **Network errors** — retried with exponential backoff
- **Timeouts** — retried with exponential backoff

**4xx client errors are never retried** — they indicate a problem with the request itself
(bad input, missing auth, document not found, etc.).

The ``recoverable`` property on error objects reflects the *server's* assessment of whether the
error is transient. You can use it for application-level retry decisions:

.. code-block:: typescript

   try {
     await db.upsert(documents);
   } catch (err) {
     if (err instanceof LocalVectorDBError && err.recoverable) {
       // Server says this might work if we try again
       await retry(() => db.upsert(documents));
     } else {
       throw err;
     }
   }
