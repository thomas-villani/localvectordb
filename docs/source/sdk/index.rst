JavaScript / TypeScript SDK
===========================

The ``@localvectordb/sdk`` package provides a TypeScript-first client for interacting with LocalVectorDB servers from
Node.js and browser applications. It mirrors the Python ``RemoteVectorDB`` client's API surface, giving JavaScript and
TypeScript developers full access to all LocalVectorDB server features.

**Key characteristics:**

- **Zero runtime dependencies** — uses only the built-in ``fetch`` API
- **TypeScript-first** — full type safety with hand-written interfaces and overloaded signatures
- **Dual format** — ships as both ESM and CommonJS
- **Cross-platform** — works in Node.js 18+, modern browsers, Deno, Bun, and edge runtimes
- **Typed errors** — error class hierarchy mirroring server error codes for precise ``catch`` handling
- **Streaming** — ``async`` generator-based SSE streaming via ``for await``

.. toctree::
   :maxdepth: 2

   quickstart
   client
   database
   streaming
   upload
   errors
   examples
