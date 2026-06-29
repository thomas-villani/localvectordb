# @localvectordb/sdk

TypeScript SDK for [LocalVectorDB](https://github.com/yourusername/localvectordb) — a document-first vector database.

Zero runtime dependencies. Works in Node.js 18+ and modern browsers.

## Installation

```bash
npm install @localvectordb/sdk
```

## Quick Start

```typescript
import { LocalVectorDBClient } from "@localvectordb/sdk";

const client = new LocalVectorDBClient({
  baseUrl: "http://localhost:5000",
  apiKey: "lvdb_your_api_key", // optional
});

// Create a database
await client.createDatabase("my_docs", {
  embedding: { provider: "ollama", model: "nomic-embed-text" },
});

// Get a database handle
const db = client.database("my_docs");

// Add documents
await db.upsert(["First document", "Second document"], {
  metadata: [{ author: "Alice" }, { author: "Bob" }],
});

// Search
const results = await db.query("search text", {
  search_type: "hybrid",
  k: 5,
});

for (const r of results.results) {
  console.log(r.id, r.score, r.content);
}
```

## Authentication

Pass an API key in the client config. It is sent as a `Bearer` token in the `Authorization` header.

```typescript
const client = new LocalVectorDBClient({
  baseUrl: "http://localhost:5000",
  apiKey: "lvdb_your_api_key",
});
```

## API Overview

### Client-level operations

```typescript
await client.health();
await client.listDatabases();
await client.createDatabase("name", options);
await client.deleteDatabase("name");
await client.globalSearch("query", options);
await client.embeddings(["text"], "ollama", "nomic-embed-text");
await client.factCheck("claim to verify", options);
```

### Database operations

```typescript
const db = client.database("my_db");

// Documents
await db.upsert(documents, options);
await db.insert(documents, options);
await db.get("doc-id");           // single document
await db.get(["id1", "id2"]);     // multiple documents
await db.update("doc-id", { content: "new text" });
await db.delete("doc-id");        // single
await db.delete(["id1", "id2"]);  // batch
await db.count();
await db.exists(["id1", "id2"]);
await db.list({ page: 1, limit: 20 });

// Search
await db.query("search text", { search_type: "hybrid", k: 10 });
await db.queryMultiColumn("search text", { columns: ["content", "title"] });
await db.filter({ author: "Alice" }, { order_by: "created_at DESC", limit: 10 });

// Schema
await db.getSchema();
await db.updateSchema({ new_field: { type: "text", indexed: true } });

// Comparison
await db.compare("doc-id-1", "doc-id-2");
await db.compareDetailed("doc-id-1", "doc-id-2");
await db.nearestNeighbors("doc-id", 5);
await db.similarityMatrix();

// Tuning & Maintenance
await db.getTuning();
await db.setTuning("write_heavy");
await db.checkpoint();
await db.optimize();
await db.vacuum();

// Fact-checking
await db.factCheck("The sky is blue");
```

## Streaming

Query results can be streamed via Server-Sent Events:

```typescript
for await (const result of db.queryStream("search text", { k: 20 })) {
  console.log(result.id, result.score);
}
```

Break out of the loop to cancel the stream.

## File Upload

### Browser

```typescript
const input = document.querySelector<HTMLInputElement>("#file-input");
const files = Array.from(input.files);

await db.upload(files, {
  metadata: { source: "web-upload" },
});
```

### Node.js

```typescript
import { readFile } from "fs/promises";

const data = await readFile("document.pdf");
await db.upload(
  [{ name: "document.pdf", data, type: "application/pdf" }],
  { use_filename_as_id: true },
);
```

## Error Handling

The SDK throws typed errors that mirror the server's error codes:

```typescript
import {
  DatabaseNotFoundError,
  DocumentNotFoundError,
  ValidationError,
  AuthenticationError,
  DuplicateDocumentError,
  EmbeddingError,
  ConnectionError,
  TimeoutError,
} from "@localvectordb/sdk";

try {
  await db.get("nonexistent");
} catch (err) {
  if (err instanceof DocumentNotFoundError) {
    console.log("Document not found:", err.message);
  } else if (err instanceof AuthenticationError) {
    console.log("Bad API key");
  } else if (err instanceof ConnectionError) {
    console.log("Server unreachable");
  }

  // All errors extend LocalVectorDBError with:
  // err.code        - machine-readable error code
  // err.statusCode  - HTTP status (0 for network errors)
  // err.details     - additional context from the server
  // err.recoverable - whether retrying might help
  // err.requestId   - for debugging with server logs
}
```

## Configuration

```typescript
const client = new LocalVectorDBClient({
  baseUrl: "http://localhost:5000",
  apiKey: "lvdb_...",       // optional
  timeout: 30000,           // request timeout in ms (default: 30s)
  maxRetries: 3,            // retry count for 5xx/network errors (default: 3)
  retryDelay: 1000,         // base delay in ms, doubles each retry (default: 1s)
});
```

## License

MIT
