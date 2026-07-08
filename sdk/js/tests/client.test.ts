import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { LocalVectorDBClient } from "../src/client.js";
import { DatabaseHandle } from "../src/database.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("LocalVectorDBClient", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("database() returns a DatabaseHandle synchronously", () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    const db = client.database("test_db");
    expect(db).toBeInstanceOf(DatabaseHandle);
    expect(db.name).toBe("test_db");
    // No fetch call should have been made
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("health() calls GET /api/v1/health", async () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ status: "ok", version: "1.0.0", ollama_available: true, timestamp: "2026-01-01" }),
    );

    const result = await client.health();
    expect(result.status).toBe("ok");

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/health");
    expect(init?.method).toBe("GET");
  });

  it("listDatabases() calls GET /api/v1/databases", async () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ databases: ["db1", "db2"], count: 2 }),
    );

    const result = await client.listDatabases();
    expect(result.databases).toEqual(["db1", "db2"]);
    expect(result.count).toBe(2);
  });

  it("createDatabase() calls POST /api/v1/databases with body", async () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "created", status: "success", config: {} }),
    );

    await client.createDatabase("new_db", {
      embedding: { provider: "ollama", model: "nomic-embed-text" },
    });

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/databases");
    const body = JSON.parse(init?.body as string);
    expect(body.name).toBe("new_db");
    expect(body.embedding.provider).toBe("ollama");
  });

  it("deleteDatabase() calls DELETE /api/v1/databases/{name}", async () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "deleted", status: "success" }),
    );

    await client.deleteDatabase("old_db");

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/databases/old_db");
    expect(init?.method).toBe("DELETE");
  });

  it("globalSearch() calls POST /api/v1/search and returns results_by_database", async () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({
        results_by_database: {
          db1: [
            { id: "1", score: 0.9, type: "document", content: "hi", metadata: {} },
          ],
        },
        search_type: "hybrid",
        return_type: "documents",
      }),
    );

    const result = await client.globalSearch("test query", {
      search_type: "hybrid",
      k: 5,
    });
    expect(result.results_by_database.db1).toHaveLength(1);

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/search");
    const body = JSON.parse(init?.body as string);
    expect(body.query).toBe("test query");
    expect(body.search_type).toBe("hybrid");
  });

  it("embeddings() calls POST /api/v1/embeddings", async () => {
    const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ embeddings: [[0.1, 0.2]], provider: "ollama", model: "nomic" }),
    );

    await client.embeddings("hello", "ollama", "nomic");

    const body = JSON.parse(
      (vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string),
    );
    expect(body.texts).toEqual(["hello"]);
    expect(body.provider).toBe("ollama");
  });
});
