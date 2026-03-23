import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { LocalVectorDBClient } from "../src/client.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function client() {
  return new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("DatabaseHandle", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // -----------------------------------------------------------------------
  // Info
  // -----------------------------------------------------------------------

  it("info() calls GET /api/v1/{name}/info", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ name: "testdb", stats: {}, config: {} }),
    );

    const result = await db.info();
    expect(result.name).toBe("testdb");

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/info");
  });

  // -----------------------------------------------------------------------
  // Upsert / Insert
  // -----------------------------------------------------------------------

  it("upsert() sends documents and options", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "ok", ids: ["1", "2"], status: "success" }),
    );

    const result = await db.upsert(["doc1", "doc2"], {
      metadata: [{ author: "Alice" }, { author: "Bob" }],
    });
    expect(result.ids).toEqual(["1", "2"]);

    const body = JSON.parse(
      vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string,
    );
    expect(body.documents).toEqual(["doc1", "doc2"]);
    expect(body.metadata).toHaveLength(2);
  });

  it("insert() sends to /documents/insert", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "ok", ids: ["1"], status: "success" }),
    );

    await db.insert("single doc", { errors: "ignore" });

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/documents/insert");
  });

  // -----------------------------------------------------------------------
  // Get (overloaded)
  // -----------------------------------------------------------------------

  it("get(string) calls GET /documents/{id}", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ id: "abc", content: "hello", metadata: {}, created_at: null, updated_at: null, content_hash: null }),
    );

    const doc = await db.get("abc");
    expect(doc.id).toBe("abc");

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/documents/abc");
  });

  it("get(string[]) calls GET /documents?ids=...", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({
        documents: [
          { id: "a", content: "1", metadata: {}, created_at: null, updated_at: null, content_hash: null },
          { id: "b", content: "2", metadata: {}, created_at: null, updated_at: null, content_hash: null },
        ],
      }),
    );

    const docs = await db.get(["a", "b"]);
    expect(docs).toHaveLength(2);

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toContain("ids=a%2Cb");
  });

  // -----------------------------------------------------------------------
  // Update
  // -----------------------------------------------------------------------

  it("update() calls PUT /documents/{id}", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "updated", status: "success", updated: true }),
    );

    await db.update("doc1", { content: "new content" });

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/documents/doc1");
    expect(init?.method).toBe("PUT");
  });

  // -----------------------------------------------------------------------
  // Delete (overloaded)
  // -----------------------------------------------------------------------

  it("delete(string) calls DELETE /documents/{id}", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "deleted", status: "success", deleted_count: 1 }),
    );

    await db.delete("doc1");

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/documents/doc1");
    expect(init?.method).toBe("DELETE");
  });

  it("delete(string[]) calls POST /documents/delete", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ message: "deleted", status: "success", deleted_count: 2, failed_ids: [] }),
    );

    await db.delete(["doc1", "doc2"]);

    const [url, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/documents/delete");
    expect(init?.method).toBe("POST");
  });

  // -----------------------------------------------------------------------
  // Count / Exists / List
  // -----------------------------------------------------------------------

  it("count() returns number", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ count: 42 }));

    const count = await db.count();
    expect(count).toBe(42);
  });

  it("exists() calls POST /documents/exists", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ exists: { doc1: true, doc2: false }, ids: ["doc1", "doc2"] }),
    );

    const result = await db.exists(["doc1", "doc2"]);
    expect(result.exists["doc1"]).toBe(true);
  });

  it("list() builds query params", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ documents: [], pagination: null }),
    );

    await db.list({ page: 2, limit: 10 });

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toContain("page=2");
    expect(url).toContain("limit=10");
  });

  // -----------------------------------------------------------------------
  // Query / Filter
  // -----------------------------------------------------------------------

  it("query() sends correct body", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ results: [], search_type: "hybrid", return_type: "documents", total_results: 0 }),
    );

    await db.query("test", { search_type: "hybrid", k: 5 });

    const body = JSON.parse(
      vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string,
    );
    expect(body.query).toBe("test");
    expect(body.search_type).toBe("hybrid");
    expect(body.k).toBe(5);
  });

  it("filter() sends where + options", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ documents: [], count: 0 }),
    );

    await db.filter({ author: "Alice" }, { order_by: "created_at DESC", limit: 5 });

    const body = JSON.parse(
      vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string,
    );
    expect(body.where).toEqual({ author: "Alice" });
    expect(body.order_by).toBe("created_at DESC");
  });

  // -----------------------------------------------------------------------
  // Schema
  // -----------------------------------------------------------------------

  it("getSchema() calls GET /schema", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ database: "testdb", schema_info: {}, status: "success" }),
    );

    await db.getSchema();

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://localhost:5000/api/v1/testdb/schema");
  });

  // -----------------------------------------------------------------------
  // Comparison
  // -----------------------------------------------------------------------

  it("compare() sends doc IDs", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ doc_id_1: "a", doc_id_2: "b", similarity: 0.85, status: "success" }),
    );

    const result = await db.compare("a", "b");
    expect(result.similarity).toBe(0.85);

    const body = JSON.parse(
      vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string,
    );
    expect(body.doc_id_1).toBe("a");
    expect(body.doc_id_2).toBe("b");
  });

  // -----------------------------------------------------------------------
  // Tuning
  // -----------------------------------------------------------------------

  it("setTuning() sends profile and options", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ database: "testdb", message: "ok", tuning: {}, status: "success" }),
    );

    await db.setTuning("write_heavy", { persist: false });

    const body = JSON.parse(
      vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string,
    );
    expect(body.profile).toBe("write_heavy");
    expect(body.persist).toBe(false);
  });

  // -----------------------------------------------------------------------
  // Fact-Check
  // -----------------------------------------------------------------------

  it("factCheck() sends text and options", async () => {
    const db = client().database("testdb");
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ verdict: "supported" }));

    await db.factCheck("The earth is round", { llm_provider: "anthropic" });

    const body = JSON.parse(
      vi.mocked(globalThis.fetch).mock.calls[0][1]?.body as string,
    );
    expect(body.text).toBe("The earth is round");
    expect(body.llm_provider).toBe("anthropic");
  });
});
