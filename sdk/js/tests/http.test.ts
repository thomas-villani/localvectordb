import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { HttpClient } from "../src/http.js";
import {
  ConnectionError,
  TimeoutError,
  ValidationError,
  ServerError,
  DatabaseNotFoundError,
  AuthenticationError,
} from "../src/errors.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeClient(overrides?: Partial<ConstructorParameters<typeof HttpClient>[0]>) {
  return new HttpClient({
    baseUrl: "http://localhost:5000",
    timeout: 5000,
    maxRetries: 2,
    retryDelay: 10, // fast for tests
    ...overrides,
  });
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function errorResponse(code: string, message: string, status: number): Response {
  return jsonResponse(
    { error: { code, message, timestamp: "2026-01-01T00:00:00Z", request_id: "req-1", details: {}, recoverable: false } },
    status,
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("HttpClient", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  // -----------------------------------------------------------------------
  // Auth headers
  // -----------------------------------------------------------------------

  it("sends Authorization header when apiKey is set", async () => {
    const client = makeClient({ apiKey: "lvdb_test123" });
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ ok: true }));

    await client.get("/api/v1/health");

    const [, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect((init?.headers as Record<string, string>)["Authorization"]).toBe(
      "Bearer lvdb_test123",
    );
  });

  it("omits Authorization header when no apiKey", async () => {
    const client = makeClient();
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ ok: true }));

    await client.get("/api/v1/health");

    const [, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect((init?.headers as Record<string, string>)["Authorization"]).toBeUndefined();
  });

  // -----------------------------------------------------------------------
  // URL construction
  // -----------------------------------------------------------------------

  it("constructs full URL from baseUrl + path", async () => {
    const client = makeClient({ baseUrl: "http://myhost:8080/" }); // trailing slash
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ ok: true }));

    await client.get("/api/v1/databases");

    const [url] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(url).toBe("http://myhost:8080/api/v1/databases");
  });

  // -----------------------------------------------------------------------
  // JSON body
  // -----------------------------------------------------------------------

  it("sends JSON body for POST", async () => {
    const client = makeClient();
    vi.mocked(globalThis.fetch).mockResolvedValue(jsonResponse({ ids: ["1"] }));

    await client.post("/api/v1/mydb/documents", { documents: ["hello"] });

    const [, init] = vi.mocked(globalThis.fetch).mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ documents: ["hello"] });
  });

  // -----------------------------------------------------------------------
  // Error mapping
  // -----------------------------------------------------------------------

  it("throws ValidationError on 400", async () => {
    const client = makeClient();
    vi.mocked(globalThis.fetch).mockResolvedValue(
      errorResponse("VALIDATION_ERROR", "bad input", 400),
    );

    await expect(client.post("/api/v1/x", {})).rejects.toThrow(ValidationError);
  });

  it("throws AuthenticationError on 401", async () => {
    const client = makeClient();
    vi.mocked(globalThis.fetch).mockResolvedValue(
      jsonResponse({ detail: "Unauthorized" }, 401),
    );

    await expect(client.get("/api/v1/x")).rejects.toThrow(AuthenticationError);
  });

  it("throws DatabaseNotFoundError for DATABASE_NOT_FOUND code", async () => {
    const client = makeClient();
    vi.mocked(globalThis.fetch).mockResolvedValue(
      errorResponse("DATABASE_NOT_FOUND", "db 'foo' not found", 404),
    );

    await expect(client.get("/api/v1/foo/info")).rejects.toThrow(
      DatabaseNotFoundError,
    );
  });

  // -----------------------------------------------------------------------
  // No retry on 4xx
  // -----------------------------------------------------------------------

  it("does NOT retry on 4xx errors", async () => {
    const client = makeClient({ maxRetries: 3 });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      errorResponse("VALIDATION_ERROR", "bad", 400),
    );

    await expect(client.post("/api/v1/x", {})).rejects.toThrow(ValidationError);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  // -----------------------------------------------------------------------
  // Retry on 5xx
  // -----------------------------------------------------------------------

  it("retries on 5xx then succeeds", async () => {
    const client = makeClient({ maxRetries: 2, retryDelay: 1 });
    vi.mocked(globalThis.fetch)
      .mockResolvedValueOnce(jsonResponse({}, 500))
      .mockResolvedValueOnce(jsonResponse({ ok: true }));

    const result = await client.get<{ ok: boolean }>("/api/v1/health");
    expect(result.ok).toBe(true);
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
  });

  it("throws ServerError after exhausting retries on 5xx", async () => {
    const client = makeClient({ maxRetries: 1, retryDelay: 1 });
    vi.mocked(globalThis.fetch)
      .mockResolvedValue(errorResponse("INTERNAL_ERROR", "boom", 500));

    await expect(client.get("/api/v1/health")).rejects.toThrow(ServerError);
    // 1 initial + 1 retry = 2
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
  });

  // -----------------------------------------------------------------------
  // Retry on network error
  // -----------------------------------------------------------------------

  it("retries on network error then succeeds", async () => {
    const client = makeClient({ maxRetries: 2, retryDelay: 1 });
    vi.mocked(globalThis.fetch)
      .mockRejectedValueOnce(new TypeError("fetch failed"))
      .mockResolvedValueOnce(jsonResponse({ ok: true }));

    const result = await client.get<{ ok: boolean }>("/api/v1/health");
    expect(result.ok).toBe(true);
  });

  it("throws ConnectionError after exhausting retries on network error", async () => {
    const client = makeClient({ maxRetries: 1, retryDelay: 1 });
    vi.mocked(globalThis.fetch).mockRejectedValue(new TypeError("fetch failed"));

    await expect(client.get("/api/v1/health")).rejects.toThrow(ConnectionError);
  });

  // -----------------------------------------------------------------------
  // Timeout
  // -----------------------------------------------------------------------

  it("throws TimeoutError when request exceeds timeout", async () => {
    const client = makeClient({ timeout: 50, maxRetries: 0 });
    vi.mocked(globalThis.fetch).mockImplementation(
      (_url, init) =>
        new Promise((_resolve, reject) => {
          // Simulate a request that never completes but respects abort
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("The operation was aborted.", "AbortError"));
          });
        }),
    );

    await expect(client.get("/api/v1/health")).rejects.toThrow(TimeoutError);
  });

  // -----------------------------------------------------------------------
  // postRaw
  // -----------------------------------------------------------------------

  it("postRaw returns raw Response", async () => {
    const client = makeClient();
    const body = JSON.stringify({ results: [] });
    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(body, { status: 200 }),
    );

    const resp = await client.postRaw("/api/v1/mydb/query/stream", { query: "test" });
    expect(resp).toBeInstanceOf(Response);
    expect(resp.status).toBe(200);
  });
});
