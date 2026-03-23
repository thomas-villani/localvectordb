import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { streamQuery } from "../src/sse.js";
import { HttpClient } from "../src/http.js";
import { ServerError } from "../src/errors.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Create a ReadableStream from SSE-formatted text chunks. */
function sseStream(...chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(encoder.encode(chunks[index]));
        index++;
      } else {
        controller.close();
      }
    },
  });
}

function makeHttpClient(): HttpClient {
  return new HttpClient({
    baseUrl: "http://localhost:5000",
    timeout: 5000,
    maxRetries: 0,
    retryDelay: 0,
  });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("streamQuery (SSE parser)", () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    globalThis.fetch = vi.fn();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("yields QueryResult objects from SSE result events", async () => {
    const sse =
      'event: result\ndata: {"id":"1","score":0.9,"type":"document","content":"hello","metadata":{}}\n\n' +
      'event: result\ndata: {"id":"2","score":0.8,"type":"document","content":"world","metadata":{}}\n\n' +
      'event: done\ndata: {"total_results":2}\n\n';

    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(sseStream(sse), { status: 200 }),
    );

    const http = makeHttpClient();
    const results = [];
    for await (const result of streamQuery(http, "testdb", "search")) {
      results.push(result);
    }

    expect(results).toHaveLength(2);
    expect(results[0].id).toBe("1");
    expect(results[0].score).toBe(0.9);
    expect(results[1].id).toBe("2");
  });

  it("handles SSE events split across chunks", async () => {
    // Split an event mid-way through a data line
    const part1 = 'event: result\ndata: {"id":"1","score":0.9,';
    const part2 = '"type":"document","content":"hello","metadata":{}}\n\n';
    const done = 'event: done\ndata: {"total_results":1}\n\n';

    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(sseStream(part1, part2, done), { status: 200 }),
    );

    const http = makeHttpClient();
    const results = [];
    for await (const result of streamQuery(http, "testdb", "search")) {
      results.push(result);
    }

    expect(results).toHaveLength(1);
    expect(results[0].id).toBe("1");
  });

  it("throws ServerError on error event", async () => {
    const sse =
      'event: error\ndata: {"error":"something went wrong"}\n\n';

    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(sseStream(sse), { status: 200 }),
    );

    const http = makeHttpClient();

    await expect(async () => {
      for await (const _result of streamQuery(http, "testdb", "search")) {
        // should not reach here
      }
    }).rejects.toThrow(ServerError);
  });

  it("terminates on done event even with no results", async () => {
    const sse = 'event: done\ndata: {"total_results":0}\n\n';

    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(sseStream(sse), { status: 200 }),
    );

    const http = makeHttpClient();
    const results = [];
    for await (const result of streamQuery(http, "testdb", "search")) {
      results.push(result);
    }

    expect(results).toHaveLength(0);
  });

  it("ignores SSE comments (lines starting with :)", async () => {
    const sse =
      ': keep-alive\n\n' +
      'event: result\ndata: {"id":"1","score":0.9,"type":"document","content":"hello","metadata":{}}\n\n' +
      'event: done\ndata: {"total_results":1}\n\n';

    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(sseStream(sse), { status: 200 }),
    );

    const http = makeHttpClient();
    const results = [];
    for await (const result of streamQuery(http, "testdb", "search")) {
      results.push(result);
    }

    expect(results).toHaveLength(1);
  });

  it("handles empty body gracefully", async () => {
    vi.mocked(globalThis.fetch).mockResolvedValue(
      new Response(null, { status: 200 }),
    );

    const http = makeHttpClient();
    const results = [];
    for await (const result of streamQuery(http, "testdb", "search")) {
      results.push(result);
    }

    expect(results).toHaveLength(0);
  });
});
