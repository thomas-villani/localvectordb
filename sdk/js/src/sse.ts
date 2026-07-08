import type { QueryResult, StreamQueryOptions } from "./types.js";
import { ServerError } from "./errors.js";
import type { HttpClient } from "./http.js";

/**
 * Stream query results from the server using POST-based Server-Sent Events.
 *
 * Yields individual {@link QueryResult} objects as they arrive.  The generator
 * terminates when the server sends an `event: done` frame.
 *
 * Breaking out of the `for await` loop will cancel the underlying stream.
 *
 * @example
 * ```ts
 * for await (const result of streamQuery(http, "mydb", "search text")) {
 *   console.log(result.id, result.score);
 * }
 * ```
 *
 * @internal Not part of the public API — consumed via
 * {@link DatabaseHandle.queryStream}.
 */
export async function* streamQuery(
  httpClient: HttpClient,
  dbName: string,
  queryText: string,
  options?: StreamQueryOptions,
): AsyncGenerator<QueryResult, void, undefined> {
  const body = {
    query: queryText,
    ...options,
  };

  // Use postRaw so we get the raw Response with a readable body stream.
  // Pass a very long timeout / no timeout — streaming can take a while.
  const response = await httpClient.postRaw(
    `/api/v1/databases/${encodeURIComponent(dbName)}/query/stream`,
    body,
  );

  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE events are separated by double newlines.
      const parts = buffer.split("\n\n");

      // The last element is either empty (if buffer ends with \n\n) or a
      // partial event that hasn't been fully received yet.
      buffer = parts.pop()!;

      for (const part of parts) {
        const event = parseSSEEvent(part);
        if (!event) continue;

        switch (event.type) {
          case "result": {
            const result = JSON.parse(event.data) as QueryResult;
            yield result;
            break;
          }
          case "done":
            return;
          case "error": {
            let message = event.data;
            try {
              const parsed = JSON.parse(event.data) as { error?: string };
              message = parsed.error ?? event.data;
            } catch {
              // data is plain text — use as-is
            }
            throw new ServerError(message, {
              code: "STREAMING_ERROR",
              statusCode: 500,
            });
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------

interface SSEEvent {
  type: string;
  data: string;
}

function parseSSEEvent(raw: string): SSEEvent | null {
  let type = "message";
  const dataLines: string[] = [];

  for (const line of raw.split("\n")) {
    if (line.startsWith(":")) {
      // SSE comment / keep-alive — skip
      continue;
    }
    if (line.startsWith("event:")) {
      type = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }

  if (dataLines.length === 0 && type === "message") {
    return null;
  }

  return { type, data: dataLines.join("\n") };
}
