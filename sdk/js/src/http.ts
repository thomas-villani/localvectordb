import type { ServerErrorPayload } from "./types.js";
import {
  ConnectionError,
  TimeoutError,
  errorFromResponse,
} from "./errors.js";

export interface HttpClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout: number;
  maxRetries: number;
  retryDelay: number;
}

interface RequestOptions {
  headers?: Record<string, string>;
  signal?: AbortSignal;
  /** Skip JSON parsing and return the raw Response. */
  raw?: boolean;
}

/**
 * Low-level HTTP client wrapping `fetch` with auth, timeout, retries, and
 * structured error mapping.
 *
 * @internal Not exported from the public API.
 */
export class HttpClient {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly timeout: number;
  private readonly maxRetries: number;
  private readonly retryDelay: number;

  constructor(config: HttpClientConfig) {
    // Strip trailing slash
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.apiKey = config.apiKey;
    this.timeout = config.timeout;
    this.maxRetries = config.maxRetries;
    this.retryDelay = config.retryDelay;
  }

  // -------------------------------------------------------------------------
  // Public convenience methods
  // -------------------------------------------------------------------------

  async get<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>("GET", path, undefined, options);
  }

  async post<T>(
    path: string,
    body?: unknown,
    options?: RequestOptions,
  ): Promise<T> {
    return this.request<T>("POST", path, body, options);
  }

  async put<T>(
    path: string,
    body?: unknown,
    options?: RequestOptions,
  ): Promise<T> {
    return this.request<T>("PUT", path, body, options);
  }

  async del<T>(path: string, options?: RequestOptions): Promise<T> {
    return this.request<T>("DELETE", path, undefined, options);
  }

  /**
   * POST that returns the raw `Response` (for SSE streams and file uploads).
   * No retry is applied — callers handle errors themselves.
   */
  async postRaw(
    path: string,
    body?: BodyInit | Record<string, unknown>,
    options?: RequestOptions,
  ): Promise<Response> {
    const url = this.baseUrl + path;
    const isFormData =
      typeof FormData !== "undefined" && body instanceof FormData;

    const headers: Record<string, string> = {
      ...this.authHeaders(),
      ...options?.headers,
    };

    // Only set Content-Type for JSON bodies; FormData sets its own boundary.
    let fetchBody: BodyInit | undefined;
    if (isFormData) {
      fetchBody = body as FormData;
    } else if (body !== undefined && body !== null) {
      headers["Content-Type"] = "application/json";
      fetchBody = JSON.stringify(body);
    }

    const controller = new AbortController();
    const timeoutId =
      this.timeout > 0
        ? setTimeout(() => controller.abort(), this.timeout)
        : undefined;

    try {
      const response = await fetch(url, {
        method: "POST",
        headers,
        body: fetchBody,
        signal: options?.signal ?? controller.signal,
      });

      if (!response.ok) {
        const parsed = await this.parseErrorBody(response);
        throw errorFromResponse(response.status, parsed);
      }

      return response;
    } catch (err) {
      throw this.wrapNetworkError(err);
    } finally {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
    }
  }

  // -------------------------------------------------------------------------
  // Core request method with retry
  // -------------------------------------------------------------------------

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    options?: RequestOptions,
  ): Promise<T> {
    const url = this.baseUrl + path;
    const maxAttempts = this.maxRetries + 1;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const controller = new AbortController();
      const timeoutId =
        this.timeout > 0
          ? setTimeout(() => controller.abort(), this.timeout)
          : undefined;

      try {
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
          ...this.authHeaders(),
          ...options?.headers,
        };

        const response = await fetch(url, {
          method,
          headers,
          body: body !== undefined ? JSON.stringify(body) : undefined,
          signal: options?.signal ?? controller.signal,
        });

        if (response.ok) {
          return (await response.json()) as T;
        }

        // 4xx: do NOT retry — throw immediately
        if (response.status >= 400 && response.status < 500) {
          const parsed = await this.parseErrorBody(response);
          throw errorFromResponse(response.status, parsed);
        }

        // 5xx: retry unless this is the last attempt
        if (attempt < maxAttempts - 1) {
          await this.backoff(attempt);
          continue;
        }

        const parsed = await this.parseErrorBody(response);
        throw errorFromResponse(response.status, parsed);
      } catch (err) {
        // Already a LocalVectorDBError — re-throw
        if (err instanceof Error && "code" in err && "statusCode" in err) {
          throw err;
        }

        // Network / timeout error — retry unless last attempt
        if (attempt < maxAttempts - 1) {
          await this.backoff(attempt);
          continue;
        }

        throw this.wrapNetworkError(err);
      } finally {
        if (timeoutId !== undefined) clearTimeout(timeoutId);
      }
    }

    // Unreachable, but TypeScript requires a return.
    throw new ConnectionError("Request failed after retries");
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  private authHeaders(): Record<string, string> {
    if (!this.apiKey) return {};
    return { Authorization: `Bearer ${this.apiKey}` };
  }

  private async parseErrorBody(
    response: Response,
  ): Promise<ServerErrorPayload | string> {
    try {
      return (await response.json()) as ServerErrorPayload;
    } catch {
      try {
        return await response.text();
      } catch {
        return `HTTP ${response.status}`;
      }
    }
  }

  private backoff(attempt: number): Promise<void> {
    const delay = this.retryDelay * 2 ** attempt;
    return new Promise((resolve) => setTimeout(resolve, delay));
  }

  private wrapNetworkError(err: unknown): Error {
    if (err instanceof Error) {
      if (err.name === "AbortError") {
        return new TimeoutError(`Request timed out after ${this.timeout}ms`);
      }
      return new ConnectionError(err.message);
    }
    return new ConnectionError(String(err));
  }
}
