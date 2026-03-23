import { DatabaseHandle } from "./database.js";
import { HttpClient } from "./http.js";
import type {
  ClientConfig,
  CreateDatabaseOptions,
  CreateDatabaseResponse,
  DatabaseListResponse,
  DeleteDatabaseResponse,
  EmbeddingsResponse,
  FactCheckResponse,
  GlobalFactCheckOptions,
  GlobalSearchOptions,
  GlobalSearchResponse,
  HealthResponse,
  SystemResourcesResponse,
} from "./types.js";

/**
 * Top-level client for a LocalVectorDB server.
 *
 * ```ts
 * const client = new LocalVectorDBClient({
 *   baseUrl: "http://localhost:5000",
 *   apiKey: "lvdb_my_secret_key",
 * });
 *
 * const db = client.database("my_database");
 * await db.upsert(["Hello world"]);
 * ```
 */
export class LocalVectorDBClient {
  private readonly http: HttpClient;

  constructor(config: ClientConfig) {
    this.http = new HttpClient({
      baseUrl: config.baseUrl,
      apiKey: config.apiKey,
      timeout: config.timeout ?? 30_000,
      maxRetries: config.maxRetries ?? 3,
      retryDelay: config.retryDelay ?? 1_000,
    });
  }

  // =========================================================================
  // Database handle
  // =========================================================================

  /**
   * Get a handle for a specific database.
   *
   * This is synchronous and makes no network call — the returned
   * {@link DatabaseHandle} is a lightweight reference.  If the database
   * does not exist, the first API call against it will fail with
   * `DatabaseNotFoundError`.
   */
  database(name: string): DatabaseHandle {
    return new DatabaseHandle(this.http, name);
  }

  // =========================================================================
  // Database management
  // =========================================================================

  /** Create a new database on the server. */
  async createDatabase(
    name: string,
    options?: CreateDatabaseOptions,
  ): Promise<CreateDatabaseResponse> {
    return this.http.post<CreateDatabaseResponse>("/api/v1/databases", {
      name,
      metadata_schema: options?.metadata_schema,
      database: options?.database,
      embedding: options?.embedding,
    });
  }

  /** List all databases on the server. */
  async listDatabases(): Promise<DatabaseListResponse> {
    return this.http.get<DatabaseListResponse>("/api/v1/databases");
  }

  /** Delete a database. */
  async deleteDatabase(name: string): Promise<DeleteDatabaseResponse> {
    return this.http.del<DeleteDatabaseResponse>(
      `/api/v1/${encodeURIComponent(name)}`,
    );
  }

  // =========================================================================
  // Health
  // =========================================================================

  /** Check server health. */
  async health(): Promise<HealthResponse> {
    return this.http.get<HealthResponse>("/api/v1/health");
  }

  /** Get system resource information. */
  async systemResources(): Promise<SystemResourcesResponse> {
    return this.http.get<SystemResourcesResponse>("/api/v1/system/resources");
  }

  // =========================================================================
  // Cross-database operations
  // =========================================================================

  /** Search across multiple databases. */
  async globalSearch(
    query: string,
    options?: GlobalSearchOptions,
  ): Promise<GlobalSearchResponse> {
    return this.http.post<GlobalSearchResponse>("/api/v1/search", {
      query,
      ...options,
    });
  }

  /** Generate embeddings using a specified provider and model. */
  async embeddings(
    texts: string | string[],
    provider: string,
    model: string,
  ): Promise<EmbeddingsResponse> {
    return this.http.post<EmbeddingsResponse>("/api/v1/embeddings", {
      texts: Array.isArray(texts) ? texts : [texts],
      provider,
      model,
    });
  }

  /** Check text for factual grounding across databases. */
  async factCheck(
    text: string,
    options?: GlobalFactCheckOptions,
  ): Promise<FactCheckResponse> {
    return this.http.post<FactCheckResponse>("/api/v1/factcheck", {
      text,
      ...options,
    });
  }
}
