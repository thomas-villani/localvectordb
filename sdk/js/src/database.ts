import type { HttpClient } from "./http.js";
import { streamQuery } from "./sse.js";
import { uploadFiles } from "./upload.js";
import type {
  AutoTuneOptions,
  AutoTuneResponse,
  BatchDeleteResponse,
  CheckpointMode,
  CompareDetailedOptions,
  CompareDetailedResponse,
  CompareResponse,
  CountOptions,
  DatabaseEmbeddingsOptions,
  DatabaseInfoResponse,
  DeleteResponse,
  Document,
  EmbeddingsResponse,
  ExistsResponse,
  FactCheckOptions,
  FactCheckResponse,
  FilterOptions,
  FilterResponse,
  InsertChunksOptions,
  InsertOptions,
  InsertResponse,
  ListDocumentsOptions,
  ListDocumentsResponse,
  MaintenanceResponse,
  MetadataFieldDefinition,
  NearestNeighborsResponse,
  QueryMultiColumnOptions,
  QueryOptions,
  QueryResponse,
  QueryResult,
  SchemaInfoResponse,
  SetTuningOptions,
  SetTuningResponse,
  SimilarityMatrixResponse,
  StreamQueryOptions,
  TuningResponse,
  UpdateResponse,
  UpdateSchemaOptions,
  UpdateSchemaResponse,
  UploadOptions,
  UploadResponse,
  UploadableFile,
  UpsertChunksOptions,
  UpsertOptions,
  UpsertResponse,
} from "./types.js";

/**
 * Handle for a single database on the server.
 *
 * Obtain an instance via {@link LocalVectorDBClient.database}:
 *
 * ```ts
 * const db = client.database("my_database");
 * await db.upsert(["Hello world"]);
 * ```
 */
export class DatabaseHandle {
  /** @internal */
  constructor(
    private readonly http: HttpClient,
    /** The database name. */
    public readonly name: string,
  ) {}

  /** URL prefix for this database. */
  private get prefix(): string {
    return `/api/v1/${encodeURIComponent(this.name)}`;
  }

  // =========================================================================
  // Info
  // =========================================================================

  /** Get database statistics and configuration. */
  async info(): Promise<DatabaseInfoResponse> {
    return this.http.get<DatabaseInfoResponse>(`${this.prefix}/info`);
  }

  // =========================================================================
  // Documents — CRUD
  // =========================================================================

  /** Upsert (insert or update) documents. */
  async upsert(
    documents: string | string[],
    options?: UpsertOptions,
  ): Promise<UpsertResponse> {
    return this.http.post<UpsertResponse>(`${this.prefix}/documents`, {
      documents,
      ...options,
    });
  }

  /** Insert new documents (fails if ID already exists, unless `errors: "ignore"`). */
  async insert(
    documents: string | string[],
    options?: InsertOptions,
  ): Promise<InsertResponse> {
    return this.http.post<InsertResponse>(`${this.prefix}/documents/insert`, {
      documents,
      ...options,
    });
  }

  /**
   * Get one or more documents by ID.
   *
   * A single string returns a single {@link Document}; an array of strings
   * returns an array.
   */
  async get(id: string): Promise<Document>;
  async get(ids: string[]): Promise<Document[]>;
  async get(ids: string | string[]): Promise<Document | Document[]> {
    if (typeof ids === "string") {
      return this.http.get<Document>(
        `${this.prefix}/documents/${encodeURIComponent(ids)}`,
      );
    }
    const params = new URLSearchParams();
    params.set("ids", ids.join(","));
    const resp = await this.http.get<ListDocumentsResponse>(
      `${this.prefix}/documents?${params.toString()}`,
    );
    return resp.documents;
  }

  /** Update a document's content and/or metadata. */
  async update(
    id: string,
    options: { content?: string; metadata?: Record<string, unknown> },
  ): Promise<UpdateResponse> {
    return this.http.put<UpdateResponse>(
      `${this.prefix}/documents/${encodeURIComponent(id)}`,
      options,
    );
  }

  /**
   * Delete one or more documents.
   *
   * A single ID uses `DELETE`; an array uses batch `POST`.
   */
  async delete(id: string): Promise<DeleteResponse>;
  async delete(ids: string[]): Promise<BatchDeleteResponse>;
  async delete(
    ids: string | string[],
  ): Promise<DeleteResponse | BatchDeleteResponse> {
    if (typeof ids === "string") {
      return this.http.del<DeleteResponse>(
        `${this.prefix}/documents/${encodeURIComponent(ids)}`,
      );
    }
    return this.http.post<BatchDeleteResponse>(
      `${this.prefix}/documents/delete`,
      { ids },
    );
  }

  /** Count documents, optionally matching a filter. */
  async count(options?: CountOptions): Promise<number> {
    const resp = await this.http.post<{ count: number }>(
      `${this.prefix}/documents/count`,
      options ?? {},
    );
    return resp.count;
  }

  /** Check if one or more documents exist. */
  async exists(ids: string | string[]): Promise<ExistsResponse> {
    return this.http.post<ExistsResponse>(
      `${this.prefix}/documents/exists`,
      { ids: Array.isArray(ids) ? ids : [ids] },
    );
  }

  /** List documents with optional pagination. */
  async list(options?: ListDocumentsOptions): Promise<ListDocumentsResponse> {
    const params = new URLSearchParams();
    if (options?.page !== undefined) params.set("page", String(options.page));
    if (options?.limit !== undefined)
      params.set("limit", String(options.limit));
    if (options?.ids !== undefined) params.set("ids", options.ids.join(","));
    const qs = params.toString();
    return this.http.get<ListDocumentsResponse>(
      `${this.prefix}/documents${qs ? `?${qs}` : ""}`,
    );
  }

  // =========================================================================
  // Documents — Chunked
  // =========================================================================

  /** Upsert documents from pre-chunked data. */
  async upsertChunks(
    chunks_by_document: Record<string, string[]>,
    options?: UpsertChunksOptions,
  ): Promise<UpsertResponse> {
    return this.http.post<UpsertResponse>(`${this.prefix}/documents/chunks`, {
      chunks_by_document,
      ...options,
    });
  }

  /** Insert documents from pre-chunked data. */
  async insertChunks(
    chunks_by_document: Record<string, string[]>,
    options?: InsertChunksOptions,
  ): Promise<InsertResponse> {
    return this.http.post<InsertResponse>(
      `${this.prefix}/documents/chunks/insert`,
      { chunks_by_document, ...options },
    );
  }

  // =========================================================================
  // Search
  // =========================================================================

  /** Unified query interface (vector, keyword, or hybrid search). */
  async query(
    queryText: string,
    options?: QueryOptions,
  ): Promise<QueryResponse> {
    return this.http.post<QueryResponse>(`${this.prefix}/query`, {
      query: queryText,
      ...options,
    });
  }

  /** Query across multiple metadata columns. */
  async queryMultiColumn(
    queryText: string,
    options?: QueryMultiColumnOptions,
  ): Promise<QueryResponse> {
    return this.http.post<QueryResponse>(
      `${this.prefix}/query-multi-column`,
      { query: queryText, ...options },
    );
  }

  /** Filter documents by metadata with MongoDB-style operators. */
  async filter(
    where: Record<string, unknown>,
    options?: FilterOptions,
  ): Promise<FilterResponse> {
    return this.http.post<FilterResponse>(`${this.prefix}/filter`, {
      where,
      ...options,
    });
  }

  /**
   * Stream query results via Server-Sent Events.
   *
   * Returns an `AsyncGenerator` — consume with `for await`:
   *
   * ```ts
   * for await (const result of db.queryStream("search text")) {
   *   console.log(result.id, result.score);
   * }
   * ```
   */
  queryStream(
    queryText: string,
    options?: StreamQueryOptions,
  ): AsyncGenerator<QueryResult, void, undefined> {
    return streamQuery(this.http, this.name, queryText, options);
  }

  // =========================================================================
  // File Upload
  // =========================================================================

  /** Upload files with automatic text extraction. */
  async upload(
    files: UploadableFile[],
    options?: UploadOptions,
  ): Promise<UploadResponse> {
    return uploadFiles(this.http, this.name, files, options);
  }

  // =========================================================================
  // Schema
  // =========================================================================

  /** Get metadata schema information. */
  async getSchema(): Promise<SchemaInfoResponse> {
    return this.http.get<SchemaInfoResponse>(`${this.prefix}/schema`);
  }

  /** Update the metadata schema. */
  async updateSchema(
    metadata_schema: Record<string, MetadataFieldDefinition>,
    options?: UpdateSchemaOptions,
  ): Promise<UpdateSchemaResponse> {
    return this.http.put<UpdateSchemaResponse>(`${this.prefix}/schema`, {
      metadata_schema,
      ...options,
    });
  }

  // =========================================================================
  // Embeddings
  // =========================================================================

  /** Get embeddings for chunk IDs or custom texts. */
  async getEmbeddings(
    options: DatabaseEmbeddingsOptions,
  ): Promise<EmbeddingsResponse> {
    return this.http.post<EmbeddingsResponse>(
      `${this.prefix}/embeddings`,
      options,
    );
  }

  // =========================================================================
  // Comparison
  // =========================================================================

  /** Compare two documents by ID (returns overall similarity score). */
  async compare(docId1: string, docId2: string): Promise<CompareResponse> {
    return this.http.post<CompareResponse>(`${this.prefix}/compare`, {
      doc_id_1: docId1,
      doc_id_2: docId2,
    });
  }

  /** Compare two documents with chunk-level analysis. */
  async compareDetailed(
    docId1: string,
    docId2: string,
    options?: CompareDetailedOptions,
  ): Promise<CompareDetailedResponse> {
    return this.http.post<CompareDetailedResponse>(
      `${this.prefix}/compare/detailed`,
      { doc_id_1: docId1, doc_id_2: docId2, ...options },
    );
  }

  /** Find the k nearest neighbor documents. */
  async nearestNeighbors(
    docId: string,
    k?: number,
  ): Promise<NearestNeighborsResponse> {
    return this.http.post<NearestNeighborsResponse>(
      `${this.prefix}/nearest-neighbors`,
      { doc_id: docId, ...(k !== undefined ? { k } : {}) },
    );
  }

  /** Compute a pairwise similarity matrix for documents. */
  async similarityMatrix(
    docIds?: string[],
  ): Promise<SimilarityMatrixResponse> {
    return this.http.post<SimilarityMatrixResponse>(
      `${this.prefix}/similarity-matrix`,
      docIds ? { doc_ids: docIds } : {},
    );
  }

  // =========================================================================
  // Tuning & Maintenance
  // =========================================================================

  /** Get current SQLite tuning configuration. */
  async getTuning(): Promise<TuningResponse> {
    return this.http.get<TuningResponse>(`${this.prefix}/tuning`);
  }

  /** Apply a SQLite tuning profile. */
  async setTuning(
    profile: string,
    options?: SetTuningOptions,
  ): Promise<SetTuningResponse> {
    return this.http.put<SetTuningResponse>(`${this.prefix}/tuning`, {
      profile,
      ...options,
    });
  }

  /** Run a SQLite WAL checkpoint. */
  async checkpoint(mode?: CheckpointMode): Promise<MaintenanceResponse> {
    return this.http.post<MaintenanceResponse>(
      `${this.prefix}/maintenance/checkpoint`,
      mode ? { mode } : {},
    );
  }

  /** Run SQLite PRAGMA optimize. */
  async optimize(): Promise<MaintenanceResponse> {
    return this.http.post<MaintenanceResponse>(
      `${this.prefix}/maintenance/optimize`,
      {},
    );
  }

  /** Run SQLite VACUUM. */
  async vacuum(): Promise<MaintenanceResponse> {
    return this.http.post<MaintenanceResponse>(
      `${this.prefix}/maintenance/vacuum`,
      {},
    );
  }

  /** Run an incremental VACUUM. */
  async incrementalVacuum(pages?: number): Promise<MaintenanceResponse> {
    return this.http.post<MaintenanceResponse>(
      `${this.prefix}/maintenance/incremental_vacuum`,
      pages !== undefined ? { pages } : {},
    );
  }

  /** Get auto-tuning recommendations (optionally apply them). */
  async autoTune(options?: AutoTuneOptions): Promise<AutoTuneResponse> {
    return this.http.post<AutoTuneResponse>(
      `${this.prefix}/auto-tune`,
      options ?? {},
    );
  }

  /** Checkpoint only if the WAL file exceeds a size threshold. */
  async checkpointIfWalLarge(
    thresholdMb?: number,
  ): Promise<MaintenanceResponse> {
    return this.http.post<MaintenanceResponse>(
      `${this.prefix}/maintenance/checkpoint_if_large`,
      thresholdMb !== undefined ? { threshold_mb: thresholdMb } : {},
    );
  }

  // =========================================================================
  // Fact-Check
  // =========================================================================

  /** Check text against this database for factual grounding. */
  async factCheck(
    text: string,
    options?: FactCheckOptions,
  ): Promise<FactCheckResponse> {
    return this.http.post<FactCheckResponse>(`${this.prefix}/factcheck`, {
      text,
      ...options,
    });
  }
}
