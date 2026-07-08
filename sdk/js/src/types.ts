// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------

export type MetadataFieldType =
  | "text"
  | "integer"
  | "real"
  | "boolean"
  | "date"
  | "json";

export type SearchType = "vector" | "keyword" | "hybrid";

export type ContextUnit = "chunks" | "tokens" | "words" | "characters";

export type ReturnType =
  | "documents"
  | "chunks"
  | "sections"
  | "context"
  | "enriched";

export type DocumentScoringMethod =
  | "best"
  | "average"
  | "worst"
  | "weighted_average"
  | "frequency_boost"
  | "harmonic_mean"
  | "diminishing_returns"
  | "statistical"
  | "robust_mean"
  | "percentile"
  | "geometric_mean";

export type QueryResultType =
  | "document"
  | "chunk"
  | "section"
  | "context"
  | "enriched"
  | "group"
  | "aggregation";

export type CheckpointMode = "PASSIVE" | "FULL" | "RESTART" | "TRUNCATE";

export type ErrorCode =
  | "VALIDATION_ERROR"
  | "INVALID_FILTER"
  | "DATABASE_NOT_FOUND"
  | "DOCUMENT_NOT_FOUND"
  | "DUPLICATE_DOCUMENT_ID"
  | "DATABASE_ALREADY_EXISTS"
  | "EMBEDDING_ERROR"
  | "OLLAMA_NOT_AVAILABLE"
  | "DATABASE_CONNECTION_ERROR"
  | "CONFIGURATION_ERROR"
  | "DATABASE_ERROR"
  | "INTERNAL_ERROR"
  | "FEATURE_NOT_AVAILABLE"
  | "BATCH_SIZE_EXCEEDED"
  | (string & {}); // allow unknown codes while preserving autocomplete

// ---------------------------------------------------------------------------
// Core data types (matching server JSON serialization)
// ---------------------------------------------------------------------------

export interface ChunkPosition {
  start: number;
  end: number;
  line: number;
  column: number;
  end_line: number;
  end_column: number;
}

export interface Chunk {
  content: string;
  position: ChunkPosition;
  tokens: number;
  index: number;
}

export interface Document {
  id: string;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  content_hash: string | null;
}

export interface QueryResult {
  id: string;
  score: number;
  type: QueryResultType;
  content: string;
  metadata: Record<string, unknown>;
  document_id?: string;
  position?: ChunkPosition;
}

export interface MetadataFieldDefinition {
  type: MetadataFieldType;
  indexed?: boolean;
  required?: boolean;
  default_value?: unknown;
  embedding_enabled?: boolean;
  fts_enabled?: boolean;
}

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

export interface ClientConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  maxRetries?: number;
  retryDelay?: number;
}

// ---------------------------------------------------------------------------
// Options types
// ---------------------------------------------------------------------------

export interface CreateDatabaseOptions {
  metadata_schema?: Record<string, MetadataFieldDefinition>;
  database?: {
    chunking_method?: string;
    chunk_size?: number;
    chunk_overlap?: number;
    enable_fts?: boolean;
    enable_gpu?: boolean;
    sqlite_profile?: string;
    sqlite_pragma_overrides?: Record<string, unknown>;
  };
  embedding?: {
    provider?: string;
    model?: string;
    config?: Record<string, unknown>;
  };
}

export interface UpsertOptions {
  metadata?: Record<string, unknown> | Record<string, unknown>[];
  ids?: string | string[];
  batch_size?: number;
  similarity_threshold?: number;
}

export interface InsertOptions extends UpsertOptions {
  errors?: "raise" | "ignore";
}

/**
 * Options common to every search surface.
 *
 * Note: `semantic_filters` lives on {@link QueryOptions} only. The server accepts
 * it on `/query` and `/search` (global) but rejects it on the streaming and
 * multi-column endpoints, whose request bodies forbid unknown fields.
 */
export interface BaseQueryOptions {
  search_type?: SearchType;
  return_type?: ReturnType;
  k?: number;
  score_threshold?: number;
  filters?: Record<string, unknown>;
  vector_weight?: number;
  /**
   * Size of the assembled context for `return_type` `"context"`/`"enriched"`,
   * measured in {@link BaseQueryOptions.context_unit}. In the default `"chunks"`
   * unit this is the number of surrounding/similar chunks; with a token/word/
   * character unit it is an approximate budget for the assembled content.
   */
  context_window?: number;
  /**
   * Unit in which `context_window` is measured. Defaults to `"chunks"`. With a
   * non-chunk unit, whole neighbouring/similar chunks are added greedily until
   * the next would exceed the budget (the matched chunk is always kept).
   */
  context_unit?: ContextUnit;
  /**
   * When `true` and `context_unit` is a token/word/character budget, hard-truncate
   * the assembled context to exactly the budget. Defaults to `false` (whole chunks
   * only). The only way to guarantee the budget is never exceeded when a single
   * chunk is larger than it.
   */
  context_truncate?: boolean;
  semantic_dedup_threshold?: number;
  document_scoring_method?: DocumentScoringMethod;
  document_scoring_options?: Record<string, unknown>;
}

export interface QueryOptions extends BaseQueryOptions {
  semantic_filters?: SemanticFilter[];
}

export interface SemanticFilter {
  field: string;
  concept: string;
  threshold?: number;
  metric?: string;
}

export interface QueryMultiColumnOptions extends BaseQueryOptions {
  columns?: string[];
}

export interface StreamQueryOptions extends BaseQueryOptions {
  batch_size?: number;
}

export interface FilterOptions {
  order_by?: string;
  limit?: number;
  offset?: number;
}

export interface ListDocumentsOptions {
  offset?: number;
  limit?: number;
  ids?: string[];
}

export interface CountOptions {
  filters?: Record<string, unknown>;
}

export interface UpsertChunksOptions {
  /**
   * Metadata applied to the reconstructed documents, keyed by document id.
   * The chunk endpoints take a single mapping (not a per-item array).
   */
  metadata?: Record<string, unknown>;
  batch_size?: number;
  similarity_threshold?: number;
}

export interface InsertChunksOptions extends UpsertChunksOptions {
  errors?: "raise" | "ignore";
}

export interface UpdateSchemaOptions {
  drop_columns?: boolean;
  column_mapping?: Record<string, string>;
}

export interface SetTuningOptions {
  overrides?: Record<string, unknown>;
  persist?: boolean;
}

/**
 * Workload characteristics used to bias auto-tuning recommendations.
 *
 * All fields are optional; omitted fields fall back to balanced server-side
 * defaults. Mirrors the server's `WorkloadProfile`.
 */
export interface WorkloadProfile {
  workload_type?:
    | "read_heavy"
    | "write_heavy"
    | "balanced"
    | "batch_ingest"
    | "real_time";
  document_size?: "small" | "medium" | "large";
  concurrent_users?: number;
  durability_level?: "critical" | "high" | "normal" | "low";
  memory_constraint?: "generous" | "moderate" | "limited";
}

export interface AutoTuneOptions {
  workload?: WorkloadProfile;
  apply?: boolean;
}

export interface CompareDetailedOptions {
  chunk_threshold?: number;
}

export interface GlobalSearchOptions extends QueryOptions {
  databases?: string[];
}

export type DatabaseEmbeddingsOptions =
  | { ids: string | string[] }
  | { texts: string | string[] };

/**
 * A file that can be uploaded to the server.
 *
 * - Browser: use native `File` from `<input type="file">` or drag-and-drop
 * - Node.js 20+: use native `File`
 * - Node.js 18+: use `{ name, data, type }` with a `Blob` or `Uint8Array`
 */
export type UploadableFile =
  | File
  | Blob
  | { name: string; data: Blob | ArrayBuffer | Uint8Array; type?: string };

export interface UploadOptions {
  metadata?: Record<string, unknown> | Record<string, unknown>[];
  batch_size?: number;
  ids?: string | string[];
  mode?: "upsert" | "insert";
  errors?: "raise" | "ignore";
  similarity_threshold?: number;
  use_filename_as_id?: boolean;
}

// ---------------------------------------------------------------------------
// Response types (matching server JSON)
// ---------------------------------------------------------------------------

/**
 * Server-authoritative database configuration echoed back on create.
 *
 * The server resolves the requested embedding/chunking settings (filling in
 * defaults and the embedding dimension) and returns them here — treat this as
 * the source of truth rather than the values sent in the request.
 */
export interface CreateDatabaseConfig {
  name: string;
  embedding_provider: string;
  embedding_model: string;
  embedding_dimension: number;
  chunking_method: string;
  chunk_size: number;
  chunk_overlap: number;
  fts_enabled: boolean;
  metadata_schema: Record<string, MetadataFieldDefinition>;
  [key: string]: unknown;
}

export interface CreateDatabaseResponse {
  message: string;
  status: string;
  config: CreateDatabaseConfig;
}

export interface DatabaseListResponse {
  databases: string[];
  count: number;
}

export interface DatabaseInfoResponse {
  name: string;
  stats: Record<string, unknown>;
  config: Record<string, unknown>;
}

export interface DeleteDatabaseResponse {
  message: string;
  status: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  ollama_available: boolean;
  timestamp: string;
}

export interface SystemResourcesResponse {
  system_resources: Record<string, unknown>;
  status: string;
}

export interface UpsertResponse {
  message: string;
  ids: string[];
}

export interface InsertResponse {
  message: string;
  ids: string[];
}

export interface UpdateResponse {
  message: string;
  updated: boolean;
}

export interface DeleteResponse {
  message: string;
  deleted_count: number;
}

export interface BatchDeleteResponse {
  message: string;
  deleted_count: number;
  failed_ids: string[];
}

export interface CountResponse {
  count: number;
}

/**
 * `exists` is a positional array aligned with `ids` (index i corresponds to
 * `ids[i]`), not a keyed map.
 */
export interface ExistsResponse {
  exists: boolean[];
  ids: string[];
}

export interface PageInfo {
  limit: number;
  offset: number;
  total: number;
  has_more: boolean;
}

export interface ListDocumentsResponse {
  documents: Document[];
  /** Present when listing (no explicit `ids`). */
  pagination?: PageInfo;
  /** Present when fetching specific `ids`. */
  returned_ids?: string[];
  /** Present when fetching specific `ids`. */
  missing_ids?: string[];
}

export interface QueryResponse {
  results: QueryResult[];
  search_type: string;
  return_type: string;
  total_results: number;
  processing_info?: Record<string, unknown>;
}

export interface FilterResponse {
  documents: Document[];
  count: number;
  filter_info?: Record<string, unknown>;
}

export interface GlobalSearchResponse {
  /** Per-database result lists, keyed by database name. */
  results_by_database: Record<string, QueryResult[]>;
  search_type: string;
  return_type: string;
}

export interface SchemaInfoResponse {
  database: string;
  schema_info: Record<string, unknown>;
  status: string;
}

export interface UpdateSchemaResponse {
  message: string;
  status: string;
  changes: Record<string, unknown>;
  new_schema: Record<string, unknown>;
}

export interface EmbeddingsResponse {
  embeddings: number[][];
  provider?: string;
  model?: string;
}

export interface CompareResponse {
  doc_id_1: string;
  doc_id_2: string;
  similarity: number;
  status: string;
}

export interface CompareDetailedResponse {
  doc_id_1: string;
  doc_id_2: string;
  overall_similarity: number;
  chunk_similarities: unknown[];
  common_themes?: string[];
  unique_to_doc1?: string[];
  unique_to_doc2?: string[];
  status: string;
}

export interface NearestNeighborsResponse {
  doc_id: string;
  k: number;
  results: QueryResult[];
  total_results: number;
  status: string;
}

export interface SimilarityMatrixResponse {
  doc_ids: string[];
  matrix: number[][];
  similarity_pairs?: Array<{
    doc_id_1: string;
    doc_id_2: string;
    similarity: number;
  }>;
  status: string;
}

export interface TuningResponse {
  database: string;
  tuning: Record<string, unknown>;
  status: string;
}

export interface SetTuningResponse {
  database: string;
  message: string;
  tuning: Record<string, unknown>;
  status: string;
}

export interface MaintenanceResponse {
  database: string;
  message: string;
  status: string;
  [key: string]: unknown;
}

export interface AutoTuneResponse {
  database: string;
  recommendation: Record<string, unknown>;
  status: string;
}

export interface UploadResponse {
  message: string;
  files_processed: number;
  document_ids: string[];
  extraction_results: unknown[];
  extraction_summary: Record<string, unknown>;
  status: string;
}

// ---------------------------------------------------------------------------
// Server error envelope
// ---------------------------------------------------------------------------

export interface ServerErrorPayload {
  error: {
    message: string;
    code: ErrorCode;
    timestamp: string;
    request_id: string;
    details: Record<string, unknown>;
    recoverable: boolean;
  };
}
