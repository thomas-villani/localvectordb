# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`query(search_level="sections"|"documents")` now raises `ValueError` on a
  database created without `hierarchical_embeddings=True`**, instead of silently
  returning chunk-level results. The old behaviour handed back plausible
  wrong-level results, which reads as "the feature does nothing" rather than
  "the feature is switched off". `"fused"` already raised; all three levels are
  now consistent, in `query()` and `query_async()` alike. If you were relying on
  the silent fallthrough, pass `search_level="chunks"` (the default) explicitly.
- `lvdb create --chunking-method` now offers every registered chunker (it was a
  hardcoded list missing `paragraphs` and `code-blocks` — the latter documented
  in the README's own code-repository example but unreachable from the CLI).
  `lvdb db <name> search --search-level` gains `fused`.

### Added

- Document **patch API** for in-place edits — change part of a stored document
  without re-sending the whole content. Exact find/replace with a uniqueness
  requirement (the contract coding agents already handle), plus `splice` /
  `append` / `prepend` ops resolved against the original content, validated
  non-overlapping, and applied atomically. Surfaced across every layer:
  - `LocalVectorDB.patch()` / `patch_async()` and `RemoteVectorDB` equivalents,
    returning `PatchResult(updated, new_hash, ops_applied)`.
  - `PATCH /databases/{db}/documents/{doc_id}` gains additive `ops` +
    `expect_hash` fields (mutually exclusive with `content`); `409 HASH_CONFLICT`
    on a stale precondition, `422 PATCH_FAILED` on an unmatched/ambiguous/
    overlapping op.
  - `patch_document` MCP tool exposing the `old_string`/`new_string` edit
    contract for agents.
  - `lvdb db <name> patch <doc_id> --find/--replace/--append/--prepend/--expect-hash`.
  - JavaScript SDK `database.patch()` with typed `PatchOp` / `PatchOptions`.
- Optional `expect_hash` precondition on patches for optimistic concurrency:
  fail instead of clobbering a concurrent write. New `PatchConflictError` and
  `PatchError` exceptions (mirrored in the JS SDK as `PatchConflictError` /
  `PatchFailedError`).
- `OllamaEmbeddings` gains `num_ctx`, `num_batch`, and `truncate` options
  (settable via `embedding_config`). Ollama's `/api/embed` caps input at
  `n_batch` (default **2048**) regardless of `num_ctx`, silently truncating
  longer inputs — so raising `num_ctx` alone does nothing for embeddings past
  2048. `num_batch` auto-defaults to `num_ctx` so a raised context actually
  takes effect (e.g. embed full 8192-token inputs with a long-context encoder).

### Fixed

- Raw-span section/document embeddings now size their pooling window to the
  encoder's own context (`num_ctx` / `max_input_tokens`) instead of a fixed
  ~24k-char (~8k-token) window. On a small-context encoder (e.g. a 2k-context
  local model) an over-long section is windowed and mean-pooled to represent it
  in full, rather than each 24k window overflowing and being silently truncated.

## [0.1.0] - 2026-07-09

### Added

- Document-first API with automatic position-aware chunking and reconstruction
- SQLite + FAISS dual storage backend for documents, metadata, and vectors
- Unified `query()` interface supporting vector, keyword (FTS5), and hybrid search
- Strongly typed metadata schema with TEXT, INTEGER, REAL, BOOLEAN, DATE, JSON types
- Pluggable embedding providers: Ollama, OpenAI, Google, Jina, HuggingFace, Sentence Transformers
- Pluggable reranker providers: Jina, Sentence Transformers, HuggingFace
- Multiple chunking strategies: sentences, tokens, words, paragraphs, sections, code blocks
- SQL-like query builder for metadata filtering
- FastAPI HTTP server with multi-database management
- API key authentication with permission levels (read-only, read-write)
- Rate limiting, CORS, and security headers middleware
- SSE streaming for query results
- File upload with text extraction via [all2md](https://all2md.readthedocs.io/):
  a single `All2MdExtractor` covering 20+ document formats and 200+ source/text
  formats, emitting Markdown to preserve document structure (headings, tables,
  lists) for better chunk boundaries. The plugin interface (`BaseExtractor`,
  `ExtractorRegistry`, the `localvectordb.file_extractors` entry-point group)
  supports custom extractors.
- Hardened extraction defaults for untrusted uploads (remote fetching and local
  file access disabled, HTML dangerous elements stripped, attachments skipped;
  file-size and ZIP-bomb guards), configurable via the `[extraction]` server
  config section and `LVDB_EXTRACTION_*` environment variables.
- `file-extraction-ocr` extra for OCR of scanned PDFs (Tesseract).
- Section detection and the `sections` chunking strategy ignore Markdown headers
  inside fenced code blocks, so code snippets don't create spurious sections.
- Raw-span section vectors for hierarchical databases: a new
  `section_vector_strategy` option (`"rawspan"` | `"centroid"`) controls how a
  section is represented in the section index. `"rawspan"` embeds the section's
  actual text (window-mean-pooled for over-long spans) instead of averaging its
  chunk vectors, which retrieves better on real, section-structured documents.
  New hierarchical databases default to `"rawspan"`; databases created before this
  option existed keep `"centroid"`, and the resolved value is persisted per
  database. Off by default (requires `hierarchical_embeddings=True`).
- `search_level="fused"` retrieval: blends chunk retrieval with section (raw-span)
  retrieval via relative-score fusion, tunable with a `section_weight` scalar
  (0 = chunk-only, 1 = section-only; default 0.65). Supports `return_type`
  `"documents"` (the measured win) and `"sections"`. Local databases only for now;
  remote/streaming raise a clear error. The default chunk-only retrieval path is
  unchanged.
- Document comparison and nearest-neighbor endpoints
- LLM-based fact-checking module
- Cursor-based pagination for async query results
- Backup and restore with incremental and point-in-time recovery
- Database migration engine and schema versioning
- SQLite tuning profiles for different workloads
- MCP (Model Context Protocol) server integration
- CLI tool (`lvdb`) for database management, server control, and configuration
- Read-only multi-worker read fan-out: a `mmap_index` setting memory-maps the
  FAISS index (`IO_FLAG_MMAP`) so many workers share one page-cached copy instead
  of each loading a private, RAM-resident copy. A memory-mapped database is
  read-only and refuses writes. A shared cachelib/Redis registry coordinates the
  set of database names across workers. The deployment model is single-writer:
  route all writes to one writer process (`mmap_index = false`).
- The FAISS index file is rewritten only when the in-memory index has actually
  changed, so a database that only served reads is never re-persisted (and, under
  read fan-out, never races another worker on the shared index file) on close or
  idle-eviction.
- A hardened `Dockerfile` (pinned base image, dependencies isolated in a virtualenv,
  non-root user, `HEALTHCHECK` against `/api/v1/health`), built and booted in CI on every
  pull request so it cannot drift.
- Comprehensive test suite with 85%+ coverage requirement
- End-to-end release-qualification suite (`scripts/e2e/`) exercising real
  embedding backends (Ollama, Sentence Transformers) and real PDF/DOCX/XLSX/
  HTML/Markdown documents against the library, file ingestion, HTTP server,
  and CLI
- Sphinx documentation with autodoc
- CI/CD pipeline with linting, type checking, security scanning, and tests

### Changed

Breaking HTTP/API contract changes finalized before the v0.1.0 freeze (relevant
to anyone tracking the pre-release):

- **HTTP routes**: all per-database endpoints moved under `/api/v1/databases/{db_name}/...`
  (for example `/api/v1/databases/{db_name}/query`). Global endpoints
  (`/api/v1/databases`, `/api/v1/search`, `/api/v1/embeddings`, `/api/v1/health`,
  `/api/v1/system/resources`, `/api/v1/upload/...`) are unchanged. Database names are
  now namespaced under `/databases/`, so no database names are reserved.
- **Global search**: `POST /api/v1/search` now returns the per-database map under
  `results_by_database` (was `results`).
- **Default `vector_weight` changed from `0.7` to `0.5`** for hybrid search (the default
  `search_type`). This changes hybrid ranking for callers who do not pass `vector_weight`
  explicitly. Once T1.1's relative-score fusion made `vector_weight` an actual blend, an
  even weighting measured better on *both* evaluation corpora — SciFact `frequency_boost`
  nDCG@10 0.6940 → 0.7090 (+2.2% relative) and NFCorpus 0.3298 → 0.3367 (+2.1%) — where it
  is also the best configuration in the entire sweep. Pass `vector_weight=0.7` to restore
  the previous behaviour. Applies to the Python API, HTTP API, MCP server, and the
  `lvdb db <name> search --vector-weight` CLI default.
- **Default server port** changed from `5000` to `8000` (5000 collides with the macOS
  AirPlay Receiver).
- Single-document delete (`DELETE /api/v1/databases/{db_name}/documents/{doc_id}`) is
  idempotent — deleting a missing document succeeds instead of erroring.

### Removed

- Remote/HTTP fact-checking: the `/factcheck` HTTP endpoints and the
  `RemoteVectorDB.fact_check()` client method are removed. Fact-checking ("reverse RAG")
  remains available as a local-only feature via the `FactChecker` class over
  `LocalVectorDB`.

### Fixed

- `server.rate_limit_storage_uri` was defined but never read, so slowapi silently fell
  back to a per-process in-memory store and the effective limit was N× the configured
  one under N workers. It is now passed to the limiter, and a shared store (e.g. Redis)
  enforces one limit across all workers.
- Backups could capture a mutually inconsistent pair of stores. SQLite and the FAISS
  index are copied separately, so a write landing between the two could produce a backup
  whose SQLite rows referenced vectors absent from the copied index (dangling rows,
  which require re-embedding to recover). Passing the live database —
  `BackupManager(path, db=db)` — now holds its write lock and flushes the index for the
  duration of the snapshot. The path-only form is unchanged and is documented as safe
  only for a quiescent or closed database.
- Persisting the index could fail with `PermissionError` on Windows. `os.replace` is the
  final step of writing the index, and it intermittently fails with `[WinError 5] Access
  is denied` when any process holds a transient handle on the target — a virus scanner,
  the search indexer, or simply another process reading the index (a backup copying it,
  a reader worker opening it). The error propagated out of `save()`/`close()`, leaving
  the index unwritten while SQLite had already committed. It is now retried with bounded
  exponential backoff.
- `PATCH /databases/{db}/documents/{doc_id}` conflated "nothing to update" with
  "document not found", inverting both outcomes. `update()` returns `False` for a no-op
  and raises `DocumentNotFoundError` for a missing document, but the route reported the
  no-op as `404 DOCUMENT_NOT_FOUND` — on a document that exists — while the missing
  document raised past the route into the generic 500 branch (`DocumentNotFoundError`
  has no mapping in `standardize_error_response`). A no-op is now `200 {"updated": false}`
  and a missing document is `404 DOCUMENT_NOT_FOUND`.
- `RemoteVectorDB.update()` / `update_async()` swallowed a 404 into a `False` return, so a
  missing document was indistinguishable from "no updates needed" and the remote backend
  diverged from `LocalVectorDB.update()`, which raises `DocumentNotFoundError`. Both now
  raise, and `False` means only "no updates needed". The JavaScript SDK's
  `database.update()` is reconciled the same way (it now throws `DocumentNotFoundError`
  instead of resolving `{updated: false}`). The `update()`/`update_async()` contract is
  now stated on the abstract base so both backends are held to it.
- `RemoteVectorDB.update()` / `update_async()` short-circuited on `if not content and not
  metadata`, so `content=""` (clear a document) and `metadata={}` were silently dropped
  client-side and never reached the server. They now test against `None`.
- The `update_document` MCP tool discarded `update()`'s return value and always reported
  success, so an agent could not distinguish "my edit landed" from "nothing changed". It
  now returns an `updated` flag.

Issues found during pre-release end-to-end qualification with real embedding
providers (the mocked test suite could not catch these):

- Server search endpoints (`/query`, `/search/*`, `/query-multi-column`,
  `/query-builder`, global `/search`) called sync query/embedding paths on the
  event loop, so vector and hybrid search failed with every real embedding
  provider; they now use the async query APIs
- SSE streaming endpoint (`/query/stream`) did not await `query_cursor_async`
  and iterated the cursor incorrectly
- Server-side database creation forwarded unset `api_key`/`base_url` to
  embedding providers that don't accept them, breaking `ollama` database
  creation over HTTP
- `/documents/count` and document listing called `db.count()` with a
  nonexistent `where` keyword and always returned HTTP 500
- Server config, request validation, and CLI rejected every embedding
  provider except `ollama`/`openai`; they now accept any provider registered
  with `EmbeddingRegistry`
- `$contains`/`$not_contains` metadata filters on JSON fields generated SQL
  with two placeholders but bound one parameter, crashing every such filter
- JSON metadata fields were returned as raw serialized strings from
  `get()`/`filter()`, which also broke partial `update()` on any document
  with a JSON-typed field
- `/health` performed an inline Ollama check with a 60-second timeout and
  three retries (minutes-long hangs when Ollama was down); it now uses a
  single 2-second attempt
- Default Ollama base URL changed from `localhost` to `127.0.0.1` (matching
  Ollama's default bind address) to avoid a ~2.5 s IPv6 resolution stall per
  connection on Windows
- README/docs metadata-filter examples used unsupported operator spellings
  (`contains`, `>=`) instead of `$contains`/`$gte`

Pre-release consistency fixes:

- `query(filters=...)`, `query_multi_column(filters=...)`, and
  `nearest_neighbors(filters=...)` silently returned no matches for filter
  fields not in the metadata schema or unsupported operators; they now raise
  `MetadataFilterError` (a `DatabaseError`/`ValueError` subclass) up front,
  matching `filter(where=...)` behavior
- Invalid filter specs over HTTP returned 500 `DATABASE_ERROR`; they now
  return 400 `INVALID_FILTER` (a client error), the Python client raises
  `MetadataFilterError` for it, and clients no longer waste retries on them
- `upsert()` silently dropped metadata fields not in the metadata schema; it
  now logs a warning naming the dropped fields
- `lvdb db <name> <cmd> --help` required the database (and DB folder) to
  exist; the database is now opened lazily on first use so help always works
- A malformed or invalid config file crashed the CLI with a raw traceback; it
  now prints a friendly error and exits with the configuration-error code (2)
- `lvdb db <name> add <file>` assigned generated `doc_N` ids while the
  library's `upsert_from_file()` used the filename stem; the CLI now also
  defaults file inputs to the filename stem (repeated stems in one batch fall
  back to generated ids)

Final pre-release contract hardening (packaging, API, HTTP, and CLI surfaces
frozen for v0.1.0):

- **Packaging**: a base `pip install localvectordb` crashed on import because
  `sqlite_tuning` imported `psutil`, which was only declared in the `[server]`
  and `[benchmark]` extras; `psutil` is now a core dependency. `click` is
  declared explicitly in `[server]`. Importing `localvectordb_server` (and the
  `lvdb` console script) without the `[server]` extra now raises a clear error
  naming the extra instead of a bare `ModuleNotFoundError`.
- **Factory**: `VectorDB(name, "http://...", timeout=...)` raised `TypeError`
  because the remote client's parameter is `request_timeout`; the factory now
  documents and forwards the real remote parameter names.
- **Remote comparison parity**: `RemoteVectorDB.compare_documents_detailed()`
  and `pairwise_similarity_matrix()` returned raw dicts (and the server
  serialized fields the result dataclass never had, dropping the real data);
  they now return the same `DocumentComparisonResult` /
  `DocumentSimilarityMatrix` dataclasses as `LocalVectorDB`. `nearest_neighbors`
  gained the `score_threshold`/`filters` parameters on the remote client and
  server. Removed the remote-only legacy `hybrid_query()`/`keyword_search()`.
- **HTTP contract**: rate-limit (429) responses now use the standard
  `{"error": {...}}` envelope (the stock slowapi body broke the client);
  `query_builder` path is hyphenated (`query-builder`); `PATCH` added to the
  default CORS methods; `DELETE` on a missing database returns 404 instead of
  200; SSE error payloads no longer leak internal exception text.
- **CLI**: failing `tuning`/`maintenance`/`backup verify`/`backup pitr`/
  `migrate`/`db get`/`db delete`/`delete` invocations now exit non-zero;
  machine-output is unified on `--format/-f {table,json}` (with `-j` as a
  shortcut for `--format json`), and `-o/--output` reserved for output files;
  `--help` works
  without a config file and `lvdb serve` falls back to localhost defaults;
  `config init --cors-origins` now persists; and `lvdb db <name> add` errors on
  a path-like argument that does not exist instead of silently storing it as
  text (use `--text` to force literal text).
