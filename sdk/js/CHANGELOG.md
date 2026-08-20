# Changelog

All notable changes to `@localvectordb/sdk` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CreateDatabaseOptions.reranker` — persist a default reranker on the database
  at creation, applied by every query server-side. The resolved (redacted)
  config comes back on `CreateDatabaseConfig.reranker`; servers older than the
  feature ignore the key, so check the echo rather than assuming it took.
- Query options gain `rerank?: boolean` — pass `false` to disable the
  database's persisted default reranker for one call.

## [0.1.1] - 2026-08-19

### Added

- Query options the server already supported but the SDK could not express:
  - `search_level` (`"chunks" | "sections" | "documents"`) — hierarchical
    retrieval granularity, previously unreachable from TypeScript.
  - `reranker_config` / `rerank_k` — cross-encoder reranking of the candidate
    pool, mirroring the library's `query(reranker_config=...)`.
- `DocumentScoringMethod` gains the missing `"auto"` (the server default:
  `best` for vector search, `frequency_boost` otherwise) and `"percentile"`
  (configured via `document_scoring_options: { percentile: 0.9 }`).

### Changed

- `return_type` documentation now states the omission semantics (the server
  answers in the natural unit of `search_level`) and that streaming rejects
  `"sections"`.

## [0.1.0] - 2026-08-19

Initial public release. Targets the LocalVectorDB **v0.1.0** server HTTP API
(all routes under `/api/v1`).

### Added

- `LocalVectorDBClient` — database management (`createDatabase`, `listDatabases`,
  `deleteDatabase`), health/system info, cross-database `globalSearch`, and
  `embeddings`.
- `DatabaseHandle` — full per-database surface: document CRUD (`upsert`, `insert`,
  `get`, `update`, `delete`, `count`, `exists`, `list`), pre-chunked ingestion
  (`upsertChunks`, `insertChunks`), search (`query`, `queryMultiColumn`, `filter`),
  SSE streaming (`queryStream`), file upload with server-side extraction (`upload`),
  schema management, embeddings, comparison, and tuning/maintenance.
- Assembled-context query options: `context_window`, `context_unit`
  (`chunks`/`tokens`/`words`/`characters`), and `context_truncate`.
- Typed error hierarchy mirroring server error codes (`LocalVectorDBError` and
  subclasses), with automatic retry for 5xx/network/timeout errors.
- Dual ESM + CommonJS builds; zero runtime dependencies; Node.js 18+, browser,
  Deno, and Bun support.

[0.1.0]: https://github.com/thomas-villani/localvectordb/releases/tag/v0.1.0
