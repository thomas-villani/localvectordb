# Changelog

All notable changes to `@localvectordb/sdk` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - Unreleased

Initial public release. Targets the LocalVectorDB **v0.1.0** server HTTP API
(all routes under `/api/v1`).

### Added

- `LocalVectorDBClient` — database management (`createDatabase`, `listDatabases`,
  `deleteDatabase`), health/system info, cross-database `globalSearch`,
  `embeddings`, and `factCheck`.
- `DatabaseHandle` — full per-database surface: document CRUD (`upsert`, `insert`,
  `get`, `update`, `delete`, `count`, `exists`, `list`), pre-chunked ingestion
  (`upsertChunks`, `insertChunks`), search (`query`, `queryMultiColumn`, `filter`),
  SSE streaming (`queryStream`), file upload with server-side extraction (`upload`),
  schema management, embeddings, comparison, tuning/maintenance, and fact-checking.
- Assembled-context query options: `context_window`, `context_unit`
  (`chunks`/`tokens`/`words`/`characters`), and `context_truncate`.
- Typed error hierarchy mirroring server error codes (`LocalVectorDBError` and
  subclasses), with automatic retry for 5xx/network/timeout errors.
- Dual ESM + CommonJS builds; zero runtime dependencies; Node.js 18+, browser,
  Deno, and Bun support.

[0.1.0]: https://github.com/thomas-villani/localvectordb/releases/tag/v0.1.0
