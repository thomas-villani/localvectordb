# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-02

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
- API key authentication with permission levels (read-only, read-write, admin)
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
- Document comparison and nearest-neighbor endpoints
- LLM-based fact-checking module
- Cursor-based pagination for async query results
- Backup and restore with incremental and point-in-time recovery
- Database migration engine and schema versioning
- SQLite tuning profiles for different workloads
- MCP (Model Context Protocol) server integration
- CLI tool (`lvdb`) for database management, server control, and configuration
- Redis integration for distributed multi-worker deployments
- Comprehensive test suite with 85%+ coverage requirement
- End-to-end release-qualification suite (`scripts/e2e/`) exercising real
  embedding backends (Ollama, Sentence Transformers) and real PDF/DOCX/XLSX/
  HTML/Markdown documents against the library, file ingestion, HTTP server,
  and CLI
- Sphinx documentation with autodoc
- CI/CD pipeline with linting, type checking, security scanning, and tests

### Fixed

Issues found during pre-release end-to-end qualification with real embedding
providers (the mocked test suite could not catch these):

- Server search endpoints (`/query`, `/search/*`, `/query-multi-column`,
  `/query_builder`, global `/search`) called sync query/embedding paths on the
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
