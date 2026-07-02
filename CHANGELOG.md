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
- Sphinx documentation with autodoc
- CI/CD pipeline with linting, type checking, security scanning, and tests
