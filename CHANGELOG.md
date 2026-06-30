# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **File extraction now uses [all2md](https://all2md.readthedocs.io/).** The
  former per-format extractors (DOCX, PPTX, XLSX, PDF, HTML, XML, EPUB, RTF,
  plain text) are replaced by a single `All2MdExtractor` that delegates to
  all2md, covering 20+ document formats and 200+ source/text formats. The
  plugin interface (`BaseExtractor`, `ExtractorRegistry`, the
  `localvectordb.file_extractors` entry-point group) is unchanged, so custom
  extractors continue to work.
- **Extracted content is now Markdown** instead of plain text, preserving
  document structure (headings, tables, lists) for better chunk boundaries.
  Re-ingesting existing corpora will change the stored text.
- Hardened extraction defaults for untrusted uploads: remote fetching and local
  file access disabled, HTML dangerous elements stripped, attachments skipped;
  file-size and ZIP-bomb guards retained.
- Section detection and the `sections` chunking strategy now ignore Markdown
  headers (`#`) that appear inside fenced code blocks (```` ``` ```` and
  `~~~`), so example snippets in extracted documents no longer create spurious
  section boundaries.

### Added

- `[extraction]` server configuration section (and `LVDB_EXTRACTION_*`
  environment variables) exposing extraction security options
  (`allow_remote_fetch`, `allowed_hosts`, `strip_dangerous_elements`,
  `attachment_mode`).
- `file-extraction-ocr` extra for OCR of scanned PDFs (Tesseract).

### Removed

- Native extractor modules (`office_extractors`, `pdf_extractors`,
  `web_extractors`, `other_extractors`, `text_extractors`) and their direct
  dependencies (`python-docx`, `python-pptx`, `openpyxl`, `pdfplumber`, `pypdf`,
  `beautifulsoup4`, `lxml`, `defusedxml`, `ebooklib`, `striprtf`); these formats
  are now handled by all2md.
- The `file-extraction-office`, `file-extraction-pdf`, `file-extraction-web`,
  and `file-extraction-rtf-epub` extras. Common formats are now covered by the
  base install; the `file-extraction` extra now installs all2md's
  extended/niche format parsers.

## [0.1.0] - 2026-06-28

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
- File upload with text extraction (PDF, DOCX, PPTX, XLSX, RTF, EPUB, HTML, XML)
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
