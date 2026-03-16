# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

In order to properly execute any of the code in this repo, you will need to invoke the executables located in the virtual environment.

The path to the python executable in the virtual environment is:
./.venv/Scripts/python.exe  (relative to project root)
/mnt/c/Users/thomas.villani/pycharmprojects/localvectordb/.venv/Scripts/python.exe  (absolute)


### Testing
```bash
# Run all tests
./.venv/Scripts/pytest.exe

# Run tests with coverage
./.venv/Scripts/pytest.exe --cov=localvectordb --cov-report=html --cov-report=term-missing

# Run fast tests only (skip slow/network tests)
./.venv/Scripts/pytest.exe -m "not slow and not network"

# Run specific test categories
./.venv/Scripts/pytest.exe -m unit                    # Unit tests only
./.venv/Scripts/pytest.exe -m integration            # Integration tests only
./.venv/Scripts/pytest.exe -m performance           # Performance tests only

# Run tests in parallel
./.venv/Scripts/pytest.exe -n auto

# Run specific test file
./.venv/Scripts/pytest.exe tests/test_core.py
```

### Linting, Formatting, and Type Checking
```bash
# Run ruff linter
./.venv/Scripts/ruff.exe check .

# Auto-fix ruff issues
./.venv/Scripts/ruff.exe check . --fix

# Run mypy type checker
./.venv/Scripts/mypy.exe src/

# Run black code formatter (check mode)
./.venv/Scripts/black.exe --check src/

# Run black code formatter (apply changes)
./.venv/Scripts/black.exe src/

# Run bandit security linter
./.venv/Scripts/bandit.exe -c pyproject.toml -r src/

# Run all lint checks (convenience script)
bash scripts/lint.sh

# Run pre-commit hooks on all files
./.venv/Scripts/pre-commit.exe run --all-files
```

### Documentation
```bash
# Build Sphinx documentation
cd docs && make html

# Clean and rebuild documentation
cd docs && make clean && make html
```

### Build and Packaging
```bash
# Build package
./.venv/Scripts/python.exe -m build

# Install in development mode
./.venv/Scripts/pip.exe install -e ".[dev,server,file-extraction]"
```

### Package Management
```bash
# Install development dependencies
pip install -e ".[dev,server,file-extraction]"

# Install specific dependency groups
pip install -e ".[server]"          # Server features
pip install -e ".[file-extraction]" # File processing
pip install -e ".[async]"          # Async support
```

### CLI Commands
```bash
# Start development server
lvdb serve --debug

# Create a test database
lvdb create testdb --embedding-model nomic-embed-text

# Run database operations
lvdb db testdb add "sample document"
lvdb db testdb search "query text" --search-type hybrid
```

## Architecture Overview

LocalVectorDB is a document-first vector database built on SQLite + FAISS with pluggable embedding providers. The architecture follows a clear separation between local and remote operations with a unified API.

### Core Components

**localvectordb/** - Core library
- `core.py` - Fundamental data structures (Document, MetadataField, etc.)
- `database/` - Modular database implementation package:
  - `base.py` - Base database interface
  - `_core.py` - Core LocalVectorDB implementation with SQLite + FAISS
  - `_crud.py` - CRUD operations
  - `_ingest.py` - Document ingestion and chunking
  - `_search.py` - Search operations (vector, keyword, hybrid)
  - `_metadata.py` - Metadata management
- `client.py` - RemoteVectorDB client for HTTP server communication
- `factory.py` - VectorDB factory function for automatic local/remote selection
- `embeddings.py` - Pluggable embedding providers (Ollama, OpenAI, Mock)
- `chunking.py` - Position-aware text chunking with multiple strategies
- `query_builder.py` - SQL-like query builder for metadata filtering

**localvectordb_server/** - HTTP server implementation
- `routes.py` - Flask REST API endpoints
- `_dbmanager.py` - Multi-database management and connection pooling
- `_auth.py` - API key authentication system
- `config.py` - Configuration management (TOML/JSON + env vars)
- `cli/` - Command-line interface (`lvdb` command)
- `extractors/` - File format extractors (PDF, DOCX, etc.)

### Key Design Patterns

**Document-First API**: Users work with complete documents rather than chunks. Chunking is handled internally with position tracking for perfect reconstruction.

**Factory Pattern**: `VectorDB()` function automatically chooses LocalVectorDB or RemoteVectorDB based on path/URL, enabling seamless local-to-remote migration.

**Plugin Architecture**: Embedding providers and file extractors use Python entry points for extensibility.

**Dual Storage**: Documents/metadata in SQLite with FTS5 for keyword search; vectors in FAISS for semantic similarity.

**Unified Search**: Single `query()` method supports vector, keyword, and hybrid search with normalized scoring.

## Development Guidelines

### Database Operations
- LocalVectorDB handles SQLite + FAISS operations directly
- RemoteVectorDB communicates with Flask server via HTTP
- All operations support both sync and async variants
- Connection pooling is managed automatically

### Testing Strategy
- Unit tests mock external dependencies (FAISS, HTTP calls)
- Integration tests use temporary databases and mock embedding providers
- Performance tests marked with `@pytest.mark.slow`
- Network tests marked with `@pytest.mark.network` (currently mocked)

### Metadata Schema
- Strongly typed with MetadataField definitions
- Supports TEXT, INTEGER, REAL, BOOLEAN, DATE, JSON types
- Indexed fields automatically get SQLite indexes
- Schema validation occurs at upsert time

### Chunking System
- Position-aware chunking preserves document reconstruction
- Multiple strategies: sentences, tokens, paragraphs, code blocks
- Configurable overlap between chunks
- Metadata inheritance from parent documents

### Embedding Providers
- Pluggable via entry points in pyproject.toml
- Support for batch processing and concurrent requests
- Automatic retry logic with exponential backoff
- Provider-specific configuration via embedding_config

### Server Architecture
- Flask application with modular blueprint structure
- Multi-database support with lazy loading
- Authentication via API keys with expiration
- Rate limiting, CORS, and caching support
- Redis integration for distributed deployments

## Configuration

The system uses hierarchical configuration:
1. Default values in code
2. Configuration files (TOML/JSON)
3. Environment variables (prefixed with `LVDB_`)
4. Command-line arguments

Key configuration sections:
- `[database]` - Storage paths, chunking settings
- `[embedding]` - Provider selection and models
- `[server]` - HTTP server settings, auth, rate limiting

## Common Tasks

### Adding New Embedding Provider
1. Create class inheriting from `EmbeddingProvider`
2. Implement `embed_documents()` and `embed_query()` methods
3. Add entry point to pyproject.toml
4. Add corresponding tests

### Adding New File Extractor
1. Create extractor class with `extract_text()` method
2. Add entry point to pyproject.toml under `localvectordb_server.file_extractors`
3. Add MIME type mapping in extractors/__init__.py

### Extending Query Builder
- Query builder supports SQL-like syntax for metadata filtering
- New operators can be added to `_filters.py`
- Integration with both local SQLite and remote HTTP queries

## Testing Notes

- Use `conftest.py` fixtures for common test setup
- Mock embedding providers with `MockEmbeddings` for deterministic tests
- Temporary directories automatically cleaned up after tests
- Coverage target is 85% overall, 90% for core modules