# LocalVectorDB

A high-performance, document-first vector database with SQLite + FAISS backend, featuring intelligent chunking, unified search, and optional HTTP server.

[![PyPI version](https://img.shields.io/pypi/v/localvectordb.svg)](https://pypi.org/project/localvectordb/)
[![Python versions](https://img.shields.io/pypi/pyversions/localvectordb.svg)](https://pypi.org/project/localvectordb/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/thomas-villani/localvectordb/actions/workflows/test.yml/badge.svg)](https://github.com/thomas-villani/localvectordb/actions/workflows/test.yml)
[![Docs](https://github.com/thomas-villani/localvectordb/actions/workflows/docs.yml/badge.svg)](https://thomas-villani.github.io/localvectordb/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> **Beyond basic RAG.** LocalVectorDB pairs a zero-infrastructure SQLite + FAISS core with capabilities most vector stores don't have:
> - 🧬 **Section-level retrieval, measured** — in a long, structured document the answer is usually spread across a whole *section*, not concentrated in one stray sentence — and flat chunking cannot see it. LocalVectorDB embeds each section's own text and retrieves it alongside chunks: **+0.03–0.08 nDCG@10** for finding the right document and **+0.07–0.17** for the right section, over a chunk-only baseline, across three local encoders on real papers (Qasper, 15 papers / 48 queries). [Read the study](https://thomas-villani.github.io/localvectordb/hierarchical-evaluation.html).
> - ✅ **Reverse-RAG fact-checking** — ground LLM-generated text against your corpus, flagging unsupported or contradicted claims with citations.
> - 📊 **Document comparison & visualization** — synteny ribbons and chord diagrams that show how two documents (or a document's own chunks) relate.

## Contents

- [Quick Start](#-quick-start) — install, index, search
- [Use with Claude Code & other AI agents](#-use-with-claude-code--other-ai-agents)
- [Features](#-features)
- [Server Deployment](#️-server-deployment)
- [TypeScript SDK](#-typescript-sdk)
- [API Reference](#-api-reference)
- [CLI Reference](#️-cli-reference)
- [Architecture](#️-architecture)
- [File Extraction](#-file-extraction)
- [Configuration Options](#️-configuration-options)
- [Production Deployment](#-production-deployment)
- [Examples](#-examples)
- [Contributing](#-contributing)

📖 **Full documentation:** <https://thomas-villani.github.io/localvectordb/>

## 🚀 Quick Start

### Installation

LocalVectorDB is a standard PyPI package — install it with [uv](https://docs.astral.sh/uv/) (recommended) or pip.

```bash
# Add to your project with uv (recommended)
uv add localvectordb

# For server features (optional)
uv add "localvectordb[server]"

# For all file extraction formats (optional)
uv add "localvectordb[server,file-extraction]"
```

Prefer pip? Every command above has a pip equivalent, e.g. `pip install "localvectordb[server,file-extraction]"`.

To run the CLI/server without adding it to a project, use uv's tool runner:

```bash
uvx --from "localvectordb[server]" lvdb serve
```

### Prerequisites

LocalVectorDB needs an embedding provider. By default it embeds through
[Ollama](https://ollama.com) running locally:

```bash
ollama pull nomic-embed-text
```

Any other provider works via `embedding_provider=` — OpenAI, Google, Jina,
HuggingFace (Inference API or local), or Sentence Transformers. See the
[Embeddings guide](https://thomas-villani.github.io/localvectordb/embeddings.html).

**No provider handy?** The built-in `mock` provider needs no service and ships in
the base install, so you can verify the install and explore the API immediately:

<!-- test: run -->
```python
from localvectordb import VectorDB

db = VectorDB("demo", ":memory:", embedding_provider="mock", embedding_model="mock")
db.upsert(["Python is a programming language"])
print(db.query("programming", k=1))
```

Mock vectors are deterministic but carry no semantic meaning, so *rankings from
the mock provider are arbitrary*. It is for wiring up code, not for judging
retrieval quality.

### Basic Usage

<!-- test: verify-api -->
```python
from localvectordb import VectorDB

# Create or connect to a database (defaults to Ollama + nomic-embed-text)
db = VectorDB("my_docs", "./data")

# Add documents
doc_ids = db.upsert([
    "Python is a programming language",
    "Machine learning with neural networks"
])

# Search documents
results = db.query("programming", k=5)
for result in results:
    print(f"{result.id}: {result.content} (score: {result.score:.3f})")

# Get specific document
doc = db.get(doc_ids[0])
print(f"Content: {doc.content}")
```

### With Metadata Schema

<!-- test: verify-api -->
```python
from localvectordb import VectorDB
from localvectordb.core import MetadataField, MetadataFieldType

# Define metadata schema
schema = {
    'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
    'created_date': MetadataField(type=MetadataFieldType.DATE, indexed=True),
    'tags': MetadataField(type=MetadataFieldType.JSON)
}

db = VectorDB("articles", "./data", metadata_schema=schema)

# Add documents with metadata
db.upsert(
    documents=["Article about Python programming"],
    metadata=[{
        'title': 'Python Guide',
        'author': 'Jane Doe',
        'created_date': '2024-01-15',
        'tags': ['python', 'programming', 'tutorial']
    }]
)

# Search with metadata filters
results = db.query(
    "programming tutorial",
    filters={'author': 'Jane Doe', 'tags': {'$contains': 'python'}}
)
```

### Remote Server Usage

<!-- test: verify-api -->
```python
from localvectordb import VectorDB

# Use HTTP server (automatically detected by URL)
db = VectorDB(
    "my_docs", 
    "http://localhost:8000",
    api_key="your_api_key"
)

# Same API as local database
doc_ids = db.upsert(["Remote document content"])
results = db.query("content", search_type="hybrid")
```

## 🤖 Use with Claude Code & other AI agents

LocalVectorDB ships a built-in [Model Context Protocol](https://modelcontextprotocol.io/)
server, so an AI agent can search your knowledge bases directly — no glue code.
It is **read-only by default**, which makes it safe to point at a corpus you care
about.

```bash
uv add "localvectordb[mcp]"     # or: pip install "localvectordb[mcp]"
```

Build a knowledge base with the CLI first (rich formats like PDF and DOCX are
extracted to Markdown automatically):

```bash
lvdb create technical_docs --embedding-model nomic-embed-text
lvdb db technical_docs add ./docs/*.md ./manual.pdf
```

### Claude Code

Register the server once, from your project directory:

```bash
claude mcp add lvdb   -e LVDB_MCP_MODE=read-only   -e LVDB_MCP_DATABASES_ROOT=/path/to/databases   -- lvdb mcp serve
```

Everything after `--` is the launch command, passed through untouched. Use
`-s project` to write a committed `.mcp.json` your team shares, `-s user` to make
it available in every project; the default scope is `local` (this project, just
you). Then `claude mcp list` to check status, `/mcp` inside a session to manage
it, and `claude mcp remove lvdb` to undo.

### Claude Desktop, and any other MCP client

Add the same server to `claude_desktop_config.json` (or your client's equivalent
config — the `mcpServers` block is a shared convention):

```json
{
  "mcpServers": {
    "lvdb": {
      "type": "stdio",
      "command": "lvdb",
      "args": ["mcp", "serve"],
      "env": {
        "LVDB_MCP_MODE": "read-only",
        "LVDB_MCP_DATABASES_ROOT": "/path/to/databases"
      }
    }
  }
}
```

### What the agent can do

The server exposes focused tools — `query_database`, `find_related_documents`,
`filter_documents`, `get_document` (a whole document, or a chunk / line range /
section of one), and `list_databases`. Tool sets are configurable per deployment,
so you can expose only what a given agent needs.

Granting writes is explicit — `lvdb mcp serve --mode read-write` (or
`LVDB_MCP_MODE=read-write`) additionally enables `upsert_documents`,
`create_database`, and friends. Only do this for databases the agent is meant to
modify.

See the [MCP documentation](https://thomas-villani.github.io/localvectordb/mcp.html)
for the full tool list, configuration reference, and security guidance.

## ✨ Features

### 🗃️ **Document-First Architecture**
- **Smart Chunking**: Position-tracking chunking — the default chunker reconstructs documents byte-for-byte from their chunks
- **Metadata Schema**: Structured, indexed metadata fields with validation
- **Unified API**: Single interface for vector, keyword, and hybrid search
- **In-place Patch API**: Edit a stored document with exact find/replace (or span splice) instead of re-sending the whole content — with an optional `expect_hash` precondition to guard against lost updates. Available in the library, HTTP API, MCP tool, CLI, and JS SDK

### 🔍 **Advanced Search**
- **Vector Search**: Semantic similarity via pluggable embedding providers — Ollama, OpenAI, Google, Jina, HuggingFace (Inference API + local), and Sentence Transformers
- **Keyword Search**: Full-text search with SQLite FTS5
- **Hybrid Search**: Combined vector + keyword with configurable weighting
- **Reranking**: Optional cross-encoder reranking via Jina, Sentence Transformers, or HuggingFace
- **Metadata Filtering**: MongoDB-style queries on structured metadata
- **Document Scoring**: Three chunk-to-document aggregation strategies (`best`, `average`, `frequency_boost`) for tuning relevance

### 🧬 **Hierarchical Retrieval**
- **Raw-Span Section Vectors**: Each section is embedded from *its own text*, not averaged from its chunks — averaging blurs away the cross-chunk structure that makes a section retrievable in the first place. This costs one extra embedding call per section at ingest (sections are far fewer than chunks, so it is modest — but it is not free). Sections longer than the encoder's context are window-pooled, never truncated
- **Search a Level, or Fuse Two**: `search_level="sections"` retrieves sections directly; `search_level="fused"` blends the section and chunk rankings with a tunable `section_weight` (default 0.65). Sections alone are the stronger choice when relevance is genuinely section-shaped; fusion leans toward document-level accuracy
- **Automatic Section Detection**: Sections derived from document structure (Markdown headings by default, custom patterns supported)
- **Measured, Not Asserted**: A controlled study across three local encoders, two chunk sizes, and real papers — raw-span sections beat the chunk-only baseline at every target, and beat the "free" centroid decisively. Full tables, methodology, and caveats: [Raw-Span Hierarchical Retrieval](https://thomas-villani.github.io/localvectordb/hierarchical-evaluation.html)
- **Check It Yourself**: Don't take the study on faith — [`examples/section_vs_chunk_retrieval.py`](examples/section_vs_chunk_retrieval.py) runs the same comparison on *your* corpus and prints the same nDCG@10 table. Your documents are the only ones that decide whether this is worth enabling
- **Section Metadata**: Pluggable extractors (heading path, keywords, word/char counts, or your own)
- **Opt-in**: Off by default (`hierarchical_embeddings=True` to enable); the flat retrieval path is unchanged. Document-level search (`search_level="documents"`) is also available

### ✅ **Reverse-RAG Fact-Checking**
- **Grounding Verification**: Check LLM-generated text against your databases claim-by-claim
- **Citations & Contradictions**: Each claim scored, cited to a source excerpt, and flagged if contradicted
- **Multi-Provider LLMs**: Works with Anthropic, OpenAI, or Gemini clients (auto-detected)

### 📊 **Document Comparison & Visualization**
- **Similarity & Neighbors**: Compare documents, find nearest neighbors, build similarity matrices
- **Embedding Maps**: t-SNE / PCA projections with clustering
- **Synteny & Chord Diagrams**: Visualize chunk-level alignment between documents or within one

### 🌐 **Flexible Deployment**
- **Local Database**: Direct SQLite + FAISS for maximum performance
- **HTTP Server**: RESTful API with permission-based authentication, rate limiting, CORS
- **Remote Client**: Seamless local/remote switching via factory pattern
- **Multi-Worker**: Redis-based coordination for distributed deployments

### 📄 **File Processing**
- **Text Extraction**: PDF, DOCX, PPTX, XLSX, RTF, EPUB support
- **Batch Upload**: Multi-file processing with metadata extraction
- **Format Detection**: Automatic MIME type detection and processing

### 🤖 **AI / LLM Integration**
- **MCP Server**: Built-in [Model Context Protocol](https://modelcontextprotocol.io/) server for Claude Desktop, Claude Code, and other MCP clients
- **Read-Only by Default**: Safe knowledge-base access; opt into read-write explicitly
- **TypeScript SDK**: First-class browser/Node client (`@localvectordb/sdk`)

### 🛠️ **Developer Experience**
- **CLI Tools**: Database management, server control, interactive shell
- **Configuration**: TOML/JSON config with environment variable support
- **Comprehensive Logging**: Structured logging with performance monitoring
- **Type Safety**: Full type annotations and validation

## 🖥️ Server Deployment

### Start the Server

```bash
# Quick start: with no config file, serves on localhost:8000 with built-in
# defaults. Run `lvdb config init` first to customize (host, auth, CORS, ...).
lvdb serve

# Production configuration
lvdb --config ./config.toml serve --host 0.0.0.0 --port 8000
```

### Configuration

Create a configuration file:

```bash
# Interactive setup wizard
lvdb config init --interactive

# Production setup with Redis
lvdb config init --redis-registry redis://localhost:6379/1 \
                  --enable-cache --cache-type redis \
                  --enable-rate-limiting --enable-cors \
                  --enable-auth
```

Example configuration (`config.toml`):

```toml
[database]
root_dir = "./databases"
chunk_size = 500
chunking_method = "sentences"
chunk_overlap = 1

[embedding]
provider = "ollama"
model = "nomic-embed-text"

[server]
host = "0.0.0.0"
port = 8000
enable_rate_limiting = true
rate_limit = "100 per minute"

# Authentication and CORS live under the [server.security] table.
[server.security]
require_api_key = true
cors_enabled = true
cors_allowed_origins = ["http://localhost:3000"]
```

### API Key Management

```bash
# Create API key with permission level
lvdb auth create-key --description "Production API" --permission-level read_write

# Create read-only key for analytics
lvdb auth create-key --description "Analytics Dashboard" --permission-level read_only

# List keys with their permissions
lvdb auth list-keys --active-only

# Revoke key
lvdb auth revoke-key key_20241201_abc123
```

## 🟦 TypeScript SDK

A zero-dependency TypeScript/JavaScript client is available for the HTTP server
(Node.js 18+ and modern browsers):

```bash
npm install @localvectordb/sdk
```

```typescript
import { LocalVectorDBClient } from "@localvectordb/sdk";

const client = new LocalVectorDBClient({ baseUrl: "http://localhost:8000" });
const db = client.database("my_docs");

await db.upsert(["First document", "Second document"]);
const results = await db.query("search text", { search_type: "hybrid", k: 5 });
```

See [`sdk/js/README.md`](sdk/js/README.md) for the full SDK API.

## 📚 API Reference

### Core Methods

#### `upsert(documents, metadata=None, ids=None)`
Insert or update documents.

<!-- test: verify-api -->
```python
# Single document
db.upsert("Document content")

# Multiple documents with metadata
doc_ids = db.upsert(
    documents=["Doc 1", "Doc 2"],
    metadata=[{"type": "article"}, {"type": "blog"}],
    ids=["doc_1", "doc_2"]
)
```

#### `query(query, search_type='hybrid', k=10, filters=None)`
Unified search interface.

<!-- test: verify-api -->
```python
# Vector search
results = db.query("search text", search_type="vector", k=5)

# Hybrid search with metadata filter
results = db.query(
    "machine learning",
    search_type="hybrid",
    vector_weight=0.5,
    filters={"category": "AI"}
)

# Keyword search
results = db.query("exact phrase", search_type="keyword")
```

> **Note:** Filter fields (and metadata keys on upsert) must be declared in the
> database's `metadata_schema`. Filtering on an undeclared field raises
> `DatabaseError`; undeclared metadata keys are dropped on upsert with a warning.

#### `get(ids)` / `delete(ids)` / `exists(ids)`
Document management.

<!-- test: verify-api -->
```python
# Single document
doc = db.get("doc_1")
exists = db.exists("doc_1")
deleted_count = db.delete("doc_1")

# Multiple documents
docs = db.get(["doc_1", "doc_2"])
exist_flags = db.exists(["doc_1", "doc_2"])
deleted_count = db.delete(["doc_1", "doc_2"])
```

#### `patch(doc_id, ops, *, expect_hash=None, metadata=None)`
Edit a document in place without re-sending its whole content. Ops resolve
against the current content, touch disjoint spans, and apply atomically.

<!-- test: verify-api -->
```python
# Exact find/replace (must match exactly `count` times, default 1)
result = db.patch("doc_1", [{"op": "replace", "find": "draft", "replace": "final"}])
print(result.updated, result.new_hash, result.ops_applied)

# Span splice + append/prepend
db.patch("doc_1", [{"op": "splice", "start": 0, "end": 5, "text": "Hello"}])
db.patch("doc_1", [{"op": "append", "text": " (revised)"}])

# Optimistic concurrency — raises PatchConflictError if the doc changed
doc = db.get("doc_1")
db.patch("doc_1", [{"op": "replace", "find": "v1", "replace": "v2"}],
         expect_hash=doc.content_hash)
```

#### `filter(where=None, order_by=None, limit=None, offset=0)`
MongoDB-style filtering on metadata.

<!-- test: verify-api -->
```python
# Simple filters
docs = db.filter(where={"author": "Jane Doe", "status": "published"})

# Complex queries with operators
docs = db.filter(
    where={"created_date": {"$gte": "2024-01-01"}},
    order_by="created_date DESC",
    limit=10
)

# Logical operators and pattern matching
docs = db.filter(
    where={"$and": [{"author": {"$like": "%Smith%"}}, {"rating": {"$gt": 4.0}}]}
)
```

### HTTP API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/databases` | List databases |
| `POST` | `/api/v1/databases` | Create database |
| `GET` | `/api/v1/databases/{db}/info` | Database info |
| `POST` | `/api/v1/databases/{db}/documents` | Upsert documents |
| `GET` | `/api/v1/databases/{db}/documents/{id}` | Get document |
| `PATCH` | `/api/v1/databases/{db}/documents/{id}` | Update document |
| `DELETE` | `/api/v1/databases/{db}/documents/{id}` | Delete document |
| `POST` | `/api/v1/databases/{db}/query` | Search documents |
| `POST` | `/api/v1/databases/{db}/query/stream` | Stream results (SSE) |
| `POST` | `/api/v1/databases/{db}/filter` | Filter documents |
| `POST` | `/api/v1/databases/{db}/upload` | Upload files |
| `POST` | `/api/v1/databases/{db}/compare` | Compare documents |
| `POST` | `/api/v1/databases/{db}/nearest-neighbors` | Find similar documents |

Example API usage:

```bash
# Create database
curl -X POST http://localhost:8000/api/v1/databases \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my_db"}'

# Search documents
curl -X POST http://localhost:8000/api/v1/databases/my_db/query \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "search_type": "hybrid", "k": 5}'
```

## 🛠️ CLI Reference

### Database Management

```bash
# List databases
lvdb list --details

# Create database
lvdb create mydb --embedding-model nomic-embed-text --chunk-size 500

# Delete database
lvdb delete mydb --confirm
```

### Database Operations

```bash
# Add documents
lvdb db mydb add document.txt
lvdb db mydb add "docs/*.py"
cat content.txt | lvdb db mydb add -

# Search documents
lvdb db mydb search "query text" --search-type hybrid --limit 10

# Get document (or a portion: --chunk / --range / --lines / --section / --outline)
lvdb db mydb get doc_1 --format json --metadata

# Find documents related to an existing one (nearest neighbours)
lvdb db mydb related doc_1 --limit 5

# Interactive shell
lvdb db mydb shell
```

### Configuration

```bash
# View configuration
lvdb config show --section database

# Update settings
lvdb config set server.port 8080
lvdb config set database.chunk_size 1000
```

### All Commands

| Command | What it does |
|---------|--------------|
| `lvdb serve` | Start the HTTP server |
| `lvdb create` / `list` / `delete` / `rename` | Database lifecycle |
| `lvdb version` | Show the installed version |
| `lvdb db <name> ...` | Operate on one database: `info`, `stats`, `search`, `add`, `get`, `related`, `update`, `patch`, `delete`, `repair`, `shell`, `schema` |
| `lvdb chunk` | Chunk text to JSONL without a database — useful for inspecting a chunking strategy |
| `lvdb backup` | `create`, `list`, `restore`, `verify`, `cleanup`, `pitr` |
| `lvdb migrate` | Metadata schema evolution: `status`, `apply`, `rollback`, `create`, `list` |
| `lvdb config` | `show`, `get`, `set`, `init` |
| `lvdb auth` | API keys: `create-key`, `list-keys`, `revoke-key`, `rotate-key`, `key-info`, `prune-expired`, `status` |
| `lvdb tuning` | SQLite tuning: `list`, `get`, `set`, `set-pragma`, `auto-tune` |
| `lvdb maintenance` | `checkpoint`, `optimize`, `vacuum`, `analyze` |
| `lvdb mcp` | MCP server: `serve`, `status`, `test`, `tools`, `config-example` |

Every command takes `--help`. Full reference with all flags and exit codes:
[CLI documentation](https://thomas-villani.github.io/localvectordb/cli.html).

## 🏗️ Architecture

### Local Architecture
```
┌─────────────────┐    ┌───────────────┐    ┌─────────────┐
│   Application   │────│ LocalVectorDB │────│   SQLite    │
└─────────────────┘    └───────────────┘    └─────────────┘
                              │
                       ┌──────────────┐
                       │    FAISS     │
                       │    Index     │
                       └──────────────┘
```

### Server Architecture
```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│     Client      │────│ HTTP Server  │────│ DB Manager  │
│ (RemoteVectorDB)│    │  (FastAPI)   │    └─────────────┘
└─────────────────┘    └──────────────┘           │
                              │            ┌───────────────┐
                       ┌──────────────┐    │Multiple DBs   │
                       │     Auth     │    │(LocalVectorDB)│
                       │  Rate Limit  │    └───────────────┘
                       │  CORS / SSE  │
                       └──────────────┘
```

### Chunking System
- **Position Tracking**: Exact character positions — every general-purpose chunker (`sentences`, `tokens`, `words`, `lines`, `characters`, `paragraphs`, `sections`) reconstructs the source byte-for-byte. The specialized `code-blocks` chunker is exact only when a document fits in a single chunk.
- **Multiple Methods**: Sentences, tokens, paragraphs, sections, code blocks
- **Overlap Support**: Configurable overlap between chunks
- **Metadata Preservation**: Document metadata inherited by all chunks

## 📁 File Extraction

Text extraction is powered by [all2md](https://all2md.readthedocs.io/), which
converts 20+ document formats and 200+ source/text formats to Markdown. Common
formats work out of the box; extended/niche formats and OCR are opt-in extras.

| Formats | Availability |
|---------|--------------|
| PDF, DOCX, PPTX, XLSX | Built-in |
| HTML, EPUB, RTF, ODT/ODP/ODS | Built-in |
| Markdown, reStructuredText, Org, CSV, JSON, YAML, `.eml`, `.ipynb` | Built-in |
| Source code & plain text (200+ extensions) | Built-in |
| LaTeX, MediaWiki, Textile, archives, `.enex`, `.fb2`, Outlook | `localvectordb[file-extraction]` |
| OCR for scanned PDFs (Tesseract) | `localvectordb[file-extraction-ocr]` |

Extracted content is **Markdown**, preserving headings, tables, and lists for
better chunk boundaries.

<!-- test: skip reason="raw HTTP against a running server; no library calls to check" -->
```python
# Upload files via HTTP API
import requests

files = {'files': open('document.pdf', 'rb')}
response = requests.post(
    'http://localhost:8000/api/v1/databases/mydb/upload',
    files=files,
    headers={'Authorization': 'Bearer your_api_key'}
)
```

## ⚙️ Configuration Options

### Database Settings
- `root_dir`: Database storage directory
- `chunk_size`: Maximum tokens per chunk
- `chunking_method`: Algorithm for splitting text
- `chunk_overlap`: Overlap between adjacent chunks
- `default_metadata_schema`: Schema for new databases

### Embedding Settings
- `provider`: Embedding provider — one of `ollama`, `openai`, `google`, `jina`, `huggingface`, `huggingface_local`, or `sentence_transformers`
- `model`: Model name (e.g., "nomic-embed-text")
- `base_url`: Custom API endpoint
- `api_key`: API key for providers requiring authentication

### Server Settings
- `host` / `port`: Server binding
- `require_api_key`: Enable authentication
- `enable_rate_limiting`: Rate limiting with configurable limits
- `cors_enabled`: CORS support for web apps
- `cache_enabled`: Response caching (memory, file, Redis)

## 🔧 Production Deployment

### Scale and limits

LocalVectorDB is built for **agent-native document memory**, not as a general-purpose
vector store at web scale. Be deliberate about the ceiling:

- **The index is exact, not approximate.** The default `IndexFlatL2` is a brute-force
  flat index: every query scans every vector, so latency grows linearly with the
  collection. You get exact recall in return — no ANN tuning, no recall cliff.
- **Vectors are RAM-resident float32, unquantized.** Budget `dimensions × 4 bytes` per
  vector: **1M × 768-dim ≈ 3.1 GB** (plus the SQLite store). Reader workers can share a
  single page-cached copy via `database.mmap_index` (see below), which cuts *per-worker*
  RAM but not the on-disk size.
- **Practical ceiling: ~10⁵–10⁶ vectors** on a normal machine. Beyond that you want
  quantization or an ANN index, which are deliberately out of scope for v0.1.

**Bulk-load in one call.** Every `upsert`/`insert` rewrites the *entire* FAISS index file
when it flushes, so ingesting N documents in N separate calls is N full-index rewrites.
Pass the whole batch to a single `upsert(documents=[...])` instead — it is dramatically
faster and rewrites the index once.

### Docker Deployment

A hardened [`Dockerfile`](Dockerfile) ships in the repo: pinned base image, dependencies
isolated in a virtualenv, non-root user, and a `HEALTHCHECK` against `/health`. It builds
from source, and CI builds it on every pull request so it cannot silently rot.

```bash
docker build -t localvectordb:local .
docker run --rm -p 8000:8000 -v lvdb-data:/data localvectordb:local
```

Databases persist in the `/data` volume (`LVDB_DATABASE_ROOT_DIR`). Configure with `LVDB_*`
environment variables, or mount a TOML config file:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/config.toml:/etc/lvdb/config.toml:ro" \
  -v lvdb-data:/data \
  localvectordb:local \
  lvdb --config /etc/lvdb/config.toml serve --host 0.0.0.0 --port 8000
```

The image installs the `server` extra only. For file upload/extraction, add
`file-extraction` to the `pip install` line in the `Dockerfile`.

### Scaling reads across workers

LocalVectorDB is **single-writer**. One process owns writes to a database; to scale
query throughput you fan out **read-only** replicas across many workers.

Build (or update) the database from a single writer, then serve reads from N
workers. Set `database.mmap_index = true` on the readers so every worker shares one
memory-mapped copy of the FAISS index through the OS page cache, instead of each
loading a private, RAM-resident copy:

```bash
# On the reader deployment: memory-map the index (read-only, shared page cache)
lvdb config set database.mmap_index true

# Optional: coordinate the set of database names across workers via a shared registry
lvdb config set server.db_registry_type "RedisCache"
lvdb config set server.db_registry_settings '{"host": "redis", "port": 6379, "db": 1}'

# Fan out read-only workers
uvicorn "localvectordb_server.app:create_app" --factory --host 0.0.0.0 --port 8000 --workers 4
```

> **⚠️ Single-writer only.** Do not send writes (upsert / insert / update / delete)
> through a multi-worker deployment. Each worker holds an independent in-memory FAISS
> index and does not observe another worker's writes, and two writers racing the
> index file will diverge or corrupt it. A database opened with `mmap_index = true`
> refuses writes outright. Route all writes to one writer process
> (`mmap_index = false`); readers observe a writer's updates only after they reload
> the index (on restart or idle-eviction).

### Environment Variables

```bash
export LVDB_SERVER_CONFIG="/path/to/config.toml"
export LVDB_DATABASE_ROOT_DIR="/data/databases"
export LVDB_EMBEDDING_PROVIDER="ollama"
export LVDB_EMBEDDING_MODEL="nomic-embed-text"
export OPENAI_API_KEY="your-openai-key"  # if using OpenAI
```

## 🧪 Examples

### Runnable scripts

The snippets below are illustrative. [`examples/`](examples/) holds complete
programs you can run, each covered by the test suite so it cannot rot:

| Script | What it does |
|---|---|
| [`section_vs_chunk_retrieval.py`](examples/section_vs_chunk_retrieval.py) | Runs this project's headline retrieval comparison — section-level vs chunk-level — **on your own corpus**, and prints nDCG@10 / recall@k per mode. Bring your documents and a small judgments file; it ships with a sample of both. |

They need a real embedding backend (Ollama or `sentence-transformers`) and
refuse the `mock` provider, because mock vectors cannot tell you whether the
right thing ranked first. See [`examples/README.md`](examples/README.md).

### Research Paper Database

<!-- test: verify-api -->
```python
from localvectordb import VectorDB
from localvectordb import get_common_metadata_schemas

# Use predefined research paper schema
schema = get_common_metadata_schemas("research_papers")
db = VectorDB("papers", "./data", metadata_schema=schema)

# Add papers
db.upsert(
    documents=["Paper content..."],
    metadata=[{
        'title': 'Attention Is All You Need',
        'authors': ['Vaswani', 'Shazeer', 'Parmar'],
        'publication_date': '2017-06-12',
        'journal': 'NIPS',
        'keywords': ['attention', 'transformer', 'neural networks']
    }]
)

# Search by topic and filter by date
results = db.query(
    "transformer architecture",
    filters={"publication_date": {"$gte": "2017-01-01"}},
    search_type="hybrid"
)
```

### Code Repository Search

<!-- test: verify-api -->
```python
import glob
import os
from datetime import datetime, timezone

from localvectordb import VectorDB, get_common_metadata_schemas

# Create database for code files. Declaring a metadata schema is what lets the
# metadata below be stored (undeclared fields are dropped with a warning).
db = VectorDB("code", "./data",
              chunking_method="code-blocks",
              chunk_size=1000,
              metadata_schema=get_common_metadata_schemas("code_repository"))

# Add Python files
for file_path in glob.glob("**/*.py", recursive=True):
    with open(file_path, "r") as f:
        content = f.read()

    db.upsert(
        documents=[content],
        metadata=[{
            "file_path": file_path,
            "language": "python",
            "last_modified": datetime.fromtimestamp(
                os.path.getmtime(file_path), tz=timezone.utc
            ).isoformat(),
        }]
    )

# Search for specific functions
results = db.query("async def", search_type="keyword")
```

### Hierarchical (Section-Level) Retrieval

<!-- test: verify-api -->
```python
from localvectordb import VectorDB

# Enable the document → section → chunk hierarchy
db = VectorDB("manuals", "./data", hierarchical_embeddings=True)
db.upsert([open("user_guide.md").read()], ids=["guide"])

# Match the most relevant *section* instead of a single chunk
results = db.query("how do I reset my password?", search_level="sections")
for r in results:
    print(r.metadata["section_heading"], f"{r.score:.3f}")
```

### Reverse-RAG Fact-Checking

<!-- test: verify-api -->
```python
import anthropic
from localvectordb import VectorDB, FactChecker

db = VectorDB("kb", "./data")
db.upsert(["The Eiffel Tower is 330 metres tall and located in Paris."])

# Ground an LLM claim against the corpus (provider auto-detected from the client)
checker = FactChecker(db, llm=anthropic.Anthropic())
result = checker.check("The Eiffel Tower is 300 metres tall and stands in Berlin.")

print(f"Overall grounding score: {result.overall_score:.2f}")
for claim in result.claims:
    print(claim.claim, "->", claim.polarity.value, f"(grounded={claim.grounded})")
```

## 🤝 Contributing

Contributions are welcome! By submitting a contribution you agree to license it under the project's [MIT License](LICENSE).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/thomas-villani/localvectordb.git
cd localvectordb

# Install dev tooling + test extras (add --extra mcp to work on the MCP server)
uv sync --dev

# Run tests
uv run pytest

# Start development server
lvdb serve --debug
```

## 📄 License

This project is licensed under the [MIT License](LICENSE) — see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/thomas-villani/localvectordb/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/thomas-villani/localvectordb/discussions)
- 📧 **Contact**: thomas.villani@gmail.com

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [SQLite](https://sqlite.org/) for the document database
- [Ollama](https://ollama.ai/) for local embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the HTTP server
- [Click](https://click.palletsprojects.com/) for the Click CLI library
- All the contributors who made this project possible