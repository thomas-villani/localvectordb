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
> - 🧬 **Hierarchical retrieval** — search a three-level *document → section → chunk* hierarchy, so you can match a whole section (not just a stray sentence) in long, structured documents.
> - ✅ **Reverse-RAG fact-checking** — ground LLM-generated text against your corpus, flagging unsupported or contradicted claims with citations.
> - 📊 **Document comparison & visualization** — synteny ribbons and chord diagrams that show how two documents (or a document's own chunks) relate.

## ✨ Features

### 🗃️ **Document-First Architecture**
- **Smart Chunking**: Position-tracking chunking with perfect document reconstruction
- **Metadata Schema**: Structured, indexed metadata fields with validation
- **Unified API**: Single interface for vector, keyword, and hybrid search

### 🔍 **Advanced Search**
- **Vector Search**: Semantic similarity via pluggable embedding providers — Ollama, OpenAI, Google, Jina, HuggingFace (Inference API + local), and Sentence Transformers
- **Keyword Search**: Full-text search with SQLite FTS5
- **Hybrid Search**: Combined vector + keyword with configurable weighting
- **Reranking**: Optional cross-encoder reranking via Jina, Sentence Transformers, or HuggingFace
- **Metadata Filtering**: MongoDB-style queries on structured metadata
- **Document Scoring**: 11 chunk-to-document aggregation strategies for tuning relevance

### 🧬 **Hierarchical Retrieval**
- **Three-Level Hierarchy**: Index and search at *document*, *section*, and *chunk* granularity
- **Automatic Section Detection**: Sections derived from document structure (Markdown headings by default, custom patterns supported)
- **No Extra Embedding Cost**: Section/document vectors are centroids of existing chunk embeddings
- **Section Metadata**: Pluggable extractors (heading path, keywords, word/char counts, or your own)

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

### Basic Usage

```python
from localvectordb import VectorDB

# Create or connect to a database
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

```python
from localvectordb import VectorDB

# Use HTTP server (automatically detected by URL)
db = VectorDB(
    "my_docs", 
    "http://localhost:5000",
    api_key="your_api_key"
)

# Same API as local database
doc_ids = db.upsert(["Remote document content"])
results = db.query("content", search_type="hybrid")
```

## 🖥️ Server Deployment

### Start the Server

```bash
# Quick start with defaults
lvdb serve

# Production configuration
lvdb --config ./config.toml serve --host 0.0.0.0 --port 5000
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
port = 5000
require_api_key = true
enable_rate_limiting = true
rate_limit = "100 per minute"
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

## 🤖 MCP Server (Claude Desktop / Claude Code)

LocalVectorDB ships a built-in [Model Context Protocol](https://modelcontextprotocol.io/)
server, so an LLM agent can search and manage your vector databases directly. It
works with **Claude Desktop**, **Claude Code**, and any other MCP client.

```bash
# Install with MCP support
pip install localvectordb[mcp]

# Start the server (read-only by default — safe for knowledge bases)
lvdb mcp serve

# Enable writes when you need them
lvdb mcp serve --mode read-write
```

Register it with Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "localvectordb": {
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

The server exposes focused tools — `query_database`, `find_related_documents`,
`filter_documents`, `get_document` (whole document or a chunk/range/section
portion), `list_databases`, and (in read-write mode) `upsert_documents`,
`create_database`, and more. Tool sets are configurable per deployment. See the
[MCP documentation](https://thomas-villani.github.io/localvectordb/mcp.html) for
the full tool list, configuration, and security guidance.

## 🟦 TypeScript SDK

A zero-dependency TypeScript/JavaScript client is available for the HTTP server
(Node.js 18+ and modern browsers):

```bash
npm install @localvectordb/sdk
```

```typescript
import { LocalVectorDBClient } from "@localvectordb/sdk";

const client = new LocalVectorDBClient({ baseUrl: "http://localhost:5000" });
const db = client.database("my_docs");

await db.upsert(["First document", "Second document"]);
const results = await db.query("search text", { search_type: "hybrid", k: 5 });
```

See [`sdk/js/README.md`](sdk/js/README.md) for the full SDK API.

## 📚 API Reference

### Core Methods

#### `upsert(documents, metadata=None, ids=None)`
Insert or update documents.

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

```python
# Vector search
results = db.query("search text", search_type="vector", k=5)

# Hybrid search with metadata filter
results = db.query(
    "machine learning",
    search_type="hybrid",
    vector_weight=0.7,
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

#### `filter(where=None, order_by=None, limit=None, offset=0)`
MongoDB-style filtering on metadata.

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
| `GET` | `/api/v1/{db}/info` | Database info |
| `POST` | `/api/v1/{db}/documents` | Upsert documents |
| `GET` | `/api/v1/{db}/documents/{id}` | Get document |
| `PATCH` | `/api/v1/{db}/documents/{id}` | Update document |
| `DELETE` | `/api/v1/{db}/documents/{id}` | Delete document |
| `POST` | `/api/v1/{db}/query` | Search documents |
| `POST` | `/api/v1/{db}/query/stream` | Stream results (SSE) |
| `POST` | `/api/v1/{db}/filter` | Filter documents |
| `POST` | `/api/v1/{db}/upload` | Upload files |
| `POST` | `/api/v1/{db}/compare` | Compare documents |
| `POST` | `/api/v1/{db}/nearest-neighbors` | Find similar documents |
| `POST` | `/api/v1/{db}/factcheck` | Fact-check text |

Example API usage:

```bash
# Create database
curl -X POST http://localhost:5000/api/v1/databases \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my_db"}'

# Search documents
curl -X POST http://localhost:5000/api/v1/my_db/query \
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
lvdb db mydb get doc_1 --json --metadata

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
- **Position Tracking**: Exact character positions for perfect reconstruction
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
| LaTeX, MediaWiki, Textile, archives, `.enex`, `.fb2`, CHM, Outlook | `localvectordb[file-extraction]` |
| OCR for scanned PDFs (Tesseract) | `localvectordb[file-extraction-ocr]` |

Extracted content is **Markdown**, preserving headings, tables, and lists for
better chunk boundaries.

```python
# Upload files via HTTP API
files = {'files': open('document.pdf', 'rb')}
response = requests.post(
    'http://localhost:5000/api/v1/mydb/upload',
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

### Docker Deployment

```dockerfile
FROM python:3.12-slim

RUN pip install localvectordb[server,file-extraction]

COPY config.toml /app/config.toml
WORKDIR /app

EXPOSE 5000
CMD ["lvdb", "--config", "config.toml", "serve"]
```

### Multi-Worker Setup

```bash
# Configure Redis registry
lvdb config set server.db_registry_type "RedisCache"
lvdb config set server.db_registry_settings '{"host": "redis", "port": 6379, "db": 1}'

# Start with multiple uvicorn workers
uvicorn "localvectordb_server.app:create_app" --factory --host 0.0.0.0 --port 5000 --workers 4
```

### Environment Variables

```bash
export LVDB_SERVER_CONFIG="/path/to/config.toml"
export LVDB_DATABASE_ROOT_DIR="/data/databases"
export LVDB_EMBEDDING_PROVIDER="ollama"
export LVDB_EMBEDDING_MODEL="nomic-embed-text"
export OPENAI_API_KEY="your-openai-key"  # if using OpenAI
```

## 🧪 Examples

### Research Paper Database

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

```python
# Create database for code files
db = VectorDB("code", "./data", 
              chunking_method="code-blocks",
              chunk_size=1000)

# Add Python files
import glob
for file_path in glob.glob("**/*.py", recursive=True):
    with open(file_path, 'r') as f:
        content = f.read()
    
    db.upsert(
        documents=[content],
        metadata=[{
            'file_path': file_path,
            'language': 'python',
            'last_modified': os.path.getmtime(file_path)
        }]
    )

# Search for specific functions
results = db.query("async def", search_type="keyword")
```

### Hierarchical (Section-Level) Retrieval

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