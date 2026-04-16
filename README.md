# LocalVectorDB

A high-performance, document-first vector database with SQLite + FAISS backend, featuring intelligent chunking, unified search, and optional HTTP server.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm%20Noncommercial-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)

## ✨ Features

### 🗃️ **Document-First Architecture**
- **Smart Chunking**: Position-tracking chunking with perfect document reconstruction
- **Metadata Schema**: Structured, indexed metadata fields with validation
- **Unified API**: Single interface for vector, keyword, and hybrid search

### 🔍 **Advanced Search**
- **Vector Search**: Semantic similarity using Ollama or OpenAI embeddings
- **Keyword Search**: Full-text search with SQLite FTS5
- **Hybrid Search**: Combined vector + keyword with configurable weighting
- **Metadata Filtering**: SQL-like queries on structured metadata

### 🌐 **Flexible Deployment**
- **Local Database**: Direct SQLite + FAISS for maximum performance
- **HTTP Server**: RESTful API with permission-based authentication, rate limiting, CORS
- **Remote Client**: Seamless local/remote switching via factory pattern
- **Multi-Worker**: Redis-based coordination for distributed deployments

### 📄 **File Processing**
- **Text Extraction**: PDF, DOCX, PPTX, XLSX, RTF, EPUB support
- **Batch Upload**: Multi-file processing with metadata extraction
- **Format Detection**: Automatic MIME type detection and processing

### 🛠️ **Developer Experience**
- **CLI Tools**: Database management, server control, interactive shell
- **Configuration**: TOML/JSON config with environment variable support
- **Comprehensive Logging**: Structured logging with performance monitoring
- **Type Safety**: Full type annotations and validation

## 🚀 Quick Start

### Installation

```bash
pip install localvectordb

# For server features (optional)
pip install localvectordb[server]

# For all file extraction formats (optional)
pip install localvectordb[server,file-extraction]
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
    filters={'author': 'Jane Doe', 'tags': {'contains': 'python'}}
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
lvdb serve --host 0.0.0.0 --port 5000 --config ./config.toml
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

#### `query(query, search_type='vector', k=10, filters=None)`
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

#### `filter(where=None, sql=None, order_by=None, limit=None)`
SQL-like filtering on metadata.

```python
# Simple filters
docs = db.filter(where={"author": "Jane Doe", "status": "published"})

# Complex queries
docs = db.filter(
    where={"created_date": {">=": "2024-01-01"}},
    order_by="created_date DESC",
    limit=10
)

# Raw SQL
docs = db.filter(sql="author LIKE '%Smith%' AND rating > 4.0")
```

### HTTP API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/databases` | List databases |
| `POST` | `/api/v1/databases` | Create database |
| `GET` | `/api/v1/{db}/info` | Database info |
| `POST` | `/api/v1/{db}/documents` | Upsert documents |
| `GET` | `/api/v1/{db}/documents/{id}` | Get document |
| `PUT` | `/api/v1/{db}/documents/{id}` | Update document |
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

# Get document
lvdb db mydb get doc_1 --json --metadata

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

Supports automatic text extraction from various formats:

| Format | Library | Status |
|--------|---------|--------|
| PDF | pdfplumber, PyPDF2 | ✅ |
| DOCX | python-docx | ✅ |
| PPTX | python-pptx | ✅ |
| XLSX | openpyxl | ✅ |
| RTF | striprtf | ✅ |
| EPUB | ebooklib | ✅ |
| TXT, MD, etc. | Built-in | ✅ |

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
- `provider`: "ollama" or "openai"
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
CMD ["lvdb", "serve", "--config", "config.toml"]
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
    filters={"publication_date": {">=": "2017-01-01"}},
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

## 🤝 Contributing

Contributions are welcome for noncommercial use under the project's license terms.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/njii-comet/localvectordb.git
cd localvectordb

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Start development server
lvdb serve --debug
```

## 📄 License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) — see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software for any noncommercial purpose. For commercial licensing inquiries, please contact thomas.villani@njii.com.

## 🆘 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/njii-comet/localvectordb/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/njii-comet/localvectordb/discussions)
- 📧 **Contact**: thomas.villani@njii.com

## 🙏 Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [SQLite](https://sqlite.org/) for the document database
- [Ollama](https://ollama.ai/) for local embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the HTTP server
- [Click](https://click.palletsprojects.com/) for the Click CLI library
- All the contributors who made this project possible