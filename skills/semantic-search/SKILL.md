---
name: semantic-search
description: Build and query semantic search systems with LocalVectorDB. Use when the user wants to create a vector database, add documents, perform semantic/keyword/hybrid search, or filter by metadata. Covers document ingestion, chunking, embedding, and retrieval.
license: MIT
compatibility: Requires Python 3.12+, faiss-cpu, and an embedding provider (Ollama, OpenAI, or mock for testing).
metadata:
  author: localvectordb
  version: "1.0"
---

# Semantic Search with LocalVectorDB

LocalVectorDB is a document-first vector database built on SQLite + FAISS with pluggable embedding providers. This skill covers creating databases, ingesting documents, and running semantic, keyword, and hybrid searches.

## Quick Start

```python
from localvectordb import LocalVectorDB
from localvectordb.core import MetadataField, MetadataFieldType

# Create a database with typed metadata schema
db = LocalVectorDB(
    name="my_docs",
    base_path="./vector_db",
    metadata_schema={
        "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        "tags": MetadataField(type=MetadataFieldType.JSON),
    },
    embedding_provider="ollama",       # or "openai", "mock"
    embedding_model="nomic-embed-text",
    chunking_method="sentences",       # or "tokens", "paragraphs", "code_blocks"
    chunk_size=500,
    chunk_overlap=1,
)
```

## Adding Documents

Use `upsert()` for insert-or-update semantics, or `insert()` for strict insert (raises on duplicates).

```python
# Add documents with metadata
doc_ids = db.upsert(
    documents=[
        "Python is a high-level programming language known for simplicity.",
        "Machine learning enables computers to learn from data.",
        "Vector databases store high-dimensional embeddings for similarity search.",
    ],
    metadata=[
        {"author": "Alice", "category": "programming", "tags": ["python"]},
        {"author": "Bob", "category": "ai", "tags": ["ml", "data"]},
        {"author": "Carol", "category": "database", "tags": ["vectors", "search"]},
    ],
    ids=["doc_python", "doc_ml", "doc_vectordb"],  # optional, auto-generated if omitted
)
```

### Batch ingestion

For large datasets, `upsert()` and `insert()` accept `batch_size` to control memory:

```python
db.upsert(documents=large_list, metadata=meta_list, batch_size=50)
```

### Deduplication

Pass `similarity_threshold` to skip near-duplicate documents:

```python
db.upsert(documents=texts, similarity_threshold=0.95)
```

## Searching

The unified `query()` method supports three search types.

### Vector search (semantic similarity)

```python
results = db.query(
    "how do neural networks learn?",
    search_type="vector",
    k=5,
    score_threshold=0.3,
)
for r in results:
    print(f"[{r.score:.3f}] {r.id}: {r.content[:80]}...")
```

### Keyword search (SQLite FTS5)

```python
results = db.query("python programming", search_type="keyword", k=5)
```

### Hybrid search (vector + keyword combined)

```python
results = db.query(
    "python machine learning",
    search_type="hybrid",
    k=10,
    hybrid_weights=(0.7, 0.3),  # (vector_weight, keyword_weight)
)
```

### Return types

Set `return_type` to control what comes back:

```python
# Get full documents (default)
results = db.query("query", return_type="documents")

# Get individual chunks with position info
results = db.query("query", return_type="chunks")
```

### QueryResult fields

Each result is a `QueryResult` with:
- `id` - Document or chunk ID
- `score` - Normalised similarity score (0-1, higher is better)
- `type` - `"document"`, `"chunk"`, `"section"`, etc.
- `content` - The matched text
- `metadata` - Document metadata dict
- `document_id` - Parent document ID (for chunk results)
- `position` - `ChunkPosition` with `start`, `end`, `line`, `column` (for chunks)

## Filtering by Metadata

```python
# Filter without search
docs = db.filter(where={"category": "ai"}, limit=10)

# Combine search with metadata filters
results = db.query(
    "deep learning",
    search_type="vector",
    k=5,
    filters={"author": "Bob"},
)
```

## CRUD Operations

```python
# Get documents by ID
doc = db.get("doc_python")
docs = db.get(["doc_python", "doc_ml"])

# Check existence
exists = db.exists("doc_python")  # True/False

# Count documents
total = db.count()
filtered = db.count(filters={"category": "ai"})

# Update content or metadata
db.update("doc_python", content="Updated content", metadata={"category": "updated"})

# Delete documents
db.delete("doc_python")
db.delete(["doc_ml", "doc_vectordb"])
```

## Factory Pattern (Local or Remote)

```python
from localvectordb import VectorDB

# Automatically picks LocalVectorDB or RemoteVectorDB
local_db = VectorDB("docs", "./local_path")
remote_db = VectorDB("docs", "http://localhost:5000", api_key="key")
```

## Embedding Providers

Available providers (configured via `embedding_provider` parameter):

| Provider | Value | Notes |
|----------|-------|-------|
| Ollama | `"ollama"` | Local models, default |
| OpenAI | `"openai"` | Requires API key in env |
| Google | `"google"` | Requires API key |
| Jina | `"jina"` | Requires API key |
| Sentence Transformers | `"sentence_transformers"` | Local, requires torch |
| Mock | `"mock"` | Deterministic, for testing |

Pass provider-specific config via `embedding_config`:

```python
db = LocalVectorDB(
    name="openai_db",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    embedding_config={"api_key": "sk-..."},
)
```

## Chunking Strategies

| Strategy | Value | Best for |
|----------|-------|----------|
| Sentences | `"sentences"` | General text (default) |
| Tokens | `"tokens"` | Precise token control |
| Paragraphs | `"paragraphs"` | Long-form documents |
| Code blocks | `"code_blocks"` | Source code |
| Lines | `"lines"` | Log files, CSVs |
| Sections | `"sections"` | Markdown with headers |

## Metadata Schema Types

| Type | Python types | Notes |
|------|-------------|-------|
| `TEXT` | `str` | Set `indexed=True` for FTS |
| `INTEGER` | `int` | |
| `REAL` | `int`, `float` | |
| `BOOLEAN` | `bool`, `int` | |
| `DATE` | `datetime`, `str` | ISO 8601 strings |
| `JSON` | `dict`, `list` | Stored as JSON |

Set `embedding_enabled=True` on TEXT/JSON fields for per-field vector search.

## In-Memory Databases

For testing or ephemeral use:

```python
db = LocalVectorDB(name=":memory:", embedding_provider="mock", embedding_model="test")
```

## Lifecycle

Always close the database when done:

```python
db.close()

# Or use as context manager pattern - close in finally block
try:
    db = LocalVectorDB(...)
    # ... use db
finally:
    db.close()
```
