---
name: semantic-search
description: Build and query semantic search systems with LocalVectorDB. Use when the user wants to create a vector database, add documents, perform semantic/keyword/hybrid search, filter by metadata, or retrieve whole documents and sub-document portions (chunk/character/line ranges, sections, outlines). Covers document ingestion, chunking, embedding, and retrieval. Available from the Python API, the `lvdb` CLI, and the MCP server.
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

## Sub-Document Retrieval

`db.get()` returns a whole document. To retrieve only *part* of a document — a chunk
range, a character/line slice, a named Markdown section, or its outline — use
`get_document_portion()`. This is the same logic the CLI `get` command and the MCP
`get_document` tool use, so all three surfaces behave identically.

```python
from localvectordb.document_portions import get_document_portion

# Stored chunk(s) by 0-based index or inclusive range "M:N"
portion = get_document_portion(db, "doc_python", chunk="2:5")
print(portion.text)                       # selected chunks joined with blank lines
print(portion.chunks[0]["index"],
      portion.chunks[0]["position"])       # each chunk: index / content / position

# Character slice "M:N" (0-based, end-exclusive)
portion = get_document_portion(db, "doc_python", char_range="0:200")

# Line range "M:N" (1-based, inclusive)
portion = get_document_portion(db, "doc_python", line_range="10:20")

# Body of a Markdown section by heading (case-insensitive)
portion = get_document_portion(db, "doc_python", section="Installation")

# Section outline (headings, levels, start/end lines)
portion = get_document_portion(db, "doc_python", outline=True)
for item in portion.outline:
    print(item["level"], item["heading"], item["start_line"])
```

The selection arguments are mutually exclusive; passing more than one raises
`ValueError`. With no argument the whole document is returned. The result is a
`DocumentPortion` with:

- `document` - the full source `Document` (for `id`, `metadata`, timestamps)
- `mode` - which selection ran: `"document"`, `"chunk"`, `"range"`, `"lines"`, `"section"`, `"outline"`
- `text` - the portion as text (`None` for `outline`; use `outline` instead)
- `chunks` - `[{index, content, position}]` for `chunk` mode (else `None`)
- `outline` - the outline items for `outline` mode (else `None`)

## Factory Pattern (Local or Remote)

```python
from localvectordb import VectorDB

# Automatically picks LocalVectorDB or RemoteVectorDB
local_db = VectorDB("docs", "./local_path")
remote_db = VectorDB("docs", "http://localhost:8000", api_key="key")
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

## CLI and MCP Access

The same search-and-retrieve flow is available without writing Python, which is
useful when driving LocalVectorDB from a shell or an MCP-connected agent.

**CLI** (`lvdb`):

```bash
# Search (vector / keyword / hybrid); return chunks-with-context if wanted
lvdb db my_docs search "how do neural networks learn?" --search-type hybrid --limit 5
lvdb db my_docs search "optimization" --return-type context --context-window 2

# Whole document, or a portion (mutually exclusive selectors)
lvdb db my_docs get doc_python --json --metadata
lvdb db my_docs get doc_python --section "Installation"
lvdb db my_docs get doc_python --range 0:200        # also --chunk, --lines, --outline

# Documents related to an existing one (nearest neighbours)
lvdb db my_docs related doc_python --limit 5
```

**MCP tools** (read-only): `query_database` (search), `get_document` (whole document
or a `chunk`/`char_range`/`line_range`/`section`/`outline` portion), and
`find_related_documents` (nearest neighbours). See the `document-comparison` skill for
the similarity/comparison side of the API.

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
