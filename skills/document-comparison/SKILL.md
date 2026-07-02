---
name: document-comparison
description: Compare documents, find similar/related documents, build similarity matrices, and visualize embedding spaces with LocalVectorDB. Use when the user wants to measure document similarity, find nearest neighbors ("more like this"), detect partial overlap between documents, cluster documents, or create embedding visualizations. Nearest-neighbour search is also available via the `lvdb ... related` CLI command and the MCP `find_related_documents` tool.
license: MIT
compatibility: Requires Python 3.12+, localvectordb. Visualization features require scikit-learn and matplotlib (pip install localvectordb[visualization]).
metadata:
  author: localvectordb
  version: "1.0"
---

# Document Comparison & Visualization with LocalVectorDB

LocalVectorDB provides built-in methods for comparing documents at both the document and chunk level, finding nearest neighbors, computing similarity matrices, and visualizing the embedding space.

## Setup

```python
from localvectordb import LocalVectorDB

db = LocalVectorDB(
    name="my_docs",
    base_path="./vector_db",
    embedding_provider="ollama",
    embedding_model="nomic-embed-text",
)

# Add some documents
db.upsert(
    documents=[
        "Python is a versatile programming language used in web development and data science.",
        "JavaScript powers interactive web applications and runs in the browser.",
        "Machine learning algorithms learn patterns from data to make predictions.",
        "Deep learning uses neural networks with multiple layers for complex tasks.",
        "SQL databases store structured data in tables with defined schemas.",
    ],
    ids=["python", "javascript", "ml", "deep-learning", "sql"],
)
```

## Comparing Two Documents

Get a single similarity score (cosine similarity normalised to [0, 1]):

```python
score = db.compare_documents("python", "javascript")
print(f"Similarity: {score:.3f}")  # e.g. 0.72

score = db.compare_documents("python", "sql")
print(f"Similarity: {score:.3f}")  # e.g. 0.45
```

## Finding Nearest Neighbors

Find the k most similar documents to a reference document:

```python
results = db.nearest_neighbors("ml", k=3)

for r in results:
    print(f"  {r.id}: {r.score:.3f} - {r.content[:60]}...")
```

### With filtering

```python
# Only return neighbours matching metadata criteria
results = db.nearest_neighbors(
    "ml",
    k=3,
    score_threshold=0.5,          # minimum similarity
    filters={"category": "ai"},   # metadata filter
)
```

### Result type

`nearest_neighbors()` returns `List[QueryResult]` with `type="document"`, sorted by score descending. The reference document is excluded from results.

### CLI and MCP

The same "find related documents" capability is available without Python:

```bash
# CLI: documents most related to a reference document
lvdb db my_docs related ml --limit 3 --score-threshold 0.5
```

From an MCP-connected agent, call the read-only `find_related_documents` tool
(arguments: `database_name`, `document_id`, `k`, `score_threshold`, `filters`).

## Pairwise Similarity Matrix

Compute an NxN similarity matrix for all or selected documents:

```python
# All documents
matrix = db.pairwise_similarity_matrix()

print(matrix.doc_ids)     # ["python", "javascript", "ml", "deep-learning", "sql"]
print(matrix.matrix)      # (5, 5) numpy array of similarity scores
print(matrix.embeddings)  # (5, D) document embeddings used

# Selected documents only
matrix = db.pairwise_similarity_matrix(doc_ids=["python", "javascript", "sql"])
# matrix.matrix is (3, 3)
```

The `DocumentSimilarityMatrix` dataclass contains:
- `matrix` - `np.ndarray` of shape (N, N) with pairwise similarity scores
- `doc_ids` - List of document IDs matching rows/columns
- `embeddings` - `np.ndarray` of shape (N, D) with the document embeddings

## Detailed Chunk-Level Comparison

Detect *partial* similarity between documents. This reveals which sections overlap and which diverge:

```python
result = db.compare_documents_detailed("python", "javascript", chunk_threshold=0.7)

print(f"Overall similarity: {result.overall_similarity:.3f}")
print(f"Matched chunks in doc 1: {result.matched_ratio_1:.1%}")
print(f"Matched chunks in doc 2: {result.matched_ratio_2:.1%}")

# See which chunks align
for alignment in result.chunk_alignments:
    print(f"  Chunk {alignment.chunk_index_1} <-> Chunk {alignment.chunk_index_2}: "
          f"{alignment.similarity:.3f}")

# See which chunks have NO good match
print(f"Unmatched in doc 1: {result.unmatched_chunks_1}")
print(f"Unmatched in doc 2: {result.unmatched_chunks_2}")
```

### Interpreting results

| Scenario | overall_similarity | matched_ratio | Interpretation |
|----------|-------------------|---------------|----------------|
| Near-identical docs | High (~0.9+) | High (~1.0) | Documents are very similar throughout |
| Shared section | Moderate (~0.6) | Low (~0.3) | Documents share some content but mostly differ |
| Completely different | Low (~0.3) | ~0.0 | No meaningful overlap |

The `DocumentComparisonResult` dataclass contains:
- `doc_id_1`, `doc_id_2` - The compared document IDs
- `overall_similarity` - Centroid-level cosine similarity
- `chunk_alignments` - `List[ChunkAlignment]` sorted by similarity descending
- `matched_ratio_1` - Fraction of doc_1 chunks matched above threshold
- `matched_ratio_2` - Fraction of doc_2 chunks matched above threshold
- `unmatched_chunks_1`, `unmatched_chunks_2` - Chunk indices with no good match

## Visualization

Install visualization dependencies first:

```bash
pip install localvectordb[visualization]
# For interactive plotly plots:
pip install localvectordb[visualization-interactive]
```

### Embedding map (2D scatter plot)

```python
fig = db.visualize_documents(method="tsne")  # or "pca"
fig.savefig("embedding_map.png")
```

### Colour by metadata

```python
fig = db.visualize_documents(method="pca", color_by="category")
```

### Cluster documents

```python
fig = db.visualize_documents(method="tsne", n_clusters=3)
```

### Query overlay

Show how queries relate to the document space:

```python
fig = db.visualize_queries(
    queries=["web development frameworks", "neural network architectures"],
    method="pca",
)
```

### Interactive plots (plotly)

```python
fig = db.visualize_documents(method="pca", interactive=True)
fig.show()  # Opens in browser
```

### Similarity heatmap

```python
from localvectordb.visualization import plot_similarity_matrix

matrix = db.pairwise_similarity_matrix()
fig = plot_similarity_matrix(matrix, title="Document Similarities")
fig.savefig("similarity_heatmap.png")
```

### Similarity graph

Visualise documents as a network where edges connect similar documents:

```python
from localvectordb.visualization import plot_similarity_graph

matrix = db.pairwise_similarity_matrix()
fig = plot_similarity_graph(matrix, threshold=0.5)
fig.savefig("similarity_graph.png")
```

### Standalone visualization API

For more control, use the visualization module directly:

```python
from localvectordb.visualization import (
    reduce_dimensions,
    cluster_embeddings,
    find_optimal_clusters,
    plot_embedding_map,
    plot_clusters,
    build_similarity_graph,
)

# Get embeddings manually
embeddings, doc_ids = db._get_document_embeddings_batch(["python", "javascript", "ml"])

# Reduce dimensions
projection = reduce_dimensions(embeddings, method="pca", doc_ids=doc_ids)

# Cluster
clusters = cluster_embeddings(embeddings, n_clusters=2)
# Or auto-detect optimal k:
optimal_k = find_optimal_clusters(embeddings)
clusters = cluster_embeddings(embeddings, n_clusters=optimal_k)

# Plot
fig = plot_clusters(projection, clusters, title="Document Clusters")

# Build graph structure (for custom processing)
graph = build_similarity_graph(matrix, threshold=0.4)
# graph = {"nodes": [...], "edges": [{"source": ..., "target": ..., "weight": ...}]}
```

## Common Patterns

### Find duplicate or near-duplicate documents

```python
matrix = db.pairwise_similarity_matrix()
threshold = 0.95

for i in range(len(matrix.doc_ids)):
    for j in range(i + 1, len(matrix.doc_ids)):
        if matrix.matrix[i, j] >= threshold:
            print(f"Near-duplicate: {matrix.doc_ids[i]} <-> {matrix.doc_ids[j]} "
                  f"({matrix.matrix[i, j]:.3f})")
```

### Group documents by topic

```python
from localvectordb.visualization import cluster_embeddings, find_optimal_clusters

embeddings, doc_ids = db._get_document_embeddings_batch(None)  # all docs
k = find_optimal_clusters(embeddings)
clusters = cluster_embeddings(embeddings, n_clusters=k)

# Print groups
for cluster_id in range(clusters.n_clusters):
    members = [doc_ids[i] for i, label in enumerate(clusters.labels) if label == cluster_id]
    print(f"Cluster {cluster_id}: {members}")
```

### Content gap analysis

Use detailed comparison to find what's missing between two documents:

```python
result = db.compare_documents_detailed("doc_v1", "doc_v2", chunk_threshold=0.6)

if result.unmatched_chunks_2:
    print(f"New content in v2 (chunks {result.unmatched_chunks_2}):")
    doc_v2 = db.get("doc_v2")
    for chunk in doc_v2.chunks or []:
        if chunk.index in result.unmatched_chunks_2:
            print(f"  [{chunk.index}] {chunk.content[:100]}...")
```
