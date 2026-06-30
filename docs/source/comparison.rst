Document Comparison & Visualization
=====================================

LocalVectorDB includes built-in methods for comparing documents at the document and chunk level,
finding nearest neighbours, computing pairwise similarity matrices, and visualising the embedding space.

.. contents:: On This Page
   :local:
   :depth: 2

Document-Level Comparison
-------------------------

Comparing Two Documents
^^^^^^^^^^^^^^^^^^^^^^^

Use ``compare_documents()`` to get the cosine similarity (normalised to [0, 1]) between two
documents based on their centroid embeddings:

.. code-block:: python

   score = db.compare_documents("doc_a", "doc_b")
   print(f"Similarity: {score:.3f}")

A score of 1.0 means the documents are identical in embedding space; 0.5 means they are
orthogonal; values approaching 0.0 mean they are as dissimilar as possible.

Finding Nearest Neighbours
^^^^^^^^^^^^^^^^^^^^^^^^^^

``nearest_neighbors()`` returns the *k* most similar documents to a reference document,
excluding the reference itself:

.. code-block:: python

   results = db.nearest_neighbors("doc_a", k=5)

   for r in results:
       print(f"  {r.id}: {r.score:.3f}")

Results are ``QueryResult`` objects with ``type="document"``, sorted by score descending.

**With filtering and thresholds:**

.. code-block:: python

   results = db.nearest_neighbors(
       "doc_a",
       k=5,
       score_threshold=0.5,               # minimum similarity to include
       filters={"category": "research"},   # metadata filter
   )

Pairwise Similarity Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute an NxN similarity matrix for all (or selected) documents:

.. code-block:: python

   # All documents
   matrix = db.pairwise_similarity_matrix()

   # Selected subset
   matrix = db.pairwise_similarity_matrix(doc_ids=["doc_a", "doc_b", "doc_c"])

The returned ``DocumentSimilarityMatrix`` contains:

- ``matrix`` -- ``np.ndarray`` of shape (N, N) with pairwise similarity scores
- ``doc_ids`` -- list of document IDs matching rows and columns
- ``embeddings`` -- ``np.ndarray`` of shape (N, D), the document embeddings used

.. code-block:: python

   print(matrix.doc_ids)   # ['doc_a', 'doc_b', 'doc_c']
   print(matrix.matrix)    # (3, 3) numpy array

Chunk-Level Comparison
----------------------

``compare_documents_detailed()`` reveals *where* two documents overlap and where they diverge
by aligning individual chunks:

.. code-block:: python

   result = db.compare_documents_detailed("doc_a", "doc_b", chunk_threshold=0.7)

   print(f"Overall similarity: {result.overall_similarity:.3f}")
   print(f"Matched in doc_a:   {result.matched_ratio_1:.1%}")
   print(f"Matched in doc_b:   {result.matched_ratio_2:.1%}")

   for a in result.chunk_alignments:
       print(f"  chunk {a.chunk_index_1} <-> chunk {a.chunk_index_2}: {a.similarity:.3f}")

   print(f"Unmatched in doc_a: {result.unmatched_chunks_1}")
   print(f"Unmatched in doc_b: {result.unmatched_chunks_2}")

How to interpret the result:

.. list-table::
   :header-rows: 1

   * - Scenario
     - overall_similarity
     - matched_ratio
     - Interpretation
   * - Near-identical
     - High (~0.9+)
     - High (~1.0)
     - Documents are very similar throughout
   * - Shared section
     - Moderate (~0.6)
     - Low (~0.3)
     - Some shared content, mostly different
   * - Completely different
     - Low (~0.3)
     - ~0.0
     - No meaningful overlap

The ``chunk_threshold`` parameter controls the minimum similarity for a chunk pair to count
as "matched".

Data Classes
^^^^^^^^^^^^

.. code-block:: python

   from localvectordb.core import (
       ChunkAlignment,
       DocumentComparisonResult,
       DocumentSimilarityMatrix,
   )

- ``ChunkAlignment`` -- ``chunk_index_1``, ``chunk_index_2``, ``similarity``
- ``DocumentComparisonResult`` -- ``doc_id_1``, ``doc_id_2``, ``overall_similarity``,
  ``chunk_alignments``, ``matched_ratio_1``, ``matched_ratio_2``,
  ``unmatched_chunks_1``, ``unmatched_chunks_2``
- ``DocumentSimilarityMatrix`` -- ``matrix``, ``doc_ids``, ``embeddings``

Visualization
-------------

The visualization module provides dimensionality reduction, clustering, and plotting utilities
for exploring the document embedding space.

Installation
^^^^^^^^^^^^

Visualization requires optional dependencies:

.. code-block:: bash

   # Core visualization (scikit-learn + matplotlib)
   pip install localvectordb[visualization]

   # Interactive plots (adds plotly)
   pip install localvectordb[visualization-interactive]

Convenience Methods
^^^^^^^^^^^^^^^^^^^

The database object provides two high-level methods that handle embedding extraction,
projection, and plotting in a single call.

**Embedding map:**

.. code-block:: python

   # 2D scatter plot of all documents
   fig = db.visualize_documents(method="tsne")
   fig.savefig("embedding_map.png")

   # Colour by a metadata field
   fig = db.visualize_documents(method="pca", color_by="category")

   # Cluster and colour by cluster
   fig = db.visualize_documents(method="tsne", n_clusters=4)

   # Interactive plotly plot
   fig = db.visualize_documents(method="pca", interactive=True)
   fig.show()

**Query overlay:**

Show how query strings relate to the document space. Query points are projected into the
same 2D space and displayed as distinct markers. Document dot sizes scale by relevance
to the queries.

.. code-block:: python

   fig = db.visualize_queries(
       queries=["web development", "neural networks"],
       method="pca",
   )

Standalone Visualization API
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For more control, use the visualization module directly.

**Dimensionality reduction:**

.. code-block:: python

   from localvectordb.visualization import reduce_dimensions

   # PCA
   projection = reduce_dimensions(embeddings, method="pca", doc_ids=ids)
   print(projection.coordinates.shape)     # (N, 2)
   print(projection.explained_variance)    # variance ratio per component

   # t-SNE
   projection = reduce_dimensions(embeddings, method="tsne", doc_ids=ids)

``reduce_dimensions()`` returns an ``EmbeddingProjection`` containing coordinates, the
fitted transformer (for projecting new points), and document IDs.

**Clustering:**

.. code-block:: python

   from localvectordb.visualization import cluster_embeddings, find_optimal_clusters

   # Auto-detect optimal k via silhouette analysis
   k = find_optimal_clusters(embeddings)
   clusters = cluster_embeddings(embeddings, n_clusters=k)

   print(clusters.labels)      # (N,) cluster assignments
   print(clusters.centroids)   # (K, D) cluster centres
   print(clusters.n_clusters)  # K

**Plotting:**

.. code-block:: python

   from localvectordb.visualization import (
       plot_embedding_map,
       plot_similarity_matrix,
       plot_clusters,
       plot_similarity_graph,
   )

   # Scatter plot
   fig = plot_embedding_map(projection, color_by=labels)

   # Similarity heatmap
   matrix = db.pairwise_similarity_matrix()
   fig = plot_similarity_matrix(matrix)

   # Cluster plot
   fig = plot_clusters(projection, clusters)

   # Similarity graph (nodes = docs, edges = similarity above threshold)
   fig = plot_similarity_graph(matrix, threshold=0.5)

**Graph structure (for custom processing):**

.. code-block:: python

   from localvectordb.visualization import build_similarity_graph

   graph = build_similarity_graph(matrix, threshold=0.4)
   # graph["nodes"] = [{"id": "doc_a", "index": 0}, ...]
   # graph["edges"] = [{"source": "doc_a", "target": "doc_b", "weight": 0.82}, ...]

**Interactive plots (plotly):**

.. code-block:: python

   from localvectordb.visualization import (
       plot_embedding_map_interactive,
       plot_similarity_matrix_interactive,
   )

   fig = plot_embedding_map_interactive(projection)
   fig.show()

   fig = plot_similarity_matrix_interactive(matrix)
   fig.show()

Common Patterns
---------------

Finding Duplicate Documents
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   matrix = db.pairwise_similarity_matrix()

   for i in range(len(matrix.doc_ids)):
       for j in range(i + 1, len(matrix.doc_ids)):
           if matrix.matrix[i, j] >= 0.95:
               print(f"Duplicate: {matrix.doc_ids[i]} <-> {matrix.doc_ids[j]}")

Topic Clustering
^^^^^^^^^^^^^^^^

.. code-block:: python

   from localvectordb.visualization import cluster_embeddings, find_optimal_clusters

   # pairwise_similarity_matrix() returns the document embeddings and IDs for all
   # documents (pass doc_ids=[...] to restrict to a subset).
   matrix = db.pairwise_similarity_matrix()
   embeddings, doc_ids = matrix.embeddings, matrix.doc_ids
   k = find_optimal_clusters(embeddings)
   clusters = cluster_embeddings(embeddings, n_clusters=k)

   for cid in range(clusters.n_clusters):
       members = [doc_ids[i] for i, l in enumerate(clusters.labels) if l == cid]
       print(f"Cluster {cid}: {members}")

Content Gap Analysis
^^^^^^^^^^^^^^^^^^^^

Use detailed comparison to find what was added between document versions:

.. code-block:: python

   result = db.compare_documents_detailed("doc_v1", "doc_v2", chunk_threshold=0.6)

   if result.unmatched_chunks_2:
       doc = db.get("doc_v2")
       # db.get() returns document content/metadata only; re-derive the chunks
       # (with their indices) using the database's chunker.
       chunks = db.chunker.chunk(doc.content)
       print("New content in v2:")
       for chunk in chunks:
           if chunk.index in result.unmatched_chunks_2:
               print(f"  [{chunk.index}] {chunk.content[:100]}...")
