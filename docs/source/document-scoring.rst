Document Scoring Methods
========================

When aggregating chunk-level search results into document-level scores, LocalVectorDB supports multiple scoring methods. Each method has different strengths depending on your use case.

Simple Methods
--------------

``"best"``
^^^^^^^^^^

Uses the highest-scoring chunk as the document score. Choose this when you want documents ranked by their single most relevant passage, regardless of overall document quality.

**Parameters:** None

``"worst"``
^^^^^^^^^^^

Uses the lowest-scoring chunk as the document score. Useful when you need documents where all content meets a minimum relevance threshold.

**Parameters:** None

``"average"``
^^^^^^^^^^^^^

Takes the arithmetic mean of all chunk scores. Good for documents where overall content quality matters more than peak relevance.

**Parameters:** None

``"weighted_average"``
^^^^^^^^^^^^^^^^^^^^^^

Computes a weighted average where chunk scores are normalized and used as weights. Emphasizes higher-scoring chunks while considering overall document quality.

**Parameters:** None

Advanced Methods
----------------

``"frequency_boost"`` (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Boosts the best chunk score based on the number of quality chunks found, rewarding documents with multiple relevant passages. Ideal for comprehensive documents where breadth of coverage indicates relevance.

**Parameters:**

* ``frequency_bias`` (0.0-1.0, default=0.3): Controls how much to boost scores based on chunk frequency. Higher values favor documents with more matching chunks.

``"harmonic_mean"``
^^^^^^^^^^^^^^^^^^^

Uses harmonic mean of top-scoring chunks with a coverage bonus for chunks above a quality threshold. Balances quality and quantity while being more conservative than arithmetic mean.

**Parameters:**

* ``max_chunks`` (int, default=5): Number of top-scoring chunks to include in harmonic mean calculation
* ``coverage_threshold`` (0.0-1.0, default=0.7): Score threshold above which chunks are considered "high-quality" and provide a coverage bonus

``"diminishing_returns"``
^^^^^^^^^^^^^^^^^^^^^^^^^

Applies exponential decay to chunk contributions, where later chunks have progressively less impact. Use when the first few relevant chunks are most important and additional matches provide diminishing value.

**Parameters:**

* ``decay_factor`` (0.0-1.0, default=0.8): Controls how quickly subsequent chunks lose influence. Lower values create steeper decay.

``"statistical"``
^^^^^^^^^^^^^^^^^

Combines multiple statistical measures: best score, mean score, consistency (inverse variance), and coverage ratio. Provides a comprehensive assessment balancing multiple quality factors.

**Parameters:**

* ``best_weight`` (0.0-1.0, default=0.6): Weight for the highest chunk score
* ``mean_weight`` (0.0-1.0, default=0.2): Weight for the average chunk score
* ``consistency_weight`` (0.0-1.0, default=0.1): Weight for score consistency (low variance bonus)
* ``coverage_weight`` (0.0-1.0, default=0.1): Weight for percentage of above-median chunks

``"robust_mean"``
^^^^^^^^^^^^^^^^^

Removes statistical outliers and applies position-based weighting to create a stable score less sensitive to anomalous chunks. Good for noisy data or when you want to avoid being skewed by a few very high/low scores.

**Parameters:**

* ``outlier_threshold`` (float, default=2.0): Z-score threshold for identifying outliers to remove
* ``position_decay`` (0.0-1.0, default=0.9): How much to penalize lower-ranked chunks in the final score

``"percentile"``
^^^^^^^^^^^^^^^^

Combines high and low percentile scores to balance peak relevance with overall quality. Useful when you want documents that have both strong matches and consistent relevance.

**Parameters:**

* ``primary_percentile`` (0.0-1.0, default=0.9): Higher percentile to sample (captures peak relevance)
* ``secondary_percentile`` (0.0-1.0, default=0.7): Lower percentile to sample (captures broader quality)
* ``primary_weight`` (0.0-1.0, default=0.7): How much to weight the primary vs secondary percentile

``"geometric_mean"``
^^^^^^^^^^^^^^^^^^^^

Uses geometric mean with stabilization to prevent zero scores from dominating. More conservative than arithmetic mean and good when you want all chunks to contribute meaningfully to the final score.

**Parameters:** None

Choosing a Method
-----------------

* **Single best passage matters most**: Use ``"best"``
* **Overall document quality important**: Use ``"average"`` or ``"weighted_average"``
* **Want to reward multiple relevant sections**: Use ``"frequency_boost"`` (default)
* **Need comprehensive quality assessment**: Use ``"statistical"``
* **Data has outliers or noise**: Use ``"robust_mean"``
* **Want balance of peak and consistent relevance**: Use ``"percentile"``
* **Conservative scoring**: Use ``"harmonic_mean"`` or ``"geometric_mean"``

How Raw Scores Are Computed
---------------------------

Before document-level aggregation is applied, each chunk receives a raw similarity
score between 0.0 and 1.0. The normalization depends on the search type.

Vector Search
^^^^^^^^^^^^^

FAISS returns raw distances which are converted to similarity scores by
``_distance_to_similarity``:

* **Inner Product (IP) index**: ``similarity = (distance + 1) / 2``, clamped to [0, 1].
  Assumes normalized embeddings where inner product ranges from -1 to 1.
* **L2 index**: ``similarity = 1 / (1 + distance)``. Larger distances map to lower
  similarity, approaching 0 for very distant vectors.

Keyword Search
^^^^^^^^^^^^^^

Keyword search uses SQLite FTS5 with the BM25 ranking function. FTS5 BM25 scores are
negative values where more negative means a better match. These are converted to
similarity scores using an exponential mapping:

.. code-block:: text

   similarity = 1.0 - min(1.0, exp(rank))

This produces scores in [0, 1] where better BM25 matches yield higher similarity.

Hybrid Search
^^^^^^^^^^^^^

Hybrid search runs vector and keyword searches independently, then merges the results
with a weighted linear combination:

.. code-block:: text

   final_score = vector_weight * vector_score + (1 - vector_weight) * keyword_score

The ``vector_weight`` parameter (default 0.7) controls the balance. Chunks appearing
in only one result set receive 0.0 for the missing component. The merged scores are
then filtered by ``score_threshold`` and passed to document-level aggregation.

Using Scoring Methods via the Server API
-----------------------------------------

All search endpoints accept ``document_scoring_method`` and ``document_scoring_options``
in the request body. These parameters are forwarded directly to the local database's
``query()`` method.

.. code-block:: bash

   # Unified query endpoint
   curl -X POST http://localhost:8000/api/v1/my_db/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning",
       "search_type": "hybrid",
       "return_type": "documents",
       "k": 10,
       "score_threshold": 0.3,
       "vector_weight": 0.7,
       "document_scoring_method": "statistical",
       "document_scoring_options": {
         "best_weight": 0.5,
         "mean_weight": 0.3,
         "consistency_weight": 0.1,
         "coverage_weight": 0.1
       }
     }'

The convenience endpoints (``/search/vector``, ``/search/keyword``, ``/search/hybrid``)
also accept these parameters with the same schema.

.. note::
   ``document_scoring_method`` and ``document_scoring_options`` only take effect when
   ``return_type`` is ``"documents"``. For chunk-level return types (``"chunks"``,
   ``"context"``, ``"enriched"``), raw chunk scores are returned directly.