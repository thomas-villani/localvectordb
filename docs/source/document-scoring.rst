Document Scoring Methods
========================

When aggregating chunk-level search results into document-level scores, LocalVectorDB supports three scoring methods. Each has a different strength depending on your use case.

.. note::

   Earlier releases exposed eight additional heuristic methods (``worst``,
   ``weighted_average``, ``harmonic_mean``, ``diminishing_returns``, ``statistical``,
   ``robust_mean``, ``percentile``, ``geometric_mean``). They were removed in v0.1.0:
   measured on the NFCorpus retrieval benchmark none of them beat the three methods
   below, they overlapped heavily, and several were non-monotonic (adding a low-scoring
   chunk could reorder documents). Passing a removed name now raises ``ValueError``.

Methods
-------

``"best"``
^^^^^^^^^^

Uses the highest-scoring chunk as the document score. Choose this when you want documents ranked by their single most relevant passage, regardless of overall document quality.

**Parameters:** None

``"average"``
^^^^^^^^^^^^^

Takes the arithmetic mean of all chunk scores. Good for documents where overall content quality matters more than peak relevance.

**Parameters:** None

``"frequency_boost"`` (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Boosts the best chunk score based on the number of quality chunks found, rewarding documents with multiple relevant passages. Ideal for comprehensive documents where breadth of coverage indicates relevance.

**Parameters:**

* ``frequency_bias`` (0.0-1.0, default=0.3): Controls how much to boost scores based on chunk frequency. Higher values favor documents with more matching chunks.

Choosing a Method
-----------------

* **Single best passage matters most**: Use ``"best"``
* **Overall document quality important**: Use ``"average"``
* **Want to reward multiple relevant sections**: Use ``"frequency_boost"`` (default)

How Raw Scores Are Computed
---------------------------

Before document-level aggregation is applied, each chunk receives a raw similarity
score between 0.0 and 1.0. The normalization depends on the search type.

Vector Search
^^^^^^^^^^^^^

FAISS returns raw distances which are converted to similarity scores by
``_distance_to_similarity``:

* **Inner Product (IP) index**: ``similarity = (distance + 1) / 2``, clamped to [0, 1].
  This mapping assumes an inner product in ``[-1, 1]``, i.e. unit-norm vectors. The
  library guarantees that by L2-normalizing at the write and query boundary whenever
  the index metric is inner product, so IP scoring is correct regardless of whether
  the embedding provider's own ``normalize`` option is set. (No normalization is
  applied to an L2 index, so its geometry -- and the ``normalize`` option's effect on
  it -- is unchanged.)
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
Ranking is unaffected by the shape of this curve, because FTS5 orders by the raw
BM25 score before the mapping is applied.

.. note::

   This mapping saturates. Any reasonably good BM25 match lands within about
   ``2e-05`` of 1.0, and past a rank of roughly ``-36`` it reaches exactly 1.0.
   Treat the absolute value of a keyword score as "matched", not as a measure of
   how well. Hybrid fusion therefore normalizes the *raw* BM25 rank, never this
   number.

Hybrid Search
^^^^^^^^^^^^^

Hybrid search runs vector and keyword searches independently, then fuses them with
**relative-score fusion**: each leg's scores are min-max normalized within the
current query's candidate pool, and the normalized values are blended:

.. code-block:: text

   v = (vector_score - min_vector) / (max_vector - min_vector)
   k = (-bm25 - min(-bm25)) / (max(-bm25) - min(-bm25))

   final_score = vector_weight * v + (1 - vector_weight) * k

Normalizing first is what makes ``vector_weight`` (default 0.5) an actual blend. The
two legs are otherwise on incompatible, corpus-dependent scales -- a bounded
similarity against raw BM25 -- and summing them directly lets whichever leg happens
to span the wider range decide the ranking. Chunks appearing in only one result set
receive 0.0 for the missing component. The fused scores are then filtered by
``score_threshold`` and passed to document-level aggregation.

.. warning::

   Hybrid scores are **relative to the query's own candidate pool**. They are
   comparable within a single result set, but not across queries, and not across
   different values of ``k`` (which changes the pool size). A ``score_threshold``
   on a hybrid query therefore selects by rank position within the pool rather than
   by absolute match quality. Two further consequences: the best chunk of a leg
   always normalizes to 1.0, and the worst normalizes to 0.0 -- indistinguishable
   from a chunk that leg never retrieved at all.

Using Scoring Methods via the Server API
-----------------------------------------

All search endpoints accept ``document_scoring_method`` and ``document_scoring_options``
in the request body. These parameters are forwarded directly to the local database's
``query()`` method.

.. code-block:: bash

   # Unified query endpoint
   curl -X POST http://localhost:8000/api/v1/databases/my_db/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "machine learning",
       "search_type": "hybrid",
       "return_type": "documents",
       "k": 10,
       "score_threshold": 0.3,
       "vector_weight": 0.5,
       "document_scoring_method": "frequency_boost",
       "document_scoring_options": {
         "frequency_bias": 0.3
       }
     }'

The convenience endpoints (``/search/vector``, ``/search/keyword``, ``/search/hybrid``)
also accept these parameters with the same schema.

.. note::
   ``document_scoring_method`` and ``document_scoring_options`` only take effect when
   ``return_type`` is ``"documents"``. For chunk-level return types (``"chunks"``,
   ``"context"``, ``"enriched"``), raw chunk scores are returned directly.