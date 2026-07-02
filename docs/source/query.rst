Query Types and Return Modes
============================

LocalVectorDB provides powerful and flexible query capabilities with multiple search types and return modes. This guide covers all available options and when to use them.

Search Types
------------

LocalVectorDB supports three complementary search approaches:

``search_type="vector"``
^^^^^^^^^^^^^^^^^^^^^^^^

Performs semantic similarity search using vector embeddings. Best for finding conceptually related content even when exact keywords don't match.

.. code-block:: python

    # Find documents about machine learning concepts
    results = db.query(
        "artificial intelligence algorithms", 
        search_type="vector"
    )

**When to use:**
- Finding conceptually similar content
- Cross-language or synonym matching
- Abstract concept queries
- When keyword matching is too restrictive

``search_type="keyword"`` 
^^^^^^^^^^^^^^^^^^^^^^^^^

Uses full-text search (FTS5) to find documents containing specific terms. Ideal for exact phrase matching and traditional text search.

.. code-block:: python

    # Find documents containing specific terms
    results = db.query(
        "machine learning", 
        search_type="keyword"
    )

**When to use:**
- Looking for specific terminology
- Exact phrase matching
- Technical terms or proper nouns
- When you need precise keyword matches

``search_type="hybrid"`` 
^^^^^^^^^^^^^^^^^^^^^^^^

Combines vector and keyword search with configurable weighting. Provides the best of both semantic understanding and precise term matching.

.. code-block:: python

    # Balanced semantic and keyword search
    results = db.query(
        "neural network architectures", 
        search_type="hybrid",
        vector_weight=0.7  # 70% vector, 30% keyword
    )

**When to use:**
- Most general-purpose searches (recommended default)
- When you want both semantic and exact matches
- Balancing precision and recall
- When unsure which search type is best

Return Types
------------

LocalVectorDB offers four return modes optimized for different use cases:

``return_type="documents"`` (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns complete documents with aggregated scores from all matching chunks. Uses document scoring methods to combine chunk-level results.
There are a number of different valid inputs for the ``document_scoring_method`` parameter, which modify how the
similarity score for the document is calculated. For a list of possible methods, see :doc:`document scoring methods<document-scoring>`.

.. code-block:: python

    # Get full documents ranked by relevance
    results = db.query(
        "machine learning", 
        return_type="documents",
        document_scoring_method="frequency_boost"
    )
    
    for doc in results:
        print(f"Document: {doc.id}")
        print(f"Score: {doc.score}")
        print(f"Content: {doc.content[:200]}...")

**When to use:**
- Want complete document context
- Need to see full content
- Ranking documents by overall relevance
- Traditional document retrieval

``return_type="chunks"``
^^^^^^^^^^^^^^^^^^^^^^^^

Returns individual matching chunks with their positions and metadata. Provides fine-grained access to specific relevant passages.

.. code-block:: python

    # Get specific matching passages
    results = db.query(
        "neural networks", 
        return_type="chunks"
    )
    
    for chunk in results:
        print(f"Chunk: {chunk.id}")
        print(f"Document: {chunk.document_id}")
        print(f"Position: {chunk.position}")
        print(f"Content: {chunk.content}")

**When to use:**
- Need specific relevant passages
- Building search result snippets
- Fine-grained relevance analysis
- When document context isn't needed

``return_type="context"``
^^^^^^^^^^^^^^^^^^^^^^^^^

Returns matching chunks enhanced with surrounding chunks for better readability. Combines the target chunk with neighboring chunks based on position.

.. code-block:: python

    # Get chunks with surrounding context
    results = db.query(
        "deep learning",
        return_type="context",
        context_window=2  # Include 2 chunks before/after
    )

    for result in results:
        print(f"Context: {result.id}")
        print(f"Original chunk: {result.metadata['_original_chunk_index']}")
        print(f"Context spans {result.metadata['_context_chunk_count']} chunks")
        print(f"Content: {result.content}")

**Sizing context by tokens, words, or characters:**

By default ``context_window`` counts *chunks*. Set ``context_unit`` to
``"tokens"``, ``"words"``, or ``"characters"`` to instead treat ``context_window``
as an approximate budget for the assembled context. Neighbouring chunks are added
whole and greedily (the matched chunk is always kept) until the next one would
exceed the budget, so the returned content never overshoots — unless a *single*
chunk is already larger than the budget.

.. code-block:: python

    # "Give me roughly 500 tokens of context around each match"
    results = db.query(
        "deep learning",
        return_type="context",
        context_window=500,
        context_unit="tokens",   # or "words" / "characters"
    )

Because whole chunks are kept, the result can fall short of the budget by up to
one chunk. To guarantee a hard upper bound (for example when packing an LLM
context window), set ``context_truncate=True`` — the assembled text is then cut to
exactly the budget (tokens via ``tiktoken``; words/characters back off to a
whitespace boundary). Truncated results are flagged with
``metadata["_context_truncated"] = True`` and their end position becomes
approximate.

.. code-block:: python

    results = db.query(
        "deep learning",
        return_type="context",
        context_window=500,
        context_unit="tokens",
        context_truncate=True,   # never exceed 500 tokens, even for one big chunk
    )

The chosen unit is always recorded on each result as
``metadata["_context_unit"]``.

**When to use:**

- Need readable context around matches
- Preserving document flow and coherence
- Creating human-readable excerpts
- When individual chunks lack sufficient context
- Packing a fixed token/character budget (e.g. an LLM prompt) — use
  ``context_unit`` + ``context_truncate``

``return_type="enriched"``
^^^^^^^^^^^^^^^^^^^^^^^^^^

**New in this release!** Returns chunks enhanced with semantically similar chunks from the same document. Uses intra-document similarity to find the most relevant related content.

.. code-block:: python

    # Get semantically enriched results
    results = db.query(
        "machine learning",
        return_type="enriched",
        context_window=3  # Include up to 3 similar chunks
    )

    # ...or size the enrichment by a token/word/character budget instead:
    results = db.query(
        "machine learning",
        return_type="enriched",
        context_window=400,
        context_unit="tokens",   # add the most-similar chunks that fit in ~400 tokens
    )
    
    for result in results:
        print(f"Enriched: {result.id}")
        print(f"Matched chunks: {result.metadata['_matched_chunk_indices']}")
        print(f"All chunks: {result.metadata['_all_chunk_indices']}")
        print(f"Similarity scores: {result.metadata['_similarity_scores']}")
        print(f"Content: {result.content}")

**Key Features:**
- **One result per document** (combines all matches)
- **Semantic similarity** within documents
- **Automatic deduplication** of chunks
- **Rich metadata** about enrichment process

**When to use:**
- Want comprehensive document excerpts
- Need related context within documents
- Building AI/RAG applications
- When topical coherence is important

``return_type="sections"``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns section-level results (one result per matching section, with
``type="section"``). Requires a database created with
``hierarchical_embeddings=True``. Pair it with ``search_level="sections"`` to
match against the section index directly:

.. code-block:: python

    results = db.query(
        "how do I rotate the API key?",
        search_level="sections",
    )
    for r in results:
        print(r.score, r.metadata["section_heading"])

See :doc:`hierarchical` for the full three-level (document → section → chunk)
retrieval model, section detection, and section metadata.

Parameters and Options
----------------------

Common Parameters
^^^^^^^^^^^^^^^^^

All query methods support these parameters:

* ``k`` (int, default=10): Maximum number of results to return
* ``score_threshold`` (float, default=0.0): Minimum similarity score (0-1, higher=better)
* ``filters`` (dict, optional): Metadata filters to apply

Search Type Specific
^^^^^^^^^^^^^^^^^^^^

**Hybrid Search:**

* ``vector_weight`` (float, default=0.7): Weight for vector vs keyword results (0.0-1.0)

**Context and Enriched:**

* ``context_window`` (int, default=2): Size of the assembled context, measured in
  ``context_unit``. In the default ``"chunks"`` unit this is the number of
  surrounding/similar chunks to include; with a budget unit it is an approximate
  token/word/character budget.
* ``context_unit`` (str, default="chunks"): One of ``"chunks"``, ``"tokens"``,
  ``"words"``, ``"characters"``. Selects how ``context_window`` is interpreted.
* ``context_truncate`` (bool, default=False): With a budget unit, hard-truncate the
  assembled context to exactly the budget (otherwise whole chunks are kept and the
  result may fall short of the budget).

**Document Return Type:**

* ``document_scoring_method`` (str, default="frequency_boost"): How to aggregate chunk scores
* ``document_scoring_options`` (dict, optional): Parameters for scoring methods

**Hierarchical Search:**

* ``search_level`` (str, default="chunks"): Which FAISS index to query — ``"chunks"``,
  ``"sections"``, or ``"documents"``. The ``"sections"`` and ``"documents"`` levels
  require ``hierarchical_embeddings=True``. See :doc:`hierarchical`.

**Advanced Options:**

* ``semantic_dedup_threshold`` (float, optional): Remove semantically similar results

Practical Examples
------------------

Multi-Modal Search Strategy
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Start with hybrid search for balanced results
    results = db.query("neural network training", search_type="hybrid")
    
    if len(results) < 5:
        # Fall back to vector search for broader matches
        results = db.query("neural network training", search_type="vector")
    
    if len(results) < 3:
        # Use keyword search for exact terms
        results = db.query("neural network", search_type="keyword")

Progressive Context Enrichment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Start with chunks for precision
    chunks = db.query("transformer architecture", return_type="chunks")
    
    if chunks:
        # Get enriched results for better context
        enriched = db.query(
            "transformer architecture", 
            return_type="enriched",
            context_window=4
        )
        
        # Compare chunk precision vs enriched comprehensiveness
        print(f"Precise chunks: {len(chunks)}")
        print(f"Enriched results: {len(enriched)}")

Adaptive Scoring
^^^^^^^^^^^^^^^^

.. code-block:: python

    # For research/comprehensive search
    scholarly_results = db.query(
        "climate change impacts",
        return_type="documents",
        document_scoring_method="statistical",
        document_scoring_options={
            "best_weight": 0.4,
            "mean_weight": 0.3,
            "consistency_weight": 0.2,
            "coverage_weight": 0.1
        }
    )
    
    # For finding best excerpts
    excerpt_results = db.query(
        "climate change impacts",
        return_type="enriched",
        context_window=3,
    )

Building RAG Applications
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def rag_query(question: str, max_context: int = 2000):
        """Get optimal context for RAG applications"""
        
        # Use enriched results for comprehensive context
        results = db.query(
            question,
            search_type="hybrid",
            return_type="enriched",
            context_window=4,
            k=3
        )
        
        # Combine results within token limit
        context_parts = []
        total_length = 0
        
        for result in results:
            if total_length + len(result.content) < max_context:
                context_parts.append(result.content)
                total_length += len(result.content)
            else:
                break
        
        return "\n\n".join(context_parts)

Performance Considerations
--------------------------

**Vector Search:**
- Requires embedding generation for queries
- Scales with FAISS index size
- CPU/GPU intensive for large collections

**Keyword Search:**
- Fast FTS5 queries
- Scales well with document count
- Limited to exact term matching

**Hybrid Search:**
- Combines both search costs
- Benefits from both search strengths
- Recommended for most use cases

**Return Types Performance:**
- ``documents``: Fastest, minimal processing
- ``chunks``: Fast, direct chunk access
- ``context``: Moderate, requires chunk assembly
- ``enriched``: Slower, requires similarity calculations

**Best Practices:**
- Use ``enriched`` for quality over speed
- Use ``chunks`` for high-volume applications
- Cache frequently-used enriched results
- Consider ``semantic_dedup_threshold`` for large result sets

Streaming Large Result Sets
---------------------------

For large-scale retrieval where loading all results into memory at once is impractical, LocalVectorDB provides
cursor-based streaming. A ``QueryCursor`` performs the FAISS/FTS search once and lazily loads content from SQLite
in batches as you iterate:

.. code-block:: python

   # Stream results in batches instead of loading all at once
   for batch in db.query_stream(
       "machine learning",
       search_type="hybrid",
       return_type="chunks",
       k=100,
       batch_size=10,
   ):
       for result in batch:
           process(result)

   # Async streaming with backpressure
   async for batch in db.query_stream_async(
       "deep learning",
       search_type="vector",
       return_type="documents",
       k=200,
       batch_size=25,
   ):
       await process_batch(batch)

For the full streaming API including ``QueryCursor`` lifecycle management, async generators, and QueryBuilder
integration, see :doc:`streaming`.

See Also
--------

* :doc:`streaming` - Cursor-based streaming for large result sets
* :doc:`document-scoring` - Document scoring methods reference
* :doc:`metadata.filtering` - Advanced filtering options
* :doc:`embeddings` - Embedding provider configuration