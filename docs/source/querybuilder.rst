QueryBuilder
============

The ``QueryBuilder`` interface provides a fluent, chainable API for constructing complex queries against your
LocalVectorDB database. This approach offers a more structured and readable alternative to dictionary-based filters,
especially for complex query conditions.

Core Concepts
-------------

The QueryBuilder pattern allows you to:

1. Chain multiple conditions with logical operators
2. Build complex nested queries
3. Mix different search types (vector, keyword, hybrid)
4. Apply metadata filters with precise control
5. Customize result handling and sorting

Basic Usage
-----------

Creating a QueryBuilder
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from localvectordb import VectorDB

   # Initialize your database
   db = VectorDB("my_database", "./vector_data")

   # Create a query builder from the database instance
   query = db.query_builder()

   # The following is equivalent:
   from localvectordb.query_builder import QueryBuilder
   query = QueryBuilder(db)


Simple Filters
^^^^^^^^^^^^^^

.. code-block:: python

   # Find documents where category equals "tech"
   results = (
       db.query_builder()
       .filter("category", "tech")  # Exact match on category field
       .execute()
   )

   # Find documents with priority of 3 or higher
   results = (
       db.query_builder()
       .filter("priority", gte_=3)  # priority >= 3
       .execute()
   )

   # Multiple conditions combined with AND logic
   results = (
       db.query_builder()
       .filter(category="tech", priority=5)  # category = "tech" AND priority = 5
       .execute()
   )

Search Operations
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Perform vector search (semantic similarity)
   results = (
       db.query_builder()
       .search("machine learning algorithms", search_type="vector")  # Find semantically similar content
       .execute()
   )

   # Perform keyword search (exact text matching)
   results = (
       db.query_builder()
       .keyword("python programming")  # Look for exact keyword matches
       .execute()
   )

   # Combine vector and keyword search with custom weights
   results = (
       db.query_builder()
       .hybrid("neural networks", vector_weight=0.7)  # 70% semantic similarity, 30% keyword matching
       .execute()
   )

   # Search within a specific metadata field (case-insensitive "contains" for
   # string values, exact match otherwise) rather than the document content
   results = (
       db.query_builder()
       .search_field("title", "transformer")  # title ILIKE '%transformer%'
       .execute()
   )


Combining Search with Filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Search for documents about ML algorithms that are in the tech category and high priority
   results = (
       db.query_builder()
       .search("machine learning algorithms")  # Find semantically similar documents
       .filter("category", "tech")  # Narrow down to tech category
       .filter("priority", gte_=3)  # Only include high priority (3+) documents
       .limit(5)  # Return only top 5 results
       .execute()
   )

.. note::

   **Filters are pushed into the search, not applied afterwards.** When you
   combine a metadata filter with a search, the filter is evaluated *inside* the
   vector and keyword search (a FAISS id-selector over the matching documents, or
   a subquery on the full-text index) so that a filter matching only a small
   fraction of the corpus still returns a full page of matches. A selective
   filter -- say one that matches 0.1% of documents -- therefore returns its best
   matches rather than only whatever happened to fall in the first candidate
   window.

   This applies to ``query()``, ``query_async()``, the streaming cursor, and the
   section/document hierarchical searches. Two cases fall back to post-filtering a
   fixed candidate pool and log a warning if fewer than ``k`` results survive: a
   database built with ``faiss_index_type="IndexLSH"`` (which cannot take a
   selector -- prefer a flat index for filtered search) and a filter that SQL
   cannot express, such as dot-notation into a JSON field (``"author.name"``).
   Results are always correct; only these two cases may under-return on a very
   selective filter.

Limiting and Pagination
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Limit to 10 results
   results = (
       db.query_builder()
       .search("machine learning")
       .limit(10)  # Only return the top 10 most relevant results
       .execute()
   )

   # Pagination example - first page (results 1-20)
   results_page1 = (
       db.query_builder()
       .filter(category="tech")  # Filter by category
       .limit(20)  # Page size of 20
       .offset(0)  # Start at the beginning (first page)
       .execute()
   )

   # Pagination example - second page (results 21-40)
   results_page2 = (
       db.query_builder()
       .filter(category="tech")  # Same filter as above
       .limit(20)  # Same page size
       .offset(20)  # Skip the first 20 results (move to second page)
       .execute()
   )

Advanced Usage
--------------

Semantic Filtering
^^^^^^^^^^^^^^^^^^

Semantic filtering allows you to refine a search by other fields. This can be particularly useful when you have
rich metadata, since you can do semantic matching of concepts across fields other than `content`.

For example, suppose you had a library of scientific publications where details about the methodology are extracted,
and you want only papers about NLP using unsupervised techniques:

.. code-block:: python

   # Find documents about NLP that use unsupervised learning approaches
   results = (
       db.query_builder()
       .search("natural language processing")  # Base query for NLP documents
       .semantic_filter(
           field="methodology",  # Look at the methodology field
           concept="unsupervised learning",  # Find semantic similarity to this concept
           threshold=0.75  # Must be at least 75% similar to match
       )
       .execute()
   )

``semantic_filter`` also accepts a ``metric`` argument selecting how similarity is
measured. Pass a :class:`~localvectordb.query_builder.SimilarityMetric` value —
``COSINE`` (default), ``EUCLIDEAN``, ``DOT_PRODUCT``, or ``MANHATTAN``:

.. code-block:: python

   from localvectordb.query_builder import SimilarityMetric

   results = (
       db.query_builder()
       .search("natural language processing")
       .semantic_filter(
           field="methodology",
           concept="unsupervised learning",
           threshold=0.75,
           metric=SimilarityMetric.DOT_PRODUCT,
       )
       .execute()
   )

Return Type Customization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Return the individual chunks instead of complete documents
   results = (
       db.query_builder()
       .search("specific technical detail")
       .chunks()  # Return individual text chunks that match
       .execute()
   )

   # Return matching chunks with surrounding context
   results = (
       db.query_builder()
       .search("neural network architecture")
       .context(window_size=3)  # Include 3 chunks before and after each match
       .execute()
   )

   # Control how document scores are calculated from multiple matching chunks
   results = (
       db.query_builder()
       .search("machine learning")
       .documents(scoring_method="average")  # Use the mean of the matching chunk scores
       .execute()
   )

Hierarchical Search
^^^^^^^^^^^^^^^^^^^

When the database was created with ``hierarchical_embeddings=True``, two extra
builder methods select the retrieval granularity (see :doc:`hierarchical`):

* ``.search_level(level)`` — which FAISS index to query: ``"chunks"`` (default),
  ``"sections"``, or ``"documents"``.
* ``.sections()`` — return section-level results (``type="section"``).

.. code-block:: python

   # Find the most relevant sections, not individual chunks
   results = (
       db.query_builder()
       .search("network timeout errors")
       .search_level("sections")
       .filter("product", "gateway")
       .limit(5)
       .execute()
   )

Ordering Results
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Order results by publication date (descending by default, newest first)
   results = (
       db.query_builder()
       .filter(category="tech")
       .order_by("publish_date")  # Default direction is "desc" -> newest to oldest
       .execute()
   )

   # Order results by publication date in ascending order (oldest first)
   results = (
       db.query_builder()
       .filter(category="tech")
       .order_by("publish_date", "asc")  # Sort from oldest to newest
       .execute()
   )

   # Order search results by relevance score
   results = (
       db.query_builder()
       .search("artificial intelligence")
       .order_by_score("desc")  # Show most relevant results first
       .execute()
   )

   # Apply multiple ordering criteria (first by year, then by citations)
   results = (
       db.query_builder()
       .search("research papers")
       .order_by("year", "desc")  # First sort by most recent year
       .order_by("citations", "desc")  # For papers from same year, sort by most citations
       .execute()
   )

Grouping and Aggregation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Group search results by category field
   results = (
       db.query_builder()
       .search("machine learning")
       .group_by("category")  # Group matching documents by their category
       .execute()
   )

   # Count documents by category for documents from 2024
   results = (
       db.query_builder()
       .filter(year=2024)  # Only include documents from 2024
       .group_by("category")  # Group by category
       .count_by("*", "document_count")  # Count documents in each group, call the results 'document_count'
       .execute()
   )

   # Calculate average priority by category and filter to high-priority categories
   results = (
       db.query_builder()
       .group_by("category")  # Group by category
       .avg_by("priority", "avg_priority")  # Calculate average priority for each category
       .having("avg_priority", "gt", 3)  # Only include categories with average priority > 3
       .execute()
   )

The full set of aggregation helpers mirrors SQL: ``count_by``, ``sum_by``,
``avg_by``, ``min_by``, and ``max_by``. Each takes a field and an optional
alias (defaulting to e.g. ``sum_<field>``):

.. code-block:: python

   # Total and lowest word_count per category
   results = (
       db.query_builder()
       .group_by("category")
       .sum_by("word_count", "total_words")   # SUM(word_count)
       .min_by("word_count", "shortest")      # MIN(word_count)
       .execute()
   )

For HAVING on a count aggregation, ``having_count(operator, value, alias="count")``
is a shorthand for ``having(alias, operator, value)``. HAVING clauses support only
the comparison operators ``eq``, ``ne``, ``gt``, ``gte``, ``lt``, and ``lte``:

.. code-block:: python

   # Only keep categories with more than 5 documents
   results = (
       db.query_builder()
       .group_by("category")
       .count_by("*", "count")
       .having_count("gt", 5)
       .execute()
   )

Result Reranking
^^^^^^^^^^^^^^^^

The QueryBuilder provides powerful reranking capabilities to improve search relevance by reordering results based on various criteria. This is particularly useful for enhancing the quality of search results beyond simple relevance scoring.

Basic Reranking
~~~~~~~~~~~~~~~

Rerank results using different strategies:

.. code-block:: python

   # Rerank by recency (newer documents ranked higher)
   results = (
       db.query_builder()
       .search("machine learning research")
       .rerank_by_recency(date_field="publish_date", weight=0.3)  # 30% weight to recency
       .execute()
   )

   # Rerank by diversity (promote variety in specified field)
   results = (
       db.query_builder()
       .search("AI applications")
       .rerank_by_diversity(field="category", weight=0.5)  # Promote diverse categories
       .execute()
   )

Cross-Encoder Reranking
~~~~~~~~~~~~~~~~~~~~~~~

Use a cross-encoder or reranking model to re-score results for higher relevance. Cross-encoders
evaluate query-document pairs jointly, producing more accurate relevance scores than bi-encoder
(embedding) similarity alone.

.. code-block:: python

   # Rerank using a local cross-encoder model (requires sentence-transformers)
   results = (
       db.query_builder()
       .search("machine learning optimization")
       .rerank_by_model(
           provider="sentence_transformers",
           model="cross-encoder/ms-marco-MiniLM-L-6-v2",
           top_k=10
       )
       .execute()
   )

   # Rerank using Jina AI reranker API
   results = (
       db.query_builder()
       .search("neural network architectures")
       .rerank_by_model(
           provider="jina",
           model="jina-reranker-v2-base-multilingual",
           top_k=5
       )
       .execute()
   )

   # Rerank using HuggingFace Inference API
   results = (
       db.query_builder()
       .search("deep learning")
       .rerank_by_model(
           provider="huggingface",
           model="BAAI/bge-reranker-v2-m3"
       )
       .execute()
   )

Cross-encoder reranking can also be applied directly via the ``query()`` method:

.. code-block:: python

   # Using reranker_config dict
   results = db.query(
       "machine learning",
       reranker_config={
           "provider": "mock",
           "model": "mock-reranker"
       }
   )

   # Or pass a reranker instance
   from localvectordb.reranking import create_reranker

   reranker = create_reranker("sentence_transformers")
   results = db.query("machine learning", reranker=reranker)

Over-fetching for reranking (``rerank_k``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reranker can only improve results it actually sees. When a reranker is
configured, ``query()`` fetches a larger candidate pool -- ``rerank_k``, default
``5 * k`` -- re-scores that pool, and returns the top ``k``. Without this
over-fetch the reranker would only ever reorder the ``k`` results the search
already selected, so it could never pull a stronger match up from just below the
cutoff.

.. code-block:: python

   # Fetch 100 candidates, rerank them, return the best 10.
   results = db.query(
       "machine learning",
       k=10,
       rerank_k=100,
       reranker_config={"provider": "jina", "model": "jina-reranker-v2-base-multilingual"},
   )

``rerank_k`` is clamped to at most 200 (a cross-encoder pass costs one model call
per candidate) and is never smaller than ``k``. It has no effect when no reranker
is supplied. The same parameter is accepted by ``query_async()`` and by the HTTP
API. Reranking is **not** available on the streaming/cursor path -- use
``query()``/``query_async()`` when a reranker is configured.

.. note::

   Each reranker leaves ``result.score`` as an **absolute** value in ``[0, 1]``
   (comparable across queries and usable with ``score_threshold``), and preserves
   the model's raw output in ``result.metadata["rerank_raw_score"]`` and the
   pre-rerank search score in ``result.metadata["original_score"]``. Jina uses its
   API's native relevance score; the SentenceTransformers and HuggingFace
   cross-encoders squash their logits with a logistic sigmoid. None uses a
   per-batch min-max, which would be pool-relative rather than absolute.

Reranker Provider Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in providers accept a few construction options (passed either through
``reranker_config`` or as keyword arguments to ``create_reranker``):

- All rerankers accept ``timeout`` (seconds, default 90) and ``max_retries``
  (default 3).
- ``sentence_transformers`` accepts ``device`` (e.g. ``"cpu"``, ``"cuda"``,
  ``"mps"``) to select where the local cross-encoder runs.
- ``huggingface`` accepts ``base_url`` to target a self-hosted Text-Embeddings-
  Inference / custom endpoint instead of the hosted Inference API.

.. code-block:: python

   from localvectordb.reranking import create_reranker, list_rerankers

   print(list_rerankers())   # ['jina', 'sentence_transformers', 'huggingface', 'mock', ...]

   reranker = create_reranker(
       "sentence_transformers",
       model="cross-encoder/ms-marco-MiniLM-L-6-v2",
       device="cuda",
   )

Custom Rerankers
~~~~~~~~~~~~~~~~~

Rerankers use the same plugin architecture as embedding providers. Subclass
:class:`~localvectordb.reranking.Reranker`, then either register it in-process
or expose it as an entry point:

.. code-block:: python

   from localvectordb.reranking import Reranker, RerankerRegistry

   class MyReranker(Reranker):
       ...

   # In-process registration
   RerankerRegistry.register("my_reranker", MyReranker)
   reranker = create_reranker("my_reranker")

To make a reranker discoverable across installs, add an entry point under the
``localvectordb.reranker_providers`` group in your ``pyproject.toml``:

.. code-block:: toml

   [project.entry-points."localvectordb.reranker_providers"]
   my_reranker = "my_package.rerankers:MyReranker"

Advanced Reranking Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the generic ``rerank()`` method for more control:

.. code-block:: python

   # Custom recency reranking with specific parameters
   results = (
       db.query_builder()
       .search("recent developments")
       .rerank("recency",
               date_field="updated_at",
               weight=0.4,
               decay_factor=0.95)  # How quickly recency importance decays
       .execute()
   )

   # Diversity reranking with threshold
   results = (
       db.query_builder()
       .search("research papers")
       .rerank("diversity",
               field="author",
               weight=0.6,
               max_per_group=2)  # Maximum 2 papers per author
       .execute()
   )

Combining Multiple Reranking Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply multiple reranking strategies in sequence:

.. code-block:: python

   # First promote diversity, then favor recent content
   results = (
       db.query_builder()
       .search("technology trends")
       .filter("category", "tech")
       .rerank_by_diversity(field="subcategory", weight=0.3)  # Promote diverse subcategories
       .rerank_by_recency(date_field="publish_date", weight=0.2)  # Then favor recent
       .order_by("score", "desc")  # Final ordering by combined score
       .limit(20)
       .execute()
   )

Use Cases for Reranking
~~~~~~~~~~~~~~~~~~~~~~~

**Recency Reranking** - Ideal for:

- News and content feeds where freshness matters
- Documentation where newer versions are preferred
- Research databases prioritizing recent findings
- Time-sensitive information retrieval

.. code-block:: python

   # News feed with recency boost
   news_results = (
       db.query_builder()
       .search("economic forecast")
       .filter("type", "news")
       .rerank_by_recency(
           date_field="publish_date",
           weight=0.5  # Strong recency preference
       )
       .limit(10)
       .execute()
   )

**Diversity Reranking** - Ideal for:

- Avoiding echo chambers by promoting varied perspectives
- Ensuring broad coverage across categories or sources
- Reducing redundancy in search results
- Providing comprehensive overviews of topics

.. code-block:: python

   # Diverse perspectives on a topic
   diverse_results = (
       db.query_builder()
       .search("climate change solutions")
       .rerank_by_diversity(
           field="source_type",  # Mix academic, news, government sources
           weight=0.4
       )
       .limit(15)
       .execute()
   )

Understanding Reranking Impact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monitor how reranking affects your results:

.. code-block:: python

   # Compare results with and without reranking
   base_results = (
       db.query_builder()
       .search("artificial intelligence")
       .limit(10)
       .execute()
   )

   reranked_results = (
       db.query_builder()
       .search("artificial intelligence")
       .rerank_by_recency(date_field="publish_date", weight=0.3)
       .limit(10)
       .execute()
   )

   # Analyze the differences
   print("Base results order:")
   for i, result in enumerate(base_results):
       print(f"{i+1}. {result.metadata.get('title', 'No title')} "
             f"(score: {result.score:.3f}, date: {result.metadata.get('publish_date')})")

   print("\nReranked results order:")
   for i, result in enumerate(reranked_results):
       print(f"{i+1}. {result.metadata.get('title', 'No title')} "
             f"(score: {result.score:.3f}, date: {result.metadata.get('publish_date')})")

Reranking with Query Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use query explanation to understand reranking decisions:

.. code-block:: python

   # Get detailed explanation of reranking process
   results = (
       db.query_builder()
       .search("data science techniques")
       .rerank_by_diversity(field="methodology", weight=0.4)
       .explain(detailed=True)
       .execute()
   )

   # Check execution plan
   for result in results:
       if "_execution_plan" in result.metadata:
           plan = result.metadata["_execution_plan"]
           if "reranking" in plan["steps"]:
               # detailed=True adds a "details" entry; "steps" is always present.
               print(f"Reranking applied. Plan details: {plan.get('details')}")

Best Practices for Reranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with Low Weights**: Begin with reranking weights of 0.1-0.3 and adjust based on results
2. **Test with Real Queries**: Use actual user queries to evaluate reranking effectiveness
3. **Monitor User Engagement**: Track click-through rates and user satisfaction
4. **Combine Strategically**: Use multiple reranking strategies thoughtfully to avoid conflicts
5. **Consider Field Quality**: Ensure reranking fields have good data quality and coverage

.. code-block:: python

   # Example of gradual reranking tuning
   def test_reranking_weights(query, weights=[0.1, 0.2, 0.3, 0.4, 0.5]):
       results_by_weight = {}

       for weight in weights:
           results = (
               db.query_builder()
               .search(query)
               .rerank_by_recency(date_field="publish_date", weight=weight)
               .limit(5)
               .execute()
           )
           results_by_weight[weight] = results

       return results_by_weight

   # Test different weights to find optimal setting
   test_results = test_reranking_weights("machine learning frameworks")

Optimizing Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Process a large result set in manageable batches. batch_size is honoured by
   # stream() (which pages through a cursor); execute() always returns the full
   # list, so pass batch_size to stream() rather than execute().
   for batch in (
       db.query_builder()
       .search("common term")  # Potentially matches many documents
       .filter(is_public=True)
       .stream(batch_size=200)  # Yields lists of up to 200 results at a time
   ):
       for result in batch:
           ...  # process each result

   # Remove semantically similar results to reduce redundancy and increase diversity of returned documents
   results = (
       db.query_builder()
       .search("neural networks")
       .semantic_dedup(threshold=0.92)  # Remove results that are >92% semantically similar
       .execute()
   )

   # Get detailed information about query execution for performance tuning
   results = (
       db.query_builder()
       .search("complex query")
       .filter(category="research")
       .explain(detailed=True)  # Include execution time and query plan details
       .execute()
   )

Streaming Results
^^^^^^^^^^^^^^^^^

The ``stream()`` method uses a cursor internally: FAISS/FTS is queried once and results are hydrated from SQLite in
batches. This is much more efficient than re-executing the full query for each page.

.. code-block:: python

   # Stream results in batches (single FAISS/FTS search, lazy SQLite loading)
   query = db.query_builder().search("machine learning").limit(200)

   for batch in query.stream(batch_size=50):
       for result in batch:
           print(f"Processing {result.id}")

For more control over the iteration lifecycle, create a ``QueryCursor`` directly:

.. code-block:: python

   # Create a cursor with explicit lifecycle management
   cursor = (
       db.query_builder()
       .hybrid("neural networks", vector_weight=0.7)
       .filter("year", gte_=2023)
       .limit(100)
       .cursor(batch_size=25, cursor_ttl=600.0)
   )

   with cursor:
       print(f"Found {cursor.total_candidates} candidates")
       first_batch = cursor.fetch_batch()
       remaining = cursor.fetch_all()  # Fetch everything else

See :doc:`streaming` for the full streaming and cursor API reference.

Counting Results
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Get the count of matching documents without retrieving all data
   count = (
       db.query_builder()
       .filter(category="tech", year=2024)  # Documents from tech category in 2024
       .count()  # Return only the count, not the actual documents
   )

   print(f"Found {count} matching documents")

Async Support
^^^^^^^^^^^^^

.. code-block:: python

   # Use async execution for better performance
   results = await (
       db.query_builder()
       .search("machine learning")
       .filter(is_published=True)
       .execute_async()
   )

   # Async streaming with cursor-based batching (single search, lazy hydration)
   async for batch in (
       db.query_builder()
       .search("deep learning")
       .limit(100)
       .stream_async(batch_size=25)
   ):
       for result in batch:
           await process_result(result)

   # Async cursor with explicit lifecycle
   cursor = await (
       db.query_builder()
       .hybrid("transformers", vector_weight=0.8)
       .limit(50)
       .cursor_async(batch_size=10)
   )

   async with cursor:
       async for result in cursor.stream_individual_async():
           await handle(result)

   # Get count asynchronously
   count = await db.query_builder().filter(year=2024).count_async()

Complex Queries
---------------

Combined Features
^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Complex multi-criteria search with filtering and ranking
   results = (
       db.query_builder()
       .hybrid("deep learning applications", vector_weight=0.7)  # Use hybrid search (70% vector, 30% keyword)
       .filter(year=2024)  # Only from 2024
       .filter("citations", gte_=10)  # With at least 10 citations
       .semantic_filter("abstract", "supervised learning", threshold=0.8)  # Using supervised learning approaches
       .documents()  # Return full documents
       .order_by("publish_date", "desc")  # Most recent first
       .limit(20)  # Top 20 results only
       .execute()
   )

   # Advanced analysis with grouping, aggregation and filtering
   results = (
       db.query_builder()
       .search("climate change")  # Find climate change related documents
       .filter("publish_date", gte_="2020-01-01")  # Published since 2020
       .group_by("author", "institution")  # Group by author and institution
       .count_by("*", "publication_count")  # Count publications per group
       .max_by("citations", "max_citations")  # Find maximum citations per group
       .having("publication_count", "gte", 5)  # Only include prolific authors (5+ publications)
       .order_by("max_citations", "desc")  # Order by most cited
       .limit(10)  # Top 10 groups only
       .execute()
   )

Debug and Diagnostics
^^^^^^^^^^^^^^^^^^^^^

The QueryBuilder provides comprehensive debugging and diagnostic capabilities to help you understand query execution, optimize performance, and troubleshoot issues.

Query Explanation
~~~~~~~~~~~~~~~~~~

The ``explain()`` method provides detailed information about how your query will be executed. It can be used in two ways: traditional mode (returns QueryBuilder with explanation enabled) or direct mode (returns execution plan immediately).

**Traditional Usage** (returns QueryBuilder with explanation enabled):

.. code-block:: python

   # Enable explanation for query execution
   results = (
       db.query_builder()
       .search("machine learning")
       .filter(category="tech")
       .rerank_by_recency(date_field="publish_date", weight=0.3)
       .explain(detailed=True)  # Enable detailed explanation
       .execute()
   )

   # Execution plan is included in result metadata
   for result in results:
       if "_execution_plan" in result.metadata:
           plan = result.metadata["_execution_plan"]
           print(f"Query type: {plan['query_type']}")
           print(f"Execution steps: {plan['steps']}")
           print(f"Estimated cost: {plan['estimated_cost']}")

**Direct Usage** (returns execution plan immediately):

.. code-block:: python

   # Get execution plan directly without executing the query
   plan = (
       db.query_builder()
       .search("machine learning")
       .filter(category="tech", year=2024)
       .rerank_by_recency(date_field="publish_date", weight=0.3)
       .explain(detailed=True, return_plan=True)  # Return plan directly
   )

   print(f"Query will execute with these steps: {plan['steps']}")
   print(f"Estimated relative cost: {plan['estimated_cost']}")
   print(f"Query type: {plan['query_type']}")
   print(f"Optimizations: {plan['optimizations']}")

   # With detailed=True, get additional information
   if 'details' in plan:
       details = plan['details']
       print(f"Search clauses: {details['search_clauses']}")
       print(f"Exact filters: {details['exact_filters']}")
       print(f"Semantic filters: {details['semantic_filters']}")
       print(f"Aggregations: {details['aggregations']}")

Execution Plan Analysis
~~~~~~~~~~~~~~~~~~~~~~~

Understanding the execution plan helps optimize query performance:

.. code-block:: python

   # Analyze different query variations
   queries = [
       db.query_builder().search("AI research"),
       db.query_builder().search("AI research").filter(year=2024),
       db.query_builder().search("AI research").semantic_filter("methodology", "deep learning", 0.8)
   ]

   for i, query in enumerate(queries):
       plan = query.explain(return_plan=True)
       print(f"\nQuery {i+1} execution plan:")
       print(f"  Steps: {' -> '.join(plan['steps'])}")
       print(f"  Estimated cost: {plan['estimated_cost']}")

       # Identify expensive operations
       if plan['estimated_cost'] > 100:
           print(f"  ⚠️  High cost query - consider optimization")

       if 'semantic_filtering' in plan['steps']:
           print(f"  🧠 Semantic filtering detected - will use embeddings")

Async Execution Plans
~~~~~~~~~~~~~~~~~~~~~

Get execution plans for async queries:

.. code-block:: python

   # Get execution plan for async query
   async def analyze_async_query():
       plan = await (
           db.query_builder()
           .search("complex async query")
           .semantic_filter("field", "concept", 0.75)
           .get_execution_plan_async(detailed=True)
       )

       print(f"Async execution plan: {plan}")
       return plan

   # Use in async context
   import asyncio
   plan = asyncio.run(analyze_async_query())

Performance Profiling
~~~~~~~~~~~~~~~~~~~~~~

Monitor query execution time and performance:

.. code-block:: python

   import time

   # Compare execution times with and without optimizations
   def profile_query_variations():
       base_query = (
           db.query_builder()
           .search("machine learning algorithms")
           .filter(year=2024)
       )

       # Test different configurations
       test_cases = [
           ("Base query", base_query),
           ("With semantic filter", base_query.semantic_filter("topic", "neural networks", 0.8)),
           ("With reranking", base_query.rerank_by_recency("publish_date", 0.3)),
           ("With both", base_query.semantic_filter("topic", "neural networks", 0.8)
                                  .rerank_by_recency("publish_date", 0.3))
       ]

       for name, query in test_cases:
           # Get execution plan first
           plan = query.explain(return_plan=True)

           # Time the actual execution
           start_time = time.time()
           results = query.execute()
           execution_time = time.time() - start_time

           print(f"\n{name}:")
           print(f"  Estimated cost: {plan['estimated_cost']}")
           print(f"  Actual time: {execution_time:.3f}s")
           print(f"  Results count: {len(results)}")
           print(f"  Steps: {' -> '.join(plan['steps'])}")

   profile_query_variations()

Debug Information
~~~~~~~~~~~~~~~~~

Get comprehensive debug information about query state:

.. code-block:: python

   # Build a complex query for debugging
   query = (
       db.query_builder()
       .search("complex query")
       .filter(category="tech")
       .semantic_filter("abstract", "machine learning", 0.7)
       .rerank_by_diversity(field="author", weight=0.4)
       .group_by("institution")
       .count_by("*", "paper_count")
       .having("paper_count", "gte", 5)
       .order_by("paper_count", "desc")
       .limit(20)
   )

   # Get detailed debug information
   debug_info = query.debug_info()
   print(f"Query complexity: {debug_info}")

   # Debug information includes:
   # - Number of search clauses, filters, aggregations
   # - Current query configuration
   # - Return type and limits
   # - Ordering and grouping information

Query Validation
~~~~~~~~~~~~~~~~~

Validate query configuration before execution:

.. code-block:: python

   # Build a potentially problematic query
   query = (
       db.query_builder()
       .semantic_filter("field", "concept", 0.8)  # Semantic filter without search
       .group_by("category")
       .having("count", "gte", 10)  # Having without aggregation
       .limit(10000)  # Very large limit
   )

   # Validate the query
   validation = query.validate()

   print(f"Query valid: {validation['valid']}")

   if validation['issues']:
       print("Issues found:")
       for issue in validation['issues']:
           print(f"  ❌ {issue}")

   if validation['warnings']:
       print("Warnings:")
       for warning in validation['warnings']:
           print(f"  ⚠️  {warning}")

   if validation['recommendations']:
       print("Recommendations:")
       for rec in validation['recommendations']:
           print(f"  💡 {rec}")

   print(f"Query complexity: {validation['query_complexity']}")

Debugging Common Issues
~~~~~~~~~~~~~~~~~~~~~~~

Use diagnostics to troubleshoot common problems:

.. code-block:: python

   # Debug slow queries
   def debug_slow_query(query_builder):
       plan = query_builder.explain(detailed=True, return_plan=True)

       # Check for expensive operations
       expensive_ops = []
       if plan['estimated_cost'] > 200:
           expensive_ops.append("High overall cost")

       if 'semantic_filtering' in plan['steps']:
           semantic_count = plan['details']['semantic_filters']
           if semantic_count > 2:
               expensive_ops.append(f"Multiple semantic filters ({semantic_count})")

       if expensive_ops:
           print("⚠️  Performance concerns:")
           for op in expensive_ops:
               print(f"   - {op}")

           print("\n💡 Optimization suggestions:")
           print("   - Reduce semantic filter count")
           print("   - Add exact filters before semantic filters")
           print("   - Consider using streaming for large result sets")

   # Example usage
   slow_query = (
       db.query_builder()
       .search("broad topic")
       .semantic_filter("field1", "concept1", 0.7)
       .semantic_filter("field2", "concept2", 0.8)
       .semantic_filter("field3", "concept3", 0.75)
       .limit(1000)
   )

   debug_slow_query(slow_query)
