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


Combining Search with Filters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Search for documents about ML algorithms that are in the tech category and high priority
   results = (
       db.query_builder()
       .search("machine learning algorithms")  # Find semantically similar documents
       .filter("category", "tech")  # Narrow down to tech category
       .filter("priority", gte=3)  # Only include high priority (3+) documents
       .limit(5)  # Return only top 5 results
       .execute()
   )

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

Return Type Customization
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Return the individual chunks instead of complete documents
   results = (
       db.query_builder()
       .search("specific technical detail")
       .return_type("chunks")  # Return individual text chunks that match
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
       .documents(scoring_method="weighted_average")  # Use weighted average of chunk scores
       .execute()
   )

Ordering Results
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Order results by publication date (ascending by default)
   results = (
       db.query_builder()
       .filter(category="tech")
       .order_by("publish_date")  # Sort from oldest to newest
       .execute()
   )

   # Order results by publication date in descending order (newest first)
   results = (
       db.query_builder()
       .filter(category="tech")
       .order_by("publish_date", "desc")  # Sort from newest to oldest
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

Optimizing Performance
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Process a large result set in manageable batches
   results = (
       db.query_builder()
       .search("common term")  # Potentially matches many documents
       .filter(is_public=True)
       .batch_size(200)  # Process in batches of 200 to reduce memory usage
       .execute()
   )

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

.. code-block:: python

   # Process a large number of results in batches to avoid memory issues
   query = db.query_builder().filter(is_archived=False)  # Could be many documents

   for batch in query.stream(batch_size=100):  # Process 100 documents at a time
       for result in batch:
           print(f"Processing {result.id}")  # Handle each document incrementally

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

   # Use async execution for better performance with remote databases
   results = await (
       db.query_builder()
       .search("machine learning")  # Find ML-related documents
       .filter(is_published=True)  # Only include published ones
       .execute_async()  # Execute asynchronously for non-blocking operation
   )

   # Process large result sets asynchronously in batches
   query = db.query_builder().filter(category="tech")

   async for batch in query.stream_async(batch_size=50):  # Process in smaller batches
       for result in batch:
           await process_result(result)  # Process each result with an async function

   # Get count asynchronously for non-blocking operation
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
       .return_type("documents")  # Return full documents
       .order_by("publish_date", "desc")  # Most recent first
       .limit(20)  # Top 20 results only
       .execute()
   )

   # Advanced analysis with grouping, aggregation and filtering
   results = (
       db.query_builder()
       .search("climate change")  # Find climate change related documents
       .filter("publish_date", gte="2020-01-01")  # Published since 2020
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

.. code-block:: python

   # Build a complex query that you want to inspect
   query = (
       db.query_builder()
       .search("complex query")  # Search term
       .filter(category="tech")  # Filter to tech category
       .hybrid(vector_weight=0.6)  # Use hybrid search
   )

   # Get detailed diagnostic information including SQL preview, filter details, etc.
   debug_info = query.debug_info()
   print(debug_info)  # Useful for debugging and optimization
