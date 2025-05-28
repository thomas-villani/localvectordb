Chunking
========

LocalVectorDB v2.0 features a sophisticated position-tracking chunking system that maintains exact character positions for perfect document reconstruction and precise highlighting.

Overview
--------

**Chunking** is the process of breaking down large documents into smaller, manageable pieces that can be efficiently processed by embedding models. LocalVectorDB’s chunking system:

- **Preserves Position Information**: Every chunk knows exactly where it came from
- **Enables Perfect Reconstruction**: Documents can be perfectly rebuilt from chunks
- **Supports Multiple Strategies**: Different methods for different content types
- **Handles Overlaps Intelligently**: Maintains context across chunk boundaries

Chunking Methods
----------------

Sentences (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Splits text at sentence boundaries while respecting token limits.

.. code-block:: python

   from localvectordb import VectorDB

   db = VectorDB(
       "docs",
       chunking_method="sentences",
       chunk_size=500,          # Max tokens per chunk
       chunk_overlap=1          # Number of sentences to overlap
   )

   # Example document
   text = """
   This is the first sentence. This is the second sentence about something important.
   This is the third sentence with more details. This concludes the paragraph.
   """

   # Results in chunks like:
   # Chunk 1: "This is the first sentence. This is the second sentence about something important."
   # Chunk 2: "This is the second sentence about something important. This is the third sentence with more details."
   # Chunk 3: "This is the third sentence with more details. This concludes the paragraph."

**Best for**: Articles, documentation, general text

Paragraphs
~~~~~~~~~~

Splits text at paragraph boundaries (double newlines).

.. code-block:: python

   db = VectorDB(
       "docs",
       chunking_method="paragraphs",
       chunk_size=800,
       chunk_overlap=1  # Number of paragraphs to overlap
   )

**Best for**: Blog posts, essays, structured documents

Tokens
~~~~~~

Splits text at token boundaries using tiktoken encoding.

.. code-block:: python

   db = VectorDB(
       "docs",
       chunking_method="tokens",
       chunk_size=400,
       chunk_overlap=50  # Number of tokens to overlap
   )

**Best for**: Maximum control over chunk size, technical content

Words
~~~~~

Splits text at word boundaries.

.. code-block:: python

   db = VectorDB(
       "docs",
       chunking_method="words",
       chunk_size=300,
       chunk_overlap=20  # Number of words to overlap
   )

**Best for**: Social media content, short-form text

Lines
~~~~~

Splits text at line boundaries.

.. code-block:: python

   db = VectorDB(
       "docs",
       chunking_method="lines",
       chunk_size=600,
       chunk_overlap=3  # Number of lines to overlap
   )

**Best for**: Log files, CSV data, structured line-based content

Sections
~~~~~~~~

Splits text at markdown-style headers (`#`, `##`, `###`).

.. code-block:: python

   db = VectorDB(
       "docs",
       chunking_method="sections",
       chunk_size=1000
       # Sections typically don't overlap
   )

**Best for**: Markdown documents, technical documentation, structured content

Code Blocks
~~~~~~~~~~~

Intelligent code-aware chunking that preserves logical code structure.

.. code-block:: python

   db = VectorDB(
       "code_db",
       chunking_method="code-blocks",
       chunk_size=800,
       chunk_overlap=2  # Number of lines to overlap
   )

**Features**:

- Detects programming language automatically
- Preserves function and class boundaries
- Handles Python, JavaScript, Java, and generic code
- Maintains proper indentation context

**Best for**: Source code, technical documentation with code

Characters
~~~~~~~~~~

Fine-grained chunking at character level.

.. code-block:: python

   db = VectorDB(
       "docs",
       chunking_method="characters",
       chunk_size=500,
       chunk_overlap=50  # Number of characters to overlap
   )

**Best for**: Non-Western languages, specialized text processing

Advanced Chunking Configuration
-------------------------------

Custom Chunk Factories
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from localvectordb.chunking import ChunkerFactory

   # Create chunker directly
   chunker = ChunkerFactory.create_chunker(
       method="sentences",
       max_tokens=400,
       overlap=2
   )

   # Use chunker manually
   chunks = chunker.chunk("Your document text here...")
   for chunk in chunks:
       print(f"Chunk {chunk.index}: {chunk.content}")
       print(f"Position: {chunk.position.start}-{chunk.position.end}")
       print(f"Tokens: {chunk.tokens}")

Language-Specific Code Chunking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Python-specific chunking
   db_python = VectorDB(
       "python_code",
       chunking_method="code-blocks",
       chunk_size=600,
       chunk_overlap=2
   )

   # JavaScript-specific chunking
   db_js = VectorDB(
       "javascript_code",
       chunking_method="code-blocks",
       chunk_size=500,
       chunk_overlap=1
   )

Position Tracking
-----------------

Every chunk maintains precise position information:

.. code-block:: python

   from localvectordb.core import ChunkPosition

   # ChunkPosition attributes:
   # - start: Character position in original document
   # - end: Character position in original document
   # - line: Line number (1-based)
   # - column: Column number (1-based)

   # Access position information
   results = db.query("search term", return_type="chunks")
   for result in results:
       pos = result.position
       print(f"Found at line {pos.line}, column {pos.column}")
       print(f"Characters {pos.start}-{pos.end}")

Document Reconstruction
-----------------------

Perfect reconstruction capabilities:

.. code-block:: python

   from localvectordb.chunking import reconstruct_document

   # Get all chunks for a document
   results = db.query("", return_type="chunks", filters={"document_id": "doc_1"})
   chunks = [result for result in results]

   # Reconstruct original document
   original_text = reconstruct_document(chunks, original_length=len(original_doc))

   # Find specific positions
   from localvectordb.chunking import find_chunk_containing_position

   chunk = find_chunk_containing_position(chunks, character_position=150)
   if chunk:
       print(f"Character 150 is in chunk {chunk.index}")

Chunking Strategies by Content Type
-----------------------------------

Academic Papers
~~~~~~~~~~~~~~~

.. code-block:: python

   # Academic papers with citations and references
   db_papers = VectorDB(
       "academic_papers",
       chunking_method="paragraphs",  # Preserve paragraph structure
       chunk_size=600,                # Larger chunks for context
       chunk_overlap=1,               # Overlap paragraphs
       metadata_schema={
           'title': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'authors': MetadataField(type=MetadataFieldType.JSON),
           'journal': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'year': MetadataField(type=MetadataFieldType.INTEGER, indexed=True)
       }
   )

Source Code Repositories
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Code repositories with multiple languages
   db_code = VectorDB(
       "source_code",
       chunking_method="code-blocks",
       chunk_size=500,
       chunk_overlap=2,
       metadata_schema={
           'file_path': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'language': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'author': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'last_modified': MetadataField(type=MetadataFieldType.DATE, indexed=True)
       }
   )

Legal Documents
~~~~~~~~~~~~~~~

.. code-block:: python

   # Legal documents requiring precise citation
   db_legal = VectorDB(
       "legal_docs",
       chunking_method="sections",    # Preserve legal structure
       chunk_size=1000,               # Large chunks for legal context
       chunk_overlap=0,               # No overlap to avoid confusion
       metadata_schema={
           'case_number': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'court': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'date_filed': MetadataField(type=MetadataFieldType.DATE, indexed=True),
           'document_type': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
       }
   )

Customer Support
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Customer support tickets and responses
   db_support = VectorDB(
       "support_tickets",
       chunking_method="sentences",   # Natural conversation flow
       chunk_size=300,                # Shorter for specific queries
       chunk_overlap=2,               # Maintain conversation context
       metadata_schema={
           'ticket_id': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'customer_id': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'priority': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
           'status': MetadataField(type=MetadataFieldType.TEXT, indexed=True)
       }
   )

Performance Considerations
--------------------------

Chunk Size Optimization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   # Small chunks (200-400 tokens)
   # + Better precision for specific queries
   # + Faster embedding generation
   # - Less context for complex topics
   # - More storage overhead

   # Medium chunks (400-600 tokens) - RECOMMENDED
   # + Good balance of precision and context
   # + Works well with most embedding models
   # + Reasonable storage requirements

   # Large chunks (600-1000+ tokens)
   # + Maximum context preservation
   # + Better for complex topics
   # - Slower search and embedding generation
   # - May miss specific details

Overlap Strategy
~~~~~~~~~~~~~~~~

.. code-block:: text

   # No overlap (overlap=0)
   # + Fastest processing
   # + Minimal storage
   # - May lose context at boundaries

   # Small overlap (1-2 units)
   # + Preserves boundary context
   # + Minimal overhead
   # - Slight storage increase

   # Large overlap (3+ units)
   # + Maximum context preservation
   # + Better for complex queries
   # - Significant storage overhead

Debugging and Analysis
----------------------

Chunk Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze chunking results
   def analyze_chunks(db, doc_id):
       # Get document
       doc = db.get(doc_id)

       # Get all chunks for document (using internal API)
       # Note: This requires access to internal chunk storage
       print(f"Document length: {len(doc.content)} characters")

       # Estimate chunk count
       estimated_chunks = len(doc.content) // (db.chunk_size * 4)  # Rough estimate
       print(f"Estimated chunks: {estimated_chunks}")

   # Test different chunking strategies
   def test_chunking_strategies(text):
       strategies = ["sentences", "paragraphs", "tokens", "words"]

       for strategy in strategies:
           chunker = ChunkerFactory.create_chunker(strategy, max_tokens=300)
           chunks = chunker.chunk(text)

           print(f"\n{strategy.title()} Chunking:")
           print(f"  Total chunks: {len(chunks)}")
           print(f"  Avg tokens per chunk: {sum(c.tokens for c in chunks) / len(chunks):.1f}")
           print(f"  Token range: {min(c.tokens for c in chunks)}-{max(c.tokens for c in chunks)}")

Chunk Visualization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def visualize_chunks(text, chunker):
       """Visualize how text is chunked"""
       chunks = chunker.chunk(text)

       print("Original text with chunk boundaries:")
       print("=" * 50)

       last_end = 0
       for i, chunk in enumerate(chunks):
           # Print any text before this chunk
           if chunk.position.start > last_end:
               gap_text = text[last_end:chunk.position.start]
               print(f"[GAP: '{gap_text}']")

           # Print chunk with markers
           print(f"[CHUNK {i}] {chunk.content}")
           last_end = chunk.position.end

       # Print any remaining text
       if last_end < len(text):
           remaining = text[last_end:]
           print(f"[REMAINING: '{remaining}']")
