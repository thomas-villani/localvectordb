Chunking
========

LocalVectorDB features a sophisticated position-tracking chunking system that maintains exact character positions for perfect document reconstruction and precise highlighting.

Overview
--------

**Chunking** is the process of breaking down large documents into smaller, manageable pieces that can be efficiently processed by embedding models. LocalVectorDB’s chunking system:

- **Preserves Position Information**: Every chunk knows exactly where it came from
- **Enables Perfect Reconstruction**: Documents can be perfectly rebuilt from chunks
- **Supports Multiple Strategies**: Different methods for different content types
- **Handles Overlaps Intelligently**: Maintains context across chunk boundaries

Chunking Methods
----------------

Sentences
~~~~~~~~~

Splits text at sentence boundaries while respecting token limits.

.. code-block:: python

   from localvectordb import VectorDB

   db = VectorDB(
       "docs",
       "./vector_storage",
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
       "./vector_storage",
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
       "./vector_storage",
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
       "./vector_storage",
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
       "./vector_storage",
       chunking_method="lines",
       chunk_size=600,
       chunk_overlap=3  # Number of lines to overlap
   )

**Best for**: Log files, CSV data, structured line-based content

Sections
~~~~~~~~

Splits text at Markdown-style headers (``#``, ``##``, ``###``), using them as
preferred break points and keeping small sections whole. Headers that appear
**inside fenced code blocks** (```` ``` ```` or ``~~~``) are ignored, so example
snippets containing ``#`` comments or shell prompts do not create spurious
breaks.

.. code-block:: python

   db = VectorDB(
       "docs",
       "./vector_storage",
       chunking_method="sections",
       chunk_size=1000
       # Sections typically don't overlap
   )

**Best for**: Markdown documents, technical documentation, structured content.

.. tip::

   Files ingested via :meth:`~localvectordb.LocalVectorDB.upsert_from_file` or
   the server ``/upload`` endpoint are extracted to **Markdown** (see
   :doc:`/file-extraction`), so ``"sections"`` is an especially good fit for
   PDFs, DOCX, and other documents whose heading structure is preserved during
   extraction.

Code Blocks
~~~~~~~~~~~

Intelligent code-aware chunking that preserves logical code structure.

.. code-block:: python

   db = VectorDB(
       "code_db",
       "./vector_storage",
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
       "./vector_storage",
       chunking_method="characters",
       chunk_size=500,
       chunk_overlap=50  # Number of characters to overlap
   )

**Best for**: Non-Western languages, specialized text processing

Custom Chunking Methods
-----------------------

You can create your own chunking strategy by subclassing ``PositionTrackingChunker`` and
registering it with the ``ChunkerFactory``.

Creating a Custom Chunker
~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the ``chunk()`` method, which must return a list of ``Chunk`` objects with accurate
position tracking. Use the helper methods from the base class (``count_tokens``,
``_create_chunk``, ``_calculate_line_column``) to stay consistent with built-in chunkers.

.. code-block:: python

   from localvectordb.chunking import PositionTrackingChunker, ChunkerFactory
   from localvectordb.core import Chunk
   from typing import List

   class RegexChunker(PositionTrackingChunker):
       """Split text at a custom regex pattern."""

       def __init__(self, max_tokens: int = 500, overlap: int = 0, pattern: str = r"\n---\n"):
           super().__init__(max_tokens, overlap)
           import re
           self.pattern = re.compile(pattern)

       def chunk(self, text: str) -> List[Chunk]:
           if not text.strip():
               return []

           parts = self.pattern.split(text)
           chunks = []
           pos = 0
           for i, part in enumerate(parts):
               start = text.index(part, pos)
               end = start + len(part)
               chunk = self._create_chunk(text, start, end, i)
               chunks.append(chunk)
               pos = end

           # Ensure all chunks respect max_tokens limit
           return self._ensure_chunks_within_limit(chunks, text)

Registering with ChunkerFactory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to use a custom chunker.

**Option 1 -- Add to the factory registry** so it can be referenced by name:

.. code-block:: python

   # Register once at startup
   ChunkerFactory.CHUNKERS["regex"] = RegexChunker

   # Now use it by name when creating a database
   db = VectorDB(
       "my_db",
       "./vector_storage",
       chunking_method="regex",
       chunk_size=400,
   )

   # Or create the chunker directly
   chunker = ChunkerFactory.create_chunker("regex", max_tokens=400)

**Option 2 -- Pass the class directly** to ``create_chunker``:

.. code-block:: python

   # No registration needed
   chunker = ChunkerFactory.create_chunker(RegexChunker, max_tokens=400)
   chunks = chunker.chunk("Part one\n---\nPart two\n---\nPart three")

You can list all registered chunking methods at any time:

.. code-block:: python

   print(ChunkerFactory.list_methods())
   # ['sentences', 'tokens', 'words', 'lines', 'characters', 'paragraphs', 'sections', 'code-blocks', 'regex']

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
       "./vector_storage",
       chunking_method="code-blocks",
       chunk_size=600,
       chunk_overlap=2
   )

   # JavaScript-specific chunking
   db_js = VectorDB(
       "javascript_code",
       "./vector_storage",
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
   # - start: Start character position in original document
   # - end: End character position in original document
   # - line: Start line number (1-based)
   # - column: Start column number (1-based)
   # - end_line: End line number (1-based)
   # - end_column: End column number (1-based)

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

   from localvectordb.chunking import ChunkerFactory, reconstruct_document

   # Chunk a document, keeping the chunk objects (each carries its position)
   original_doc = "Your original document text..."
   chunker = ChunkerFactory.create_chunker("sentences", max_tokens=200)
   chunks = chunker.chunk(original_doc)

   # Reconstruct original document from its position-tracked chunks
   original_text = reconstruct_document(chunks, original_length=len(original_doc))

   # Find specific chunk by iterating through chunks
   target_position = 150
   found_chunk = None
   for chunk in chunks:
       if chunk.position.start <= target_position < chunk.position.end:
           found_chunk = chunk
           break
   
   if found_chunk:
       print(f"Character {target_position} is in chunk {found_chunk.index}")

Section Detection (Hierarchical)
--------------------------------

Section detection is an **overlay** on top of chunking. Rather than changing how
a document is split, it identifies the document's section structure (from
Markdown headings) and groups the resulting chunks under those sections. This
provides a mid-level abstraction between whole documents and individual chunks,
enabling hierarchical retrieval — for example, finding the most relevant
*section* and then drilling into its chunks.

It is implemented by :class:`~localvectordb.section_detection.SectionDetector`,
which scans the text for headings and emits
:class:`~localvectordb.core.SectionBoundary` objects:

.. code-block:: python

   from localvectordb.section_detection import SectionDetector

   detector = SectionDetector()           # defaults to Markdown headers (# … ######)
   sections = detector.detect_sections(document_text)

   for s in sections:
       print(s.heading_level, s.heading, s.start_pos, s.end_pos)

Key behaviors:

- **Preamble handling**: text before the first heading becomes a leading section
  with ``heading=None``.
- **Code-fence aware**: ``#`` lines inside fenced code blocks (```` ``` ````/
  ``~~~``) are not treated as headings. This matters because extracted content
  is now Markdown and frequently contains fenced code. The helper
  :func:`~localvectordb.section_detection.find_code_fence_spans` computes the
  fenced regions used to exclude these false headers.
- **Custom patterns**: pass a ``pattern`` with two capture groups (level
  indicator, heading text) to detect non-Markdown structures, e.g.
  ``SectionDetector(pattern=r"^(SECTION \d+): (.+)$")``.

Chunks are mapped to their containing section by position with
:meth:`SectionDetector.assign_chunks_to_sections`. Hierarchical (section-level)
embeddings are an opt-in database feature; when enabled, section centroids are
indexed alongside chunk vectors so queries can search at the section level. See
:doc:`hierarchical` for enabling section- and document-level indices, section
metadata extractors, and querying across levels.

Chunking Strategies by Content Type
-----------------------------------

Academic Papers
~~~~~~~~~~~~~~~

.. code-block:: python

   from localvectordb import VectorDB
   from localvectordb.core import MetadataField, MetadataFieldType

   # Academic papers with citations and references
   db_papers = VectorDB(
       "academic_papers",
       "./vector_storage",
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
       "./vector_storage",
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
       "./vector_storage",
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
       "./vector_storage",
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
