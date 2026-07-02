Hierarchical Embeddings
=======================

By default LocalVectorDB embeds and indexes documents at a single granularity:
each document is chunked, and every chunk is embedded and stored in one FAISS
index. **Hierarchical embeddings** add two coarser levels on top of that, giving
a three-level retrieval hierarchy:

.. code-block:: text

   document   ← whole-document centroid (coarsest)
     └── section   ← group of chunks under one heading
           └── chunk   ← the usual fine-grained unit (default)

Each level has its own FAISS index, so a single database can be searched at
whichever granularity fits the question:

- **chunks** — precise passage retrieval (the default, unchanged).
- **sections** — find the most relevant *section* of a document, then optionally
  drill into its chunks. Useful for long, structured documents (manuals,
  papers, books) where the right *region* matters more than the exact sentence.
- **documents** — coarse "which document is this about?" retrieval, matching
  against a single centroid per document.

Sections are an **overlay** on top of chunking, not a different chunking
strategy: the document is chunked exactly as before, sections are detected from
the document's heading structure, and each chunk is assigned to its containing
section. Section and document vectors are **centroids** (the mean of their
member chunk embeddings), so no extra embedding calls are made during ingestion.

The feature is fully opt-in. With ``hierarchical_embeddings=False`` (the
default) nothing changes and no extra indices are built.

Enabling hierarchical embeddings
--------------------------------

Pass ``hierarchical_embeddings=True`` when constructing the database:

.. code-block:: python

   from localvectordb import VectorDB

   db = VectorDB(
       "manuals",
       "./vector_storage",
       hierarchical_embeddings=True,
       # optional: customise how sections are detected
       section_pattern=r"^(#{1,6})\s+(.+)$",          # default: Markdown headers
       # optional: attach metadata to every detected section
       section_metadata_extractors=["heading_path", "word_count", "keywords"],
   )

   db.upsert_document("guide.md", open("guide.md").read())

Constructor parameters:

``hierarchical_embeddings`` : bool, default ``False``
    Enable section- and document-level indices. When ``True``, two extra FAISS
    sidecar files are created next to the main index: ``<name>_sections.faiss``
    and ``<name>_documents.faiss``.

``section_pattern`` : str, default ``r"^(#{1,6})\s+(.+)$"``
    Regex (MULTILINE) used to detect section headings. It must expose two
    capture groups — group 1 is the level indicator (e.g. the ``#`` characters),
    group 2 is the heading text. The default matches Markdown headings, which
    suits extracted content (the file extractors emit Markdown).

``section_metadata_extractors`` : list, optional
    Built-in extractor names and/or :class:`~localvectordb.section_metadata.SectionMetadataExtractor`
    instances to run against each section during ingestion. See
    `Section metadata`_ below.

Section detection
-----------------

Detection is implemented by
:class:`~localvectordb.section_detection.SectionDetector`, which scans text for
headings and emits :class:`~localvectordb.core.SectionBoundary` objects. You can
use it directly to preview how a document will be split:

.. code-block:: python

   from localvectordb.section_detection import SectionDetector

   detector = SectionDetector()           # defaults to Markdown headers (# … ######)
   sections = detector.detect_sections(document_text)

   for s in sections:
       print(s.heading_level, s.heading, s.start_pos, s.end_pos)

Key behaviours:

- **Preamble handling** — text before the first heading becomes a leading
  section with ``heading=None``.
- **Code-fence aware** — ``#`` lines inside fenced code blocks (triple backticks
  or ``~~~``) are *not* treated as headings, so shell prompts and comments in
  example snippets do not create spurious sections. The helper
  :func:`~localvectordb.section_detection.find_code_fence_spans` computes the
  fenced regions that are excluded.
- **Custom patterns** — pass a ``section_pattern`` with two capture groups to
  detect non-Markdown structures, e.g.
  ``section_pattern=r"^(SECTION \d+): (.+)$"``.

Section metadata
----------------

Section metadata extractors run once per detected section during ingestion and
merge their output into that section's metadata, where it is returned on
section-level query results. Built-in extractors (all text-based, no external
dependencies) are referenced by name:

- ``"word_count"`` — adds ``word_count``: number of words in the section.
- ``"char_count"`` — adds ``char_count``: number of characters in the section.
- ``"keywords"`` — adds ``keywords``: top-N keywords by frequency (stop-words
  removed).
- ``"heading_path"`` — adds ``heading_path``: the nested heading trail, e.g.
  ``"Chapter 1 > Introduction > Background"``.

.. code-block:: python

   db = VectorDB(
       "manuals", "./vector_storage",
       hierarchical_embeddings=True,
       section_metadata_extractors=["heading_path", "keywords"],
   )

Write a custom extractor by subclassing
:class:`~localvectordb.section_metadata.SectionMetadataExtractor` and passing an
instance:

.. code-block:: python

   from localvectordb.section_metadata import SectionMetadataExtractor

   class ReadingTimeExtractor(SectionMetadataExtractor):
       name = "reading_time"

       def extract(self, section_text, heading, context):
           minutes = max(1, round(len(section_text.split()) / 200))
           return {"reading_time_min": minutes}

   db = VectorDB(
       "manuals", "./vector_storage",
       hierarchical_embeddings=True,
       section_metadata_extractors=[ReadingTimeExtractor()],
   )

Querying across levels
----------------------

Two independent query options control the hierarchy. Both are available on
:meth:`~localvectordb.database.LocalVectorDB.query` and on the
:doc:`query builder <querybuilder>`.

``search_level`` — *which index to search*
    ``"chunks"`` (default), ``"sections"``, or ``"documents"``. Selects the
    FAISS index the query vector is matched against. ``"sections"`` and
    ``"documents"`` require ``hierarchical_embeddings=True``.

``return_type="sections"`` — *section-grouped results*
    Alongside the existing ``"documents"``/``"chunks"``/``"context"``/``"enriched"``
    modes (see :doc:`query`), ``"sections"`` returns section-level results.

.. code-block:: python

   # Search the section index directly — one result per matching section
   results = db.query(
       "how do I rotate the API key?",
       search_level="sections",
   )

   for r in results:
       print(r.score, r.metadata["section_heading"], r.metadata.get("heading_path"))

   # Coarse document-level retrieval
   docs = db.query("billing and invoices", search_level="documents")

Section results carry ``type="section"`` and expose the section's structure in
their metadata (``section_heading``, ``section_level``, ``section_index``, plus
anything added by section metadata extractors) along with a
:class:`~localvectordb.core.ChunkPosition` spanning the section. Document-level
results carry ``type="document"``.

The same options are available on the fluent query builder via
:meth:`~localvectordb.query_builder.QueryBuilder.search_level` and
:meth:`~localvectordb.query_builder.QueryBuilder.sections`:

.. code-block:: python

   results = (
       db.query_builder()
       .search("network timeout errors")
       .search_level("sections")
       .filter("product", "gateway")
       .limit(5)
       .execute()
   )

Metadata filters apply at every level: for section and document searches the
filter is matched against the parent document's metadata.

Rebuilding indices for existing databases
-----------------------------------------

Section and document centroids are computed during ingestion. If you enable
hierarchical embeddings on a database that already contains documents (or need
to rebuild after corruption), call
:meth:`~localvectordb.database.LocalVectorDB.rebuild_hierarchical_embeddings`:

.. code-block:: python

   db = VectorDB("existing_db", "./vector_storage", hierarchical_embeddings=True)
   db.rebuild_hierarchical_embeddings()   # re-detects sections and rebuilds both indices

This re-runs section detection over every stored document and rebuilds both
sidecar indices from the existing chunk embeddings without re-embedding. It
raises ``ValueError`` if ``hierarchical_embeddings`` is not enabled.

Notes
-----

- **Backward compatible** — leaving ``hierarchical_embeddings=False`` (the
  default) preserves the original single-index behaviour exactly.
- **Persistence** — the ``hierarchical_embeddings`` setting is saved with the
  database; reopening a database re-loads the section/document sidecar indices
  automatically. Renaming a database via the CLI moves the sidecar files too.
- **No extra embedding cost** — section and document vectors are centroids of
  chunk embeddings, so enabling the feature does not add embedding-provider
  calls during ingestion.
