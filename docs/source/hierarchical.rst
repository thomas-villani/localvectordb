Hierarchical Embeddings
=======================

.. note::

   This page is the user-facing reference. For the technique explained from
   first principles, plus a controlled evaluation across local embedding
   encoders, see :doc:`hierarchical-evaluation`.

By default LocalVectorDB embeds and indexes documents at a single granularity:
each document is chunked, and every chunk is embedded and stored in one FAISS
index. **Hierarchical embeddings** add two coarser levels on top of that, giving
a three-level retrieval hierarchy:

.. code-block:: text

   document   ← whole-document vector (coarsest)
     └── section   ← a run of chunks under one heading
           └── chunk   ← the usual fine-grained unit (default)

Each level has its own FAISS index, so a single database can be searched at
whichever granularity fits the question:

- **chunks** — precise passage retrieval (the default, unchanged).
- **sections** — find the most relevant *section* of a document. Useful for long,
  structured documents (manuals, papers, books) where the right *region* matters
  more than the exact sentence.
- **documents** — coarse "which document is this about?" retrieval.
- **fused** — blend chunk retrieval with section retrieval into one ranking. This
  is the mode that improves retrieval *quality* on real, structured documents
  (see `Fused retrieval`_).

Sections are an **overlay** on top of chunking, not a different chunking
strategy: the document is chunked exactly as before, sections are detected from
the document's heading structure, and each chunk is assigned to its containing
section.

The feature is fully opt-in. With ``hierarchical_embeddings=False`` (the
default) nothing changes, no extra indices are built, and the default retrieval
path is byte-for-byte identical.

How a section is represented: raw-span vs centroid
--------------------------------------------------

The interesting question is *what vector stands in for a section*. There are two
strategies, selected by the ``section_vector_strategy`` constructor argument.

``"rawspan"`` (default for new hierarchical databases)
    The section's vector is the embedding of the **section's own text** — the
    concatenated span of every chunk under that heading, embedded directly.

``"centroid"``
    The section's vector is the **mean** of its member chunk embeddings. This is
    free (no extra embedding calls) but is the weaker representation; it is the
    legacy behaviour and the automatic choice for databases created before the
    strategy option existed.

Why the default is raw-span
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A centroid is a *blur*: averaging unit vectors throws away every cross-chunk
interaction — coreference, topic composition, the way an argument builds across
paragraphs — and leaves only a mean direction. Embedding the section's actual
text keeps that structure.

We measured this before changing the default, on real embeddings across Qasper
(real NLP papers) and synthetic section corpora built from BEIR — and again
across several local encoders. Raw-span sections beat both the chunk baseline and
the centroid at every meaningful target; averaging genuinely discards signal the
span embedding keeps, and on some corpora the centroid section scored *below*
chunk-only while the raw-span section scored well above it. The full tables,
encoders, and methodology are in :doc:`hierarchical-evaluation`.

Cost and back-compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Raw-span costs one extra embedding call per section at ingest.** Sections are
  far fewer than chunks, so this is modest, but it is not free the way a centroid
  is. It also means ingest calls the embedding provider for the section text.
- **The document-level vector is always a centroid**, regardless of strategy. The
  document level carries little retrieval signal (see `Limitations`_), is not
  used by fused retrieval, and is not worth an extra embedding call.
- **The strategy is persisted with the database** and resolves on reopen with a
  strict precedence, so existing indices are never silently reinterpreted:

  1. a persisted ``section_vector_strategy`` value always wins on reopen;
  2. a hierarchical database created *before* this option existed resolves to
     ``"centroid"`` — its stored vectors are centroids and must be scored as
     such;
  3. a brand-new hierarchical database (or an existing non-hierarchical one
     turned hierarchical) defaults to ``"rawspan"``.

  To migrate an existing centroid database to raw-span, pass
  ``section_vector_strategy="rawspan"`` and call
  :meth:`~localvectordb.database.LocalVectorDB.rebuild_hierarchical_embeddings`
  (see `Rebuilding indices for existing databases`_).

Enabling hierarchical embeddings
--------------------------------

Pass ``hierarchical_embeddings=True`` when constructing the database:

.. code-block:: python

   from localvectordb import VectorDB

   db = VectorDB(
       "manuals",
       "./vector_storage",
       hierarchical_embeddings=True,
       section_vector_strategy="rawspan",              # default for new hierarchical DBs
       # optional: customise how sections are detected
       section_pattern=r"^(#{1,6})\s+(.+)$",           # default: Markdown headers
       # optional: attach metadata to every detected section
       section_metadata_extractors=["heading_path", "word_count", "keywords"],
   )

   db.upsert(open("guide.md").read(), ids="guide.md")

Constructor parameters:

``hierarchical_embeddings`` : bool, default ``False``
    Enable section- and document-level indices. When ``True``, two extra FAISS
    sidecar files are created next to the main index: ``<name>_sections.faiss``
    and ``<name>_documents.faiss``.

``section_vector_strategy`` : ``"rawspan"`` | ``"centroid"``, optional
    How section vectors are built (see
    `How a section is represented: raw-span vs centroid`_). Defaults
    to ``"rawspan"`` for a new hierarchical database and ``"centroid"`` for a
    legacy one; once set it is persisted and wins on reopen.

``section_pattern`` : str, default ``r"^(#{1,6})\s+(.+)$"``
    Regex (MULTILINE) used to detect section headings. It must expose two
    capture groups — group 1 is the level indicator (e.g. the ``#`` characters),
    group 2 is the heading text. The default matches Markdown headings, which
    suits extracted content (the file extractors emit Markdown).

``section_metadata_extractors`` : list, optional
    Built-in extractor names and/or
    :class:`~localvectordb.section_metadata.SectionMetadataExtractor` instances
    to run against each section during ingestion. See `Section metadata`_.

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

Because raw-span section quality depends on sections being *meaningful* spans,
detection quality matters more with ``"rawspan"`` than with ``"centroid"``. A
document with no headings becomes a single whole-document section; a document
whose headings carve it into coherent topics is where raw-span pays off most.

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

The ``search_level`` option on
:meth:`~localvectordb.database.LocalVectorDB.query` (and on the
:doc:`query builder <querybuilder>`) selects *how* the query is matched:

``"chunks"`` (default)
    The normal chunk search. Unchanged by hierarchical embeddings.

``"sections"``
    Search the section index directly — one result per matching section.
    Requires ``hierarchical_embeddings=True``.

``"documents"``
    Coarse document-index retrieval against the per-document centroid.

``"fused"``
    Blend chunk and section retrieval into a single ranking (see
    `Fused retrieval`_). Requires ``hierarchical_embeddings=True``.

.. code-block:: python

   # Search the section index directly — one result per matching section
   results = db.query(
       "how do I rotate the API key?",
       search_level="sections",
   )

   for r in results:
       print(r.score, r.metadata["section_heading"], r.metadata.get("heading_path"))

Section results carry ``type="section"`` and expose the section's structure in
their metadata (``section_heading``, ``section_level``, ``section_index``, plus
anything added by section metadata extractors) along with a
:class:`~localvectordb.core.ChunkPosition` spanning the section. Document-level
results carry ``type="document"``.

Metadata filters apply at every level: for section, document, and fused searches
the filter is matched against the parent document's metadata.

Fused retrieval
---------------

``search_level="fused"`` is the mode that improves retrieval *quality* rather
than just changing granularity. It runs **both** a chunk search and a
section (raw-span) search for the query, maps every hit up to its target unit,
and merges the two rankings:

.. code-block:: python

   # Fuse chunk + section signal; return the best documents.
   docs = db.query(
       "what latency does the gateway add under load?",
       search_level="fused",
       return_type="documents",   # or "sections"
       section_weight=0.65,       # weight on the section leg (0–1)
   )

It supports two targets via ``return_type``:

- ``"documents"`` — a chunk hit and a section hit both credit their parent
  document; the fused score ranks documents. **This is where the measured win
  lives.**
- ``"sections"`` — chunk hits are rolled up to their containing section (best
  chunk score wins) and fused with the section-index hits; the fused score ranks
  sections.

Why fusion beats either level alone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two levels make different mistakes. A chunk match is precise but narrow — it
can miss a document whose relevance is spread thinly across a section rather than
concentrated in one passage. A section match captures that diffuse relevance but
is coarse. Fusing them recovers documents that neither level ranks well alone, and
on real long documents fusing raw-span sections with chunks beat chunk-only
retrieval at the document target. The absolute gains on real documents are
**modest** — a genuine quality-per-cost judgement, not a slam dunk — and much
larger on section-dense corpora. See :doc:`hierarchical-evaluation` for the
per-corpus numbers.

The fusion itself is **relative-score fusion**: each leg's scores are min-max
normalised within that query's own candidate pool (putting the chunk cosine and
the section cosine on a common 0–1 scale), then blended by ``section_weight``.
This is the same normalise-then-blend rule LocalVectorDB uses for hybrid
vector+keyword search.

Tuning ``section_weight``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``section_weight`` is the weight on the **section leg**:

- ``0.0`` — chunk-only (fusion contributes nothing).
- ``1.0`` — section-only.
- ``0.5`` — an equal blend.

The default is **0.65**, tuned by sweeping the weight on the real long-document
set: leaning slightly toward the section leg retrieved best there. Equal-weight
fusion (``0.5``) still beats chunk-only everywhere we measured, but it is
*suboptimal* when one leg clearly dominates — an equal blend drags a strong
section score back toward a weaker chunk score. If your corpus is more like a
collection of short, single-topic passages than long structured documents, a
lower ``section_weight`` (closer to chunk-only) is the safer choice; for long,
well-sectioned documents, the default or slightly higher works best.

Treat ``section_weight`` as corpus-dependent and tune it on your own relevance
judgments if you have them. Because fused scores are relative to each query's
candidate pool (exactly like hybrid search), a ``score_threshold`` on a fused
query behaves as a **rank-position** cutoff, not an absolute match-quality gate.

The query builder exposes the same controls:

.. code-block:: python

   results = (
       db.query_builder()
       .search("network timeout errors")
       .search_level("fused", section_weight=0.65)
       .filter("product", "gateway")
       .limit(5)
       .execute()
   )

Choosing a level: match granularity to the answer
--------------------------------------------------

The single most important rule from the experiments: **retrieval granularity
should match relevance granularity.**

- When the answer is a **single passage**, the chunk level is unbeatable and
  adding coarser levels only dilutes it. Fusion does not help here; use the
  default ``search_level="chunks"``.
- When the answer is a **region** of a long document — the relevant material is
  spread across a section rather than sitting in one sentence — a section-aware
  representation wins, and ``search_level="fused"`` is the mode to use.

So fused retrieval is a tool for long, structured corpora (papers, manuals,
books, reports), not for short-passage collections where chunk retrieval already
has the right granularity.

Limitations
-----------

**Raw-span needs a long-context embedding model.** A section can be much larger
than a chunk. If a section fits inside the encoder's context window it is
embedded whole; if it is longer, LocalVectorDB embeds it in overlapping windows
and mean-pools the results rather than truncating, so the whole section is
represented. But a **short-context encoder is a poor fit**: on a 512-token model,
long sections are dominated by pooling artefacts and raw-span can lose to the
centroid. Use a long-context embedding model (e.g. OpenAI ``text-embedding-3-*``
at 8k tokens, ``nomic-embed-text``, or another 8k+ encoder) when you enable
``"rawspan"``. :doc:`hierarchical-evaluation` reports results across OpenAI
``text-embedding-3-small`` and several local encoders.

**The document level carries little signal.** Whole-document-centroid retrieval
(``search_level="documents"``) was the weakest arm in every experiment — a single
vector for a long document is too blurry to rank well. Fused retrieval
deliberately ignores the document index. Prefer ``"sections"`` or ``"fused"`` for
coarse retrieval, not ``"documents"``.

**Hierarchical vectors are built on the synchronous ingest path only.** Section
and document vectors are computed during a normal synchronous ``upsert``. The
async ingest path and the "from chunks" ingest path do not build them; if you
ingest through those, run
:meth:`~localvectordb.database.LocalVectorDB.rebuild_hierarchical_embeddings`
afterwards to populate the sidecar indices.

**Fused retrieval is local-only for now.** ``search_level="fused"`` is a
:class:`~localvectordb.database.LocalVectorDB` feature. Remote databases
(``RemoteVectorDB``) raise ``NotImplementedError`` for a fused query, and the
streaming/cursor query paths do not support it — use a materialised
:meth:`~localvectordb.database.LocalVectorDB.query` instead.

**Gains are real but modest, and corpus-dependent.** On real long documents the
improvement is a low single-digit nDCG@10 gain (see :doc:`hierarchical-evaluation`
for exact figures). Measure on your own data before committing to a
``section_weight`` far from the default, and keep chunk-only retrieval for
short-passage corpora.

Rebuilding indices for existing databases
-----------------------------------------

Section and document vectors are computed during ingestion. If you enable
hierarchical embeddings on a database that already contains documents, switch a
database from centroid to raw-span, or need to rebuild after corruption, call
:meth:`~localvectordb.database.LocalVectorDB.rebuild_hierarchical_embeddings`:

.. code-block:: python

   db = VectorDB(
       "existing_db", "./vector_storage",
       hierarchical_embeddings=True,
       section_vector_strategy="rawspan",   # e.g. migrating from centroid
   )
   db.rebuild_hierarchical_embeddings()      # re-detects sections and rebuilds both indices

This re-runs section detection over every stored document and rebuilds both
sidecar indices. With ``"centroid"`` it recomputes the means from the existing
chunk embeddings without re-embedding; with ``"rawspan"`` it re-embeds each
section's text (embedding-provider calls, one batch per document). The
document-level vector is a centroid in both cases. It raises ``ValueError`` if
``hierarchical_embeddings`` is not enabled.

Notes
-----

- **Backward compatible** — leaving ``hierarchical_embeddings=False`` (the
  default) preserves the original single-index behaviour exactly, and the
  default chunk-only retrieval path is unchanged even when hierarchical
  embeddings are enabled.
- **Persistence** — the ``hierarchical_embeddings`` and
  ``section_vector_strategy`` settings are saved with the database; reopening
  re-loads the section/document sidecar indices automatically. Renaming a
  database via the CLI moves the sidecar files too.
- **Embedding cost** — ``"centroid"`` adds no embedding-provider calls;
  ``"rawspan"`` adds one section embedding per section at ingest (sections are
  far fewer than chunks, so the overhead is modest).
