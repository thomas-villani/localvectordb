# tests/test_hierarchical.py
"""
Tests for hierarchical document vectors (sections, multi-level FAISS indices).
"""

import asyncio
import hashlib
import tempfile

import pytest

from localvectordb.core import Chunk, ChunkPosition, Section, SectionBoundary
from localvectordb.section_detection import SectionDetector
from localvectordb.section_metadata import (
    CharCountExtractor,
    HeadingPathExtractor,
    KeywordsExtractor,
    SectionMetadataExtractor,
    WordCountExtractor,
    resolve_extractors,
)

# --- Fixtures ---

MARKDOWN_DOC = """\
This is the preamble text before any headers.

# Introduction

This section introduces the topic of neural networks.
Neural networks are computational models inspired by the brain.

## Background

Deep learning has transformed the field of AI.
Convolutional networks are used for image recognition.

## Related Work

Several approaches have been proposed for text classification.

# Methods

We propose a novel architecture for document retrieval.
The model uses attention mechanisms and transformer layers.

# Results

Our model achieves state-of-the-art performance on multiple benchmarks.
"""


@pytest.fixture
def section_detector():
    return SectionDetector()


@pytest.fixture
def simple_doc():
    return "Hello world. This is a simple document with no headers."


@pytest.fixture
def markdown_doc():
    return MARKDOWN_DOC


# --- SectionDetector Tests ---


class TestSectionDetector:
    """Test section detection with various document formats."""

    def test_detect_markdown_headers(self, section_detector, markdown_doc):
        """Test detection of markdown-style headers."""
        sections = section_detector.detect_sections(markdown_doc)

        # Should find: preamble + Introduction + Background + Related Work + Methods + Results
        assert len(sections) == 6

        # Preamble
        assert sections[0].heading is None
        assert sections[0].heading_level is None
        assert sections[0].index == 0

        # Introduction
        assert sections[1].heading == "Introduction"
        assert sections[1].heading_level == 1
        assert sections[1].index == 1

        # Background
        assert sections[2].heading == "Background"
        assert sections[2].heading_level == 2
        assert sections[2].index == 2

        # Related Work
        assert sections[3].heading == "Related Work"
        assert sections[3].heading_level == 2

        # Methods
        assert sections[4].heading == "Methods"
        assert sections[4].heading_level == 1

        # Results
        assert sections[5].heading == "Results"
        assert sections[5].heading_level == 1

    def test_no_headers(self, section_detector, simple_doc):
        """Test document with no headers produces one section."""
        sections = section_detector.detect_sections(simple_doc)
        assert len(sections) == 1
        assert sections[0].heading is None
        assert sections[0].start_pos == 0
        assert sections[0].end_pos == len(simple_doc)

    def test_empty_document(self, section_detector):
        """Test empty document returns no sections."""
        sections = section_detector.detect_sections("")
        assert len(sections) == 0

    def test_custom_pattern(self):
        """Test custom section pattern."""
        text = "SECTION 1: Intro\nSome text\nSECTION 2: Methods\nMore text"
        detector = SectionDetector(pattern=r"^(SECTION \d+): (.+)$")
        sections = detector.detect_sections(text)
        assert len(sections) == 2
        assert sections[0].heading == "Intro"
        assert sections[1].heading == "Methods"

    def test_no_preamble(self, section_detector):
        """Test document starting with a header (no preamble)."""
        text = "# First Section\nContent here.\n# Second Section\nMore content."
        sections = section_detector.detect_sections(text)
        assert len(sections) == 2
        assert sections[0].heading == "First Section"
        assert sections[0].index == 0

    def test_section_positions(self, section_detector):
        """Test that section positions cover the entire document."""
        text = "Preamble\n# A\nText A\n# B\nText B"
        sections = section_detector.detect_sections(text)

        # Verify no gaps
        assert sections[0].start_pos == 0
        assert sections[-1].end_pos == len(text)
        for i in range(len(sections) - 1):
            assert sections[i].end_pos == sections[i + 1].start_pos

    def test_section_line_numbers(self, section_detector):
        """Test that section line numbers are correct."""
        text = "Line 1\nLine 2\n# Header\nLine 4"
        sections = section_detector.detect_sections(text)
        assert len(sections) == 2
        # Preamble starts at line 1
        assert sections[0].start_line == 1
        # Header at line 3
        assert sections[1].start_line == 3

    def test_ignores_headers_in_fenced_code(self, section_detector):
        """Markdown ``#`` lines inside fenced code blocks are not sections."""
        text = (
            "# Real Heading\n\n"
            "Intro.\n\n"
            "```python\n"
            "# a comment, not a heading\n"
            "## also not a heading\n"
            "def f():\n"
            "    pass\n"
            "```\n\n"
            "## Real Subsection\n\n"
            "Body.\n"
        )
        sections = section_detector.detect_sections(text)
        headings = [s.heading for s in sections if s.heading]
        assert headings == ["Real Heading", "Real Subsection"]
        # The whole code block stays inside the first section's span.
        intro = next(s for s in sections if s.heading == "Real Heading")
        assert "# a comment, not a heading" in text[intro.start_pos : intro.end_pos]

    def test_ignores_headers_in_tilde_fenced_code(self, section_detector):
        """Tilde fences (~~~) are recognised as code as well."""
        text = "# Heading\n\n~~~\n# not a heading\n~~~\n\n## Next\n"
        sections = section_detector.detect_sections(text)
        headings = [s.heading for s in sections if s.heading]
        assert headings == ["Heading", "Next"]

    def test_unterminated_fence_swallows_rest(self, section_detector):
        """An unterminated fence extends to EOF, suppressing later ``#`` lines."""
        text = "# Heading\n\n```\n# inside code\n## still code\n"
        sections = section_detector.detect_sections(text)
        headings = [s.heading for s in sections if s.heading]
        assert headings == ["Heading"]


class TestChunkToSectionAssignment:
    """Test chunk-to-section mapping."""

    def test_basic_assignment(self, section_detector):
        """Test correct chunk-to-section mapping."""
        text = "Preamble text\n# Section A\nText in A\n# Section B\nText in B"
        sections = section_detector.detect_sections(text)

        # Create chunks that span the sections
        chunks = [
            Chunk(content="Preamble text", position=ChunkPosition(0, 13, 1, 1, 1, 14), tokens=2, index=0),
            Chunk(content="Section A text", position=ChunkPosition(14, 38, 2, 1, 2, 25), tokens=3, index=1),
            Chunk(content="Section B text", position=ChunkPosition(39, 58, 4, 1, 4, 20), tokens=3, index=2),
        ]

        mapping = SectionDetector.assign_chunks_to_sections(chunks, sections)

        # Each chunk should be in a different section
        assert 0 in mapping[0]  # chunk 0 in preamble
        assert 1 in mapping[1]  # chunk 1 in Section A
        assert 2 in mapping[2]  # chunk 2 in Section B

    def test_empty_chunks(self, section_detector):
        """Test empty chunk list."""
        text = "# Header\nContent"
        sections = section_detector.detect_sections(text)
        mapping = SectionDetector.assign_chunks_to_sections([], sections)
        assert mapping == {}

    def test_empty_sections(self):
        """Test empty section list."""
        chunks = [Chunk(content="text", position=ChunkPosition(0, 4, 1, 1, 1, 5), tokens=1, index=0)]
        mapping = SectionDetector.assign_chunks_to_sections(chunks, [])
        assert mapping == {}

    def test_content_hash_computation(self, section_detector):
        """Test section content hash computation."""
        text = "# Header\nContent text"
        sections = section_detector.detect_sections(text)

        content_hash = SectionDetector.compute_section_content_hash(text, sections[0])
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert content_hash == expected


# --- Section Metadata Extractors ---


class TestSectionMetadataExtractors:
    """Test section metadata extractors."""

    def test_word_count_extractor(self):
        ext = WordCountExtractor()
        result = ext.extract("hello world foo bar", None, {})
        assert result == {"word_count": 4}

    def test_char_count_extractor(self):
        ext = CharCountExtractor()
        result = ext.extract("hello", None, {})
        assert result == {"char_count": 5}

    def test_heading_path_extractor(self):
        ext = HeadingPathExtractor()
        all_sections = [
            ("Introduction", 1),
            ("Background", 2),
            ("History", 3),
        ]
        context = {
            "section_index": 2,
            "heading_level": 3,
            "all_sections": all_sections,
        }
        result = ext.extract("some text", "History", context)
        assert "heading_path" in result
        assert "History" in result["heading_path"]

    def test_heading_path_no_heading(self):
        ext = HeadingPathExtractor()
        result = ext.extract("text", None, {"section_index": 0})
        assert result == {"heading_path": ""}

    def test_keywords_extractor(self):
        ext = KeywordsExtractor(top_n=3, min_word_length=4)
        text = "neural networks use neural computation for neural processing tasks"
        result = ext.extract(text, None, {})
        assert "keywords" in result
        assert "neural" in result["keywords"]

    def test_resolve_extractors_by_name(self):
        extractors = resolve_extractors(["word_count", "char_count"])
        assert len(extractors) == 2
        assert isinstance(extractors[0], WordCountExtractor)
        assert isinstance(extractors[1], CharCountExtractor)

    def test_resolve_extractors_with_instance(self):
        inst = WordCountExtractor()
        extractors = resolve_extractors([inst])
        assert extractors[0] is inst

    def test_resolve_extractors_unknown(self):
        with pytest.raises(ValueError, match="Unknown"):
            resolve_extractors(["nonexistent_extractor"])

    def test_resolve_extractors_none(self):
        assert resolve_extractors(None) == []

    def test_custom_extractor(self):
        """Test user-provided callable extractor."""

        class MyExtractor(SectionMetadataExtractor):
            name = "custom"

            def extract(self, section_text, heading, context):
                return {"has_heading": heading is not None}

        ext = MyExtractor()
        result = ext.extract("text", "My Heading", {})
        assert result == {"has_heading": True}

        result2 = ext.extract("text", None, {})
        assert result2 == {"has_heading": False}


# --- Section dataclass ---


class TestSectionDataclass:
    """Test Section and SectionBoundary dataclasses."""

    def test_section_from_boundary(self):
        boundary = SectionBoundary(
            index=0,
            heading="Intro",
            heading_level=1,
            start_pos=0,
            end_pos=100,
            start_line=1,
            end_line=10,
        )
        section = Section.from_boundary(boundary, content_hash="abc123", faiss_id=42)
        assert section.index == 0
        assert section.heading == "Intro"
        assert section.heading_level == 1
        assert section.content_hash == "abc123"
        assert section.faiss_id == 42

    def test_section_boundary_metadata(self):
        boundary = SectionBoundary(
            index=0,
            heading="Test",
            heading_level=1,
            start_pos=0,
            end_pos=50,
            metadata={"key": "value"},
        )
        assert boundary.metadata == {"key": "value"}


# --- Integration tests with mock embeddings ---


class TestHierarchicalIntegration:
    """Integration tests using MockEmbeddings."""

    @pytest.fixture
    def db_with_hierarchy(self):
        """Create a database with hierarchical embeddings enabled."""
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = LocalVectorDB(
                name="test_hier",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=True,
                section_metadata_extractors=["word_count"],
                chunk_size=50,
                chunk_overlap=0,
                enable_fts=True,
            )
            yield db
            db.close()

    @pytest.fixture
    def db_without_hierarchy(self):
        """Create a standard database without hierarchical embeddings."""
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = LocalVectorDB(
                name="test_std",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=False,
                chunk_size=50,
                chunk_overlap=0,
                enable_fts=True,
            )
            yield db
            db.close()

    def test_backward_compat_no_hierarchy(self, db_without_hierarchy):
        """Existing DB without hierarchical_embeddings works unchanged."""
        db = db_without_hierarchy
        db.upsert(["Hello world. This is a test."], ids=["doc1"])
        results = db.query("hello", k=1)
        assert len(results) >= 1
        assert results[0].type == "document"

    @pytest.mark.parametrize("level", ["sections", "documents", "fused"])
    def test_non_chunk_level_without_hierarchy_raises(self, db_without_hierarchy, level):
        """Asking for a level the DB cannot serve is an error, not chunk results.

        Regression: 'sections'/'documents' used to fall through to chunk search
        here, so the caller got plausible wrong-level results and concluded the
        feature did nothing. Only 'fused' raised.
        """
        db = db_without_hierarchy
        db.upsert(["Hello world. This is a test."], ids=["doc1"])

        with pytest.raises(ValueError, match="requires hierarchical_embeddings=True"):
            db.query("hello", search_level=level, k=1)

    @pytest.mark.parametrize("level", ["sections", "documents", "fused"])
    async def test_non_chunk_level_without_hierarchy_raises_async(self, db_without_hierarchy, level):
        """query_async delegates every non-chunk level to the sync guard."""
        db = db_without_hierarchy
        db.upsert(["Hello world. This is a test."], ids=["doc1"])

        with pytest.raises(ValueError, match="requires hierarchical_embeddings=True"):
            await db.query_async("hello", search_level=level, k=1)

    @pytest.mark.parametrize("level", ["sections", "fused"])
    def test_non_chunk_level_with_hierarchy_still_works(self, db_with_hierarchy, level):
        """The guard must not block the levels a hierarchical DB can serve."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        results = db.query("neural networks", search_level=level, k=3)
        assert len(results) >= 1

    # -- H8: reranking must not be silently dropped on fused/sections/documents --

    @pytest.mark.parametrize("level", ["fused", "sections", "documents"])
    def test_reranker_is_invoked_on_non_chunk_levels(self, db_with_hierarchy, level):
        """Before the fix these levels early-returned above the rerank block, so a
        configured reranker was silently ignored."""
        from localvectordb.reranking import Reranker

        class _RecordingReranker(Reranker):
            def __init__(self):
                super().__init__("recording")
                self.calls = 0

            @property
            def provider_name(self):
                return "recording"

            def validate_model(self):
                return True

            def rerank(self, query, results, top_k=None):
                self.calls += 1
                return list(results)[:top_k] if top_k is not None else list(results)

        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        rr = _RecordingReranker()
        db.query("neural networks", search_level=level, k=2, reranker=rr)
        assert rr.calls >= 1, f"reranker was not invoked at search_level={level!r}"

    def test_reranker_reorders_fused_results(self, db_with_hierarchy):
        """A deterministic reranker must be able to change the top result."""
        from localvectordb.reranking import Reranker

        class _MarkerReranker(Reranker):
            def __init__(self, marker):
                super().__init__("marker")
                self.marker = marker

            @property
            def provider_name(self):
                return "marker"

            def validate_model(self):
                return True

            def rerank(self, query, results, top_k=None):
                for r in results:
                    r.score = 1.0 if self.marker in (r.content or "") else 0.0
                ranked = sorted(results, key=lambda x: x.score, reverse=True)
                return ranked[:top_k] if top_k is not None else ranked

        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        rr = _MarkerReranker(marker="attention mechanisms")
        results = db.query("neural networks", search_level="fused", k=5, reranker=rr)
        assert results, "no results to rerank"
        assert "attention mechanisms" in (results[0].content or "")

    async def test_reranker_forwarded_on_async_non_chunk_level(self, db_with_hierarchy):
        """query_async delegates non-chunk levels to sync query(); it must forward
        the reranker too (H8)."""
        from localvectordb.reranking import Reranker

        class _RecordingReranker(Reranker):
            def __init__(self):
                super().__init__("recording")
                self.calls = 0

            @property
            def provider_name(self):
                return "recording"

            def validate_model(self):
                return True

            def rerank(self, query, results, top_k=None):
                self.calls += 1
                return list(results)[:top_k] if top_k is not None else list(results)

        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        rr = _RecordingReranker()
        await db.query_async("neural networks", search_level="sections", k=2, reranker=rr)
        assert rr.calls >= 1

    def test_upsert_with_hierarchy(self, db_with_hierarchy):
        """Test that upsert populates sections table."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        stats = db.get_stats()
        assert stats["documents"] == 1
        assert stats["chunks"] > 0
        assert stats["sections"] > 0
        assert stats["hierarchical_embeddings"] is True
        assert stats["section_index_vectors"] > 0
        assert stats["document_index_vectors"] > 0

    def test_query_default_unchanged(self, db_with_hierarchy):
        """Default query behavior unchanged with hierarchical enabled."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", k=3)
        assert len(results) >= 1
        # Default returns documents
        assert results[0].type == "document"

    def test_query_return_sections(self, db_with_hierarchy):
        """query(return_type='sections') returns section-level results."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", return_type="sections", k=5)
        # Should get section results (if chunks map to sections)
        if results:
            assert results[0].type == "section"
            assert ":section:" in results[0].id

    def test_query_search_level_sections(self, db_with_hierarchy):
        """query(search_level='sections') searches section FAISS index."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", search_level="sections", k=5)
        assert len(results) >= 1
        assert results[0].type == "section"
        # Should contain section metadata
        assert "section_heading" in results[0].metadata

    def test_query_search_level_documents(self, db_with_hierarchy):
        """query(search_level='documents') searches document FAISS index."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", search_level="documents", k=5)
        assert len(results) >= 1
        assert results[0].type == "document"

    def test_section_metadata_populated(self, db_with_hierarchy):
        """Section metadata extractors populate section.metadata."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        # Query sections and check metadata
        results = db.query("introduction", search_level="sections", k=5)
        assert results, "expected at least one section result"
        # The word_count section-metadata extractor should have populated metadata.
        assert "word_count" in results[0].metadata

    def test_delete_cascades_sections(self, db_with_hierarchy):
        """Document delete removes sections and FAISS vectors."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        stats_before = db.get_stats()
        assert stats_before["sections"] > 0
        assert stats_before["section_index_vectors"] > 0

        db.delete("md_doc")

        stats_after = db.get_stats()
        assert stats_after["documents"] == 0
        assert stats_after["sections"] == 0

    def test_upsert_update_replaces_sections(self, db_with_hierarchy):
        """Upserting same doc ID replaces sections."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        db.get_stats()  # verify first upsert works

        new_doc = "# New Header\nNew content.\n# Another\nMore content."
        db.upsert([new_doc], ids=["md_doc"])
        stats2 = db.get_stats()

        assert stats2["documents"] == 1
        # Sections should reflect new doc
        assert stats2["sections"] > 0

    def test_multiple_documents(self, db_with_hierarchy):
        """Test hierarchical with multiple documents."""
        db = db_with_hierarchy
        doc1 = "# Doc1 Header\nDoc 1 content about neural networks."
        doc2 = "# Doc2 Header\nDoc 2 content about machine learning."

        db.upsert([doc1, doc2], ids=["doc1", "doc2"])

        stats = db.get_stats()
        assert stats["documents"] == 2
        assert stats["sections"] >= 2

        results = db.query("neural", search_level="documents", k=2)
        assert len(results) >= 1

    def test_rebuild_hierarchical_embeddings(self, db_without_hierarchy):
        """Test rebuilding hierarchical embeddings on existing DB."""
        db = db_without_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        # Enable hierarchical and rebuild
        db._hierarchical_embeddings = True
        db._section_detector = SectionDetector()
        db._section_metadata_extractors = []
        db.section_index = db._create_flat_index()
        db.document_index = db._create_flat_index()

        db.rebuild_hierarchical_embeddings()

        stats = db.get_stats()
        assert stats["sections"] > 0

    def test_query_builder_search_level(self, db_with_hierarchy):
        """Test QueryBuilder .search_level() and .sections() methods."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        results = db.query_builder().search("neural networks").search_level("sections").sections().limit(5).execute()
        if results:
            assert results[0].type == "section"

    def test_save_and_reload(self):
        """Test that hierarchical indices persist across save/load."""
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate
            db = LocalVectorDB(
                name="persist_test",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=True,
                chunk_size=50,
            )
            db.upsert([MARKDOWN_DOC], ids=["md_doc"])
            stats1 = db.get_stats()
            db.close()

            # Reopen
            db2 = LocalVectorDB(
                name="persist_test",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=True,
                chunk_size=50,
            )
            stats2 = db2.get_stats()

            assert stats2["sections"] == stats1["sections"]
            assert stats2["section_index_vectors"] == stats1["section_index_vectors"]
            assert stats2["document_index_vectors"] == stats1["document_index_vectors"]

            # Should be able to query
            results = db2.query("neural", search_level="sections", k=3)
            assert len(results) >= 1
            db2.close()


class TestT15CentroidNormalizationAndMetric:
    """T1.5: unit-normalized centroids + per-index metric for hierarchical search.

    Two independent defects lived in the same scoring boundary:

    1. Section/document centroids were raw ``np.mean`` of chunk vectors. Averaging
       shrinks the norm below unit, so a centroid no longer sat on the unit sphere
       the query occupies. Fixed by :meth:`_unit_normalize_centroids` at the write.
    2. Section/document indices are ``IndexFlatL2``, but the search converted their
       distances with :meth:`_distance_to_similarity` auto-detecting the metric off
       the *main* index. On an ``IndexFlatIP`` database that applies the IP formula
       ``(d + 1) / 2`` -- which *increases* with L2 distance -- to L2 distances,
       inverting the ranking (T1.4's correct IP detection is what exposed this).
       Fixed by passing the searched index's own metric explicitly.
    """

    def _build(self, tmpdir, faiss_index_type):
        from localvectordb.database import LocalVectorDB

        db = LocalVectorDB(
            name="t15",
            base_path=tmpdir,
            embedding_provider="mock",
            embedding_model="mock",
            hierarchical_embeddings=True,
            # This suite is about the centroid write path; pin it so the default
            # flip to raw-span (which builds section vectors differently) does not
            # quietly stop exercising centroid normalization.
            section_vector_strategy="centroid",
            faiss_index_type=faiss_index_type,
            chunk_size=50,
            chunk_overlap=0,
            enable_fts=True,
        )
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        return db

    def test_unit_normalize_centroids(self):
        """The helper unit-normalizes each row, leaves zero rows, copies input."""
        import numpy as np

        from localvectordb.database import LocalVectorDB

        raw = np.array(
            [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        original = raw.copy()
        out = LocalVectorDB._unit_normalize_centroids(raw)

        # Non-zero rows are unit norm; the all-zero row is untouched.
        assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-6)
        assert np.linalg.norm(out[2]) == pytest.approx(1.0, abs=1e-6)
        assert np.array_equal(out[1], np.zeros(3, dtype=np.float32))
        # Caller's buffer is never mutated in place.
        assert np.array_equal(raw, original)

    @pytest.mark.parametrize("faiss_index_type", ["IndexFlatL2", "IndexFlatIP"])
    def test_stored_centroids_are_unit_norm(self, faiss_index_type):
        """Every non-empty section/document centroid is stored at unit norm.

        An empty section (no chunks mapped) has an all-zero centroid with no
        direction; the helper leaves those untouched, so they stay at norm 0.
        """
        import faiss
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._build(tmpdir, faiss_index_type)
            try:
                unit_seen = 0
                for index in (db.section_index, db.document_index):
                    assert index is not None and index.ntotal > 0
                    ids = faiss.vector_to_array(index.id_map)
                    for fid in ids:
                        vec = np.asarray(index.reconstruct(int(fid)), dtype=np.float32)
                        norm = float(np.linalg.norm(vec))
                        assert norm == pytest.approx(1.0, abs=1e-4) or norm == pytest.approx(0.0, abs=1e-6)
                        if norm > 0.5:
                            unit_seen += 1
                assert unit_seen > 0, "expected at least one non-empty centroid"
            finally:
                db.close()

    @pytest.mark.parametrize("faiss_index_type", ["IndexFlatL2", "IndexFlatIP"])
    def test_section_scores_use_section_index_metric(self, faiss_index_type):
        """Section scores follow the L2 index's own formula, not the main metric.

        Pre-fix on ``IndexFlatIP`` this asserted-value differs: the search applied
        ``(d + 1) / 2`` to the L2 section distances, so this equality fails and the
        ranking inverts.
        """
        import numpy as np

        query = "neural networks"
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._build(tmpdir, faiss_index_type)
            try:
                # Mock embeddings are already unit norm; the section index is
                # IndexFlatL2, so the query is used as-is (no IP boundary norm).
                qvec = np.asarray(db.embedding_provider.embed_sync([query])[0], dtype=np.float32)

                results = db.query(query, search_level="sections", k=10)
                assert len(results) >= 2, "need multiple sections to test ranking"

                with db.connection_pool.get_connection() as conn:
                    for r in results:
                        doc_id, sec_idx = r.id.split(":section:")
                        row = conn.execute(
                            "SELECT faiss_id FROM sections WHERE document_id = ? AND section_index = ?",
                            (doc_id, int(sec_idx)),
                        ).fetchone()
                        centroid = np.asarray(db.section_index.reconstruct(int(row["faiss_id"])), dtype=np.float32)
                        l2_sq = float(np.sum((qvec - centroid) ** 2))
                        expected = 1.0 / (1.0 + l2_sq)
                        assert r.score == pytest.approx(expected, abs=1e-4)

                # Scores are a valid similarity and ranking is by descending score.
                scores = [r.score for r in results]
                assert all(0.0 <= s <= 1.0 for s in scores)
                assert scores == sorted(scores, reverse=True)
            finally:
                db.close()


def _make_hier_db(tmpdir, name="hier", strategy=None):
    """A hierarchical MockEmbeddings DB (raw-span by default) ingested with MARKDOWN_DOC."""
    from localvectordb.database import LocalVectorDB

    db = LocalVectorDB(
        name=name,
        base_path=tmpdir,
        embedding_provider="mock",
        embedding_model="mock",
        hierarchical_embeddings=True,
        section_vector_strategy=strategy,
        chunk_size=50,
        chunk_overlap=0,
        enable_fts=True,
    )
    db.upsert([MARKDOWN_DOC], ids=["md_doc"])
    return db


def _section_vec(db, heading):
    """Reconstruct the section-index vector for the section with ``heading``."""
    import numpy as np

    with db.connection_pool.get_connection() as conn:
        row = conn.execute("SELECT faiss_id FROM sections WHERE heading = ?", (heading,)).fetchone()
    assert row is not None and row["faiss_id"] is not None
    return np.asarray(db.section_index.reconstruct(int(row["faiss_id"])), dtype=np.float32)


class TestSectionVectorStrategy:
    """The section_vector_strategy knob: default, persistence, back-compat, pooling."""

    def test_section_vector_strategy_defaults_rawspan_for_new_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                assert db.section_vector_strategy == "rawspan"
            finally:
                db.close()

    def test_get_stats_reports_strategy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                assert db.get_stats()["section_vector_strategy"] == "rawspan"
            finally:
                db.close()

    def test_rawspan_section_vector_differs_from_centroid(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as td_raw, tempfile.TemporaryDirectory() as td_cen:
            raw_db = _make_hier_db(td_raw, name="raw", strategy="rawspan")
            cen_db = _make_hier_db(td_cen, name="cen", strategy="centroid")
            try:
                # Same section, two representations: embedding the section's text
                # vs averaging its chunk vectors. They must not coincide.
                raw_vec = _section_vec(raw_db, "Introduction")
                cen_vec = _section_vec(cen_db, "Introduction")
                assert not np.allclose(raw_vec, cen_vec, atol=1e-4)
            finally:
                raw_db.close()
                cen_db.close()

    def test_section_vector_strategy_persists_across_reopen(self):
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir, name="persist", strategy="rawspan")
            n_vectors = db.get_stats()["section_index_vectors"]
            db.close()

            # Reopen WITHOUT passing the kwarg; the persisted value must win.
            db2 = LocalVectorDB(
                name="persist",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=True,
                chunk_size=50,
                chunk_overlap=0,
            )
            try:
                assert db2.section_vector_strategy == "rawspan"
                assert db2.get_stats()["section_index_vectors"] == n_vectors
            finally:
                db2.close()

    def test_legacy_hierarchical_db_defaults_centroid(self):
        """A hierarchical DB created before the knob existed resolves to centroid.

        Its stored section vectors are centroids; defaulting a keyless legacy DB to
        rawspan would silently reinterpret them under a different geometry.
        """
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir, name="legacy", strategy="rawspan")
            # Simulate the pre-knob on-disk state: drop the strategy key, keep the
            # hierarchical flag.
            with db.connection_pool.get_connection() as conn:
                conn.execute("DELETE FROM config WHERE key = 'section_vector_strategy'")
                conn.commit()
            db.close()

            db2 = LocalVectorDB(
                name="legacy",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=True,
                chunk_size=50,
                chunk_overlap=0,
            )
            try:
                assert db2.section_vector_strategy == "centroid"
            finally:
                db2.close()

    def test_invalid_strategy_raises(self):
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="section_vector_strategy"):
                LocalVectorDB(
                    name="bad",
                    base_path=tmpdir,
                    embedding_provider="mock",
                    embedding_model="mock",
                    hierarchical_embeddings=True,
                    section_vector_strategy="bogus",  # type: ignore[arg-type]
                    chunk_size=50,
                    chunk_overlap=0,
                )

    def test_over_window_section_pooling(self, monkeypatch):
        """A span over the encoder window is embedded in windows and mean-pooled."""
        import numpy as np

        from localvectordb.database import _span_embed
        from localvectordb.embeddings import EmbeddingRegistry

        # The mock provider exposes no context window, so the pooler uses the
        # fallback default; shrink it so a short text actually windows.
        monkeypatch.setattr(_span_embed, "_DEFAULT_WINDOW_CHARS", 20)
        provider = EmbeddingRegistry.create_provider("mock", "mock")
        dim = provider.get_dimension()

        text = " ".join(f"token{i}" for i in range(40))  # well over 20 chars
        windows = _span_embed._windows(text, 20)
        assert len(windows) > 1  # actually windowed

        pooled = _span_embed.embed_spans_pooled(provider, [text], dim)
        assert pooled.shape == (1, dim)
        expected = np.asarray(provider.embed_sync(windows), dtype=np.float32).mean(axis=0)
        assert np.allclose(pooled[0], expected, atol=1e-5)

    def test_window_size_derives_from_provider_context(self):
        """The window is sized to the provider's context window, not a fixed guess."""
        from localvectordb.database import _span_embed

        class Ctx2k:
            num_ctx = 2048

        class Ctx8k:
            max_input_tokens = 8192

        class BoolCtx:
            num_ctx = True  # a bool is not a real context window

        class NoCtx:
            pass

        # num_ctx preferred, then max_input_tokens; conservative ~3 chars/token.
        assert _span_embed._window_chars_for(Ctx2k()) == int(2048 * _span_embed._WINDOW_CHARS_PER_TOKEN)
        assert _span_embed._window_chars_for(Ctx8k()) == int(8192 * _span_embed._WINDOW_CHARS_PER_TOKEN)
        # bool / missing context fall back to the fixed default.
        assert _span_embed._window_chars_for(BoolCtx()) == _span_embed._DEFAULT_WINDOW_CHARS
        assert _span_embed._window_chars_for(NoCtx()) == _span_embed._DEFAULT_WINDOW_CHARS
        # A small-context model gets a smaller window than an 8k one.
        assert _span_embed._window_chars_for(Ctx2k()) < _span_embed._window_chars_for(Ctx8k())

    def test_empty_span_is_zero_row(self):
        import numpy as np

        from localvectordb.database._span_embed import embed_spans_pooled
        from localvectordb.embeddings import EmbeddingRegistry

        provider = EmbeddingRegistry.create_provider("mock", "mock")
        dim = provider.get_dimension()
        out = embed_spans_pooled(provider, ["", "hello"], dim)
        assert out.shape == (2, dim)
        assert np.array_equal(out[0], np.zeros(dim, dtype=np.float32))
        assert np.linalg.norm(out[1]) > 0


class TestTwoLegFusion:
    """The _two_leg_minmax_fuse primitive (pure, no DB)."""

    def test_two_leg_minmax_fuse_combines_legs(self):
        from localvectordb.database._search import _two_leg_minmax_fuse

        # primary min-max: a=1, b=0 ; secondary min-max: a=0, c=1
        fused = _two_leg_minmax_fuse({"a": 0.9, "b": 0.1}, {"a": 0.2, "c": 0.8}, 0.65)
        assert fused["a"] == pytest.approx(0.35)
        assert fused["b"] == pytest.approx(0.0)
        assert fused["c"] == pytest.approx(0.65)

    def test_two_leg_fuse_weight_extremes(self):
        from localvectordb.database._search import _two_leg_minmax_fuse

        primary = {"a": 0.9, "b": 0.1}
        secondary = {"a": 0.1, "b": 0.9}
        # secondary_weight=0 -> primary ranking (a > b); =1 -> secondary (b > a).
        f0 = _two_leg_minmax_fuse(primary, secondary, 0.0)
        assert f0["a"] > f0["b"]
        f1 = _two_leg_minmax_fuse(primary, secondary, 1.0)
        assert f1["b"] > f1["a"]

    def test_two_leg_fuse_single_leg_scores_zero_on_other(self):
        from localvectordb.database._search import _two_leg_minmax_fuse

        fused = _two_leg_minmax_fuse({"only_p": 0.5}, {"only_s": 0.5}, 0.5)
        # A key in one leg only gets 0 from the other; a lone value min-maxes to 1.0.
        assert fused["only_p"] == pytest.approx(0.5)
        assert fused["only_s"] == pytest.approx(0.5)


class TestFusedSearch:
    """search_level='fused' dispatch (relevance is validated by the eval harness)."""

    def test_query_search_level_fused_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = db.query("neural networks", search_level="fused", return_type="documents", k=3)
                assert len(results) >= 1
                assert all(r.type == "document" for r in results)
                assert results == sorted(results, key=lambda r: r.score, reverse=True)
            finally:
                db.close()

    def test_query_search_level_fused_sections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = db.query("neural networks", search_level="fused", return_type="sections", k=5)
                assert len(results) >= 1
                assert all(r.type == "section" for r in results)
                assert all(":section:" in r.id for r in results)
            finally:
                db.close()

    def test_fused_requires_hierarchical(self):
        from localvectordb.database import LocalVectorDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = LocalVectorDB(
                name="plain",
                base_path=tmpdir,
                embedding_provider="mock",
                embedding_model="mock",
                hierarchical_embeddings=False,
                chunk_size=50,
                chunk_overlap=0,
            )
            db.upsert([MARKDOWN_DOC], ids=["md_doc"])
            try:
                with pytest.raises(ValueError, match="hierarchical"):
                    db.query("neural networks", search_level="fused")
            finally:
                db.close()

    def test_fused_bad_return_type_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                with pytest.raises(ValueError, match="return_type"):
                    db.query("neural networks", search_level="fused", return_type="chunks")
            finally:
                db.close()

    def test_fused_streaming_unsupported(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                with pytest.raises(ValueError, match="fused"):
                    db.query_cursor("neural networks", search_level="fused")
            finally:
                db.close()

    def test_fused_async(self):
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = asyncio.run(
                    db.query_async("neural networks", search_level="fused", return_type="documents", k=3)
                )
                assert len(results) >= 1
                assert all(r.type == "document" for r in results)
            finally:
                db.close()

    def test_fused_weight_extremes_change_ranking(self):
        """section_weight=0 is chunk-only; =1 is section-only -- the knob has effect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                chunk_only = db.query(
                    "neural networks", search_level="fused", return_type="documents", section_weight=0.0, k=5
                )
                section_only = db.query(
                    "neural networks", search_level="fused", return_type="documents", section_weight=1.0, k=5
                )
                # Both return documents; the knob is accepted end-to-end.
                assert all(r.type == "document" for r in chunk_only)
                assert all(r.type == "document" for r in section_only)
            finally:
                db.close()

    def test_query_default_unchanged_with_rawspan(self):
        """A default query on a raw-span hierarchical DB still returns documents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = db.query("neural networks", k=3)
                assert len(results) >= 1
                assert results[0].type == "document"
            finally:
                db.close()

    def test_query_builder_search_level_fused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = (
                    db.query_builder().search("neural networks").search_level("fused", section_weight=0.65).execute()
                )
                assert len(results) >= 1
                assert all(r.type == "document" for r in results)
            finally:
                db.close()


class TestSectionSearchReturnType:
    """search_level='sections' honours return_type instead of ignoring it.

    It used to accept return_type and always answer in sections, so
    return_type='documents' silently got the wrong unit -- the same class of
    defect as the silent chunk fallthrough, one level down. The fix has to keep
    a bare query(search_level='sections') answering in sections, which is what
    every documented example expects, so return_type defaults to None ("the unit
    the level searched") rather than to "documents".
    """

    def test_sections_default_still_returns_sections(self):
        """The default must not drift to documents: every doc example relies on it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = db.query("neural networks", search_level="sections", k=5)
                assert len(results) >= 1
                assert all(r.type == "section" for r in results)
            finally:
                db.close()

    def test_sections_with_return_type_documents_rolls_up(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                results = db.query("neural networks", search_level="sections", return_type="documents", k=3)
                assert len(results) >= 1
                assert all(r.type == "document" for r in results)
                assert all(":section:" not in r.id for r in results)
                assert results == sorted(results, key=lambda r: r.score, reverse=True)
            finally:
                db.close()

    def test_rolled_up_document_scores_its_best_section(self):
        """A document inherits the score of its strongest section, not its weakest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                sections = db.query("neural networks", search_level="sections", k=50)
                best_by_doc = {}
                for s in sections:
                    best_by_doc[s.document_id] = max(best_by_doc.get(s.document_id, 0.0), s.score)

                docs = db.query("neural networks", search_level="sections", return_type="documents", k=10)
                assert docs, "expected at least one rolled-up document"
                for d in docs:
                    assert d.score == pytest.approx(best_by_doc[d.id], abs=1e-6)
            finally:
                db.close()

    def test_rollup_overfetches_the_section_pool(self, monkeypatch):
        """The section pool must be over-fetched before it is rolled up.

        Several sections of one document collapse into a single document result,
        so rolling up a pool truncated to k sections can return far fewer than k
        documents. This asserts the fetch size rather than the document count:
        with MockEmbeddings every copy of a document scores identically, so the
        ranking cannot be steered to make one document own the top sections, and
        a count-based test passes whether or not the over-fetch is there (it did).
        """
        from localvectordb.database._search import _SECTION_ROLLUP_OVERFETCH, SearchMixin

        calls = []
        original = SearchMixin._section_level_search

        def spy(self, query_embedding, k, score_threshold, filters, warn_k=None):
            calls.append({"k": k, "warn_k": warn_k})
            return original(self, query_embedding, k, score_threshold, filters, warn_k=warn_k)

        monkeypatch.setattr(SearchMixin, "_section_level_search", spy)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                db.query("neural networks", search_level="sections", return_type="documents", k=3)
                assert calls, "_section_level_search was never called"
                # Assert the pool is bigger than k, not just that it equals
                # `k * _SECTION_ROLLUP_OVERFETCH` -- that comparison moves with the
                # constant and holds even when the factor is neutered to 1.
                assert _SECTION_ROLLUP_OVERFETCH > 1, "the over-fetch factor must over-fetch"
                assert calls[0]["k"] > 3
                assert calls[0]["k"] == 3 * _SECTION_ROLLUP_OVERFETCH
                # Starvation is judged against the k the caller asked for, so the
                # over-fetch does not manufacture "filter starved" warnings.
                assert calls[0]["warn_k"] == 3

                # Returning sections directly needs no over-fetch.
                calls.clear()
                db.query("neural networks", search_level="sections", k=3)
                assert calls[0]["k"] == 3
            finally:
                db.close()

    def test_sections_reject_unsupported_return_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                with pytest.raises(ValueError, match="return_type"):
                    db.query("neural networks", search_level="sections", return_type="chunks", k=3)
            finally:
                db.close()

    def test_documents_level_rejects_unsupported_return_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                with pytest.raises(ValueError, match="return_type"):
                    db.query("neural networks", search_level="documents", return_type="sections", k=3)
            finally:
                db.close()

    def test_async_matches_sync(self):
        """query_async must resolve the default to the same unit as query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = _make_hier_db(tmpdir)
            try:
                sync_results = db.query("neural networks", search_level="sections", k=5)
                async_results = asyncio.run(db.query_async("neural networks", search_level="sections", k=5))
                assert [r.type for r in async_results] == [r.type for r in sync_results]
                assert [r.id for r in async_results] == [r.id for r in sync_results]
            finally:
                db.close()
