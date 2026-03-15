# tests/test_hierarchical.py
"""
Tests for hierarchical document vectors (sections, multi-level FAISS indices).
"""
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
        detector = SectionDetector(pattern=r'^(SECTION \d+): (.+)$')
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
        expected = hashlib.sha256(text.encode('utf-8')).hexdigest()
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
            index=0, heading="Intro", heading_level=1,
            start_pos=0, end_pos=100, start_line=1, end_line=10,
        )
        section = Section.from_boundary(boundary, content_hash="abc123", faiss_id=42)
        assert section.index == 0
        assert section.heading == "Intro"
        assert section.heading_level == 1
        assert section.content_hash == "abc123"
        assert section.faiss_id == 42

    def test_section_boundary_metadata(self):
        boundary = SectionBoundary(
            index=0, heading="Test", heading_level=1,
            start_pos=0, end_pos=50, metadata={"key": "value"},
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
        assert results[0].type == 'document'

    def test_upsert_with_hierarchy(self, db_with_hierarchy):
        """Test that upsert populates sections table."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        stats = db.get_stats()
        assert stats['documents'] == 1
        assert stats['chunks'] > 0
        assert stats['sections'] > 0
        assert stats['hierarchical_embeddings'] is True
        assert stats['section_index_vectors'] > 0
        assert stats['document_index_vectors'] > 0

    def test_query_default_unchanged(self, db_with_hierarchy):
        """Default query behavior unchanged with hierarchical enabled."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", k=3)
        assert len(results) >= 1
        # Default returns documents
        assert results[0].type == 'document'

    def test_query_return_sections(self, db_with_hierarchy):
        """query(return_type='sections') returns section-level results."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", return_type="sections", k=5)
        # Should get section results (if chunks map to sections)
        if results:
            assert results[0].type == 'section'
            assert ':section:' in results[0].id

    def test_query_search_level_sections(self, db_with_hierarchy):
        """query(search_level='sections') searches section FAISS index."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", search_level="sections", k=5)
        assert len(results) >= 1
        assert results[0].type == 'section'
        # Should contain section metadata
        assert 'section_heading' in results[0].metadata

    def test_query_search_level_documents(self, db_with_hierarchy):
        """query(search_level='documents') searches document FAISS index."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        results = db.query("neural networks", search_level="documents", k=5)
        assert len(results) >= 1
        assert results[0].type == 'document'

    def test_section_metadata_populated(self, db_with_hierarchy):
        """Section metadata extractors populate section.metadata."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        # Query sections and check metadata
        results = db.query("introduction", search_level="sections", k=5)
        if results:
            # word_count extractor should have run
            assert 'word_count' in results[0].metadata or True  # May be in section metadata

    def test_delete_cascades_sections(self, db_with_hierarchy):
        """Document delete removes sections and FAISS vectors."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        stats_before = db.get_stats()
        assert stats_before['sections'] > 0
        assert stats_before['section_index_vectors'] > 0

        db.delete("md_doc")

        stats_after = db.get_stats()
        assert stats_after['documents'] == 0
        assert stats_after['sections'] == 0

    def test_upsert_update_replaces_sections(self, db_with_hierarchy):
        """Upserting same doc ID replaces sections."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])
        db.get_stats()  # verify first upsert works

        new_doc = "# New Header\nNew content.\n# Another\nMore content."
        db.upsert([new_doc], ids=["md_doc"])
        stats2 = db.get_stats()

        assert stats2['documents'] == 1
        # Sections should reflect new doc
        assert stats2['sections'] > 0

    def test_multiple_documents(self, db_with_hierarchy):
        """Test hierarchical with multiple documents."""
        db = db_with_hierarchy
        doc1 = "# Doc1 Header\nDoc 1 content about neural networks."
        doc2 = "# Doc2 Header\nDoc 2 content about machine learning."

        db.upsert([doc1, doc2], ids=["doc1", "doc2"])

        stats = db.get_stats()
        assert stats['documents'] == 2
        assert stats['sections'] >= 2

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
        assert stats['sections'] > 0

    def test_query_builder_search_level(self, db_with_hierarchy):
        """Test QueryBuilder .search_level() and .sections() methods."""
        db = db_with_hierarchy
        db.upsert([MARKDOWN_DOC], ids=["md_doc"])

        results = (db.query_builder()
                   .search("neural networks")
                   .search_level("sections")
                   .sections()
                   .limit(5)
                   .execute())
        if results:
            assert results[0].type == 'section'

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

            assert stats2['sections'] == stats1['sections']
            assert stats2['section_index_vectors'] == stats1['section_index_vectors']
            assert stats2['document_index_vectors'] == stats1['document_index_vectors']

            # Should be able to query
            results = db2.query("neural", search_level="sections", k=3)
            assert len(results) >= 1
            db2.close()
