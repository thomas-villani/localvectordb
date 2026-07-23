"""
Tests for localvectordb.chunking module.
"""

import logging

import pytest

from localvectordb.chunking import (
    CharChunker,
    ChunkerFactory,
    CodeBlockChunker,
    DelimiterChunker,
    LineChunker,
    ParagraphChunker,
    PositionTrackingChunker,
    SectionChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
    reconstruct_document,
)
from localvectordb.core import Chunk, ChunkPosition


@pytest.mark.unit
@pytest.mark.chunking
class TestPositionTrackingChunker:
    """Test base PositionTrackingChunker class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            PositionTrackingChunker()

    def test_count_tokens(self):
        """Test token counting."""

        class TestChunker(PositionTrackingChunker):
            def chunk(self, text):
                return []

        chunker = TestChunker()
        # Mock returns list with length equal to word count
        assert chunker.count_tokens("hello world test") == 3

    def test_calculate_line_column(self):
        """Test line and column calculation."""

        class TestChunker(PositionTrackingChunker):
            def chunk(self, text):
                return []

        chunker = TestChunker()

        text = "Line 1\nLine 2\nLine 3"

        # Start of text
        line, col = chunker._calculate_line_column(text, 0)
        assert line == 1
        assert col == 1

        # Start of second line
        line, col = chunker._calculate_line_column(text, 7)
        assert line == 2
        assert col == 1

        # Middle of second line
        line, col = chunker._calculate_line_column(text, 10)
        assert line == 2
        assert col == 4

        # Beyond text length
        line, col = chunker._calculate_line_column(text, 100)
        assert line == 3
        assert col == 7

    def test_create_chunk(self):
        """Test chunk creation with position tracking."""

        class TestChunker(PositionTrackingChunker):
            def chunk(self, text):
                return []

        chunker = TestChunker()
        text = "Hello world\nThis is a test"

        chunk = chunker._create_chunk(text, 12, 26, 1)

        assert chunk.content == "This is a test"
        assert chunk.position.start == 12
        assert chunk.position.end == 26
        assert chunk.position.line == 2
        assert chunk.position.column == 1
        assert chunk.index == 1
        assert chunk.tokens == 4  # Mock returns word count

    def test_ensure_chunks_within_limit_no_oversized(self):
        """Test that _ensure_chunks_within_limit passes through valid chunks."""

        class TestChunker(PositionTrackingChunker):
            def chunk(self, text):
                return []

        chunker = TestChunker(max_tokens=100)
        text = "Hello world. This is a test."

        # Create chunks that are within the limit
        chunk1 = chunker._create_chunk(text, 0, 12, 0)  # "Hello world."
        chunk2 = chunker._create_chunk(text, 13, 28, 1)  # "This is a test."

        chunks = [chunk1, chunk2]
        result = chunker._ensure_chunks_within_limit(chunks, text)

        # All chunks should pass through unchanged
        assert len(result) == 2
        assert result[0].content == "Hello world."
        assert result[1].content == "This is a test."

    def test_ensure_chunks_within_limit_splits_oversized(self):
        """Test that _ensure_chunks_within_limit splits oversized chunks."""

        class TestChunker(PositionTrackingChunker):
            def chunk(self, text):
                return []

        # Use a very small max_tokens to force splitting
        chunker = TestChunker(max_tokens=3)

        # Create a text that will produce a chunk exceeding 3 tokens
        text = "This is a long text that exceeds the limit."

        # Create a chunk that exceeds the limit
        chunk = chunker._create_chunk(text, 0, len(text), 0)

        # The chunk should have more than 3 tokens
        assert chunk.tokens > 3

        # Ensure chunks within limit should split it
        result = chunker._ensure_chunks_within_limit([chunk], text)

        # Should have multiple chunks now
        assert len(result) > 1

        # All resulting chunks should be within the limit
        for c in result:
            assert c.tokens <= chunker.max_tokens or len(c.content) == 1  # Single char edge case

        # Indices should be renumbered sequentially
        for i, c in enumerate(result):
            assert c.index == i


@pytest.mark.unit
@pytest.mark.chunking
class TestSentenceChunker:
    """Test SentenceChunker class."""

    def test_create_chunker(self):
        """Test creating sentence chunker."""
        chunker = SentenceChunker(max_tokens=100, overlap=2)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 2

    def test_chunk_simple_text(self):
        """Test chunking simple text with sentences."""
        chunker = SentenceChunker(max_tokens=10, overlap=0)
        text = "First sentence. Second sentence. Third sentence."

        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Each chunk should be within token limit
        for chunk in chunks:
            assert chunk.tokens <= 10

        # Check positions are correct
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert chunk.position.start < chunk.position.end

    def test_chunk_with_overlap(self):
        """Test chunking with sentence overlap."""
        chunker = SentenceChunker(max_tokens=5, overlap=1)
        text = "First. Second. Third. Fourth. Fifth."

        chunks = chunker.chunk(text)
        # Should have multiple chunks with overlap
        assert len(chunks) == 4

        # Verify overlap by checking if some content appears in multiple chunks
        contents = [chunk.content for chunk in chunks]
        # With overlap, some sentences should appear in multiple chunks
        all_content = " ".join(contents)
        assert len(all_content) > len(text)  # Due to overlap

    def test_chunk_empty_text(self):
        """Test chunking empty and whitespace-only text (L1).

        Truly empty text yields no chunks (its reconstruction is ``""``), but
        whitespace-only text must still emit a single chunk so the document
        reconstructs byte-for-byte instead of collapsing to ``""``.
        """
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

        ws_chunks = chunker.chunk("   ")
        assert len(ws_chunks) == 1
        # Byte-for-byte reconstruction: the chunk spans the whole input.
        assert "".join(c.content for c in ws_chunks) == "   "

    def test_chunk_single_long_sentence(self):
        """Test chunking when single sentence exceeds token limit."""
        chunker = SentenceChunker(max_tokens=3, overlap=0)
        text = "This is a very long sentence that exceeds the token limit."

        chunks = chunker.chunk(text)

        # Should split the long sentence into multiple chunks
        assert len(chunks) > 1

        # Each chunk should be within token limit or be a forced split
        total_content = "".join(chunk.content for chunk in chunks)
        assert len(total_content) >= len(text.strip())

    def test_split_into_sentences(self):
        """Test sentence splitting method."""
        chunker = SentenceChunker()
        text = "First sentence. Second sentence! Third sentence? Fourth."

        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 4
        # Check that positions are correct
        for start, end, content in sentences:
            assert text[start:end].strip() == content.strip()

    def test_sentence_pattern_matching(self):
        """Test sentence pattern recognition."""
        chunker = SentenceChunker()

        # Test various sentence endings
        test_cases = [
            "Simple. Two sentences.",
            "Question? Another sentence.",
            "Exclamation! Next sentence.",
            'Quoted "sentence." Next one.',
            "Abbreviation like Dr. Smith went home.",
            "Multiple\nlines\nwith periods.",
        ]

        for text in test_cases:
            sentences = chunker._split_into_sentences(text)
            assert len(sentences) > 0

            # Verify all text is covered
            total_length = sum(end - start for start, end, _ in sentences)
            assert total_length <= len(text)


@pytest.mark.unit
@pytest.mark.chunking
class TestTokenChunker:
    """Test TokenChunker class."""

    def test_create_chunker(self):
        """Test creating token chunker."""
        chunker = TokenChunker(max_tokens=100, overlap=10)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 10

    def test_chunk_simple_text(self):
        """Test chunking simple text by tokens."""
        chunker = TokenChunker(max_tokens=10, overlap=1)
        text = "This is a test document with many words"

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        # Each chunk should be within token limit
        for chunk in chunks:
            assert chunk.tokens <= 10

        # Check positions
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert 0 <= chunk.position.start < chunk.position.end <= len(text)

    def test_chunk_short_text(self):
        """Test chunking text shorter than max tokens."""
        chunker = TokenChunker(max_tokens=100)
        text = "Short text"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].position.start == 0
        assert chunks[0].position.end == len(text)

    def test_chunk_with_overlap(self):
        """Test chunking with token overlap."""
        chunker = TokenChunker(max_tokens=3, overlap=1)
        text = "One two three four five six seven"

        chunks = chunker.chunk(text)
        # Should have multiple chunks
        assert len(chunks) == 3
        # assert chunks[0].split()[-1] == chunks[1].split()[0]

    def test_estimate_position(self):
        """Test position estimation from token index."""
        chunker = TokenChunker()
        text = "Hello world this is a test"
        tokens = list(range(len(text.split())))  # Mock tokens

        # Test position estimation
        pos = chunker._estimate_position(text, tokens, 0)
        assert pos == 0

        pos = chunker._estimate_position(text, tokens, 2)
        assert pos >= 0
        assert pos <= len(text)


@pytest.mark.unit
@pytest.mark.chunking
class TestWordChunker:
    """Test WordChunker class."""

    def test_create_chunker(self):
        """Test creating word chunker."""
        chunker = WordChunker(max_tokens=50, overlap=5)
        assert chunker.max_tokens == 50
        assert chunker.overlap == 5

    def test_chunk_by_words(self):
        """Test chunking by word boundaries."""
        chunker = WordChunker(max_tokens=5, overlap=1)
        text = "One two three four five six seven eight"

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        # Each chunk should respect word boundaries
        for chunk in chunks:
            # Content should start and end on word boundaries
            assert not chunk.content.startswith(" ")
            assert not chunk.content.endswith(" ") or chunk.content.strip() != ""

        # Check token limits
        for chunk in chunks:
            assert chunk.tokens <= 5

    def test_chunk_with_punctuation(self):
        """Test chunking text with punctuation."""
        chunker = WordChunker(max_tokens=10)
        text = "Hello, world! This is a test. How are you?"

        chunks = chunker.chunk(text)

        # Should handle punctuation correctly
        for chunk in chunks:
            assert chunk.position.start >= 0
            assert chunk.position.end <= len(text)
            assert len(chunk.content.strip()) > 0

    def test_word_pattern_matching(self):
        """Test word pattern recognition."""
        chunker = WordChunker()
        text = "Word1 word2, word3! word4? word5."

        chunks = chunker.chunk(text)

        # Should identify words correctly including punctuation
        all_content = "".join(chunk.content for chunk in chunks)
        # Should capture most of the original text
        assert len(all_content.strip()) > 0


@pytest.mark.unit
@pytest.mark.chunking
class TestLineChunker:
    """Test LineChunker class."""

    def test_create_chunker(self):
        """Test creating line chunker."""
        chunker = LineChunker(max_tokens=100, overlap=3)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 3

    def test_chunk_multiline_text(self):
        """Test chunking multiline text."""
        chunker = LineChunker(max_tokens=10, overlap=1)
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        # Each chunk should contain complete lines
        for chunk in chunks:
            lines = chunk.content.split("\n")
            # Each line should be complete (not cut off in middle)
            assert all(len(line.strip()) == 0 or not line.endswith(" ") for line in lines)

    def test_chunk_single_line(self):
        """Test chunking single line text."""
        chunker = LineChunker(max_tokens=100)
        text = "This is a single line of text"

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_long_line(self):
        """Test chunking when single line exceeds token limit."""
        chunker = LineChunker(max_tokens=3, overlap=0)
        text = "This is a very long line that exceeds the token limit\nShort line"

        chunks = chunker.chunk(text)

        # Should split the long line
        assert len(chunks) > 1

        # Verify total content is preserved
        total_chars = sum(len(chunk.content) for chunk in chunks)
        assert total_chars >= len(text)

    def test_line_preservation(self):
        """Test that line endings are preserved."""
        chunker = LineChunker(max_tokens=50)
        text = "Line 1\nLine 2\r\nLine 3\n"

        chunks = chunker.chunk(text)

        # Should preserve line endings
        reconstructed = "".join(chunk.content for chunk in chunks)
        assert "\n" in reconstructed


@pytest.mark.unit
@pytest.mark.chunking
class TestCharChunker:
    """Test CharChunker class."""

    def test_create_chunker(self):
        """Test creating character chunker."""
        chunker = CharChunker(max_tokens=100, overlap=20)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 20

    def test_chunk_by_characters(self):
        """Test chunking by character boundaries."""
        chunker = CharChunker(max_tokens=10, overlap=2)
        text = "This is a test document"

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        # Each chunk should be within token limit
        for chunk in chunks:
            assert chunk.tokens <= 10

        # Check character-level precision
        for chunk in chunks:
            assert chunk.position.end - chunk.position.start == len(chunk.content)

    def test_chunk_with_overlap(self):
        """Test character chunking with overlap."""
        chunker = CharChunker(max_tokens=2, overlap=2)
        text = "Hello world test"

        chunks = chunker.chunk(text)

        # Should have overlap
        assert len(chunks) > 1

        # Verify overlap exists
        total_chars = sum(len(chunk.content) for chunk in chunks)
        assert total_chars > len(text)
        assert chunks[0].content[-2:] == chunks[1].content[:2]

    def test_chunk_exact_boundaries(self):
        """Test that chunk boundaries are exact."""
        chunker = CharChunker(max_tokens=5, overlap=0)
        text = "0123456789"

        chunks = chunker.chunk(text)

        # Verify exact character boundaries
        for chunk in chunks:
            extracted = text[chunk.position.start : chunk.position.end]
            assert extracted == chunk.content


@pytest.mark.unit
@pytest.mark.chunking
class TestParagraphChunker:
    """Test ParagraphChunker class."""

    def test_create_chunker(self):
        """Test creating paragraph chunker."""
        chunker = ParagraphChunker(max_tokens=100, overlap=1)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 1

    def test_chunk_paragraphs(self):
        """Test chunking by paragraph boundaries."""
        chunker = ParagraphChunker(max_tokens=3, overlap=0)
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        # Each chunk should contain complete paragraphs
        for chunk in chunks:
            # Should not break in middle of paragraphs
            assert "\n\n" not in chunk.content or chunk.content.count("\n\n") <= 1

    def test_split_into_paragraphs(self):
        """Test paragraph splitting method."""
        chunker = ParagraphChunker(max_tokens=3)
        text = "Para 1.\n\nPara 2.\n\n\nPara 3."

        paragraphs = chunker._split_into_paragraphs(text)

        assert len(paragraphs) == 3
        # Check positions
        for start, end, content in paragraphs:
            assert start < end
            assert len(content.strip()) > 0

    def test_long_paragraph_splitting(self):
        """Test splitting long paragraphs."""
        chunker = ParagraphChunker(max_tokens=5)
        long_paragraph = "This is a very long paragraph that exceeds the token limit and should be split."
        text = f"{long_paragraph}\n\nShort para."

        chunks = chunker.chunk(text)

        # Should split the long paragraph
        assert len(chunks) > 1


@pytest.mark.unit
@pytest.mark.chunking
class TestDelimiterChunker:
    """Test DelimiterChunker class."""

    def test_create_chunker(self):
        """Test creating a delimiter chunker with a custom delimiter."""
        chunker = DelimiterChunker(max_tokens=100, overlap=0, delimiter="\n---\n")
        assert chunker.max_tokens == 100
        assert chunker.delimiter == "\n---\n"

    def test_default_delimiter_is_blank_line(self):
        """The default delimiter is a blank line."""
        assert DelimiterChunker().delimiter == "\n\n"

    def test_empty_delimiter_rejected(self):
        """An empty delimiter cannot make progress and is rejected."""
        with pytest.raises(ValueError):
            DelimiterChunker(delimiter="")

    def test_via_factory_forwards_delimiter(self):
        """The factory forwards ``delimiter`` to the delimiter chunker."""
        chunker = ChunkerFactory.create_chunker("delimiter", max_tokens=50, delimiter="||")
        assert isinstance(chunker, DelimiterChunker)
        assert chunker.delimiter == "||"

    def test_splits_on_delimiter(self):
        """Segments are cut on the delimiter and packed up to max_tokens."""
        chunker = DelimiterChunker(max_tokens=6, delimiter="\n---\n")
        text = "Section one.\n---\nSection two is here.\n---\nThree."
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_reconstruction_is_exact(self):
        """Contiguous spans mean reconstruct_document returns the original text."""
        chunker = DelimiterChunker(max_tokens=6, delimiter="\n---\n")
        text = "Section one.\n---\nSection two is here.\n---\nThree."
        chunks = chunker.chunk(text)
        assert reconstruct_document(chunks, len(text)) == text

    def test_custom_multichar_delimiter_reconstructs(self):
        """A multi-character non-newline delimiter also reconstructs exactly."""
        chunker = DelimiterChunker(max_tokens=4, delimiter="<SEP>")
        text = "alpha<SEP>beta gamma<SEP>delta"
        chunks = chunker.chunk(text)
        assert reconstruct_document(chunks, len(text)) == text

    def test_oversized_segment_falls_back_to_chars(self):
        """A single segment larger than max_tokens is split character-by-character."""
        big = "x " * 400
        text = f"short.\n\n{big}"
        chunker = DelimiterChunker(max_tokens=20, delimiter="\n\n")
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        assert all(c.tokens <= 20 for c in chunks)
        assert reconstruct_document(chunks, len(text)) == text

    def test_empty_and_whitespace_guards(self):
        """Empty input yields no chunks; whitespace-only yields one (reconstructible)."""
        chunker = DelimiterChunker(delimiter="\n\n")
        assert chunker.chunk("") == []
        ws = chunker.chunk("  \n\n  ")
        assert len(ws) == 1
        assert reconstruct_document(ws, len("  \n\n  ")) == "  \n\n  "

    def test_no_delimiter_present(self):
        """Text without the delimiter becomes a single chunk (if it fits)."""
        chunker = DelimiterChunker(max_tokens=100, delimiter="\n---\n")
        text = "No delimiter anywhere in this text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].content == text


@pytest.mark.unit
@pytest.mark.chunking
class TestSectionChunker:
    """Test SectionChunker class."""

    def test_create_chunker(self):
        """Test creating section chunker."""
        chunker = SectionChunker(max_tokens=100)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 0  # Sections don't typically overlap

    def test_chunk_markdown_sections(self):
        """Test chunking markdown-style sections."""
        chunker = SectionChunker(max_tokens=50)
        text = """# Main Title
Introduction text.

## Section 1
Content for section 1.

### Subsection 1.1
Detailed content.

## Section 2
Content for section 2."""

        chunks = chunker.chunk(text)

        assert len(chunks) > 0

        # Each chunk should contain complete sections
        for chunk in chunks:
            # Should start with header or content
            lines = chunk.content.strip().split("\n")
            first_line = lines[0] if lines else ""
            # Should be either a header or regular content
            assert len(first_line) > 0

    def test_split_into_sections(self):
        """Test section splitting method."""
        chunker = SectionChunker()
        text = """# Title 1
Content 1.

## Title 2
Content 2.

# Title 3
Content 3."""

        sections = chunker._split_into_sections(text)

        # Should identify sections by headers
        assert len(sections) >= 2  # At least main sections

        # Check that sections have correct structure
        for start, end, content, level in sections:
            assert start < end
            assert level >= 1  # Header levels start at 1
            assert len(content.strip()) > 0

    def test_header_level_detection(self):
        """Test header level detection."""
        chunker = SectionChunker()
        text = """# Level 1
## Level 2
### Level 3
#### Level 4"""

        sections = chunker._split_into_sections(text)

        # Should detect different header levels
        levels = [level for _, _, _, level in sections]
        assert 1 in levels
        assert max(levels) >= 2

    def test_ignores_headers_in_fenced_code(self):
        """``#`` lines inside fenced code blocks are not treated as headers."""
        chunker = SectionChunker()
        text = """# Real Heading
Intro.

```python
# not a heading
## also not a heading
code = 1
```

## Real Subsection
Body."""

        sections = chunker._split_into_sections(text)
        headers = [content.splitlines()[0] for _, _, content, _ in sections]
        # Only the two genuine headers are detected as section starts.
        assert any("Real Heading" in h for h in headers)
        assert any("Real Subsection" in h for h in headers)
        assert not any("not a heading" in h for h in headers)


@pytest.mark.unit
@pytest.mark.chunking
class TestCodeBlockChunker:
    """Test CodeBlockChunker class."""

    def test_create_chunker(self):
        """Test creating code block chunker."""
        chunker = CodeBlockChunker(max_tokens=100, overlap=2, language="python")
        assert chunker.max_tokens == 100
        assert chunker.overlap == 2
        assert chunker.language == "python"

    def test_detect_python_language(self):
        """Test Python language detection."""
        code = """def hello():
    print("Hello world")
    return True

if __name__ == "__main__":
    hello()"""

        chunker = CodeBlockChunker()
        detected = chunker._detect_language(code)
        assert detected == "python"

    def test_detect_javascript_language(self):
        """Test JavaScript language detection."""
        code = """function hello() {
    console.log("Hello world");
    return true;
}

const value = 42;
let result = hello();"""

        chunker = CodeBlockChunker()
        detected = chunker._detect_language(code)
        assert detected == "javascript"

    def test_chunk_python_code(self):
        """Test chunking Python code."""
        chunker = CodeBlockChunker(max_tokens=20, language="python", overlap=0)
        code = """def function1():
    print("Function 1")
    return 1

def function2():
    print("Function 2")
    return 2

class MyClass:
    def method(self):
        return "Hello"
"""

        chunks = chunker.chunk(code)

        assert len(chunks) > 0

        # Should respect code structure
        for chunk in chunks:
            # Should not break in middle of functions/classes when possible
            assert chunk.tokens <= 20

    def test_identify_python_blocks(self):
        """Test identifying Python code blocks."""
        chunker = CodeBlockChunker(language="python")
        code = """def func1():
    pass

def func2():
    return True

class TestClass:
    def method(self):
        pass"""

        lines = code.splitlines(keepends=True)
        starts, ends = chunker._get_language_patterns("python")
        blocks = chunker._identify_code_blocks(lines, "python", starts, ends)

        # Should identify function and class definitions
        assert len(blocks) >= 2  # At least functions and class

    def test_identify_brace_blocks(self):
        """Test identifying brace-based code blocks."""
        chunker = CodeBlockChunker(language="javascript")
        code = """function test() {
    console.log("test");
}

if (condition) {
    doSomething();
}"""

        lines = code.splitlines(keepends=True)

        starts, ends = chunker._get_language_patterns("javascript")
        blocks = chunker._identify_code_blocks(lines, "javascript", starts, ends)

        # Should identify function and if blocks
        assert len(blocks) >= 1


@pytest.mark.unit
@pytest.mark.chunking
class TestChunkerFactory:
    """Test ChunkerFactory class."""

    def test_available_chunkers(self):
        """Test that all expected chunkers are available."""
        expected_chunkers = {
            "sentences",
            "tokens",
            "words",
            "lines",
            "characters",
            "paragraphs",
            "sections",
            "code-blocks",
        }

        available = set(ChunkerFactory.CHUNKERS.keys())
        assert expected_chunkers.issubset(available)

    def test_create_sentence_chunker(self):
        """Test creating sentence chunker via factory."""
        chunker = ChunkerFactory.create_chunker("sentences", max_tokens=100, overlap=2)
        assert isinstance(chunker, SentenceChunker)
        assert chunker.max_tokens == 100
        assert chunker.overlap == 2

    def test_create_token_chunker(self):
        """Test creating token chunker via factory."""
        chunker = ChunkerFactory.create_chunker("tokens", max_tokens=50, overlap=10)
        assert isinstance(chunker, TokenChunker)
        assert chunker.max_tokens == 50
        assert chunker.overlap == 10

    def test_create_word_chunker(self):
        """Test creating word chunker via factory."""
        chunker = ChunkerFactory.create_chunker("words", max_tokens=200, overlap=5)
        assert isinstance(chunker, WordChunker)
        assert chunker.max_tokens == 200
        assert chunker.overlap == 5

    def test_create_line_chunker(self):
        """Test creating line chunker via factory."""
        chunker = ChunkerFactory.create_chunker("lines", max_tokens=75, overlap=3)
        assert isinstance(chunker, LineChunker)
        assert chunker.max_tokens == 75
        assert chunker.overlap == 3

    def test_create_character_chunker(self):
        """Test creating character chunker via factory."""
        chunker = ChunkerFactory.create_chunker("characters", max_tokens=150, overlap=20)
        assert isinstance(chunker, CharChunker)
        assert chunker.max_tokens == 150
        assert chunker.overlap == 20

    def test_create_paragraph_chunker(self):
        """Test creating paragraph chunker via factory."""
        chunker = ChunkerFactory.create_chunker("paragraphs", max_tokens=300, overlap=1)
        assert isinstance(chunker, ParagraphChunker)
        assert chunker.max_tokens == 300
        assert chunker.overlap == 1

    def test_create_section_chunker(self):
        """Test creating section chunker via factory."""
        chunker = ChunkerFactory.create_chunker("sections", max_tokens=500)
        assert isinstance(chunker, SectionChunker)
        assert chunker.max_tokens == 500
        assert chunker.overlap == 0  # Sections don't overlap

    def test_create_code_chunker(self):
        """Test creating code block chunker via factory."""
        chunker = ChunkerFactory.create_chunker("code-blocks", max_tokens=250, overlap=2, language="python")
        assert isinstance(chunker, CodeBlockChunker)
        assert chunker.max_tokens == 250
        assert chunker.overlap == 2

    def test_create_unknown_chunker(self):
        """Test creating unknown chunker raises error."""
        with pytest.raises(ValueError, match="Unknown chunking method: unknown"):
            ChunkerFactory.create_chunker("unknown")

    def test_list_methods(self):
        """Test listing available chunking methods."""
        methods = ChunkerFactory.list_methods()
        assert "sentences" in methods
        assert "tokens" in methods
        assert "words" in methods
        assert isinstance(methods, list)


@pytest.mark.unit
@pytest.mark.chunking
class TestChunkerOverlapValidation:
    """Validation and unit-confusion warnings for chunk_overlap in the factory."""

    def test_negative_overlap_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            ChunkerFactory.create_chunker("sentences", max_tokens=100, overlap=-1)

    def test_bool_overlap_rejected(self):
        # bool is an int subclass but is not a valid overlap.
        with pytest.raises(TypeError, match="non-negative"):
            ChunkerFactory.create_chunker("sentences", max_tokens=100, overlap=True)

    def test_tokens_overlap_at_or_above_size_rejected(self):
        with pytest.raises(ValueError, match="less than chunk_size"):
            ChunkerFactory.create_chunker("tokens", max_tokens=100, overlap=100)

    def test_tokens_overlap_below_size_ok(self):
        chunker = ChunkerFactory.create_chunker("tokens", max_tokens=100, overlap=99)
        assert chunker.overlap == 99

    def test_large_sentence_overlap_warns(self, caplog):
        # 50 sentences of overlap in a ~500-token chunk is the classic foot-gun.
        with caplog.at_level(logging.WARNING, logger="localvectordb.chunking"):
            ChunkerFactory.create_chunker("sentences", max_tokens=500, overlap=50)
        assert "chunk_overlap=50" in caplog.text
        assert "sentences" in caplog.text

    def test_small_sentence_overlap_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="localvectordb.chunking"):
            ChunkerFactory.create_chunker("sentences", max_tokens=500, overlap=2)
        assert "is large for chunking_method" not in caplog.text

    def test_sections_ignore_overlap_without_error(self):
        # Sections force overlap to 0 and are not subject to the count heuristic.
        chunker = ChunkerFactory.create_chunker("sections", max_tokens=500, overlap=99)
        assert chunker.overlap == 0


@pytest.mark.unit
@pytest.mark.chunking
class TestUtilityFunctions:
    """Test utility functions for chunk manipulation."""

    def test_reconstruct_document_empty(self):
        """Test reconstructing from empty chunk list."""
        result = reconstruct_document([], 0)
        assert result == ""

    def test_reconstruct_document_single_chunk(self):
        """Test reconstructing from single chunk."""
        chunk = Chunk(
            content="Hello world",
            position=ChunkPosition(start=0, end=11, line=1, column=1, end_line=1, end_column=11),
            tokens=2,
            index=0,
        )

        result = reconstruct_document([chunk], 11)
        assert result == "Hello world"

    def test_reconstruct_document_multiple_chunks(self):
        """Test reconstructing from multiple chunks."""
        chunks = [
            Chunk(
                content="Hello ",
                position=ChunkPosition(start=0, end=6, line=1, column=1, end_line=1, end_column=6),
                tokens=1,
                index=0,
            ),
            Chunk(
                content="world",
                position=ChunkPosition(start=6, end=11, line=1, column=7, end_line=1, end_column=11),
                tokens=1,
                index=1,
            ),
        ]

        result = reconstruct_document(chunks, 11)
        assert result == "Hello world"

    def test_reconstruct_document_overlapping_chunks(self):
        """Test reconstructing with overlapping chunks."""
        chunks = [
            Chunk(
                content="Hello world",
                position=ChunkPosition(start=0, end=11, line=1, column=1, end_line=1, end_column=11),
                tokens=2,
                index=0,
            ),
            Chunk(
                content="world test",
                position=ChunkPosition(start=6, end=16, line=1, column=7, end_line=1, end_column=16),
                tokens=2,
                index=1,
            ),
        ]

        result = reconstruct_document(chunks, 16)
        # Should handle overlapping chunks (later chunks overwrite)
        assert "Hello" in result
        assert "world" in result
        assert "test" in result

    def test_reconstruct_document_out_of_order(self):
        """Test reconstructing from out-of-order chunks."""
        chunks = [
            Chunk(
                content="world",
                position=ChunkPosition(start=6, end=11, line=1, column=7, end_line=1, end_column=11),
                tokens=1,
                index=1,
            ),
            Chunk(
                content="Hello ",
                position=ChunkPosition(start=0, end=6, line=1, column=1, end_line=1, end_column=6),
                tokens=1,
                index=0,
            ),
        ]

        result = reconstruct_document(chunks, 11)
        assert result == "Hello world"


# Text variants that exercise the reconstruction guarantee. The required matrix
# (trailing newline, trailing spaces, no trailing newline, multi-chunk, unicode)
# plus the leading-whitespace and inter-unit-separator cases that a measurement
# pass showed were silently lossy before T1.7.
_ROUNDTRIP_TEXTS = {
    "trailing_newline": "First sentence here. Second one follows.\n",
    "trailing_spaces": "First sentence here. Second one follows.   ",
    "no_trailing_newline": "Alpha beta gamma. Delta epsilon zeta. Eta theta iota",
    "leading_whitespace": "   Leading whitespace here. And a second sentence follows.",
    "unicode": "Café résumé naïve. 日本語 のテスト。 Emoji 🎉 done here now.",
    "multi_paragraph": "Para one line one.\nline two.\n\nPara two here.\n\n\nPara three last.",
    "blank_lines": "One.\n\n\nTwo.\n\n\n\nThree here.",
    "tabs": "Col1\tCol2\tCol3.\tAnother\tsentence here now.",
    # Long enough to force multiple chunks at a small max_tokens (see below), so
    # the inter-chunk separators are exercised, not just the single-chunk path.
    "multi_chunk": " ".join(f"This is sentence number {i} in a longer document." for i in range(40)),
    "multi_chunk_trailing": " ".join(f"Paragraph {i} has content." for i in range(30)) + "\n\n\n",
}


# Every chunker except ``code-blocks`` guarantees byte-exact reconstruction of
# arbitrary text. ``code-blocks`` is specialised for splitting source code: its
# multi-chunk path is line-oriented (splitlines + reconstructed positions) and
# remaps sub-chunks by content search, so it only guarantees reconstruction when
# the whole input fits a single chunk (its fast path). See TestCodeBlockReconstruction.
_RECONSTRUCTING_METHODS = [m for m in ChunkerFactory.CHUNKERS if m != "code-blocks"]


@pytest.mark.unit
@pytest.mark.chunking
class TestReconstructionFidelity:
    """T1.7: ``reconstruct_document`` must rebuild the source verbatim for every
    general-purpose chunker, across trailing/leading whitespace, unicode, and
    multi-chunk docs.

    Before T1.7 the ``sentences`` (the default) and ``paragraphs`` chunkers ended
    each span before its separator (so inter-unit separators belonged to no
    chunk), and ``words`` dropped leading whitespace -- all silently lossy on
    reconstruction even though the module advertised "perfect" reconstruction.
    """

    @pytest.mark.parametrize("method", _RECONSTRUCTING_METHODS)
    @pytest.mark.parametrize("variant", list(_ROUNDTRIP_TEXTS.keys()))
    def test_roundtrip_default_size(self, method, variant):
        """Round-trip at the library default chunk size (mostly single-chunk)."""
        text = _ROUNDTRIP_TEXTS[variant]
        chunker = ChunkerFactory.create_chunker(method, max_tokens=500, overlap=0)
        chunks = chunker.chunk(text)
        assert reconstruct_document(chunks, len(text)) == text

    @pytest.mark.parametrize("method", _RECONSTRUCTING_METHODS)
    @pytest.mark.parametrize("variant", list(_ROUNDTRIP_TEXTS.keys()))
    def test_roundtrip_small_size_forces_multichunk(self, method, variant):
        """A small ``max_tokens`` forces multiple chunks, exercising every
        inter-chunk separator on the split paths."""
        text = _ROUNDTRIP_TEXTS[variant]
        chunker = ChunkerFactory.create_chunker(method, max_tokens=8, overlap=0)
        chunks = chunker.chunk(text)
        assert reconstruct_document(chunks, len(text)) == text

    @pytest.mark.parametrize("method", _RECONSTRUCTING_METHODS)
    def test_roundtrip_with_overlap(self, method):
        """Overlap re-covers positions but must never leave a gap."""
        text = _ROUNDTRIP_TEXTS["multi_chunk"]
        chunker = ChunkerFactory.create_chunker(method, max_tokens=8, overlap=1)
        chunks = chunker.chunk(text)
        assert reconstruct_document(chunks, len(text)) == text


@pytest.mark.unit
@pytest.mark.chunking
class TestCodeBlockReconstruction:
    """``code-blocks`` reconstructs exactly only on its single-chunk fast path
    (the whole input fits ``max_tokens``). Its multi-chunk split path is
    code-specialised and does not guarantee byte-exact reconstruction of
    arbitrary text; this test pins that scoped guarantee so a future change that
    silently widens or narrows it is caught.
    """

    @pytest.mark.parametrize("variant", list(_ROUNDTRIP_TEXTS.keys()))
    def test_single_chunk_fast_path_is_exact(self, variant):
        text = _ROUNDTRIP_TEXTS[variant]
        chunker = ChunkerFactory.create_chunker("code-blocks", max_tokens=100000, overlap=0)
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert reconstruct_document(chunks, len(text)) == text
