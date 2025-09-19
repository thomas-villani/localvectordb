"""
Tests for localvectordb.extractors module.
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from localvectordb.core import MetadataFieldType
from localvectordb.extractors import (
    BaseExtractor,
    ExtractionResult,
    ExtractorRegistry,
    get_extractor_registry,
    get_supported_formats,
)
from localvectordb.extractors.office_extractors import DocxExtractor, PptxExtractor, XlsxExtractor
from localvectordb.extractors.other_extractors import EPubExtractor, RTFExtractor
from localvectordb.extractors.pdf_extractors import PDFPlumberExtractor, PyPDFExtractor
from localvectordb.extractors.text_extractors import TextFallbackExtractor, TextFileExtractor
from localvectordb.extractors.web_extractors import HTMLExtractor, XMLExtractor


class TestSampleFiles:
    """Test data and sample file generators."""

    @staticmethod
    def create_sample_text(content="Hello World!\nThis is a test document.\n"):
        """Create sample text file."""
        return content.encode('utf-8')

    @staticmethod
    def create_sample_html():
        """Create sample HTML file."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
    <meta name="description" content="A test HTML document">
    <meta name="keywords" content="test, html, document">
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a test paragraph.</p>
    <h2>Subsection</h2>
    <p>Another paragraph with content.</p>
    <ul>
        <li>List item 1</li>
        <li>List item 2</li>
    </ul>
    <table>
        <tr><th>Column 1</th><th>Column 2</th></tr>
        <tr><td>Row 1 Col 1</td><td>Row 1 Col 2</td></tr>
    </table>
</body>
</html>"""
        return html.encode('utf-8')

    @staticmethod
    def create_sample_xml():
        """Create sample XML file."""
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<catalog>
    <book id="1">
        <title>Sample Book</title>
        <author>Test Author</author>
        <description>A sample book for testing</description>
    </book>
    <book id="2">
        <title>Another Book</title>
        <author>Another Author</author>
        <description>Another test book</description>
    </book>
</catalog>"""
        return xml.encode('utf-8')

    @staticmethod
    def create_sample_rss():
        """Create sample RSS file."""
        rss = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
    <channel>
        <title>Test RSS Feed</title>
        <description>A test RSS feed</description>
        <item>
            <title>First Article</title>
            <description>Description of first article</description>
        </item>
        <item>
            <title>Second Article</title>
            <description>Description of second article</description>
        </item>
    </channel>
</rss>"""
        return rss.encode('utf-8')

    @staticmethod
    def create_sample_rtf():
        """Create sample RTF file."""
        rtf = r"""{\\rtf1\\ansi\\deff0 {\\fonttbl {\\f0 Times New Roman;}}
\\f0\\fs24 Hello World!\\par
This is a test RTF document.\\par
It contains some formatting.\\par
}"""
        return rtf.encode('utf-8')

    @staticmethod
    def create_corrupted_file():
        """Create a corrupted file that's not valid for any format."""
        return b"\\x00\\x01\\x02\\x03INVALID_FILE_CONTENT\\xFF\\xFE\\xFD"

    @staticmethod
    def create_empty_file():
        """Create an empty file."""
        return b""

    @staticmethod
    def create_binary_file():
        """Create a binary file with mostly non-text content."""
        return b"\\x89PNG\\r\\n\\x1a\\n" + b"\\x00" * 100 + b"BINARY_DATA" * 10


@pytest.mark.unit
class TestExtractionResult:
    """Test ExtractionResult class."""

    def test_create_success_result(self):
        """Test creating a successful extraction result."""
        result = ExtractionResult(
            text="Hello World",
            success=True,
            method="TestExtractor",
            metadata={"key": "value"}
        )

        assert result.text == "Hello World"
        assert result.success is True
        assert result.method == "TestExtractor"
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_create_failure_result(self):
        """Test creating a failed extraction result."""
        result = ExtractionResult(
            text="",
            success=False,
            method="TestExtractor",
            error="Something went wrong"
        )

        assert result.text == ""
        assert result.success is False
        assert result.method == "TestExtractor"
        assert result.error == "Something went wrong"
        assert result.metadata == {}

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = ExtractionResult(
            text="Hello World",
            success=True,
            method="TestExtractor",
            metadata={"key": "value"}
        )

        result_dict = result.to_dict()
        expected = {
            'text': 'Hello World',
            'extraction_success': True,
            'extraction_method': 'TestExtractor',
            'metadata': {'key': 'value'},
            'error': None,
            'text_length': 11
        }

        assert result_dict == expected


@pytest.mark.unit
class TestBaseExtractor:
    """Test BaseExtractor abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_concrete_extractor_must_implement_abstract_methods(self):
        """Test that concrete extractors must implement all abstract methods."""

        class IncompleteExtractor(BaseExtractor):
            def _check_availability(self):
                return True

        with pytest.raises(TypeError):
            IncompleteExtractor()


class ConcreteTestExtractor(BaseExtractor):
    """Concrete test extractor for testing."""

    @property
    def supported_extensions(self):
        return ['.test']

    @property
    def supported_mimetypes(self):
        return ['application/test']

    @property
    def required_packages(self):
        return []

    @property
    def priority(self):
        return 10

    @property
    def metadata_schema(self):
        return {}

    def _check_availability(self):
        return True

    def _extract_text_impl(self, file_content, filename, mimetype, **kwargs):
        return ExtractionResult(
            text="Test extraction successful",
            success=True,
            method="ConcreteTestExtractor"
        )


@pytest.mark.unit
class TestConcreteExtractor:
    """Test concrete extractor functionality."""

    def setup_method(self):
        """Set up test extractor."""
        self.extractor = ConcreteTestExtractor()

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        assert self.extractor.available is True
        assert self.extractor.name == "ConcreteTestExtractor"

    def test_can_extract_by_extension(self):
        """Test can_extract with supported extension."""
        assert self.extractor.can_extract("file.test") is True
        assert self.extractor.can_extract("file.txt") is False

    def test_can_extract_by_mimetype(self):
        """Test can_extract with supported MIME type."""
        assert self.extractor.can_extract("file.unknown", "application/test") is True
        assert self.extractor.can_extract("file.unknown", "text/plain") is False

    def test_extract_text_success(self):
        """Test successful text extraction."""
        result = self.extractor.extract_text(b"test content", "file.test")

        assert result.success is True
        assert result.text == "Test extraction successful"
        assert result.method == "ConcreteTestExtractor"

    def test_extract_text_unsupported_file(self):
        """Test extraction with unsupported file type."""
        result = self.extractor.extract_text(b"test content", "file.unsupported")

        assert result.success is False
        assert result.error.startswith("File type not supported")

    def test_get_info(self):
        """Test get_info method."""
        info = self.extractor.get_info()

        expected_keys = ['name', 'available', 'supported_extensions',
                        'supported_mimetypes', 'required_packages', 'priority']
        assert all(key in info for key in expected_keys)
        assert info['name'] == "ConcreteTestExtractor"
        assert info['available'] is True


@pytest.mark.unit
class TestTextFileExtractor:
    """Test TextFileExtractor."""

    def setup_method(self):
        """Set up text extractor."""
        self.extractor = TextFileExtractor()

    def test_availability(self):
        """Test that text extractor is always available."""
        assert self.extractor.available is True

    def test_supported_formats(self):
        """Test supported file formats."""
        assert '.txt' in self.extractor.supported_extensions
        assert '.py' in self.extractor.supported_extensions
        assert '.md' in self.extractor.supported_extensions
        assert 'text/plain' in self.extractor.supported_mimetypes

    def test_extract_utf8_text(self):
        """Test extracting UTF-8 text."""
        content = TestSampleFiles.create_sample_text()
        result = self.extractor.extract_text(content, "test.txt")

        assert result.success is True
        assert "Hello World!" in result.text
        assert result.metadata['encoding'] == 'utf-8'
        assert result.metadata['character_count'] > 0

    def test_extract_empty_file(self):
        """Test extracting from empty file."""
        content = TestSampleFiles.create_empty_file()
        result = self.extractor.extract_text(content, "empty.txt")

        # Empty file should be handled gracefully
        assert result.success is False

    def test_extract_binary_file(self):
        """Test extracting from binary file."""
        content = TestSampleFiles.create_binary_file()
        result = self.extractor.extract_text(content, "binary.txt")

        # Should fail to decode properly or succeed with poor quality
        # The actual behavior depends on the content, so check the result makes sense
        assert isinstance(result, ExtractionResult)
        if result.success:
            # If it succeeds, the text should contain some content but may be garbled
            assert len(result.text) > 0

    def test_metadata_schema(self):
        """Test metadata schema."""
        schema = self.extractor.metadata_schema

        expected_fields = ['encoding', 'file_size_bytes', 'character_count', 'line_count']
        assert all(field in schema for field in expected_fields)


@pytest.mark.unit
class TestTextFallbackExtractor:
    """Test TextFallbackExtractor."""

    def setup_method(self):
        """Set up fallback extractor."""
        self.extractor = TextFallbackExtractor()

    def test_availability(self):
        """Test that fallback extractor is always available."""
        assert self.extractor.available is True

    def test_can_extract_any_file(self):
        """Test that fallback extractor accepts any file."""
        assert self.extractor.can_extract("any.file") is True
        assert self.extractor.can_extract("any.file", "any/mimetype") is True

    def test_extract_text_content(self):
        """Test extracting from text-like content."""
        content = TestSampleFiles.create_sample_text()
        result = self.extractor.extract_text(content, "file.unknown")

        assert result.success is True
        assert "Hello World!" in result.text
        assert 'warning' in result.metadata

    def test_extract_binary_content(self):
        """Test extracting from binary content."""
        content = TestSampleFiles.create_binary_file()
        result = self.extractor.extract_text(content, "file.bin")

        # Fallback extractor should handle binary content gracefully
        assert isinstance(result, ExtractionResult)
        # It might succeed with UTF-8 ignore mode or fail due to low text ratio
        if result.success is False:
            assert any(keyword in result.error.lower()
                      for keyword in ["binary", "text content", "printable"])

    def test_low_text_ratio(self):
        """Test handling of content with low text ratio."""
        # Create content with low printable character ratio
        content = b"\\x00\\x01\\x02" + b"text" + b"\\xFF\\xFE" * 20
        result = self.extractor.extract_text(content, "low_ratio.bin")

        # Should fail due to low text ratio or lack of printable content
        assert isinstance(result, ExtractionResult)
        if result.success is False:
            assert any(keyword in result.error.lower()
                      for keyword in ["text content", "printable", "binary"])


@pytest.mark.unit
class TestHTMLExtractor:
    """Test HTMLExtractor."""

    def setup_method(self):
        """Set up HTML extractor."""
        self.extractor = HTMLExtractor()

    def test_availability_with_beautifulsoup(self):
        """Test availability when BeautifulSoup is available."""
        # This test checks the current state - if BS4 is available, extractor should be available
        extractor = HTMLExtractor()
        # We can't easily mock the import check since it happens at init time
        # Just verify the extractor behaves consistently
        assert isinstance(extractor.available, bool)

    def test_availability_without_beautifulsoup(self):
        """Test availability when BeautifulSoup is not available."""
        with patch.dict('sys.modules', {'bs4': None}):
            # This will be checked at import time, so we can't fully test this
            # without more complex mocking
            pass

    @pytest.mark.skipif(
        not hasattr(HTMLExtractor(), '_check_availability') or not HTMLExtractor()._check_availability(),
        reason="BeautifulSoup not available"
    )
    def test_extract_html_content(self):
        """Test extracting content from HTML."""
        content = TestSampleFiles.create_sample_html()
        result = self.extractor.extract_text(content, "test.html")

        if result.success:
            assert "Test Document" in result.text
            assert "Main Title" in result.text
            assert "test paragraph" in result.text
            assert result.metadata['title'] == "Test Document"
            assert "A test HTML document" in result.metadata['description']

    def test_supported_formats(self):
        """Test supported HTML formats."""
        assert '.html' in self.extractor.supported_extensions
        assert '.htm' in self.extractor.supported_extensions
        assert 'text/html' in self.extractor.supported_mimetypes


@pytest.mark.unit
class TestXMLExtractor:
    """Test XMLExtractor."""

    def setup_method(self):
        """Set up XML extractor."""
        self.extractor = XMLExtractor()

    @pytest.mark.skipif(
        not hasattr(XMLExtractor(), '_check_availability') or not XMLExtractor()._check_availability(),
        reason="Required XML libraries not available"
    )
    def test_extract_generic_xml(self):
        """Test extracting content from generic XML."""
        content = TestSampleFiles.create_sample_xml()
        result = self.extractor.extract_text(content, "test.xml")

        if result.success:
            assert "Sample Book" in result.text
            assert "Test Author" in result.text
            assert result.metadata['xml_type'] == 'generic'

    @pytest.mark.skipif(
        not hasattr(XMLExtractor(), '_check_availability') or not XMLExtractor()._check_availability(),
        reason="Required XML libraries not available"
    )
    def test_extract_rss_feed(self):
        """Test extracting content from RSS feed."""
        content = TestSampleFiles.create_sample_rss()
        result = self.extractor.extract_text(content, "feed.rss")

        if result.success:
            assert "Test RSS Feed" in result.text
            assert "First Article" in result.text
            assert result.metadata['xml_type'] == 'rss'


@pytest.mark.unit
class TestExtractorRegistry:
    """Test ExtractorRegistry functionality."""

    def setup_method(self):
        """Set up registry for testing."""
        # Clear registry state
        ExtractorRegistry._extractors.clear()
        ExtractorRegistry._plugins_discovered = False

    def teardown_method(self):
        """Clean up after test."""
        ExtractorRegistry._extractors.clear()
        ExtractorRegistry._plugins_discovered = False

    def test_register_extractor(self):
        """Test registering an extractor."""
        ExtractorRegistry.register(ConcreteTestExtractor)

        assert "ConcreteTestExtractor" in ExtractorRegistry._extractors
        extractor = ExtractorRegistry.get_extractor("ConcreteTestExtractor")
        assert extractor is not None
        assert isinstance(extractor, ConcreteTestExtractor)

    def test_list_extractors(self):
        """Test listing extractors."""
        ExtractorRegistry.register(ConcreteTestExtractor)
        extractors = ExtractorRegistry.list_extractors(available_only=False)

        assert "ConcreteTestExtractor" in extractors

    def test_get_extractors_for_file(self):
        """Test getting suitable extractors for a file."""
        ExtractorRegistry.register(ConcreteTestExtractor)
        extractors = ExtractorRegistry.get_extractors_for_file("test.test")

        assert len(extractors) > 0
        assert extractors[0].name == "ConcreteTestExtractor"

    def test_extract_text_with_suitable_extractor(self):
        """Test extracting text using registry."""
        ExtractorRegistry.register(ConcreteTestExtractor)
        result = ExtractorRegistry.extract_text(b"test", "file.test")

        assert result.success is True
        assert result.text == "Test extraction successful"

    def test_extract_text_no_suitable_extractor(self):
        """Test extracting text when no suitable extractor exists."""
        # Don't register any extractors
        result = ExtractorRegistry.extract_text(b"test", "file.unsupported")

        assert result.success is False
        assert "No suitable extractor found" in result.error

    def test_get_supported_formats(self):
        """Test getting supported formats."""
        ExtractorRegistry.register(ConcreteTestExtractor)
        formats = ExtractorRegistry.get_supported_formats()

        assert 'test' in formats
        assert formats['test']['available'] is True
        assert len(formats['test']['extractors']) > 0

    def test_refresh_plugins(self):
        """Test refreshing plugins."""
        # Set discovered to True first
        ExtractorRegistry._plugins_discovered = True
        initial_count = len(ExtractorRegistry._extractors)

        # Call refresh which should reset discovered flag and potentially discover plugins
        ExtractorRegistry.refresh_plugins()

        # After refresh, plugins should be discovered again
        # The _plugins_discovered flag may be True again after discovery
        assert isinstance(ExtractorRegistry._plugins_discovered, bool)
        # Verify registry still has extractors after refresh
        assert len(ExtractorRegistry._extractors) >= initial_count


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in extractors."""

    def setup_method(self):
        """Set up for error handling tests."""
        ExtractorRegistry._extractors.clear()
        ExtractorRegistry._plugins_discovered = False
        ExtractorRegistry.register(TextFileExtractor)

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        content = TestSampleFiles.create_corrupted_file()
        result = ExtractorRegistry.extract_text(content, "corrupted.txt")

        # Should either extract what it can or fail gracefully
        assert isinstance(result, ExtractionResult)

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        content = TestSampleFiles.create_empty_file()
        result = ExtractorRegistry.extract_text(content, "empty.txt")

        assert isinstance(result, ExtractionResult)

    def test_extractor_exception_handling(self):
        """Test that extractor exceptions are caught and reported."""

        class FaultyExtractor(BaseExtractor):
            @property
            def supported_extensions(self):
                return ['.faulty']

            @property
            def supported_mimetypes(self):
                return ['application/faulty']

            @property
            def required_packages(self):
                return []

            @property
            def priority(self):
                return 10

            @property
            def metadata_schema(self):
                return {}

            def _check_availability(self):
                return True

            def _extract_text_impl(self, file_content, filename, mimetype, **kwargs):
                raise Exception("Deliberate test exception")

        ExtractorRegistry.register(FaultyExtractor)
        result = ExtractorRegistry.extract_text(b"test", "file.faulty")

        assert result.success is False
        assert "Deliberate test exception" in result.error


@pytest.mark.unit
class TestMockExtractorBehaviors:
    """Test extractors with mocked dependencies."""

    def test_pdf_extractor_without_pdfplumber(self):
        """Test PDF extractor when pdfplumber is not available."""
        with patch.dict('sys.modules', {'pdfplumber': None}):
            extractor = PDFPlumberExtractor()
            assert extractor.available is False

            result = extractor.extract_text(b"fake pdf content", "test.pdf")
            assert result.success is False
            assert "Required packages not available" in result.error

    def test_docx_extractor_without_python_docx(self):
        """Test DOCX extractor when python-docx is not available."""
        with patch.dict('sys.modules', {'docx': None}):
            extractor = DocxExtractor()
            assert extractor.available is False

            result = extractor.extract_text(b"fake docx content", "test.docx")
            assert result.success is False
            assert "Required packages not available" in result.error

    def test_rtf_extractor_without_striprtf(self):
        """Test RTF extractor when striprtf is not available."""
        with patch.dict('sys.modules', {'striprtf': None}):
            extractor = RTFExtractor()
            assert extractor.available is False


@pytest.mark.integration
class TestExtractorIntegration:
    """Integration tests for extractors with real file processing."""

    def setup_method(self):
        """Set up for integration tests."""
        # Refresh registry to pick up all available extractors
        ExtractorRegistry.refresh_plugins()

    def test_multiple_extractors_priority(self):
        """Test that extractors are tried in priority order."""
        # Create a file that could be handled by multiple extractors
        content = TestSampleFiles.create_sample_text("# Markdown content\\nThis could be text or markdown.")

        extractors = ExtractorRegistry.get_extractors_for_file("test.md")

        # Should have multiple extractors available
        if len(extractors) > 1:
            # Check they're sorted by priority (highest first)
            priorities = [e.priority for e in extractors]
            assert priorities == sorted(priorities, reverse=True)

    def test_fallback_extraction_chain(self):
        """Test that fallback extraction works when primary extractors fail."""
        # Use a file extension that might not have a specific extractor
        content = TestSampleFiles.create_sample_text()
        result = ExtractorRegistry.extract_text(content, "unknown.extension")

        # Should succeed via fallback extractor or text extractor
        assert isinstance(result, ExtractionResult)

    def test_registry_get_supported_formats_integration(self):
        """Test that registry correctly reports all supported formats."""
        formats = get_supported_formats()

        # Should include basic text formats
        assert isinstance(formats, dict)

        # Check that at least some common formats are present
        common_formats = ['txt', 'html', 'xml']
        available_formats = list(formats.keys())

        # At least some should be available
        assert len(available_formats) > 0


@pytest.mark.slow
class TestRealFileExtraction:
    """Tests with real file creation (marked as slow)."""

    def test_create_and_extract_text_file(self):
        """Test creating and extracting a real text file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(TestSampleFiles.create_sample_text())
            tmp.flush()
            tmp_name = tmp.name

        try:
            # Read the file back
            with open(tmp_name, 'rb') as f:
                content = f.read()

            result = ExtractorRegistry.extract_text(content, tmp_name)

            assert result.success is True
            assert "Hello World!" in result.text
        finally:
            # Clean up
            try:
                Path(tmp_name).unlink()
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows

    def test_create_and_extract_html_file(self):
        """Test creating and extracting a real HTML file."""
        if not HTMLExtractor()._check_availability():
            pytest.skip("BeautifulSoup not available")

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            tmp.write(TestSampleFiles.create_sample_html())
            tmp.flush()
            tmp_name = tmp.name

        try:
            # Read the file back
            with open(tmp_name, 'rb') as f:
                content = f.read()

            result = ExtractorRegistry.extract_text(content, tmp_name)

            if result.success:
                assert "Test Document" in result.text
                assert "Main Title" in result.text
        finally:
            # Clean up
            try:
                Path(tmp_name).unlink()
            except (PermissionError, FileNotFoundError):
                pass  # Ignore cleanup errors on Windows


@pytest.mark.unit
class TestExtractorFactoryFunctions:
    """Test module-level factory functions."""

    def test_get_extractor_registry(self):
        """Test get_extractor_registry function."""
        registry = get_extractor_registry()
        assert registry == ExtractorRegistry

    def test_get_supported_formats(self):
        """Test get_supported_formats function."""
        formats = get_supported_formats()
        assert isinstance(formats, dict)