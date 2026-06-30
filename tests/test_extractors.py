"""
Tests for localvectordb.extractors module.

File extraction is delegated to the all2md library via a single
:class:`~localvectordb.extractors.all2md_extractor.All2MdExtractor`. These tests
cover the format-agnostic plumbing (ExtractionResult, BaseExtractor contract,
ExtractorRegistry) plus the all2md adapter itself.
"""

import pytest

from localvectordb.extractors import (
    BaseExtractor,
    ExtractionResult,
    ExtractorRegistry,
    get_extractor_registry,
    get_supported_formats,
)
from localvectordb.extractors.all2md_extractor import All2MdExtractor


class TestSampleFiles:
    """Sample file generators for constructible text-based formats."""

    @staticmethod
    def markdown() -> bytes:
        return b"# Title\n\nHello **world**, this is _markdown_.\n\n- one\n- two\n"

    @staticmethod
    def csv() -> bytes:
        return b"name,age\nAlice,30\nBob,25\n"

    @staticmethod
    def html() -> bytes:
        return (
            b"<!DOCTYPE html><html><head><title>Test Document</title></head>"
            b"<body><h1>Main Title</h1><p>A test paragraph.</p>"
            b"<script>alert('xss')</script></body></html>"
        )

    @staticmethod
    def empty() -> bytes:
        return b""

    @staticmethod
    def binary() -> bytes:
        return b"\x89PNG\r\n\x1a\n" + bytes(200)


@pytest.mark.unit
class TestExtractionResult:
    """Test ExtractionResult class."""

    def test_create_success_result(self):
        result = ExtractionResult(text="Hello World", success=True, method="TestExtractor", metadata={"key": "value"})
        assert result.text == "Hello World"
        assert result.success is True
        assert result.method == "TestExtractor"
        assert result.metadata == {"key": "value"}
        assert result.error is None

    def test_create_failure_result(self):
        result = ExtractionResult(text="", success=False, method="TestExtractor", error="Something went wrong")
        assert result.text == ""
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.metadata == {}

    def test_to_dict(self):
        result = ExtractionResult(text="Hello World", success=True, method="TestExtractor", metadata={"key": "value"})
        assert result.to_dict() == {
            "text": "Hello World",
            "extraction_success": True,
            "extraction_method": "TestExtractor",
            "metadata": {"key": "value"},
            "error": None,
            "text_length": 11,
        }


@pytest.mark.unit
class TestBaseExtractor:
    """Test BaseExtractor abstract base class."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseExtractor()

    def test_concrete_extractor_must_implement_abstract_methods(self):
        class IncompleteExtractor(BaseExtractor):
            def _check_availability(self):
                return True

        with pytest.raises(TypeError):
            IncompleteExtractor()


class ConcreteTestExtractor(BaseExtractor):
    """Concrete test extractor for exercising the base class and registry."""

    @property
    def supported_extensions(self):
        return [".test"]

    @property
    def supported_mimetypes(self):
        return ["application/test"]

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
        return ExtractionResult(text="Test extraction successful", success=True, method="ConcreteTestExtractor")


@pytest.mark.unit
class TestConcreteExtractor:
    """Test concrete extractor functionality."""

    def setup_method(self):
        self.extractor = ConcreteTestExtractor()

    def test_extractor_initialization(self):
        assert self.extractor.available is True
        assert self.extractor.name == "ConcreteTestExtractor"

    def test_can_extract_by_extension(self):
        assert self.extractor.can_extract("file.test") is True
        assert self.extractor.can_extract("file.txt") is False

    def test_can_extract_by_mimetype(self):
        assert self.extractor.can_extract("file.unknown", "application/test") is True
        assert self.extractor.can_extract("file.unknown", "text/plain") is False

    def test_extract_text_success(self):
        result = self.extractor.extract_text(b"test content", "file.test")
        assert result.success is True
        assert result.text == "Test extraction successful"

    def test_extract_text_unsupported_file(self):
        result = self.extractor.extract_text(b"test content", "file.unsupported")
        assert result.success is False
        assert result.error.startswith("File type not supported")

    def test_get_info(self):
        info = self.extractor.get_info()
        expected_keys = (
            "name",
            "available",
            "supported_extensions",
            "supported_mimetypes",
            "required_packages",
            "priority",
        )
        for key in expected_keys:
            assert key in info
        assert info["name"] == "ConcreteTestExtractor"


@pytest.mark.unit
class TestAll2MdExtractor:
    """Test the all2md-backed extractor."""

    def setup_method(self):
        self.extractor = All2MdExtractor()

    def test_available(self):
        # all2md is a core dependency, so the adapter should always be available.
        assert self.extractor.available is True
        assert self.extractor.required_packages == ["all2md"]
        assert self.extractor.priority == 10

    def test_supported_extensions_include_common_formats(self):
        exts = self.extractor.supported_extensions
        for ext in (".pdf", ".docx", ".md", ".csv", ".html", ".pptx", ".xlsx", ".epub", ".rst"):
            assert ext in exts, f"expected {ext} to be supported"

    def test_extension_maps_markdown_not_plaintext(self):
        # Regression: .md must resolve to the markdown parser, not plaintext
        # (otherwise markdown syntax gets escaped on round-trip).
        index = self.extractor._index()
        assert index["ext_to_format"].get(".md") == "markdown"

    def test_extract_markdown_preserves_syntax(self):
        result = self.extractor.extract_text(TestSampleFiles.markdown(), "doc.md")
        assert result.success is True
        assert result.method.endswith(":markdown")
        # Markdown must NOT be escaped (no backslash-escaped # or *).
        assert "# Title" in result.text
        assert "**world**" in result.text
        assert "\\#" not in result.text

    def test_extract_csv_to_markdown_table(self):
        result = self.extractor.extract_text(TestSampleFiles.csv(), "data.csv")
        assert result.success is True
        assert "| name | age |" in result.text or "name" in result.text
        assert "Alice" in result.text

    def test_extract_html_strips_dangerous_and_extracts_title(self):
        result = self.extractor.extract_text(TestSampleFiles.html(), "page.html")
        assert result.success is True
        assert "Main Title" in result.text
        # Dangerous script content must be stripped by default.
        assert "alert" not in result.text
        assert result.metadata.get("title") == "Test Document"

    def test_metadata_always_includes_core_fields(self):
        result = self.extractor.extract_text(TestSampleFiles.markdown(), "doc.md")
        for key in ("filename", "source_format", "file_size_bytes", "character_count"):
            assert key in result.metadata
        assert result.metadata["filename"] == "doc.md"
        assert result.metadata["source_format"] == "markdown"
        assert result.metadata["file_size_bytes"] == len(TestSampleFiles.markdown())

    def test_unsupported_extension_fails_gracefully(self):
        # An unrecognised extension is rejected by the base-class can_extract
        # gate before reaching all2md.
        result = self.extractor.extract_text(TestSampleFiles.binary(), "mystery.unknownext")
        assert result.success is False
        assert "not supported" in result.error.lower()

    def test_invalid_zip_archive_rejected(self):
        # A file with the ZIP magic header but garbage contents must be rejected
        # by the ZIP-safety guard rather than crashing the parser.
        result = self.extractor.extract_text(b"PK\x03\x04garbage-not-a-real-zip", "fake.docx")
        assert result.success is False
        assert "Invalid archive" in result.error or "ZIP" in result.error

    def test_security_defaults_are_safe(self):
        opts = self.extractor._build_parser_options("html")
        assert opts is not None
        assert opts.strip_dangerous_elements is True
        assert opts.network.allow_remote_fetch is False
        assert opts.local_files.allow_local_files is False

    def test_security_overrides_via_kwargs(self):
        opts = self.extractor._build_parser_options("html", allow_remote_fetch=True, strip_dangerous_elements=False)
        assert opts.network.allow_remote_fetch is True
        assert opts.strip_dangerous_elements is False

    def test_non_html_format_has_no_special_options(self):
        # Formats without security-relevant options use all2md's safe defaults.
        assert self.extractor._build_parser_options("markdown") is None

    @pytest.mark.integration
    def test_docx_round_trip(self):
        docx = pytest.importorskip("docx")
        import io

        document = docx.Document()
        document.add_heading("Quarterly Report", level=1)
        document.add_paragraph("Revenue grew this quarter.")
        buffer = io.BytesIO()
        document.save(buffer)

        result = self.extractor.extract_text(buffer.getvalue(), "report.docx")
        assert result.success is True
        assert result.method.endswith(":docx")
        assert "Quarterly Report" in result.text
        assert "Revenue grew" in result.text


@pytest.mark.unit
class TestExtractorRegistry:
    """Test ExtractorRegistry functionality."""

    def setup_method(self):
        ExtractorRegistry._extractors.clear()
        ExtractorRegistry._plugins_discovered = False

    def teardown_method(self):
        ExtractorRegistry._extractors.clear()
        ExtractorRegistry._plugins_discovered = False
        ExtractorRegistry.refresh_plugins()

    def test_register_extractor(self):
        ExtractorRegistry.register(ConcreteTestExtractor)
        assert "ConcreteTestExtractor" in ExtractorRegistry._extractors
        assert isinstance(ExtractorRegistry.get_extractor("ConcreteTestExtractor"), ConcreteTestExtractor)

    def test_get_extractors_for_file(self):
        ExtractorRegistry.register(ConcreteTestExtractor)
        extractors = ExtractorRegistry.get_extractors_for_file("test.test")
        assert len(extractors) > 0
        assert extractors[0].name == "ConcreteTestExtractor"

    def test_extract_text_with_suitable_extractor(self):
        ExtractorRegistry.register(ConcreteTestExtractor)
        result = ExtractorRegistry.extract_text(b"test", "file.test")
        assert result.success is True
        assert result.text == "Test extraction successful"

    def test_extract_text_forwards_kwargs(self):
        captured = {}

        class KwargCapturingExtractor(ConcreteTestExtractor):
            def _extract_text_impl(self, file_content, filename, mimetype, **kwargs):
                captured.update(kwargs)
                return ExtractionResult(text="ok", success=True, method=self.name)

        ExtractorRegistry.register(KwargCapturingExtractor)
        ExtractorRegistry.extract_text(b"x", "file.test", allow_remote_fetch=True)
        assert captured.get("allow_remote_fetch") is True

    def test_extract_text_no_suitable_extractor(self):
        result = ExtractorRegistry.extract_text(b"test", "file.unsupported")
        assert result.success is False
        assert "No suitable extractor found" in result.error

    def test_get_supported_formats(self):
        ExtractorRegistry.register(ConcreteTestExtractor)
        formats = ExtractorRegistry.get_supported_formats()
        assert "test" in formats
        assert formats["test"]["available"] is True
        assert len(formats["test"]["extractors"]) > 0

    def test_extractor_exception_handling(self):
        class FaultyExtractor(ConcreteTestExtractor):
            @property
            def supported_extensions(self):
                return [".faulty"]

            def _extract_text_impl(self, file_content, filename, mimetype, **kwargs):
                raise Exception("Deliberate test exception")

        ExtractorRegistry.register(FaultyExtractor)
        result = ExtractorRegistry.extract_text(b"test", "file.faulty")
        assert result.success is False
        assert "Deliberate test exception" in result.error


@pytest.mark.integration
class TestExtractorIntegration:
    """Integration tests exercising the real plugin-discovered registry."""

    def setup_method(self):
        ExtractorRegistry.refresh_plugins()

    def test_all2md_extractor_is_discovered(self):
        assert "All2MdExtractor" in ExtractorRegistry.list_extractors()

    def test_registry_extracts_via_all2md(self):
        result = ExtractorRegistry.extract_text(TestSampleFiles.markdown(), "notes.md")
        assert result.success is True
        assert "# Title" in result.text

    def test_supported_formats_non_empty(self):
        formats = get_supported_formats()
        assert isinstance(formats, dict)
        assert len(formats) > 0


@pytest.mark.unit
class TestExtractorFactoryFunctions:
    """Test module-level factory functions."""

    def test_get_extractor_registry(self):
        assert get_extractor_registry() is ExtractorRegistry

    def test_get_supported_formats(self):
        assert isinstance(get_supported_formats(), dict)
