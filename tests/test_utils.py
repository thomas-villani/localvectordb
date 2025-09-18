"""
Tests for localvectordb.utils module.
"""

from unittest.mock import patch

import pytest

from localvectordb.utils import get_system_version, make_filename_safe


class TestGetSystemVersion:
    """Test get_system_version function."""

    @patch('importlib.metadata.version')
    def test_get_version_success(self, mock_version):
        """Test successful version retrieval."""
        mock_version.return_value = "2.0.1"

        version = get_system_version()

        assert version == "2.0.1"
        mock_version.assert_called_once_with("localvectordb")

    @patch('importlib.metadata.version')
    def test_get_version_not_found(self, mock_version):
        """Test version retrieval when package not found."""
        from importlib.metadata import PackageNotFoundError
        mock_version.side_effect = PackageNotFoundError("Package not found")

        assert get_system_version() == "dev"

    @patch('importlib.metadata.version')
    def test_get_version_different_versions(self, mock_version):
        """Test with different version formats."""
        test_versions = [
            "1.0.0",
            "2.1.3",
            "0.1.0-alpha",
            "1.0.0-beta.1",
            "2.0.0-rc.1",
            "1.0.0+build.1"
        ]

        for version in test_versions:
            mock_version.return_value = version
            result = get_system_version()
            assert result == version

    @patch('importlib.metadata.version')
    def test_get_version_empty_string(self, mock_version):
        """Test with empty version string."""
        mock_version.return_value = ""

        version = get_system_version()

        assert version == ""

    @patch('importlib.metadata.version')
    def test_get_version_with_spaces(self, mock_version):
        """Test version with leading/trailing spaces."""
        mock_version.return_value = "  1.0.0  "

        version = get_system_version()

        assert version == "  1.0.0  "  # Should preserve as-is


class TestMakeFilenameSafe:
    """Test make_filename_safe function."""

    def test_simple_safe_filename(self):
        """Test with already safe filename."""
        safe_names = [
            "simple",
            "test_file",
            "document-1",
            "file.txt",
            "my_document_2024"
        ]

        for name in safe_names:
            result = make_filename_safe(name)
            assert result == name

    def test_windows_invalid_characters(self):
        """Test Windows invalid character replacement."""
        with patch('os.name', 'nt'):
            test_cases = [
                ("file<name", "file_name"),
                ("file>name", "file_name"),
                ("file:name", "file_name"),
                ("file\"name", "file_name"),
                ("file/name", "file_name"),
                ("file\\name", "file_name"),
                ("file|name", "file_name"),
                ("file?name", "file_name"),
                ("file*name", "file_name"),
                ("file<>:\"/\\|?*name", "file_________name")
            ]

            for input_name, expected in test_cases:
                result = make_filename_safe(input_name)
                assert result == expected

    def test_posix_invalid_characters(self):
        """Test POSIX invalid character replacement."""
        with patch('os.name', 'posix'):
            test_cases = [
                ("file/name", "file_name"),
                ("file:name", "file_name"),
                ("file/:name", "file__name"),
                ("normal_file", "normal_file"),
                ("file.txt", "file.txt")
            ]

            for input_name, expected in test_cases:
                result = make_filename_safe(input_name)
                assert result == expected

    def test_windows_reserved_names(self):
        """Test Windows reserved name handling."""
        with patch('os.name', 'nt'):
            reserved_names = [
                "CON", "PRN", "AUX", "NUL",
                "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
                "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
            ]

            for name in reserved_names:
                result = make_filename_safe(name)
                assert result == name + "_safe"

                # Test case variations
                result_lower = make_filename_safe(name.lower())
                assert result_lower == name.lower() + "_safe"

    def test_posix_no_reserved_names(self):
        """Test that POSIX systems don't have reserved names."""
        with patch('os.name', 'posix'):
            # These are reserved on Windows but not POSIX
            names = ["CON", "PRN", "AUX", "COM1", "LPT1"]

            for name in names:
                result = make_filename_safe(name)
                assert result == name  # Should remain unchanged

    def test_windows_strip_spaces_and_periods(self):
        """Test Windows space and period stripping."""
        with patch('os.name', 'nt'):
            test_cases = [
                ("  filename  ", "filename"),
                ("...filename...", "filename"),
                ("  ...filename...  ", "filename"),
                ("filename.", "filename"),
                ("filename ", "filename"),
                (" .filename. ", "filename"),
                ("   ", ""),  # Edge case
                ("...", "")  # Edge case
            ]

            for input_name, expected in test_cases:
                result = make_filename_safe(input_name)
                assert result == expected

    def test_posix_only_strip_spaces(self):
        """Test that POSIX only strips spaces, not periods."""
        with patch('os.name', 'posix'):
            test_cases = [
                ("  filename  ", "filename"),
                ("...filename...", "...filename..."),  # Periods preserved
                ("  ...filename...  ", "...filename..."),
                ("filename.", "filename."),
                ("filename ", "filename"),
                (" .filename. ", ".filename."),
                ("   ", ""),  # Edge case
            ]

            for input_name, expected in test_cases:
                result = make_filename_safe(input_name)
                assert result == expected

    def test_max_length_truncation(self):
        """Test filename length truncation."""
        long_name = "a" * 300  # Longer than default max_length

        result = make_filename_safe(long_name)

        assert len(result) == 255  # Default max_length
        assert result == "a" * 255

    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        long_name = "a" * 100
        custom_max = 50

        result = make_filename_safe(long_name, max_length=custom_max)

        assert len(result) == custom_max
        assert result == "a" * custom_max

    def test_empty_string_handling(self):
        """Test empty string handling."""
        result = make_filename_safe("")
        assert result == ""

    def test_complex_filename_windows(self):
        """Test complex filename with multiple issues on Windows."""
        with patch('os.name', 'nt'):
            # Filename with multiple issues
            complex_name = '  CON<file>name:with"invalid/chars\\and|extra?spaces*  '

            result = make_filename_safe(complex_name)

            # Should replace invalid chars, handle reserved name, and strip spaces
            expected = "CON_file_name_with_invalid_chars_and_extra_spaces_"
            assert result == expected

    def test_complex_filename_posix(self):
        """Test complex filename with multiple issues on POSIX."""
        with patch('os.name', 'posix'):
            # Filename with POSIX invalid chars
            complex_name = '  file/name:with:invalid/chars  '

            result = make_filename_safe(complex_name)

            # Should replace invalid chars and strip spaces
            expected = "file_name_with_invalid_chars"
            assert result == expected

    def test_unicode_characters(self):
        """Test Unicode character handling."""
        unicode_names = [
            "файл.txt",  # Cyrillic
            "文件.txt",  # Chinese
            "ファイル.txt",  # Japanese
            "αρχείο.txt",  # Greek
            "café.txt",  # Accented characters
            "file_😀.txt"  # Emoji
        ]

        for name in unicode_names:
            result = make_filename_safe(name)
            # Should not crash and should return a string
            assert isinstance(result, str)
            # For this basic implementation, Unicode chars should be preserved
            # (more advanced implementations might transliterate)

    def test_mixed_invalid_and_unicode(self):
        """Test mixed invalid characters and Unicode."""
        with patch('os.name', 'nt'):
            name = "файл<test>café:file.txt"
            result = make_filename_safe(name)

            # Invalid chars should be replaced, Unicode preserved
            assert "<" not in result
            assert ">" not in result
            assert ":" not in result
            assert "файл" in result
            assert "café" in result

    def test_edge_case_only_invalid_chars(self):
        """Test filename consisting only of invalid characters."""
        with patch('os.name', 'nt'):
            name = "<>:\"/\\|?*"
            result = make_filename_safe(name)

            # Should be replaced with underscores
            assert result == "_________"

    def test_edge_case_reserved_with_extension(self):
        """Test reserved name with file extension."""
        with patch('os.name', 'nt'):
            name = "CON"
            result = make_filename_safe(name)

            # Should add _safe to avoid reserved name
            assert result == "CON_safe"

    def test_very_long_filename_with_issues(self):
        """Test very long filename with multiple issues."""
        with patch('os.name', 'nt'):
            # Create a long name with invalid characters
            base = "file<with>invalid:chars"
            long_name = base * 15  # Make it very long

            result = make_filename_safe(long_name, max_length=100)

            # Should be truncated and cleaned
            assert len(result) == 100
            assert "<" not in result
            assert ">" not in result
            assert ":" not in result
            assert "_" in result  # Replacements should be present

    def test_fallback_for_empty_result(self):
        """Test fallback name when result is empty."""
        # This might happen if filename only contains invalid chars that get stripped
        with patch('os.name', 'nt'):
            name = "   ...   "  # Only spaces and periods
            result = make_filename_safe(name)

            # Should return empty string (or could implement fallback)
            assert result == ""

    def test_max_length_zero(self):
        """Test edge case with max_length=0."""
        name = "filename"
        result = make_filename_safe(name, max_length=0)

        assert result == ""

    def test_max_length_one(self):
        """Test edge case with max_length=1."""
        name = "filename"
        result = make_filename_safe(name, max_length=1)

        assert result == "f"
        assert len(result) == 1

    @pytest.mark.parametrize("os_name,input_name,expected", [
        ('nt', 'file<name', 'file_name'),
        ('posix', 'file<name', 'file<name'),
        ('nt', 'file/name', 'file_name'),
        ('posix', 'file/name', 'file_name'),
        ('nt', 'CON', 'CON_safe'),
        ('posix', 'CON', 'CON'),
    ])
    def test_parametrized_os_behavior(self, os_name, input_name, expected):
        """Test behavior across different operating systems."""
        with patch('os.name', os_name):
            result = make_filename_safe(input_name)
            assert result == expected


class TestUtilsIntegration:
    """Test integration between utility functions."""


    def test_version_with_invalid_chars_in_filename(self):
        """Test version with invalid characters in filename."""
        with patch('importlib.metadata.version', return_value="2.0.1-beta+build"):
            version = get_system_version()
            filename = f"db<{version}>.sqlite"

            with patch('os.name', 'nt'):
                safe_filename = make_filename_safe(filename)

                # Invalid chars should be replaced
                assert "<" not in safe_filename
                assert ">" not in safe_filename
                assert "2.0.1-beta+build" in safe_filename

    def test_utils_error_handling(self):
        """Test error handling in utility functions."""
        # Test that make_filename_safe handles None gracefully
        try:
            # This should either handle None or raise TypeError
            result = make_filename_safe(None)
            # If it doesn't raise, result should be a string
            assert isinstance(result, str)
        except TypeError:
            # This is also acceptable behavior
            pass

    def test_utils_with_different_locales(self):
        """Test utility functions with different locale settings."""
        # Test with various inputs that might behave differently in different locales
        test_names = [
            "ÄÖÜäöüß.txt",  # German
            "ñáéíóú.txt",  # Spanish
            "çãõ.txt",  # Portuguese
            "æøå.txt",  # Norwegian/Danish
        ]

        for name in test_names:
            result = make_filename_safe(name)
            # Should not crash and should return a string
            assert isinstance(result, str)
            # Should preserve most Unicode characters in basic implementation
            assert len(result) > 0


class TestUtilsPerformance:
    """Test performance characteristics of utility functions."""

    def test_make_filename_safe_performance(self):
        """Test performance with various input sizes."""
        import time

        # Test with different sizes
        sizes = [10, 100, 1000, 10000]

        for size in sizes:
            large_name = "a" * size

            start_time = time.time()
            result = make_filename_safe(large_name)
            end_time = time.time()

            # Should complete reasonably quickly (under 1 second for these sizes)
            duration = end_time - start_time
            assert duration < 1.0

            # Result should be correctly truncated if needed
            if size > 255:
                assert len(result) == 255
            else:
                assert len(result) == size

    def test_repeated_calls_performance(self):
        """Test performance of repeated function calls."""
        import time

        # Test repeated version calls
        start_time = time.time()
        for _ in range(100):
            with patch('importlib.metadata.version', return_value="1.0.0"):
                get_system_version()
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 1.0

        # Test repeated filename safe calls
        start_time = time.time()
        for i in range(100):
            make_filename_safe(f"test_file_{i}.txt")
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 1.0
