# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/extractors/text_extractors.py
"""
Text file extractors for plain text and fallback handling.
"""

import logging
from typing import List, Optional

from localvectordb import MetadataField
from localvectordb.core import MetadataFieldType
from localvectordb_server.extractors import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class TextFileExtractor(BaseExtractor):
    """
    Extractor for plain text files and structured text formats.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [
            '.txt', '.text',
            '.md', '.markdown',
            '.rst', '.asciidoc',
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h',
            '.html', '.htm', '.xml', '.css',
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg',
            '.csv', '.tsv',
            '.log', '.conf',
            '.sh', '.bat', '.ps1',
            '.sql', '.r', '.m', '.scala'
        ]

    @property
    def supported_mimetypes(self) -> List[str]:
        return [
            'text/plain',
            'text/markdown',
            'text/x-python',
            'text/javascript',
            'application/javascript',
            'text/html',
            'application/xml',
            'text/xml',
            'text/css',
            'application/json',
            'text/csv',
            'application/x-yaml',
            'text/x-yaml'
        ]

    @property
    def required_packages(self) -> List[str]:
        return []  # No additional packages required

    @property
    def priority(self) -> int:
        return 20  # High priority for text files

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return {
            "encoding": MetadataField(type=MetadataFieldType.TEXT, indexed=False, required=False),
            "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "line_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
        }

    def _check_availability(self) -> bool:
        return True  # Always available

    def _extract_text_impl(self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs) -> ExtractionResult:
        """Extract text from plain text files."""
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                text = file_content.decode(encoding)

                # Basic validation - ensure we got meaningful text
                if len(text.strip()) == 0:
                    continue

                metadata = {
                    'encoding': encoding,
                    'file_size_bytes': len(file_content),
                    'character_count': len(text),
                    'line_count': text.count('\n') + 1
                }

                return ExtractionResult(
                    text=text,
                    success=True,
                    method=f'TextFileExtractor_{encoding}',
                    metadata=metadata
                )

            except UnicodeDecodeError:
                continue

        # All encodings failed
        return ExtractionResult(
            text="",
            success=False,
            method='TextFileExtractor',
            error="Could not decode file with any supported encoding"
        )


class TextFallbackExtractor(BaseExtractor):
    """
    Fallback extractor that attempts to extract text from any file.
    This should have the lowest priority and be used as a last resort.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.*']  # Supports any extension as fallback

    @property
    def supported_mimetypes(self) -> List[str]:
        return ['*/*']  # Supports any MIME type as fallback

    @property
    def required_packages(self) -> List[str]:
        return []

    @property
    def priority(self) -> int:
        return 1  # Lowest priority - only used as fallback

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return {
            "encoding": MetadataField(type=MetadataFieldType.TEXT, indexed=False, required=False),
            "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "printable_ratio": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
        }

    def _check_availability(self) -> bool:
        return True

    def can_extract(self, filename: str, mimetype: Optional[str] = None) -> bool:
        """Always returns True as this is the fallback extractor."""
        return True

    def _extract_text_impl(self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs) -> ExtractionResult:
        """Attempt to extract text using fallback methods."""

        # Try UTF-8 with error handling
        try:
            text = file_content.decode('utf-8', errors='ignore')

            # Check if we got meaningful text (at least 10 printable characters)
            printable_chars = sum(1 for c in text if c.isprintable() or c.isspace())

            if printable_chars < 10:
                return ExtractionResult(
                    text="",
                    success=False,
                    method='TextFallbackExtractor',
                    error="File appears to be binary with no extractable text"
                )

            # Check ratio of printable to total characters
            text_ratio = printable_chars / len(text) if text else 0

            if text_ratio < 0.7:  # Less than 70% printable characters
                return ExtractionResult(
                    text="",
                    success=False,
                    method='TextFallbackExtractor',
                    error=f"Low text content ratio ({text_ratio:.2%}), likely binary file"
                )

            metadata = {
                'encoding': 'utf-8_with_ignore',
                'file_size_bytes': len(file_content),
                'character_count': len(text),
                'printable_ratio': text_ratio,
                'warning': 'Fallback extraction used - text quality may be poor'
            }

            return ExtractionResult(
                text=text,
                success=True,
                method='TextFallbackExtractor',
                metadata=metadata
            )

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='TextFallbackExtractor',
                error=f"Fallback extraction failed: {str(e)}"
            )