# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/extractors/other_extractors.py
"""
Extractors for miscellaneous file formats.
"""

import logging
from typing import List, Optional

from localvectordb import MetadataField
from localvectordb.core import MetadataFieldType
from localvectordb.extractors import BaseExtractor, ExtractionResult, ZipBombError, validate_zip_safety

logger = logging.getLogger(__name__)


class RTFExtractor(BaseExtractor):
    """
    Extractor for Rich Text Format documents (.rtf).
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.rtf']

    @property
    def supported_mimetypes(self) -> List[str]:
        return ['application/rtf', 'text/rtf']

    @property
    def required_packages(self) -> List[str]:
        return ['striprtf']

    @property
    def priority(self) -> int:
        return 15

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return {
            "filename": MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=False),
            "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "rtf_size": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "compression_ratio": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
        }

    def _check_availability(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("striprtf") is not None

    def _extract_text_impl(
            self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs
    ) -> ExtractionResult:
        """Extract text from RTF files."""
        try:
            from striprtf.striprtf import rtf_to_text

            # RTF files are typically encoded as text
            try:
                rtf_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try other encodings
                for encoding in ['latin-1', 'cp1252']:
                    try:
                        rtf_content = file_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    return ExtractionResult(
                        text="",
                        success=False,
                        method='RTFExtractor',
                        error="Could not decode RTF file with supported encodings"
                    )

            # Extract plain text from RTF
            plain_text = rtf_to_text(rtf_content)

            if not plain_text or not plain_text.strip():
                return ExtractionResult(
                    text="",
                    success=False,
                    method='RTFExtractor',
                    error="No text content found in RTF document"
                )

            metadata = {
                'filename': filename,
                'file_size_bytes': len(file_content),
                'character_count': len(plain_text),
                'rtf_size': len(rtf_content),
                'compression_ratio': len(plain_text) / len(rtf_content) if rtf_content else 0,
                'extraction_library': 'striprtf'
            }

            return ExtractionResult(
                text=plain_text,
                success=True,
                method='RTFExtractor',
                metadata=metadata
            )

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='RTFExtractor',
                error=f"RTF extraction failed: {str(e)}"
            )


class EPubExtractor(BaseExtractor):
    """
    Extractor for EPUB e-book files (.epub).
    This is an example of how to add support for additional formats.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.epub']

    @property
    def supported_mimetypes(self) -> List[str]:
        return ['application/epub+zip']

    @property
    def required_packages(self) -> List[str]:
        return ['ebooklib']

    @property
    def priority(self) -> int:
        return 15

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return {
            "filename": MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=False),
            "chapter_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
            "book_metadata": MetadataField(type=MetadataFieldType.JSON, indexed=False, required=False),
            "extraction_library": MetadataField(type=MetadataFieldType.TEXT, indexed=False, required=False),
        }

    def _check_availability(self) -> bool:
        import importlib.util
        return importlib.util.find_spec("ebooklib") is not None

    def _extract_text_impl(
            self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs
    ) -> ExtractionResult:
        """Extract text from EPUB files."""
        try:
            import io

            # Validate ZIP archive safety before processing (EPUB files are ZIP archives)
            try:
                validate_zip_safety(file_content)
            except ZipBombError as e:
                logger.warning(f"ZIP bomb detected in '{filename}': {e}")
                return ExtractionResult(
                    text="",
                    success=False,
                    method='EPubExtractor',
                    error=f"ZIP bomb protection triggered: {str(e)}"
                )
            except ValueError as e:
                return ExtractionResult(
                    text="",
                    success=False,
                    method='EPubExtractor',
                    error=f"Invalid EPUB file: {str(e)}"
                )

            import ebooklib
            from ebooklib import epub

            # EPUB files are ZIP archives, need to handle as such
            with io.BytesIO(file_content) as file_buffer:
                book = epub.read_epub(file_buffer)

                text_parts = []
                chapter_count = 0

                # Extract text from all items in the book
                for item in book.get_items():
                    if item.get_type() == ebooklib.ITEM_DOCUMENT:
                        chapter_count += 1

                        # Parse HTML content and extract text
                        try:
                            import re
                            from html import unescape

                            content = item.get_content().decode('utf-8')

                            # Remove HTML tags (basic implementation)
                            text = re.sub('<[^<]+?>', '', content)
                            text = unescape(text)
                            text = re.sub(r'\s+', ' ', text).strip()

                            if text:
                                text_parts.append(f"[Chapter {chapter_count}]\n{text}")

                        except Exception as e:
                            logger.warning(f"Error extracting text from EPUB chapter: {e}")
                            continue

                if not text_parts:
                    return ExtractionResult(
                        text="",
                        success=False,
                        method='EPubExtractor',
                        error="No text content found in EPUB file"
                    )

                full_text = '\n\n'.join(text_parts)

                # Get metadata from book
                book_metadata = {}
                try:
                    title_meta = book.get_metadata('DC', 'title')
                    author_meta = book.get_metadata('DC', 'creator')
                    lang_meta = book.get_metadata('DC', 'language')
                    book_metadata = {
                        'title': title_meta[0][0] if title_meta else 'Unknown',
                        'author': author_meta[0][0] if author_meta else 'Unknown',
                        'language': lang_meta[0][0] if lang_meta else 'Unknown',
                    }
                except Exception:
                    pass

                metadata = {
                    'filename': filename,
                    'chapter_count': chapter_count,
                    'file_size_bytes': len(file_content),
                    'character_count': len(full_text),
                    'book_metadata': book_metadata,
                    'extraction_library': 'ebooklib'
                }

                return ExtractionResult(
                    text=full_text,
                    success=True,
                    method='EPubExtractor',
                    metadata=metadata
                )

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='EPubExtractor',
                error=f"EPUB extraction failed: {str(e)}"
            )
