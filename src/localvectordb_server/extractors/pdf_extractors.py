# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/extractors/pdf_extractors.py
"""
PDF file extractors using different libraries.
"""

import io
import logging
from typing import List, Optional

from localvectordb import MetadataField
from localvectordb.core import MetadataFieldType
from localvectordb_server.extractors import BaseExtractor, ExtractionResult

logger = logging.getLogger(__name__)


_PDF_METADATA_SCHEMA = {
    "total_pages": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
    "pages_with_text": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
    "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
    "character_count": MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False),
    "page_details": MetadataField(type=MetadataFieldType.JSON, indexed=False, required=False),
}


class PDFPlumberExtractor(BaseExtractor):
    """
    PDF extractor using pdfplumber library (preferred for better text extraction).
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.pdf']

    @property
    def supported_mimetypes(self) -> List[str]:
        return ['application/pdf']

    @property
    def required_packages(self) -> List[str]:
        return ['pdfplumber']

    @property
    def priority(self) -> int:
        return 15  # Higher priority than pypdf

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return _PDF_METADATA_SCHEMA

    def _check_availability(self) -> bool:
        try:
            import pdfplumber
            return True
        except ImportError:
            return False

    def _extract_text_impl(self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs) -> ExtractionResult:
        """Extract text from PDF using pdfplumber."""
        try:
            import pdfplumber

            with io.BytesIO(file_content) as file_buffer:
                with pdfplumber.open(file_buffer) as pdf:
                    text_parts = []
                    page_info = []

                    for page_num, page in enumerate(pdf.pages, 1):
                        try:
                            page_text = page.extract_text()
                            if page_text and page_text.strip():
                                text_parts.append(page_text)
                                page_info.append({
                                    'page': page_num,
                                    'text_length': len(page_text),
                                    'has_text': True
                                })
                            else:
                                page_info.append({
                                    'page': page_num,
                                    'text_length': 0,
                                    'has_text': False
                                })
                        except Exception as e:
                            logger.warning(f"Error extracting text from page {page_num}: {e}")
                            page_info.append({
                                'page': page_num,
                                'error': str(e),
                                'has_text': False
                            })

                    if not text_parts:
                        return ExtractionResult(
                            text="",
                            success=False,
                            method='PDFPlumberExtractor',
                            error="No text content found in PDF (may be image-based)"
                        )

                    # Join pages with double newlines
                    full_text = '\n\n'.join(text_parts)

                    metadata = {
                        'total_pages': len(pdf.pages),
                        'pages_with_text': len(text_parts),
                        'file_size_bytes': len(file_content),
                        'character_count': len(full_text),
                        'page_details': page_info,
                    }

                    return ExtractionResult(
                        text=full_text,
                        success=True,
                        method='PDFPlumberExtractor',
                        metadata=metadata
                    )

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='PDFPlumberExtractor',
                error=f"pdfplumber extraction failed: {str(e)}"
            )


class PyPDFExtractor(BaseExtractor):
    """
    PDF extractor using PyPDF2 library (fallback option).
    """

    @property
    def supported_extensions(self) -> List[str]:
        return ['.pdf']

    @property
    def supported_mimetypes(self) -> List[str]:
        return ['application/pdf']

    @property
    def required_packages(self) -> List[str]:
        return ['pypdf']

    @property
    def priority(self) -> int:
        return 12  # Lower priority than pdfplumber

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return _PDF_METADATA_SCHEMA

    def _check_availability(self) -> bool:
        try:
            import pypdf
            return True
        except ImportError:
            return False

    def _extract_text_impl(self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs) -> ExtractionResult:
        """Extract text from PDF using PyPDF2."""
        try:
            import pypdf

            with io.BytesIO(file_content) as file_buffer:
                pdf_reader = pypdf.PdfReader(file_buffer)
                text_parts = []
                page_info = []

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text(extraction_mode="layout")
                        if page_text and page_text.strip():
                            text_parts.append(page_text)
                            page_info.append({
                                'page': page_num,
                                'text_length': len(page_text),
                                'has_text': True
                            })
                        else:
                            page_info.append({
                                'page': page_num,
                                'text_length': 0,
                                'has_text': False
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                        page_info.append({
                            'page': page_num,
                            'error': str(e),
                            'has_text': False
                        })

                if not text_parts:
                    return ExtractionResult(
                        text="",
                        success=False,
                        method='PyPDFExtractor',
                        error="No text content found in PDF (may be image-based)"
                    )

                # Join pages with double newlines
                full_text = '\n\n'.join(text_parts)

                metadata = {
                    'total_pages': len(pdf_reader.pages),
                    'pages_with_text': len(text_parts),
                    'file_size_bytes': len(file_content),
                    'character_count': len(full_text),
                    'page_details': page_info,
                    'extraction_library': 'PyPDF2',
                    'encrypted': False
                }

                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    metadata['encrypted'] = True

                return ExtractionResult(
                    text=full_text,
                    success=True,
                    method='PyPDFExtractor',
                    metadata=metadata
                )

        except Exception as e:
            return ExtractionResult(
                text="",
                success=False,
                method='PyPDFExtractor',
                error=f"PyPDF2 extraction failed: {str(e)}"
            )