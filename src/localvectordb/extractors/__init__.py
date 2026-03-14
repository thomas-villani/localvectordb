# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/extractors/__init__.py
"""
File content extraction plugin system for LocalVectorDB Server.

This module provides a plugin-based architecture for text extraction from various file formats.
All dependencies are optional and gracefully degrade if not available.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from localvectordb.core import MetadataField

logger = logging.getLogger(__name__)

# Maximum file size for extraction (100 MB default)
# This prevents DoS attacks via memory exhaustion from extremely large files
MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024

# ZIP bomb protection constants
# These prevent decompression attacks (zip bombs) in ZIP-based formats like DOCX, XLSX, PPTX, EPUB
MAX_ZIP_DECOMPRESSED_SIZE: int = 1024 * 1024 * 1024  # 1 GB maximum decompressed size
MAX_ZIP_COMPRESSION_RATIO: int = 100  # Maximum compression ratio (100:1)
MAX_ZIP_FILE_COUNT: int = 10000  # Maximum number of files in archive


class ZipBombError(Exception):
    """Exception raised when a potential ZIP bomb is detected."""
    pass


def validate_zip_safety(
    file_content: bytes,
    max_decompressed_size: int = MAX_ZIP_DECOMPRESSED_SIZE,
    max_compression_ratio: int = MAX_ZIP_COMPRESSION_RATIO,
    max_file_count: int = MAX_ZIP_FILE_COUNT
) -> None:
    """
    Validate a ZIP file for potential ZIP bomb attacks.

    This function checks for common ZIP bomb attack patterns:
    - Excessive decompressed size (billion laughs style)
    - High compression ratios indicating recursive compression
    - Excessive number of files (many small files attack)

    Parameters
    ----------
    file_content : bytes
        Raw ZIP file content
    max_decompressed_size : int
        Maximum allowed total decompressed size in bytes (default 1GB)
    max_compression_ratio : int
        Maximum allowed compression ratio (default 100:1)
    max_file_count : int
        Maximum allowed number of files in archive (default 10,000)

    Raises
    ------
    ZipBombError
        If the ZIP file exhibits characteristics of a ZIP bomb
    ValueError
        If the file is not a valid ZIP archive
    """
    import io
    import zipfile

    compressed_size = len(file_content)

    try:
        with zipfile.ZipFile(io.BytesIO(file_content), 'r') as zf:
            # Check file count
            file_count = len(zf.namelist())
            if file_count > max_file_count:
                raise ZipBombError(
                    f"ZIP archive contains {file_count} files, exceeding limit of {max_file_count}. "
                    "This may indicate a ZIP bomb attack."
                )

            # Calculate total decompressed size and check compression ratio
            total_uncompressed = 0
            for info in zf.infolist():
                total_uncompressed += info.file_size

                # Check for individual files with suspicious compression ratios
                if info.compress_size > 0:
                    file_ratio = info.file_size / info.compress_size
                    if file_ratio > max_compression_ratio:
                        raise ZipBombError(
                            f"File '{info.filename}' has compression ratio {file_ratio:.1f}:1, "
                            f"exceeding limit of {max_compression_ratio}:1. "
                            "This may indicate a ZIP bomb attack."
                        )

                # Early exit if total size exceeds limit
                if total_uncompressed > max_decompressed_size:
                    size_mb = total_uncompressed / (1024 * 1024)
                    limit_mb = max_decompressed_size / (1024 * 1024)
                    raise ZipBombError(
                        f"ZIP archive decompressed size ({size_mb:.1f} MB) exceeds "
                        f"limit of {limit_mb:.1f} MB. This may indicate a ZIP bomb attack."
                    )

            # Final check on overall compression ratio
            if compressed_size > 0:
                overall_ratio = total_uncompressed / compressed_size
                if overall_ratio > max_compression_ratio:
                    raise ZipBombError(
                        f"ZIP archive overall compression ratio {overall_ratio:.1f}:1 "
                        f"exceeds limit of {max_compression_ratio}:1. "
                        "This may indicate a ZIP bomb attack."
                    )

            logger.debug(
                f"ZIP safety check passed: {file_count} files, "
                f"{total_uncompressed / (1024 * 1024):.2f} MB uncompressed, "
                f"ratio {total_uncompressed / compressed_size:.1f}:1"
            )

    except zipfile.BadZipFile as e:
        raise ValueError(f"Invalid ZIP archive: {e}") from e


class ExtractionResult:
    """
    Result of text extraction with metadata.
    """

    def __init__(
            self,
            text: str,
            success: bool = True,
            method: str = None,
            metadata: Optional[Dict[str, Any]] = None,
            error: Optional[str] = None
    ):
        self.text = text
        self.success = success
        self.method = method
        self.metadata = metadata or {}
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'text': self.text,
            'extraction_success': self.success,
            'extraction_method': self.method,
            'metadata': self.metadata,
            'error': self.error,
            'text_length': len(self.text)
        }


class BaseExtractor(ABC):
    """
    Abstract base class for file content extractors.
    """

    def __init__(self, max_file_size_bytes: Optional[int] = None):
        """
        Initialize the extractor.

        Parameters
        ----------
        max_file_size_bytes : Optional[int]
            Maximum file size in bytes. If None, uses the module default MAX_FILE_SIZE_BYTES.
            Set to 0 to disable file size checking (not recommended for production).
        """
        self._is_available = self._check_availability()
        self.name = self.__class__.__name__
        self._max_file_size = max_file_size_bytes if max_file_size_bytes is not None else MAX_FILE_SIZE_BYTES

    @property
    def available(self) -> bool:
        return self._is_available

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum file size this extractor will process."""
        return self._max_file_size

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of file extensions this extractor supports (with dots, e.g., ['.pdf'])."""
        pass

    @property
    @abstractmethod
    def supported_mimetypes(self) -> List[str]:
        """List of MIME types this extractor supports."""
        pass

    @property
    @abstractmethod
    def required_packages(self) -> List[str]:
        """List of Python packages required for this extractor."""
        pass

    @property
    @abstractmethod
    def priority(self) -> int:
        """Priority level for this extractor (higher = preferred). Default should be 10."""
        pass

    @property
    @abstractmethod
    def metadata_schema(self) -> dict[str, MetadataField]:
        pass

    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if required dependencies are available."""
        pass

    @abstractmethod
    def _extract_text_impl(
            self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs
    ) -> ExtractionResult:
        """
        Implementation-specific text extraction.

        Parameters
        ----------
        file_content : bytes
            Raw file content
        filename : str
            Original filename
        mimetype : Optional[str]
            MIME type hint
        kwargs
            Optional keyword arguments

        Returns
        -------
        ExtractionResult
            Extraction result with text and metadata
        """
        pass

    def can_extract(self, filename: str, mimetype: Optional[str] = None) -> bool:
        """
        Check if this extractor can handle the given file.

        Parameters
        ----------
        filename : str
            Filename to check
        mimetype : Optional[str]
            MIME type hint

        Returns
        -------
        bool
            True if this extractor can handle the file
        """
        if not self.available:
            return False

        extension = Path(filename).suffix.lower()

        # Check extension
        if extension in self.supported_extensions:
            return True

        # Check MIME type if provided
        if mimetype and mimetype in self.supported_mimetypes:
            return True

        return False

    def extract_text(
            self, file_content: bytes, filename: str, mimetype: Optional[str] = None, **kwargs
    ) -> ExtractionResult:
        """
        Extract text from file content.

        Parameters
        ----------
        file_content : bytes
            Raw file content
        filename : str
            Original filename
        mimetype : Optional[str]
            MIME type hint
        kwargs
            Optional keyword args passed to the extractor.

        Returns
        -------
        ExtractionResult
            Extraction result
        """
        if not self.available:
            return ExtractionResult(
                text="",
                success=False,
                method=self.name,
                error=f"Required packages not available: {', '.join(self.required_packages)}"
            )

        if not self.can_extract(filename, mimetype):
            return ExtractionResult(
                text="",
                success=False,
                method=self.name,
                error=f"File type not supported by {self.name}"
            )

        # Check file size limit to prevent DoS via memory exhaustion
        file_size = len(file_content)
        if self._max_file_size > 0 and file_size > self._max_file_size:
            size_mb = file_size / (1024 * 1024)
            limit_mb = self._max_file_size / (1024 * 1024)
            logger.warning(
                f"File '{filename}' rejected: size {size_mb:.2f} MB exceeds limit of {limit_mb:.2f} MB"
            )
            return ExtractionResult(
                text="",
                success=False,
                method=self.name,
                error=f"File size ({size_mb:.2f} MB) exceeds maximum allowed size ({limit_mb:.2f} MB)"
            )

        try:
            return self._extract_text_impl(file_content, filename, mimetype, **kwargs)
        except Exception as e:
            logger.error(f"Error in {self.name} extraction: {e}")
            return ExtractionResult(
                text="",
                success=False,
                method=self.name,
                error=str(e)
            )

    def get_info(self) -> Dict[str, Any]:
        """Get information about this extractor."""
        return {
            'name': self.name,
            'available': self.available,
            'supported_extensions': self.supported_extensions,
            'supported_mimetypes': self.supported_mimetypes,
            'required_packages': self.required_packages,
            'priority': self.priority,
            'max_file_size_bytes': self._max_file_size
        }


class ExtractorRegistry:
    """
    Registry for file content extractors.
    """

    _plugins_discovered = False
    _extractors: Dict[str, BaseExtractor] = {}

    @classmethod
    def register(cls, extractor: Type[BaseExtractor]) -> None:
        """Register a new extractor."""
        extractor_obj = extractor()
        cls._extractors[extractor_obj.name] = extractor_obj
        logger.debug(f"Registered extractor: {extractor_obj.name} (available: {extractor_obj.available})")

    @classmethod
    def get_extractor(cls, name: str) -> Optional[BaseExtractor]:
        """Get an extractor by name."""
        return cls._extractors.get(name)

    @classmethod
    def list_extractors(cls, available_only: bool = True) -> List[str]:
        """List all registered extractors."""
        cls._discover_plugins()
        if available_only:
            return [name for name, extractor in cls._extractors.items() if extractor.available]
        return list(cls._extractors.keys())

    @classmethod
    def refresh_plugins(cls):
        """Force re-discovery of plugins (useful for testing)"""
        cls._plugins_discovered = False
        cls._discover_plugins()

    @classmethod
    def _discover_plugins(cls):
        """Discover file extractor plugins using entry points"""
        if cls._plugins_discovered:
            return
        from importlib.metadata import entry_points

        # Look for entry points in the 'localvectordb.embedding_providers' group
        extractor_eps = entry_points(group='localvectordb.file_extractors')

        for ep in extractor_eps:
            try:
                provider_class = ep.load()
                cls.register(provider_class)
                logger.info(f"Discovered file extractor plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load file extractor plugin {ep.name}: {e}")

        cls._plugins_discovered = True

    @classmethod
    def get_extractors_for_file(cls, filename: str, mimetype: Optional[str] = None) -> List[BaseExtractor]:
        """
        Get suitable extractors for a file, sorted by priority.

        Parameters
        ----------
        filename : str
            Filename to check
        mimetype : Optional[str]
            MIME type hint

        Returns
        -------
        List[BaseExtractor]
            List of suitable extractors, sorted by priority (highest first)
        """
        cls._discover_plugins()

        suitable = []

        for extractor in cls._extractors.values():
            if extractor.can_extract(filename, mimetype):
                suitable.append(extractor)

        # Sort by priority (highest first)
        suitable.sort(key=lambda x: x.priority, reverse=True)
        return suitable

    @classmethod
    def extract_text(cls, file_content: bytes, filename: str, mimetype: Optional[str] = None) -> ExtractionResult:
        """
        Extract text using the best available extractor.

        Parameters
        ----------
        file_content : bytes
            Raw file content
        filename : str
            Original filename
        mimetype : Optional[str]
            MIME type hint

        Returns
        -------
        ExtractionResult
            Extraction result from the best available extractor
        """
        extractors = cls.get_extractors_for_file(filename, mimetype)

        if not extractors:
            # No specific extractor found, try fallback
            fallback = cls._extractors.get('TextFallbackExtractor')
            if fallback and fallback.available:
                return fallback.extract_text(file_content, filename, mimetype)

            return ExtractionResult(
                text="",
                success=False,
                method="none",
                error=f"No suitable extractor found for file: {filename}"
            )

        # Try extractors in priority order
        last_error = None
        for extractor in extractors:
            result = extractor.extract_text(file_content, filename, mimetype)
            if result.success:
                return result
            last_error = result.error

        # All extractors failed
        return ExtractionResult(
            text="",
            success=False,
            method="failed",
            error=f"All extractors failed. Last error: {last_error}"
        )

    @classmethod
    def get_supported_formats(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all supported formats."""
        cls._discover_plugins()
        formats = {}

        for extractor in cls._extractors.values():
            if not extractor.available:
                continue

            for ext in extractor.supported_extensions:
                ext_key = ext.lstrip('.')
                if ext_key not in formats:
                    formats[ext_key] = {
                        'extensions': [ext],
                        'mimetypes': [],
                        'extractors': [],
                        'available': False
                    }

                formats[ext_key]['extractors'].append({
                    'name': extractor.name,
                    'priority': extractor.priority,
                    'required_packages': extractor.required_packages
                })
                formats[ext_key]['available'] = True

                # Add MIME types
                for mimetype in extractor.supported_mimetypes:
                    if mimetype not in formats[ext_key]['mimetypes']:
                        formats[ext_key]['mimetypes'].append(mimetype)

        # Sort extractors by priority for each format
        for format_info in formats.values():
            format_info['extractors'].sort(key=lambda x: x['priority'], reverse=True)

        return formats


def get_extractor_registry():
    """Get the global extractor registry."""
    return ExtractorRegistry


def get_supported_formats() -> Dict[str, Any]:
    """Get currently supported file formats."""
    registry = get_extractor_registry()
    return registry.get_supported_formats()
