"""
all2md-backed file content extractor.

This single extractor replaces the former per-format extractors (DOCX, PPTX,
XLSX, PDF, HTML, XML, EPUB, RTF, plain text) by delegating to the `all2md`
library, which converts 20+ document formats and 200+ source/text formats to
Markdown.

The extractor is dependency-aware: the set of extensions/mimetypes it reports as
supported reflects which all2md format parsers actually have their optional
dependencies installed (via ``all2md.registry.check_dependencies``). Formats
whose extras are missing are simply not advertised, so the registry will fall
back to any other registered extractor.

Output is Markdown (not plain text). This preserves document structure
(headings, tables, lists), which downstream chunking/section detection can
exploit for better boundaries.

Security
--------
Conversion of untrusted uploads is hardened by default:

* Remote document/asset fetching is disabled (no SSRF surface).
* Local ``file://`` access is disabled.
* For HTML, dangerous elements (scripts, event handlers) are stripped and
  attachments are skipped.
* The base-class file-size guard and a ZIP-bomb guard (for ZIP-based formats
  such as DOCX/XLSX/PPTX/EPUB/ODF) run before content is handed to all2md.

These defaults can be overridden per call via keyword arguments (plumbed from
server configuration); see :meth:`All2MdExtractor._build_parser_options`.
"""

import logging
from typing import Any, Dict, List, Optional, cast

from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.extractors import (
    BaseExtractor,
    ExtractionResult,
    ZipBombError,
    validate_zip_safety,
)

logger = logging.getLogger(__name__)

# ZIP local-file-header magic; ZIP-based formats (docx/xlsx/pptx/epub/odf/zip)
# all start with this. Used to decide when to run the ZIP-bomb guard.
_ZIP_MAGIC = b"PK\x03\x04"

# all2md format names that we do not want to advertise for ingestion even when
# their parser is importable. ``ast`` round-trips all2md's own JSON AST and has
# no meaning as an uploaded document format.
_EXCLUDED_FORMATS = frozenset({"ast"})


class All2MdExtractor(BaseExtractor):
    """Extractor that delegates to the all2md library for all supported formats."""

    # Lower than a typical custom plugin (which can use >10) so user-registered
    # extractors override all2md for the formats they claim, while all2md
    # remains the catch-all default for everything else.
    _PRIORITY = 10

    def __init__(self, max_file_size_bytes: Optional[int] = None):
        # Index populated lazily on first access; building it requires probing
        # every all2md format's optional dependencies, which we only want to do
        # once. Holds the dep-available extensions, mimetypes, and an
        # extension -> format-name map (highest-priority format wins per ext).
        self._format_index: Optional[Dict[str, Any]] = None
        super().__init__(max_file_size_bytes=max_file_size_bytes)

    # ------------------------------------------------------------------ #
    # Availability
    # ------------------------------------------------------------------ #
    def _check_availability(self) -> bool:
        import importlib.util

        return importlib.util.find_spec("all2md") is not None

    @property
    def required_packages(self) -> List[str]:
        return ["all2md"]

    @property
    def priority(self) -> int:
        return self._PRIORITY

    # ------------------------------------------------------------------ #
    # Dependency-aware format discovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_ext(ext: str) -> str:
        ext = ext.lower()
        return ext if ext.startswith(".") else f".{ext}"

    def _build_format_index(self) -> Dict[str, Any]:
        """Probe every all2md format and index the ones whose parser deps exist.

        Returns a dict with ``extensions`` (sorted list), ``mimetypes`` (sorted
        list), and ``ext_to_format`` (extension -> format name, highest-priority
        format winning when several claim the same extension).
        """
        if not self.available:
            return {"extensions": [], "mimetypes": [], "ext_to_format": {}}

        from all2md import registry

        exts: set[str] = set()
        mimes: set[str] = set()
        # ext -> (priority, format_name); keep the highest-priority claimant.
        ext_to_format: Dict[str, tuple[int, str]] = {}

        for fmt in registry.list_formats():
            if fmt in _EXCLUDED_FORMATS:
                continue
            try:
                missing = registry.check_dependencies(fmt, operation="parse")
            except Exception:  # pragma: no cover - defensive; treat as unavailable
                continue
            # check_dependencies returns a mapping of category -> [missing pkgs];
            # an empty mapping (or all-empty values) means the parser is usable.
            if missing and any(pkgs for pkgs in missing.values()):
                continue
            for info in registry.get_format_info(fmt) or []:
                fmt_name = getattr(info, "format_name", fmt)
                priority = int(getattr(info, "priority", 0) or 0)
                for raw_ext in getattr(info, "extensions", []) or []:
                    ext = self._normalize_ext(raw_ext)
                    exts.add(ext)
                    current = ext_to_format.get(ext)
                    if current is None or priority > current[0]:
                        ext_to_format[ext] = (priority, fmt_name)
                for mime in getattr(info, "mime_types", []) or []:
                    mimes.add(mime)

        return {
            "extensions": sorted(exts),
            "mimetypes": sorted(mimes),
            "ext_to_format": {ext: name for ext, (_prio, name) in ext_to_format.items()},
        }

    def _index(self) -> Dict[str, Any]:
        if self._format_index is None:
            self._format_index = self._build_format_index()
        return self._format_index

    @property
    def supported_extensions(self) -> List[str]:
        return list(self._index()["extensions"])

    @property
    def supported_mimetypes(self) -> List[str]:
        return list(self._index()["mimetypes"])

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        # all2md returns a free-form metadata dict whose keys vary by format.
        # We declare a generous superset of common fields; the upload router and
        # ingest path only persist metadata keys that exist in the target DB
        # schema, so unknown keys are harmless.
        text = lambda indexed=False: MetadataField(  # noqa: E731
            type=MetadataFieldType.TEXT, indexed=indexed, required=False
        )
        integer = MetadataField(type=MetadataFieldType.INTEGER, indexed=False, required=False)
        return {
            "filename": text(indexed=True),
            "source_format": text(indexed=True),
            "title": text(),
            "author": text(),
            "language": text(),
            "created": text(),
            "modified": text(),
            "page_count": integer,
            "word_count": integer,
            "file_size_bytes": integer,
            "character_count": integer,
        }

    # ------------------------------------------------------------------ #
    # Security options
    # ------------------------------------------------------------------ #
    def _build_parser_options(self, source_format: str, **kwargs: Any) -> Any:
        """Build hardened, format-specific parser options.

        Safe defaults are applied for untrusted input. Callers (e.g. the server,
        from configuration) may override via keyword arguments:

        * ``allow_remote_fetch`` (bool, default False) - allow fetching remote
          assets referenced by the document.
        * ``allowed_hosts`` (list[str] | None) - host allowlist when remote
          fetching is enabled.
        * ``strip_dangerous_elements`` (bool, default True) - HTML only; strip
          scripts/event handlers.
        * ``attachment_mode`` (str, default "skip") - how attachments/embedded
          assets are handled.

        Returns ``None`` for formats that have no security-relevant options, in
        which case all2md's safe defaults apply.
        """
        allow_remote_fetch = bool(kwargs.get("allow_remote_fetch", False))
        allowed_hosts = kwargs.get("allowed_hosts")
        strip_dangerous = bool(kwargs.get("strip_dangerous_elements", True))
        attachment_mode = kwargs.get("attachment_mode", "skip")

        # HTML/MHTML carry the bulk of the attack surface (scripts, remote
        # assets, embedded resources), so they get explicit hardening.
        if source_format in ("html", "mhtml", "webarchive"):
            from all2md import LocalFileAccessOptions, NetworkFetchOptions
            from all2md.options import HtmlOptions

            return HtmlOptions(
                strip_dangerous_elements=strip_dangerous,
                attachment_mode=attachment_mode,
                network=NetworkFetchOptions(
                    allow_remote_fetch=allow_remote_fetch,
                    allowed_hosts=allowed_hosts,
                ),
                local_files=LocalFileAccessOptions(
                    allow_local_files=False,
                    allow_cwd_files=False,
                ),
            )

        return None

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    def _resolve_format(self, file_content: bytes, filename: str) -> Optional[str]:
        """Determine the all2md source format for a file.

        The filename extension takes precedence over content sniffing. all2md's
        content-based auto-detection uses magic bytes plus a heuristic that can
        misclassify extension-less markup (e.g. Markdown as plain text), so we
        prefer the explicit extension hint when we recognise it and only fall
        back to content detection for unknown/extension-less inputs.
        """
        from pathlib import Path

        ext = Path(filename).suffix.lower()
        # Handle common multi-part extensions (e.g. .tar.gz) by also probing the
        # full suffix chain against the known extension map.
        ext_to_format: Dict[str, str] = self._index()["ext_to_format"]
        if ext in ext_to_format:
            return ext_to_format[ext]
        suffixes = "".join(Path(filename).suffixes).lower()
        if suffixes in ext_to_format:
            return ext_to_format[suffixes]

        # Fall back to content-based detection for unknown extensions.
        from all2md import registry
        from all2md.exceptions import FormatError

        try:
            detected: str = registry.detect_format(file_content, hint=filename)
            return detected
        except FormatError:
            return None

    def _extract_text_impl(
        self, file_content: bytes, filename: str, mimetype: Optional[str], **kwargs: Any
    ) -> ExtractionResult:
        import all2md
        from all2md.exceptions import DependencyError, FormatError, ParsingError
        from all2md.utils.input_sources import NamedBytesIO

        # ZIP-bomb guard for ZIP-based container formats. We run our own check
        # before all2md opens the archive, mirroring the protection the former
        # native extractors provided.
        if file_content[:4] == _ZIP_MAGIC:
            try:
                validate_zip_safety(file_content)
            except ZipBombError as e:
                logger.warning(f"ZIP bomb detected in '{filename}': {e}")
                return ExtractionResult(
                    text="", success=False, method=self.name, error=f"ZIP bomb protection triggered: {e}"
                )
            except ValueError as e:
                return ExtractionResult(text="", success=False, method=self.name, error=f"Invalid archive: {e}")

        # Resolve the format (extension first, content sniffing as fallback) so
        # we can build matching, hardened parser options and pass an explicit
        # hint to all2md instead of relying on its auto-detection heuristic.
        source_format = self._resolve_format(file_content, filename)
        if source_format is None:
            return ExtractionResult(
                text="", success=False, method=self.name, error=f"Unsupported or undetectable format: {filename}"
            )

        parser_options = self._build_parser_options(source_format, **kwargs)

        source = NamedBytesIO(file_content, name=filename)
        try:
            # Parse once to an AST (carries metadata), then render that AST to
            # Markdown - no second parse. source_format is a plain str here;
            # all2md types it as a Literal of format names but accepts any str.
            document = all2md.to_ast(source, source_format=cast(Any, source_format), parser_options=parser_options)
            markdown = all2md.to_markdown(document)
        except DependencyError as e:
            return ExtractionResult(
                text="",
                success=False,
                method=self.name,
                error=f"Missing optional dependency for '{source_format}': {e}",
            )
        except (ParsingError, FormatError) as e:
            return ExtractionResult(
                text="", success=False, method=self.name, error=f"Failed to convert '{source_format}': {e}"
            )

        if not markdown or not markdown.strip():
            return ExtractionResult(
                text="",
                success=False,
                method=f"{self.name}:{source_format}",
                error=f"No text content extracted from '{filename}'",
            )

        metadata = self._build_metadata(document, source_format, filename, file_content, markdown)
        return ExtractionResult(
            text=markdown,
            success=True,
            method=f"{self.name}:{source_format}",
            metadata=metadata,
        )

    @staticmethod
    def _build_metadata(
        document: Any, source_format: str, filename: str, file_content: bytes, markdown: str
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        doc_meta = getattr(document, "metadata", None)
        if isinstance(doc_meta, dict):
            # Carry through all2md's document metadata (title, author, etc.).
            # Only flat, JSON-friendly scalars are kept; nested structures are
            # skipped to stay compatible with the metadata schema.
            for key, value in doc_meta.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
        metadata.update(
            {
                "filename": filename,
                "source_format": source_format,
                "file_size_bytes": len(file_content),
                "character_count": len(markdown),
            }
        )
        return metadata
