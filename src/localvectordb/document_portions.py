"""Sub-document ("portion") retrieval shared by the CLI and MCP surfaces.

The core :class:`~localvectordb.database.LocalVectorDB` exposes whole-document
:meth:`get` and per-chunk :meth:`get_chunks`. This module builds the higher-level
"give me *part* of a document" operations on top of those two primitives so that
every surface (the ``lvdb ... get`` CLI command and the MCP ``get_document`` tool)
shares one implementation instead of duplicating the slicing logic.

Selection modes (mutually exclusive):

* ``chunk``   -- one or more stored chunks, by 0-based index or inclusive range.
* ``char_range`` -- a character slice ``M:N`` (0-based, end-exclusive).
* ``line_range`` -- a line range ``M:N`` (1-based, inclusive).
* ``section`` -- the body of the Markdown section whose heading matches a name.
* ``outline`` -- the document's section outline (headings, levels, start lines).

When no mode is selected the whole document is returned.

All user-input problems (a malformed range, an unknown section, an empty chunk
selection) raise :class:`ValueError` with a human-readable message; callers
translate that into their own error channel (a CLI exit, an MCP error dict).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from localvectordb.section_detection import SectionDetector

if TYPE_CHECKING:
    from localvectordb.core import Document


def parse_range_spec(spec: str, *, allow_single: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """Parse a ``"M:N"`` range specification into a ``(start, end)`` tuple.

    Accepts ``"M:N"``, open-ended ``"M:"`` / ``":N"`` / ``":"`` (``None`` marks an
    open end), and -- when ``allow_single`` is true -- a bare ``"M"`` meaning the
    single value ``(M, M)``.

    Parameters
    ----------
    spec : str
        The range specification to parse.
    allow_single : bool, keyword-only
        When true, a bare integer (no ``":"``) parses to ``(M, M)``. When false,
        a bare integer is rejected.

    Returns
    -------
    tuple[int | None, int | None]
        The ``(start, end)`` bounds; ``None`` marks an open end.

    Raises
    ------
    ValueError
        If ``spec`` is empty, contains a non-integer part, has more than one
        ``":"``, or is a bare value while ``allow_single`` is false.
    """
    if spec is None or spec.strip() == "":
        raise ValueError("range specification is empty")
    raw = spec.strip()

    def _to_int(part: str, label: str) -> Optional[int]:
        part = part.strip()
        if part == "":
            return None
        try:
            return int(part)
        except ValueError as e:
            raise ValueError(f"{label} of range must be an integer, got {part!r}") from e

    def _reject_negative(value: Optional[int], label: str) -> None:
        # Positions are non-negative (0-based chars/chunks, 1-based lines). A
        # negative bound would otherwise do a silent Python-negative slice
        # (``content[-3:]``) instead of the wrong-input error the caller expects.
        if value is not None and value < 0:
            raise ValueError(f"{label} of range must be non-negative, got {value}")

    if ":" not in raw:
        if not allow_single:
            raise ValueError(f"expected a range 'M:N', got {raw!r}")
        value = _to_int(raw, "value")
        if value is None:  # pragma: no cover - guarded by the empty check above
            raise ValueError("range specification is empty")
        _reject_negative(value, "value")
        return value, value

    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError(f"range must contain a single ':', got {raw!r}")
    start, end = _to_int(parts[0], "start"), _to_int(parts[1], "end")
    _reject_negative(start, "start")
    _reject_negative(end, "end")
    # A reversed range ("5:2") would silently slice to the empty string; reject
    # it so the caller sees a real error instead of an empty result.
    if start is not None and end is not None and start > end:
        raise ValueError(f"range start ({start}) must not exceed end ({end})")
    return start, end


def _slice_lines(content: str, start: Optional[int], end: Optional[int]) -> str:
    """Return the 1-based, inclusive line range ``start:end`` of ``content``."""
    lines = content.splitlines(keepends=True)
    lo = 1 if start is None else max(start, 1)
    hi = len(lines) if end is None else end
    return "".join(lines[lo - 1 : hi])


def _select_section(content: str, name: str) -> str:
    """Return the body of the section whose heading matches ``name``.

    Sections are detected on the fly from ``content`` (Markdown headers,
    code-fence aware), so this works regardless of whether hierarchical
    embeddings were enabled at ingest.

    Raises
    ------
    ValueError
        When no heading matches ``name`` (the message lists the available
        headings when there are any).
    """
    boundaries = SectionDetector().detect_sections(content)
    target = name.strip().lower()
    for b in boundaries:
        if b.heading is not None and b.heading.strip().lower() == target:
            return content[b.start_pos : b.end_pos]

    headings = [b.heading for b in boundaries if b.heading]
    if headings:
        available = ", ".join(repr(h) for h in headings)
        raise ValueError(f"Section {name!r} not found. Available sections: {available}.")
    raise ValueError(f"Section {name!r} not found: document has no Markdown headings.")


def _build_outline(content: str) -> List[Dict[str, Any]]:
    """Build a flat outline (one entry per detected section) from ``content``."""
    return [
        {
            "index": b.index,
            "heading": b.heading,
            "level": b.heading_level,
            "start_line": b.start_line,
            "end_line": b.end_line,
        }
        for b in SectionDetector().detect_sections(content)
    ]


@dataclass
class DocumentPortion:
    """A resolved slice of a document, ready for text or JSON rendering.

    Attributes
    ----------
    document : Document
        The full source document (exposes ``id``, ``metadata``, timestamps).
    mode : str
        The selection mode that produced this portion: ``"document"``,
        ``"chunk"``, ``"range"``, ``"lines"``, ``"section"``, or ``"outline"``.
    label : str
        A short human-readable suffix describing the selection (e.g.
        ``"chunk 2:5"``), or ``""`` for the whole document.
    text : str | None
        The portion rendered as text. For ``"chunk"`` this is the selected
        chunks joined with blank lines; for ``"outline"`` it is ``None`` (use
        :attr:`outline` instead).
    chunks : list[dict] | None
        For ``"chunk"`` mode, one ``{"index", "content", "position"}`` entry per
        selected chunk; ``None`` otherwise.
    outline : list[dict] | None
        For ``"outline"`` mode, the section outline items; ``None`` otherwise.
    """

    document: "Document"
    mode: str
    label: str
    text: Optional[str]
    chunks: Optional[List[Dict[str, Any]]] = None
    outline: Optional[List[Dict[str, Any]]] = None

    @property
    def doc_id(self) -> str:
        return self.document.id


def _count_selected_modes(
    chunk: Optional[str],
    char_range: Optional[str],
    line_range: Optional[str],
    section: Optional[str],
    outline: bool,
) -> int:
    return sum(1 for v in (chunk, char_range, line_range, section, outline) if v)


def get_document_portion(
    db: Any,
    doc_id: str,
    *,
    chunk: Optional[str] = None,
    char_range: Optional[str] = None,
    line_range: Optional[str] = None,
    section: Optional[str] = None,
    outline: bool = False,
) -> DocumentPortion:
    """Retrieve a document, or a selected portion of it, as a :class:`DocumentPortion`.

    Exactly one selection mode may be active; passing more than one raises
    :class:`ValueError`. With no mode active the whole document is returned
    (``mode="document"``).

    Parameters
    ----------
    db : LocalVectorDB
        A database exposing ``get(doc_id)`` and ``get_chunks(doc_id)``.
    doc_id : str
        The document to retrieve.
    chunk : str, optional
        Chunk selector: a 0-based index (``"3"``) or inclusive range (``"2:5"``).
    char_range : str, optional
        Character slice ``"M:N"`` (0-based, end-exclusive; open ends allowed).
    line_range : str, optional
        Line range ``"M:N"`` (1-based, inclusive; open ends allowed).
    section : str, optional
        Name of the Markdown heading whose section body to return
        (case-insensitive).
    outline : bool, optional
        When true, return the document's section outline.

    Returns
    -------
    DocumentPortion

    Raises
    ------
    ValueError
        On more than one selection mode, a malformed range, an unknown section,
        or an empty/out-of-range chunk selection.
    localvectordb.exceptions.DocumentNotFoundError
        If ``doc_id`` does not exist (propagated from ``db.get``).
    """
    if _count_selected_modes(chunk, char_range, line_range, section, outline) > 1:
        raise ValueError("Only one of chunk, char_range, line_range, section, outline may be selected")

    doc = db.get(doc_id)
    if doc is None:
        from localvectordb.exceptions import DocumentNotFoundError

        raise DocumentNotFoundError(f"Document '{doc_id}' not found")

    content = doc.content

    if chunk is not None:
        start, end = parse_range_spec(chunk, allow_single=True)
        all_chunks = db.get_chunks(doc_id)
        if not all_chunks:
            raise ValueError(f"Document '{doc_id}' has no stored chunks")
        max_index = all_chunks[-1].index
        lo = 0 if start is None else start
        hi = max_index if end is None else end
        selected = [c for c in all_chunks if lo <= c.index <= hi]
        if not selected:
            raise ValueError(f"No chunks in range {chunk!r} (document has chunks 0..{max_index})")
        chunk_dicts = [{"index": c.index, "content": c.content, "position": c.position.to_dict()} for c in selected]
        return DocumentPortion(
            document=doc,
            mode="chunk",
            label=f"chunk {chunk}",
            text="\n\n".join(c.content for c in selected),
            chunks=chunk_dicts,
        )

    if char_range is not None:
        start, end = parse_range_spec(char_range)
        return DocumentPortion(
            document=doc,
            mode="range",
            label=f"chars {char_range}",
            text=content[start:end],
        )

    if line_range is not None:
        start, end = parse_range_spec(line_range)
        return DocumentPortion(
            document=doc,
            mode="lines",
            label=f"lines {line_range}",
            text=_slice_lines(content, start, end),
        )

    if section is not None:
        return DocumentPortion(
            document=doc,
            mode="section",
            label=f"section {section!r}",
            text=_select_section(content, section),
        )

    if outline:
        return DocumentPortion(
            document=doc,
            mode="outline",
            label="outline",
            text=None,
            outline=_build_outline(content),
        )

    return DocumentPortion(document=doc, mode="document", label="", text=content)
