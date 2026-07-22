"""
Section detection for hierarchical document embeddings.

This module provides a SectionDetector that identifies section boundaries
in documents based on configurable patterns (e.g., markdown headers).
Sections are an overlay on top of existing chunking - they group chunks
by document structure for mid-level retrieval.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import re
from typing import Dict, List, Optional

from localvectordb.core import Chunk, SectionBoundary

logger = logging.getLogger(__name__)

DEFAULT_SECTION_PATTERN = r"^(#{1,6})\s+(.+)$"

# A fence line is up to three leading spaces followed by 3+ backticks or 3+
# tildes. An *opening* fence may carry an info string (e.g. ```python); a
# *closing* fence is the same character, at least as long, and bare. This
# matters now that extracted content is Markdown: a ``#`` comment or shell
# prompt inside a fenced code block must not be mistaken for a section header.
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")


def find_code_fence_spans(text: str) -> List[tuple[int, int]]:
    r"""Return ``(start, end)`` character spans covered by fenced code blocks.

    Both backtick (:literal:`\`\`\``) and tilde (``~~~``) fences are recognised,
    including unterminated fences (which extend to end of text) following
    CommonMark behaviour. Spans are returned in document order and are used to
    suppress header matches that fall inside code, so example Markdown or shell
    snippets in extracted documents do not create spurious section boundaries.
    """
    spans: List[tuple[int, int]] = []
    open_fence: Optional[str] = None
    open_start = 0
    pos = 0

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n").rstrip("\r")
        if open_fence is None:
            m = _FENCE_OPEN_RE.match(stripped)
            if m:
                open_fence = m.group(1)
                open_start = pos
        else:
            cm = _FENCE_CLOSE_RE.match(stripped)
            # A closing fence must use the same character and be at least as long.
            if cm and cm.group(1)[0] == open_fence[0] and len(cm.group(1)) >= len(open_fence):
                spans.append((open_start, pos + len(line)))
                open_fence = None
        pos += len(line)

    if open_fence is not None:
        spans.append((open_start, len(text)))

    return spans


def _position_in_spans(position: int, spans: List[tuple[int, int]]) -> bool:
    """Return True if ``position`` falls within any ``(start, end)`` span."""
    return any(start <= position < end for start, end in spans)


class SectionDetector:
    """Detects section boundaries in documents using regex patterns.

    By default, detects markdown-style headers (# through ######). Headers
    inside fenced code blocks are ignored, so example snippets in extracted
    Markdown do not create spurious sections.
    Custom patterns can be provided for other document formats.

    Parameters
    ----------
    pattern : str
        Regex pattern for detecting section headers. Must use MULTILINE mode.
        The pattern should have two capture groups:
        - Group 1: heading level indicator (e.g., '#' characters)
        - Group 2: heading text
    """

    def __init__(self, pattern: str = DEFAULT_SECTION_PATTERN):
        self.pattern = pattern
        self._compiled = re.compile(pattern, re.MULTILINE)

    def detect_sections(self, text: str) -> List[SectionBoundary]:
        """Detect section boundaries in the given text.

        Parameters
        ----------
        text : str
            The full document text to scan for sections.

        Returns
        -------
        List[SectionBoundary]
            List of detected section boundaries, ordered by position.
            Text before the first header becomes a "preamble" section
            (index 0, heading=None).
        """
        if not text:
            return []

        # Exclude headers that live inside fenced code blocks: extracted content
        # is now Markdown, so a ``# comment`` in a ```` ``` ```` block is text,
        # not a section boundary.
        fence_spans = find_code_fence_spans(text)
        matches = [m for m in self._compiled.finditer(text) if not _position_in_spans(m.start(), fence_spans)]

        if not matches:
            # No headers found: entire document is one section
            start_line = 1
            end_line = text.count("\n") + 1
            return [
                SectionBoundary(
                    index=0,
                    heading=None,
                    heading_level=None,
                    start_pos=0,
                    end_pos=len(text),
                    start_line=start_line,
                    end_line=end_line,
                )
            ]

        sections: List[SectionBoundary] = []
        section_index = 0

        # Check for preamble (text before first header)
        first_match_start = matches[0].start()
        if first_match_start > 0:
            preamble_text = text[:first_match_start].strip()
            if preamble_text:
                start_line = 1
                # end_pos is the next header's start, so the raw slice runs to
                # the blank line before that header; report the last line that
                # actually holds preamble content, not the header's own line.
                end_line = text[:first_match_start].rstrip().count("\n") + 1
                sections.append(
                    SectionBoundary(
                        index=section_index,
                        heading=None,
                        heading_level=None,
                        start_pos=0,
                        end_pos=first_match_start,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )
                section_index += 1

        # Process each header match
        for i, match in enumerate(matches):
            level_indicator = match.group(1)
            heading_text = match.group(2).strip()

            # Determine heading level from the indicator
            heading_level = len(level_indicator)

            start_pos = match.start()

            # Section ends at the next header or end of text
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)

            start_line = text[:start_pos].count("\n") + 1
            # end_pos is the *next* header's start (kept contiguous for chunk
            # assignment), so deriving end_line straight from it overshoots into
            # the next header's line. Count lines within this section's own body,
            # ignoring the trailing blank lines that lead up to the next header.
            end_line = start_line + text[start_pos:end_pos].rstrip().count("\n")

            sections.append(
                SectionBoundary(
                    index=section_index,
                    heading=heading_text,
                    heading_level=heading_level,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            section_index += 1

        return sections

    @staticmethod
    def assign_chunks_to_sections(
        chunks: List[Chunk],
        sections: List[SectionBoundary],
    ) -> Dict[int, List[int]]:
        """Map each chunk to its containing section by position overlap.

        Uses binary search for efficiency with large numbers of chunks/sections.

        Parameters
        ----------
        chunks : List[Chunk]
            The chunks to assign to sections.
        sections : List[SectionBoundary]
            The detected section boundaries.

        Returns
        -------
        Dict[int, List[int]]
            Mapping from section index to list of chunk indices (chunk.index values).
        """
        if not chunks or not sections:
            return {}

        # Build sorted list of section start positions for binary search
        section_starts = [s.start_pos for s in sections]
        section_map: Dict[int, List[int]] = {s.index: [] for s in sections}

        for chunk in chunks:
            chunk_mid = (chunk.position.start + chunk.position.end) // 2

            # Binary search: find the rightmost section whose start_pos <= chunk_mid
            idx = bisect.bisect_right(section_starts, chunk_mid) - 1
            if idx < 0:
                idx = 0

            section = sections[idx]
            section_map[section.index].append(chunk.index)

        return section_map

    @staticmethod
    def compute_section_content_hash(text: str, section: SectionBoundary) -> str:
        """Compute SHA-256 hash of a section's content."""
        section_text = text[section.start_pos : section.end_pos]
        return hashlib.sha256(section_text.encode("utf-8")).hexdigest()
