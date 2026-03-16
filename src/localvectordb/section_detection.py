# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/section_detection.py
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
from typing import Dict, List

from localvectordb.core import Chunk, SectionBoundary

logger = logging.getLogger(__name__)

DEFAULT_SECTION_PATTERN = r"^(#{1,6})\s+(.+)$"


class SectionDetector:
    """Detects section boundaries in documents using regex patterns.

    By default, detects markdown-style headers (# through ######).
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

        matches = list(self._compiled.finditer(text))

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
                end_line = text[:first_match_start].count("\n") + 1
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
            end_line = text[:end_pos].count("\n") + 1

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
