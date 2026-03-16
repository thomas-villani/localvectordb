# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/section_metadata.py
"""
Section metadata extractors for hierarchical document embeddings.

Extractors automatically populate section metadata during ingestion.
Built-in extractors are text-based (no external dependencies).
Users can provide custom extractors by subclassing SectionMetadataExtractor.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union


class SectionMetadataExtractor(ABC):
    """Base class for section metadata extractors."""

    name: str = ""
    requires_llm: bool = False

    @abstractmethod
    def extract(self, section_text: str, heading: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a section.

        Parameters
        ----------
        section_text : str
            The full text of the section.
        heading : Optional[str]
            The section heading, if any.
        context : Dict[str, Any]
            Additional context (e.g., document metadata, other sections).

        Returns
        -------
        Dict[str, Any]
            Metadata key-value pairs to merge into section.metadata.
        """
        ...


class WordCountExtractor(SectionMetadataExtractor):
    """Extracts word count for a section."""

    name = "word_count"

    def extract(self, section_text: str, heading: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
        return {"word_count": len(section_text.split())}


class CharCountExtractor(SectionMetadataExtractor):
    """Extracts character count for a section."""

    name = "char_count"

    def extract(self, section_text: str, heading: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
        return {"char_count": len(section_text)}


class HeadingPathExtractor(SectionMetadataExtractor):
    """Extracts hierarchical heading path (e.g., 'Chapter 1 > Introduction > Background').

    Requires 'all_sections' key in context with list of (heading, heading_level) tuples.
    """

    name = "heading_path"

    def extract(self, section_text: str, heading: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
        all_sections = context.get("all_sections", [])
        current_index = context.get("section_index", 0)

        if heading is None:
            return {"heading_path": ""}

        current_level = context.get("heading_level", 1)

        # Walk backward through sections to build the path
        path_parts = []
        target_level = current_level
        for i in range(current_index, -1, -1):
            if i < len(all_sections):
                s_heading, s_level = all_sections[i]
                if s_heading is not None and s_level is not None:
                    if s_level < target_level:
                        path_parts.append(s_heading)
                        target_level = s_level
                    elif i == current_index:
                        path_parts.append(s_heading)

        path_parts.reverse()
        return {"heading_path": " > ".join(path_parts)}


class KeywordsExtractor(SectionMetadataExtractor):
    """Extracts top-N keywords from a section using simple word frequency.

    Parameters
    ----------
    top_n : int
        Number of top keywords to extract. Default: 10.
    min_word_length : int
        Minimum word length to consider. Default: 3.
    """

    name = "keywords"

    def __init__(self, top_n: int = 10, min_word_length: int = 3):
        self.top_n = top_n
        self.min_word_length = min_word_length
        # Common English stop words
        self._stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "has",
            "have",
            "been",
            "from",
            "will",
            "with",
            "they",
            "this",
            "that",
            "each",
            "which",
            "their",
            "said",
            "them",
            "than",
            "its",
            "into",
            "more",
            "other",
            "some",
            "very",
            "when",
            "come",
            "could",
            "now",
            "would",
            "make",
            "like",
            "just",
            "over",
            "such",
            "also",
            "about",
            "know",
            "most",
            "only",
            "then",
            "these",
            "being",
            "does",
            "what",
            "there",
            "where",
            "how",
        }

    def extract(self, section_text: str, heading: Optional[str], context: Dict[str, Any]) -> Dict[str, Any]:
        words = re.findall(r"\b[a-zA-Z]+\b", section_text.lower())
        filtered = [w for w in words if len(w) >= self.min_word_length and w not in self._stop_words]
        counter = Counter(filtered)
        top_keywords = [word for word, _ in counter.most_common(self.top_n)]
        return {"keywords": top_keywords}


# Registry of built-in extractors
BUILTIN_EXTRACTORS: Dict[str, type] = {
    "word_count": WordCountExtractor,
    "char_count": CharCountExtractor,
    "heading_path": HeadingPathExtractor,
    "keywords": KeywordsExtractor,
}


def resolve_extractors(
    extractors: Optional[List[Union[str, SectionMetadataExtractor]]],
) -> List[SectionMetadataExtractor]:
    """Resolve a list of extractor names/instances to extractor instances.

    Parameters
    ----------
    extractors : Optional[List[Union[str, SectionMetadataExtractor]]]
        List of extractor names (for built-ins) or instances.

    Returns
    -------
    List[SectionMetadataExtractor]
        List of ready-to-use extractor instances.
    """
    if not extractors:
        return []

    resolved = []
    for ext in extractors:
        if isinstance(ext, str):
            if ext not in BUILTIN_EXTRACTORS:
                raise ValueError(
                    f"Unknown section metadata extractor: '{ext}'. " f"Available: {list(BUILTIN_EXTRACTORS.keys())}"
                )
            resolved.append(BUILTIN_EXTRACTORS[ext]())
        elif isinstance(ext, SectionMetadataExtractor):
            resolved.append(ext)
        else:
            raise TypeError(f"Expected str or SectionMetadataExtractor, got {type(ext)}")
    return resolved
