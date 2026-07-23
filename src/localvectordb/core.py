"""
LocalVectorDB v1.0 Core Components

This module contains the foundational classes and data structures for the new
document-first architecture.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import numpy as np

from .utils import parse_iso8601

logger = logging.getLogger(__name__)

# Type alias for the document-score aggregation that occurs when querying
DocumentScoringMethod = Literal[
    "best",
    "average",
    "frequency_boost",
]
"""
Document scoring methods for aggregating chunk scores into a document score:

- "best": Highest chunk score (single best passage matters)
- "average": Mean of all chunk scores (overall quality)
- "frequency_boost": Boosts the best chunk score by the number of quality chunks
  (default, good for comprehensive docs)
  - frequency_bias (0.3): How much to boost based on chunk count
"""


def _adapt_datetime_with_tz(dt: datetime) -> str:
    return str(dt.isoformat())


def _convert_datetime_with_tz(dt) -> Optional[datetime]:
    """
    Convert a SQLite timestamp bytes value to a Python datetime.

    Returns None if the value cannot be parsed as a valid ISO8601 datetime,
    which can happen when SQLite type detection is overly aggressive.
    This maintains type consistency (always Optional[datetime]) rather than
    silently returning a string on parse failure.
    """
    s = dt.decode("utf-8")
    try:
        return parse_iso8601(s)
    except ValueError:
        # Log a warning for debugging - this indicates a schema/data mismatch
        logger.warning(
            "Failed to parse timestamp value '%s' as datetime. "
            "This may indicate SQLite type detection mismatch. Returning None.",
            s,
        )
        return None


def _adapt_json(json_data: Any) -> str:
    return str(json.dumps(json_data))


def _convert_json(json_data: bytes) -> dict[Any, Any] | list[Any]:
    result: dict[Any, Any] | list[Any] = json.loads(json_data.decode("utf-8"))
    return result


sqlite3.register_adapter(datetime, _adapt_datetime_with_tz)
sqlite3.register_converter("timestamp", _convert_datetime_with_tz)
sqlite3.register_adapter(dict, _adapt_json)
sqlite3.register_adapter(list, _adapt_json)
sqlite3.register_converter("json", _convert_json)


class MetadataFieldType(str, Enum):
    TEXT = "text"
    INTEGER = "integer"
    REAL = "real"
    BOOLEAN = "boolean"
    DATE = "date"
    JSON = "json"

    def valid_types(self) -> Tuple[Type[Any], ...]:
        if self.value == "text":
            return (str,)
        elif self.value == "integer":
            return (int,)
        elif self.value == "real":
            return (int, float)
        elif self.value == "boolean":
            return (bool, int)
        elif self.value == "date":
            return (datetime, str)
        elif self.value == "json":
            return (dict, list)
        return ()


@dataclass
class MetadataField:
    """
    Defines a metadata field for documents.

    Parameters
    ----------
    type : MetadataFieldType or str or Type
        The type of the metadata field.
    indexed : bool, optional
        Whether the field is indexed in the database, by default False.
    required : bool, optional
        Whether the field is required, by default False.
    default_value : Any, optional
        Default value for the field if not provided, by default None.
    embedding_enabled : bool, optional
        Whether this field should have its own embeddings for vector search.
        Only applicable to TEXT and JSON fields, by default False.
    fts_enabled : bool, optional
        Whether this field should have full-text search enabled.
        Only applicable to TEXT fields, by default False.

    """

    type: MetadataFieldType | str | Type
    indexed: bool = False
    required: bool = False
    default_value: Any = None
    embedding_enabled: bool = False
    fts_enabled: bool = False

    def __post_init__(self) -> None:
        """
        Post-initialization processing to resolve type into MetadataFieldType.

        Converts string or builtin types to corresponding MetadataFieldType.
        Validates embedding_enabled and fts_enabled based on field type.

        Returns
        -------
        None
        """
        if isinstance(self.type, str):
            self.type = MetadataFieldType(self.type)
        elif self.type is str:
            self.type = MetadataFieldType.TEXT
        elif self.type is int:
            self.type = MetadataFieldType.INTEGER
        elif self.type is float:
            self.type = MetadataFieldType.REAL
        elif self.type is bool:
            self.type = MetadataFieldType.BOOLEAN
        elif self.type in (dict, list):
            self.type = MetadataFieldType.JSON

        # Validate embedding_enabled - only TEXT and JSON fields can have embeddings
        if self.embedding_enabled and self.type not in (MetadataFieldType.TEXT, MetadataFieldType.JSON):
            raise ValueError(f"embedding_enabled can only be True for TEXT or JSON fields, not {self.type}")

        # Validate fts_enabled - only TEXT fields can have FTS
        if self.fts_enabled and self.type != MetadataFieldType.TEXT:
            raise ValueError(f"fts_enabled can only be True for TEXT fields, not {self.type}")

        # Auto-enable FTS for indexed TEXT fields if not explicitly set
        if self.type == MetadataFieldType.TEXT and self.indexed and not self.fts_enabled:
            self.fts_enabled = True


@dataclass
class ChunkPosition:
    """Exact position tracking for a chunk in the original document.

    Parameters
    ----------
    start : int
        Character start position in the original document.
    end : int
        Character end position in the original document.
    line : int
        Line number in the original document (1-based).
    column : int
        Column number in the original document (1-based).
    end_line : int
        Line number of end of chunk in original document (1-based)
    end_column : int
        Column number of end of chunk in original document (1-based)
    """

    start: int
    end: int  # Character position in original

    line: int  # Line number (1-based)
    column: int  # Column number (1-based)

    end_line: int
    end_column: int

    def to_dict(self) -> dict:
        """Convert the ChunkPosition to a dictionary.

        Returns
        -------
        dict
            Dictionary representation with keys 'start', 'end', 'line', 'column'.

        Examples
        --------
        >>> pos = ChunkPosition(start=0, end=10, line=1, column=1, end_line=1, end_column=10)
        >>> pos.to_dict()
        {'start': 0, 'end': 10, 'line': 1, 'column': 1, 'end_line': 1, 'end_column': 10}
        """
        return {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChunkPosition":
        """Create a ChunkPosition instance from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys 'start', 'end', 'line', 'column'.

        Returns
        -------
        ChunkPosition
            The constructed ChunkPosition object.

        Examples
        --------
        >>> data = {'start': 0, 'end': 10, 'line': 1, 'column': 1, 'end_line': 1, 'end_column': 10}
        >>> ChunkPosition.from_dict(data)
        ChunkPosition(start=0, end=10, line=1, column=1, end_line=1, end_column=10)
        """
        return cls(**data)


@dataclass
class Chunk:
    """Internal representation of a document chunk.

    Encapsulates the content, position metadata, token count, and
    optional FAISS index identifier for a text segment.

    Parameters
    ----------
    content : str
        The text content of the chunk.
    position : ChunkPosition
        The location of this chunk in the original document.
    tokens : int
        Number of tokens in this chunk.
    index : int
        Sequential index of the chunk within the document.
    faiss_id : int, optional
        Identifier in the FAISS index, if applicable.

    """

    content: str
    position: ChunkPosition
    tokens: int
    index: int  # Chunk index within document
    faiss_id: Optional[int] = None  # Maps to FAISS index position
    content_hash: Optional[str] = None  # SHA-256 hash of content

    def __post_init__(self) -> None:
        if self.content_hash is None:
            self.content_hash = self.calculate_content_hash()

    def calculate_content_hash(self) -> str:
        """Calculate SHA-256 hash of chunk content"""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def content_equals(self, other: "Chunk") -> bool:
        """Check if this chunk has the same content as another chunk"""
        return self.content_hash == other.content_hash

    def get_context(self, original: str, window: int = 100) -> str:
        """Get chunk with surrounding context from original document"""
        start = max(0, self.position.start - window)
        end = min(len(original), self.position.end + window)

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(original) else ""

        return f"{prefix}{original[start:end]}{suffix}"

    def highlight_in_original(self, original: str) -> str:
        """Return original text with chunk highlighted"""
        before = original[: self.position.start]
        chunk_text = original[self.position.start : self.position.end]
        after = original[self.position.end :]

        return f"{before}<<<{chunk_text}>>>{after}"


@dataclass
class SectionBoundary:
    """Boundary information for a detected section in a document.

    Used during ingestion to track where sections start and end in the original text.
    """

    index: int
    heading: Optional[str]
    heading_level: Optional[int]
    start_pos: int
    end_pos: int
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Section:
    """A section within a document, grouping multiple chunks.

    Sections are an overlay on top of existing chunking. They provide
    a mid-level abstraction between documents and chunks for hierarchical
    retrieval.
    """

    index: int
    heading: Optional[str]
    heading_level: Optional[int]
    start_pos: int
    end_pos: int
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    faiss_id: Optional[int] = None
    chunks: Optional[List[Chunk]] = None

    def __post_init__(self) -> None:
        if self.content_hash is None and self.start_pos is not None and self.end_pos is not None:
            # Content hash will be calculated from actual text during ingestion
            pass

    @classmethod
    def from_boundary(
        cls,
        boundary: SectionBoundary,
        content_hash: str,
        faiss_id: Optional[int] = None,
        chunks: Optional[List[Chunk]] = None,
    ) -> "Section":
        """Create a Section from a SectionBoundary."""
        return cls(
            index=boundary.index,
            heading=boundary.heading,
            heading_level=boundary.heading_level,
            start_pos=boundary.start_pos,
            end_pos=boundary.end_pos,
            start_line=boundary.start_line,
            end_line=boundary.end_line,
            content_hash=content_hash,
            metadata=boundary.metadata,
            faiss_id=faiss_id,
            chunks=chunks,
        )


@dataclass
class Document:
    """A document in the vector database"""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    content_hash: Optional[str] = None
    chunks: Optional[List[Chunk]] = None
    sections: Optional[List[Section]] = None

    def __post_init__(self) -> None:
        if self.content_hash is None:
            self.content_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of document content"""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def needs_update(self, new_content: str) -> bool:
        """Check if document content has changed"""
        new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        return new_hash != self.content_hash

    @classmethod
    def from_dict(cls, data: dict) -> Optional["Document"]:
        """Create a Document from a dictionary response"""
        if not data:
            return None

        # Parse datetime fields
        created_at = None
        if data.get("created_at"):
            created_at = parse_iso8601(data["created_at"])

        updated_at = None
        if data.get("updated_at"):
            updated_at = parse_iso8601(data["updated_at"])

        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
            content_hash=data.get("content_hash"),
            chunks=data.get("chunks", []),
        )


@dataclass
class GrepMatch:
    """A single lexical (grep-style) match within a document's content.

    Line-oriented, mirroring command-line ``grep``: each match records the 1-based
    line number, the full matching line, the column span of the match within that
    line, and optional context lines before/after.
    """

    doc_id: str
    line_number: int  # 1-based line number within the document
    line: str  # the full matching line (without its trailing newline)
    start: int  # 0-based column where the match starts within the line
    end: int  # 0-based column where the match ends (exclusive)
    match: str  # the matched substring
    before: List[str] = field(default_factory=list)  # preceding context lines
    after: List[str] = field(default_factory=list)  # following context lines

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "doc_id": self.doc_id,
            "line_number": self.line_number,
            "line": self.line,
            "start": self.start,
            "end": self.end,
            "match": self.match,
            "before": list(self.before),
            "after": list(self.after),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrepMatch":
        """Create a GrepMatch from a dictionary response."""
        return cls(
            doc_id=data["doc_id"],
            line_number=int(data["line_number"]),
            line=data["line"],
            start=int(data["start"]),
            end=int(data["end"]),
            match=data["match"],
            before=list(data.get("before", [])),
            after=list(data.get("after", [])),
        )


@dataclass
class PrefixEntry:
    """A single child of a document-id prefix.

    Either a virtual "folder" (a common prefix shared by one or more documents,
    ``is_prefix=True``) or a leaf document that lives directly at the queried
    level (``is_prefix=False``). Modelled on S3's ``CommonPrefixes`` / ``Contents``
    split -- there are no real directories, only document ids that share a prefix.
    """

    name: str  # segment relative to the queried prefix, e.g. "reports/" or "readme"
    path: str  # full id prefix, e.g. "docs/reports/" (folder) or "docs/readme" (document)
    is_prefix: bool  # True for a virtual folder, False for a leaf document
    count: int  # documents at or beneath this entry (1 for a leaf document)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {"name": self.name, "path": self.path, "is_prefix": self.is_prefix, "count": self.count}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrefixEntry":
        """Create a PrefixEntry from a dictionary response."""
        return cls(
            name=data["name"],
            path=data["path"],
            is_prefix=bool(data["is_prefix"]),
            count=int(data["count"]),
        )


@dataclass
class PrefixListing:
    """S3-style listing of the immediate children of a document-id prefix.

    ``prefixes`` holds the virtual sub-folders (common prefixes) and ``documents``
    holds the leaf documents directly at this level. Use the returned ``path`` of a
    prefix entry as the ``prefix`` argument of a subsequent call to descend.
    """

    prefix: str
    delimiter: str
    prefixes: List["PrefixEntry"] = field(default_factory=list)
    documents: List["PrefixEntry"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "prefix": self.prefix,
            "delimiter": self.delimiter,
            "prefixes": [e.to_dict() for e in self.prefixes],
            "documents": [e.to_dict() for e in self.documents],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrefixListing":
        """Create a PrefixListing from a dictionary response."""
        return cls(
            prefix=data.get("prefix", ""),
            delimiter=data.get("delimiter", "/"),
            prefixes=[PrefixEntry.from_dict(e) for e in data.get("prefixes", [])],
            documents=[PrefixEntry.from_dict(e) for e in data.get("documents", [])],
        )


@dataclass
class QueryResult:
    """Result from a search query"""

    id: str
    score: float  # Normalized 0-1, higher=better
    type: Literal["document", "chunk", "section", "context", "enriched", "group", "aggregation"]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Additional fields for chunks
    document_id: Optional[str] = None
    position: Optional[ChunkPosition] = None

    def get_context(self, original: str, window: int = 100) -> Optional[str]:
        """Get context around chunk (only for chunk results)"""
        if self.type == "chunk" and self.position:
            start = max(0, self.position.start - window)
            end = min(len(original), self.position.end + window)

            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(original) else ""

            return f"{prefix}{original[start:end]}{suffix}"
        return None

    @classmethod
    def from_dict(cls, data: dict) -> Optional["QueryResult"]:
        """Create a QueryResult from a dictionary response"""
        if not data:
            return None

        # Parse position if present
        position = None
        if data.get("position"):
            position = ChunkPosition.from_dict(data["position"])

        q_type = data.get("type", "document")
        valid_types = ("document", "chunk", "section", "context", "enriched", "group", "aggregation")
        if q_type not in valid_types:
            raise ValueError(f"`type` must be one of {valid_types}")

        return cls(
            id=data["id"],
            score=data.get("score", 0.0),
            type=q_type,
            content=data["content"],
            metadata=data.get("metadata", {}),
            document_id=data.get("document_id"),
            position=position,
        )


# Future: QueryResultList class with composable subfiltering (.filter, .limit, etc.)
# and conversion utilities (e.g., numpy array of embeddings).
#
# class QueryResultList(UserList):
#
#     def __init__(self, initlist: List[QueryResult] = None):
#         initlist = initlist or []
#         if initlist and not all(isinstance(a, QueryResult) for a in initlist):
#             raise TypeError("`initlist` must be a list of `QueryResult`")
#
#         super().__init__(initlist)
#
#
#     def filter(self,
#                field, value, operator
#                where: Optional[Dict[str, Any]] = None, order_by: Optional[str] = None, limit: Optional[int] = None,
#                offset: int = 0) -> QueryResultList:
#         pass
#
#     def semantic_filter(self,) -> QueryResultList:
#         pass
#
#     def limit(self, n: int) -> QueryResultList:
#         pass
#
#     def order_by(self, ) -> QueryResultList:
#         pass


@dataclass
class ChunkAlignment:
    """Alignment between a chunk in one document and its best match in another.

    Parameters
    ----------
    chunk_index_1 : int
        Chunk index in the first document.
    chunk_index_2 : int
        Best-matching chunk index in the second document.
    similarity : float
        Cosine similarity between the two chunks.
    """

    chunk_index_1: int
    chunk_index_2: int
    similarity: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "chunk_index_1": self.chunk_index_1,
            "chunk_index_2": self.chunk_index_2,
            "similarity": self.similarity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkAlignment":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            chunk_index_1=int(data["chunk_index_1"]),
            chunk_index_2=int(data["chunk_index_2"]),
            similarity=float(data["similarity"]),
        )


@dataclass
class DocumentComparisonResult:
    """Rich comparison result between two documents.

    Parameters
    ----------
    doc_id_1 : str
        ID of the first document.
    doc_id_2 : str
        ID of the second document.
    overall_similarity : float
        Centroid-level cosine similarity between documents.
    chunk_alignments : List[ChunkAlignment]
        Best match per chunk in doc_1, sorted by similarity descending.
    matched_ratio_1 : float
        Fraction of doc_1 chunks with a match >= threshold.
    matched_ratio_2 : float
        Fraction of doc_2 chunks with a match >= threshold.
    unmatched_chunks_1 : List[int]
        Chunk indices in doc_1 with no match >= threshold.
    unmatched_chunks_2 : List[int]
        Chunk indices in doc_2 with no match >= threshold.
    """

    doc_id_1: str
    doc_id_2: str
    overall_similarity: float
    chunk_alignments: List[ChunkAlignment]
    matched_ratio_1: float
    matched_ratio_2: float
    unmatched_chunks_1: List[int]
    unmatched_chunks_2: List[int]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "doc_id_1": self.doc_id_1,
            "doc_id_2": self.doc_id_2,
            "overall_similarity": self.overall_similarity,
            "chunk_alignments": [a.to_dict() for a in self.chunk_alignments],
            "matched_ratio_1": self.matched_ratio_1,
            "matched_ratio_2": self.matched_ratio_2,
            "unmatched_chunks_1": list(self.unmatched_chunks_1),
            "unmatched_chunks_2": list(self.unmatched_chunks_2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentComparisonResult":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            doc_id_1=data["doc_id_1"],
            doc_id_2=data["doc_id_2"],
            overall_similarity=float(data["overall_similarity"]),
            chunk_alignments=[ChunkAlignment.from_dict(a) for a in data.get("chunk_alignments", [])],
            matched_ratio_1=float(data["matched_ratio_1"]),
            matched_ratio_2=float(data["matched_ratio_2"]),
            unmatched_chunks_1=[int(i) for i in data.get("unmatched_chunks_1", [])],
            unmatched_chunks_2=[int(i) for i in data.get("unmatched_chunks_2", [])],
        )


@dataclass
class DocumentSimilarityMatrix:
    """NxN similarity matrix for a set of documents.

    Parameters
    ----------
    matrix : np.ndarray
        (N, N) array of pairwise similarity scores.
    doc_ids : List[str]
        Ordered document IDs matching rows/columns.
    embeddings : np.ndarray
        (N, D) document embeddings used to compute the matrix.
    """

    matrix: np.ndarray
    doc_ids: List[str]
    embeddings: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict (arrays become nested lists)."""
        return {
            "matrix": self.matrix.tolist(),
            "doc_ids": list(self.doc_ids),
            "embeddings": self.embeddings.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSimilarityMatrix":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            matrix=np.array(data.get("matrix", []), dtype=float),
            doc_ids=list(data.get("doc_ids", [])),
            embeddings=np.array(data.get("embeddings", []), dtype=float),
        )


@dataclass
class ChunkSimilarityMatrix:
    """Chunk-level pairwise similarity matrix between (or within) documents.

    For cross-document comparison, ``doc_id_1`` and ``doc_id_2`` differ.
    For self-comparison (chord diagrams), they are the same.

    Parameters
    ----------
    matrix : np.ndarray
        (C1, C2) array of pairwise chunk similarity scores.
    doc_id_1 : str
        First document ID.
    doc_id_2 : str
        Second document ID (same as ``doc_id_1`` for self-comparison).
    chunk_indices_1 : List[int]
        Chunk indices for rows.
    chunk_indices_2 : List[int]
        Chunk indices for columns.
    """

    matrix: np.ndarray
    doc_id_1: str
    doc_id_2: str
    chunk_indices_1: List[int]
    chunk_indices_2: List[int]
