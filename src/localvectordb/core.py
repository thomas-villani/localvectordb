# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/core.py
"""
LocalVectorDB v1.0 Core Components

This module contains the foundational classes and data structures for the new
document-first architecture.
"""
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

logger = logging.getLogger(__name__)

# Type alias for the document-score aggregation that occurs when querying
DocumentScoringMethod = Literal["best", "average", "worst", "weighted_average", "frequency_boost",
                                         "harmonic_mean", "diminishing_returns", "statistical", "robust_mean",
                                         "percentile", "geometric_mean"]
"""
Document scoring methods for aggregating chunk scores:

- "best": Highest chunk score (single best passage matters)
- "worst": Lowest chunk score (all content must meet threshold)
- "average": Mean of all chunk scores (overall quality)
- "weighted_average": Score-weighted average (emphasizes top chunks)
- "frequency_boost": Boosts score by number of quality chunks (default, good for comprehensive docs)
  - frequency_bias (0.3): How much to boost based on chunk count
- "harmonic_mean": Conservative mean with coverage bonus
  - max_chunks (5): Top chunks for calculation
  - coverage_threshold (0.7): Quality threshold for bonus
- "diminishing_returns": Exponential decay for later chunks
  - decay_factor (0.8): How quickly impact decreases
- "statistical": Multi-factor scoring (best + mean + consistency + coverage)
  - best_weight (0.6), mean_weight (0.2), consistency_weight (0.1), coverage_weight (0.1)
- "robust_mean": Outlier-resistant with position weighting
  - outlier_threshold (2.0): Z-score for outlier removal
  - position_decay (0.9): Penalty for lower-ranked chunks
- "percentile": Combines high/low percentiles for balanced scoring
  - primary_percentile (0.9), secondary_percentile (0.7), primary_weight (0.7)
- "geometric_mean": Conservative mean (all chunks contribute meaningfully)
"""


def _adapt_datetime_with_tz(dt) -> str:
    return dt.isoformat()

def _convert_datetime_with_tz(dt) -> datetime:
    s = dt.decode("utf-8")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # If it's not a valid datetime string, just return the original string
        # This can happen when SQLite type detection is overly aggressive
        return s

def _adapt_json(json_data) -> str:
    return json.dumps(json_data)

def _convert_json(json_data) -> dict | list:
    return json.loads(json_data.decode("utf-8"))

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

    def valid_types(self) -> Tuple[type, ...]:
        if self.value == "text":
            return (str, )
        elif self.value == "integer":
            return (int, )
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
            'start': self.start,
            'end': self.end,
            'line': self.line,
            'column': self.column,
            'end_line': self.end_line,
            'end_column': self.end_column
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChunkPosition':
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
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()

    def content_equals(self, other: 'Chunk') -> bool:
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
        before = original[:self.position.start]
        chunk_text = original[self.position.start:self.position.end]
        after = original[self.position.end:]

        return f"{before}<<<{chunk_text}>>>{after}"


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

    def __post_init__(self) -> None:
        if self.content_hash is None:
            self.content_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of document content"""
        return hashlib.sha256(self.content.encode('utf-8')).hexdigest()

    def needs_update(self, new_content: str) -> bool:
        """Check if document content has changed"""
        new_hash = hashlib.sha256(new_content.encode('utf-8')).hexdigest()
        return new_hash != self.content_hash

    @classmethod
    def from_dict(cls, data: dict) -> Optional['Document']:
        """Create a Document from a dictionary response"""
        if not data:
            return None

        # Parse datetime fields
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])

        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])

        return cls(
            id=data['id'],
            content=data['content'],
            metadata=data.get('metadata', {}),
            created_at=created_at,
            updated_at=updated_at,
            content_hash=data.get('content_hash'),
            chunks=data.get('chunks', [])
        )

# TODO: create a class to handle list of QueryResult with composable subfiltering options, but still acting like a list.
#   This would allow for further processing like `.semantic_filter` and `.filter`
#   And also fancier stuff like converting to a numpy array of embeddings
@dataclass
class QueryResult:
    """Result from a search query"""
    id: str
    score: float  # Normalized 0-1, higher=better
    type: Literal['document', 'chunk', 'context', 'enriched', 'group', 'aggregation']
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Additional fields for chunks
    document_id: Optional[str] = None
    position: Optional[ChunkPosition] = None

    def get_context(self, original: str, window: int = 100) -> Optional[str]:
        """Get context around chunk (only for chunk results)"""
        if self.type == 'chunk' and self.position:
            start = max(0, self.position.start - window)
            end = min(len(original), self.position.end + window)

            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(original) else ""

            return f"{prefix}{original[start:end]}{suffix}"
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "QueryResult":
        """Create a QueryResult from a dictionary response"""
        if not data:
            return None

        # Parse position if present
        position = None
        if data.get("position"):
            position = ChunkPosition.from_dict(data["position"])

        q_type = data.get("type", "document")
        if q_type not in ("document", "chunk", "context", "enriched", "group", "aggregation"):
            raise ValueError("`type` must be 'document', 'chunk', 'context', 'enriched', 'group', or 'aggregation'")

        return cls(
            id=data["id"],
            score=data.get("score", 0.0),
            type=q_type,
            content=data["content"],
            metadata=data.get("metadata", {}),
            document_id=data.get("document_id"),
            position=position,
        )
