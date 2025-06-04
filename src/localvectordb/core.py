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
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal, Type, Generator

from localvectordb.exceptions import ConnectionPoolError


class MetadataFieldType(str, Enum):
    TEXT = "text"
    INTEGER = "integer"
    REAL = "real"
    BOOLEAN = "boolean"
    DATE = "date"
    JSON = "json"


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

    """
    type: MetadataFieldType | str | Type
    indexed: bool = False
    required: bool = False
    default_value: Any = None

    def __post_init__(self):
        """
        Post-initialization processing to resolve type into MetadataFieldType.

        Converts string or builtin types to corresponding MetadataFieldType.

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

    def __post_init__(self):
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

    def __post_init__(self):
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
    def from_dict(cls, data: dict) -> 'Document':
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


@dataclass
class QueryResult:
    """Result from a search query"""
    id: str
    score: float  # Normalized 0-1, higher=better
    type: Literal['document', 'chunk']
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
        if q_type not in ("document", "chunk"):
            raise ValueError("`type` must be 'document' or 'chunk'")

        return cls(
            id=data["id"],
            score=data.get("score", 0.0),
            type=q_type,
            content=data["content"],
            metadata=data.get("metadata", {}),
            document_id=data.get("document_id"),
            position=position,
        )

class DatabaseSchema:
    """
    Manages the SQLite database schema for LocalVectorDB, including support for custom document metadata fields.

    This class is responsible for creating and updating the core tables (`documents`, `chunks`, etc.) as well as
    dynamically managing user-defined metadata fields for documents. Metadata fields are stored in a dedicated
    `metadata_schema` table and added as columns to the `documents` table.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file.

    Attributes
    ----------
    db_path : Path
        Path to the SQLite database file.
    metadata_fields : dict
        Dictionary mapping metadata field names to `MetadataField` definitions.

    Examples
    --------
    Basic Initialization::

        from localvectordb.core import DatabaseSchema, MetadataField, MetadataFieldType

        # Define a custom metadata schema
        metadata_schema = {
            "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True, required=True),
            "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "published": MetadataField(type=MetadataFieldType.DATE, indexed=True, required=False),
            "tags": MetadataField(type=MetadataFieldType.JSON),
            "rating": MetadataField(type=MetadataFieldType.REAL, default_value=0.0)
        }

        db_schema = DatabaseSchema("mydb.sqlite")
        db_schema.initialize(metadata_schema=metadata_schema)

    **Shorthand Metadata Schema Definitions**

    You can also use string or tuple shorthands for field definitions::

        metadata_schema = {
            "title": "text",  # Equivalent to MetadataField(type=MetadataFieldType.TEXT)
            "author": ("text", True),  # (type, indexed)
            "published": ("date", True, False),  # (type, indexed, required)
        }
        db_schema.initialize(metadata_schema=metadata_schema)

    Using Predefined Schemas::

        from localvectordb.core import get_common_metadata_schemas

        # Get a predefined schema for research papers
        research_schema = get_common_metadata_schemas("research_papers")
        db_schema.initialize(metadata_schema=research_schema)

    Adding a Metadata Field After Initialization::

        db_schema.add_metadata_field(
            "reviewed", MetadataField(type=MetadataFieldType.BOOLEAN, default_value=False)
        )

    Loading Metadata Schema from the Database::

        loaded_schema = db_schema.load_metadata_schema()
        print(loaded_schema)

    Notes
    -----
    - Reserved column names (`id`, `content`, `content_hash`, `created_at`, `updated_at`) cannot be used as metadata fields.
    - Supported metadata field types are: `'text'`, `'integer'`, `'real'`, `'boolean'`, `'date'`, `'json'`.
    - The `initialize` method creates all necessary tables and indexes, and adds metadata columns as needed.
    - Metadata fields can be indexed for faster search and can have default values.

    See Also
    --------
    MetadataField : Structure for defining metadata field properties.
    get_common_metadata_schemas : Helper for common metadata schema templates.
    """

    BASE_DOCUMENTS_SCHEMA = """CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""

    BASE_CHUNKS_SCHEMA = """CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        start_pos INTEGER NOT NULL,
        end_pos INTEGER NOT NULL,
        start_line INTEGER,
        start_col INTEGER,
        end_line INTEGER,
        end_col INTEGER,
        tokens INTEGER NOT NULL,
        faiss_id INTEGER,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        UNIQUE(document_id, chunk_index)
    )"""

    BASE_METADATA_SCHEMA = """CREATE TABLE IF NOT EXISTS metadata_schema (
        field_name TEXT PRIMARY KEY,
        field_type TEXT NOT NULL CHECK(field_type IN ('text', 'integer', 'real', 'boolean', 'date', 'json')),
        indexed BOOLEAN DEFAULT FALSE,
        required BOOLEAN DEFAULT FALSE,
        default_value TEXT
    )"""

    BASE_SCHEMA = {
        "documents": BASE_DOCUMENTS_SCHEMA,
        "chunks": BASE_CHUNKS_SCHEMA,
        "metadata_schema": BASE_METADATA_SCHEMA,
        "config": """CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    }

    # Updated base indexes to include content_hash
    BASE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)",  # New index
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)",
        "CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at)"
    ]

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.metadata_fields: Dict[str, MetadataField] = {}
        self._lock = threading.RLock()

    def initialize(self, metadata_schema: Optional[Dict[str, MetadataField]] = None, db_connection = None):
        """Initialize database schema"""
        with self._lock:
            if db_connection is None:
                db_connection = sqlite3.connect(self.db_path)
            with db_connection as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")

                # Create base tables
                for table_name, ddl in self.BASE_SCHEMA.items():
                    conn.execute(ddl)

                # Create base indexes
                for index_ddl in self.BASE_INDEXES:
                    conn.execute(index_ddl)

                # Set up metadata schema if provided
                if metadata_schema:
                    self._setup_metadata_schema(conn, metadata_schema)

                conn.commit()

    def _setup_metadata_schema(self, conn: sqlite3.Connection, schema: Dict[str, MetadataField]):
        """Set up metadata schema and add columns to documents table"""
        # Define reserved column names that cannot be used for metadata fields
        RESERVED_COLUMNS = {
            "id", "content", "content_hash", "created_at", "updated_at"
        }

        # Validate that no metadata field names conflict with reserved columns
        for field_name in schema.keys():
            if field_name.lower() in RESERVED_COLUMNS:
                raise ValueError(
                    f"Metadata field name '{field_name}' conflicts with reserved column name. "
                    f"Reserved columns are: {", ".join(sorted(RESERVED_COLUMNS))}"
                )

        for field_name, field_def in schema.items():
            if isinstance(field_def, str):
                field_def = MetadataField(MetadataFieldType(field_def), False, required=False)
            elif isinstance(field_def, tuple):
                if len(field_def) == 2:
                    field_type, should_index = field_def
                    required = False
                elif len(field_def) == 3:
                    field_type, should_index, required = field_def
                else:
                    raise ValueError(f"Schema definition tuple must be 2 or 3 items, found: {len(field_def)}")
                field_def = MetadataField(MetadataFieldType(field_type), indexed=should_index, required=required)

            # Store schema definition
            conn.execute("""INSERT OR REPLACE INTO metadata_schema 
                (field_name, field_type, indexed, required, default_value)
                VALUES (?, ?, ?, ?, ?)
            """, (
                field_name,
                field_def.type.value,
                field_def.indexed,
                field_def.required,
                json.dumps(field_def.default_value) if field_def.default_value is not None else None
            ))

            # Add column to documents table
            self._add_metadata_column(conn, field_name, field_def)

        self.metadata_fields = schema

    def _add_metadata_column(self, conn: sqlite3.Connection, field_name: str, field_def: MetadataField):
        """Add a metadata column to the documents table"""
        # Map field types to SQLite types
        sqlite_type_map = {
            MetadataFieldType.TEXT: "TEXT",
            MetadataFieldType.INTEGER: "INTEGER",
            MetadataFieldType.REAL: "REAL",
            MetadataFieldType.BOOLEAN: "BOOLEAN",
            MetadataFieldType.DATE: "TEXT",  # Store as ISO string
            MetadataFieldType.JSON: "TEXT"  # Store as JSON string
        }

        sqlite_type = sqlite_type_map[field_def.type]

        # Check if column already exists
        cursor = conn.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if field_name not in existing_columns:
            # Add the column
            default_clause = ""
            if field_def.default_value is not None:
                if field_def.type in (MetadataFieldType.TEXT, MetadataFieldType.DATE):
                    default_clause = f" DEFAULT '{field_def.default_value}'"
                elif field_def.type == MetadataFieldType.JSON:
                    default_clause = f" DEFAULT '{json.dumps(field_def.default_value)}'"
                else:
                    default_clause = f" DEFAULT {field_def.default_value}"

            ddl = f'ALTER TABLE documents ADD COLUMN {field_name} {sqlite_type}{default_clause}'
            conn.execute(ddl)

            # Create index if requested
            if field_def.indexed:
                index_name = f'idx_documents_{field_name}'
                conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({field_name})')

    def load_metadata_schema(self, db_connection=None) -> Dict[str, MetadataField]:
        """Load metadata schema from database"""
        if db_connection is None:
            db_connection = sqlite3.connect(self.db_path)

        with db_connection as conn:
            cursor = conn.execute("SELECT * FROM metadata_schema")
            schema = {}

            for row in cursor.fetchall():
                field_name, field_type, indexed, required, default_value = row

                # Parse default value
                parsed_default = None
                if default_value is not None:
                    try:
                        parsed_default = json.loads(default_value)
                    except json.JSONDecodeError:
                        parsed_default = default_value

                schema[field_name] = MetadataField(
                    type=MetadataFieldType(field_type),
                    indexed=bool(indexed),
                    required=bool(required),
                    default_value=parsed_default
                )

            self.metadata_fields = schema
            return schema

    def add_metadata_field(self, field_name: str, field_def: MetadataField, db_connection=None):
        """Add a new metadata field to the schema"""
        with self._lock:
            if db_connection is None:
                db_connection = sqlite3.connect(self.db_path)

            with db_connection as conn:
                self._setup_metadata_schema(conn, {field_name: field_def})
                conn.commit()

                self.metadata_fields[field_name] = field_def



class PooledConnection:
    """Wrapper for a pooled connection that handles automatic return to pool"""

    def __init__(self, connection: sqlite3.Connection, pool: "ConnectionPool"):
        self.connection = connection
        self.pool = pool
        self._closed = False

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying connection"""
        return getattr(self.connection, name)

    def __enter__(self):
        self._closed = False
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Return connection to pool on exit"""
        if not self._closed:
            # print("Returning connection to pool")
            self.pool.return_connection(self.connection)
            self._closed = True

    def close(self):
        """Manually return connection to pool"""
        if not self._closed:
            self.pool.return_connection(self.connection)
            self._closed = True


class ConnectionPool:
    """Thread-safe connection pool for SQLite with proper context manager support"""

    def __init__(self, db_path: Union[str, Path], max_connections: int = 10):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.RLock()
        self._created_connections = 0

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with proper settings"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        self._created_connections += 1
        return conn

    @property
    def closed(self):
        return self._created_connections == 0

    def get_connection(self) -> PooledConnection:
        """Get a connection from the pool (wrapped for automatic return)"""
        with self._lock:
            if self._pool:
                # Reuse existing connection from pool
                conn = self._pool.pop()
                # Verify connection is still valid
                try:
                    if conn.in_transaction:
                        conn.rollback()
                    conn.execute("SELECT 1")
                    return PooledConnection(conn, self)
                except sqlite3.Error:
                    # Connection is invalid, create a new one
                    conn.close()
                    self._created_connections -= 1

            # Create new connection if pool is empty or connection was invalid
            if self._created_connections < self.max_connections:
                conn = self._create_connection()
                return PooledConnection(conn, self)
            else:
                raise ConnectionPoolError("No connections available!")

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        with self._lock:
            if len(self._pool) < self.max_connections:
                # Check if connection is still valid before returning to pool
                try:
                    if conn.in_transaction:
                        conn.rollback()
                    conn.execute("SELECT 1")
                    self._pool.append(conn)
                except sqlite3.Error:
                    # Connection is invalid, close it
                    conn.close()
                    self._created_connections -= 1
            else:
                # Pool is full, close the connection
                conn.close()
                self._created_connections -= 1

    @contextmanager
    def get_connection_context(self) -> Generator[sqlite3.Connection, None, None]:
        """Alternative context manager interface"""
        pooled_conn = self.get_connection()
        try:
            yield pooled_conn.connection
        finally:
            pooled_conn.close()

    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
            self._created_connections = 0

    @property
    def stats(self) -> dict:
        """Get pool statistics for debugging"""
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "max_connections": self.max_connections,
                "created_connections": self._created_connections,
                "available_connections": len(self._pool)
            }

    def __del__(self):
        """Cleanup on garbage collection"""
        try:
            self.close_all()
        except:
            pass  # Ignore errors during cleanup
            

def get_common_metadata_schemas(schema: str = None) -> Dict[str, Dict[str, MetadataField]] | Dict[str, MetadataField]:
    """Get predefined metadata schemas for common use cases"""

    schemas = {
        "files": {
            "file_path": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "created_at": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "last_modified": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "mimetype": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "file_size_bytes": MetadataField(type=MetadataFieldType.INTEGER),
            "tags": MetadataField(type=MetadataFieldType.JSON),
        },
        "documents": {
            "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "date": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "tags": MetadataField(type=MetadataFieldType.JSON),
            "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        },
        "research_papers": {
            "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "authors": MetadataField(type=MetadataFieldType.JSON, indexed=False),
            "abstract": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "publication_date": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "journal": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "doi": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "keywords": MetadataField(type=MetadataFieldType.JSON),
            "citation_count": MetadataField(type=MetadataFieldType.INTEGER),
        },
        "code_repository": {
            "file_path": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "language": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "last_modified": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "file_size": MetadataField(type=MetadataFieldType.INTEGER),
            "is_test": MetadataField(type=MetadataFieldType.BOOLEAN, default_value=False),
            "complexity_score": MetadataField(type=MetadataFieldType.REAL),
        },
        "customer_support": {
            "ticket_id": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "customer_id": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "priority": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "status": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "created_date": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "resolved_date": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "satisfaction_score": MetadataField(type=MetadataFieldType.INTEGER),
        }
    }

    if not schema:
        return schemas
    else:
        if schema not in schemas:
            raise KeyError(f"Schema `{schema}` was not found in predefined schema templates. Available options: "
                           f"{", ".join(schemas.keys())}")
        return schemas.get(schema)
