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

import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Literal, Type
import hashlib
import json


class MetadataFieldType(str, Enum):
    """Supported metadata field types"""
    TEXT = "text"
    INTEGER = "integer"
    REAL = "real"
    BOOLEAN = "boolean"
    DATE = "date"
    JSON = "json"


@dataclass
class MetadataField:
    """Definition of a metadata field"""
    type: MetadataFieldType | str | Type
    indexed: bool = False
    required: bool = False
    default_value: Any = None

    def __post_init__(self):
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
    """Exact position tracking for a chunk in the original document"""
    start: int  # Character position in original
    end: int  # Character position in original
    line: int  # Line number (1-based)
    column: int  # Column number (1-based)

    def to_dict(self) -> dict:
        return {
            'start': self.start,
            'end': self.end,
            'line': self.line,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChunkPosition':
        return cls(**data)


@dataclass
class Chunk:
    """Internal representation of a document chunk"""
    content: str
    position: ChunkPosition
    tokens: int
    index: int  # Chunk index within document
    faiss_id: Optional[int] = None  # Maps to FAISS index position

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
    highlights: List[dict] = field(default_factory=list)

    def get_context(self, original: str, window: int = 100) -> Optional[str]:
        """Get context around chunk (only for chunk results)"""
        if self.type == 'chunk' and self.position:
            start = max(0, self.position.start - window)
            end = min(len(original), self.position.end + window)

            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(original) else ""

            return f"{prefix}{original[start:end]}{suffix}"
        return None


class DatabaseSchema:
    """Manages the SQLite database schema"""

    # Base schema DDL
    BASE_SCHEMA = {
        'documents': '''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''',

        'chunks': '''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                start_line INTEGER,
                start_col INTEGER,
                tokens INTEGER NOT NULL,
                faiss_id INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            )
        ''',

        'metadata_schema': '''
            CREATE TABLE IF NOT EXISTS metadata_schema (
                field_name TEXT PRIMARY KEY,
                field_type TEXT NOT NULL CHECK(field_type IN ('text', 'integer', 'real', 'boolean', 'date', 'json')),
                indexed BOOLEAN DEFAULT FALSE,
                required BOOLEAN DEFAULT FALSE,
                default_value TEXT
            )
        ''',

        'config': '''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        '''
    }

    # Base indexes
    BASE_INDEXES = [
        'CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)',
        'CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)',
        'CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)',
        'CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at)'
    ]

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.metadata_fields: Dict[str, MetadataField] = {}
        self._lock = threading.RLock()

    def initialize(self, metadata_schema: Optional[Dict[str, MetadataField]] = None):
        """Initialize database schema"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign keys
                conn.execute('PRAGMA foreign_keys = ON')

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
            'id', 'content', 'content_hash', 'created_at', 'updated_at'
        }

        # Validate that no metadata field names conflict with reserved columns
        for field_name in schema.keys():
            if field_name.lower() in RESERVED_COLUMNS:
                raise ValueError(
                    f"Metadata field name '{field_name}' conflicts with reserved column name. "
                    f"Reserved columns are: {', '.join(sorted(RESERVED_COLUMNS))}"
                )

        for field_name, field_def in schema.items():
            # Store schema definition
            conn.execute('''
                INSERT OR REPLACE INTO metadata_schema 
                (field_name, field_type, indexed, required, default_value)
                VALUES (?, ?, ?, ?, ?)
            ''', (
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
            MetadataFieldType.TEXT: 'TEXT',
            MetadataFieldType.INTEGER: 'INTEGER',
            MetadataFieldType.REAL: 'REAL',
            MetadataFieldType.BOOLEAN: 'BOOLEAN',
            MetadataFieldType.DATE: 'TEXT',  # Store as ISO string
            MetadataFieldType.JSON: 'TEXT'  # Store as JSON string
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

    def load_metadata_schema(self) -> Dict[str, MetadataField]:
        """Load metadata schema from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM metadata_schema')
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

    def add_metadata_field(self, field_name: str, field_def: MetadataField):
        """Add a new metadata field to the schema"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                self._setup_metadata_schema(conn, {field_name: field_def})
                conn.commit()

                self.metadata_fields[field_name] = field_def

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn


class ConnectionPool:
    """Simple connection pool for SQLite"""

    def __init__(self, db_path: Union[str, Path], max_connections: int = 10):
        self.db_path = Path(db_path)
        self.max_connections = max_connections
        self._pool: List[sqlite3.Connection] = []
        self._lock = threading.Lock()

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool"""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute('PRAGMA foreign_keys = ON')
                conn.row_factory = sqlite3.Row
                return conn

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool"""
        with self._lock:
            if len(self._pool) < self.max_connections:
                self._pool.append(conn)
            else:
                conn.close()

    def close_all(self):
        """Close all connections in the pool"""
        with self._lock:
            for conn in self._pool:
                conn.close()
            self._pool.clear()
            

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
                           f"{', '.join(schemas.keys())}")
        return schemas.get(schema)
