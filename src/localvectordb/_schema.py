import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import aiosqlite

from localvectordb._pools import ReadWriteLock
from localvectordb._sqlite_uri import is_sqlite_uri, normalize_db_path
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database._utils import AsyncDatabaseExecutor, SyncDatabaseExecutor
from localvectordb.versioning import DatabaseVersion, VersionManager

logger = logging.getLogger(__name__)


def validate_sql_identifier(identifier: str) -> None:
    """
    Validate that a string is a safe SQL identifier for use in DDL operations.

    Parameters
    ----------
    identifier : str
        The identifier to validate

    Raises
    ------
    ValueError
        If the identifier is not safe for SQL DDL operations

    Notes
    -----
    Safe SQL identifiers must:
    - Start with a letter (A-Z, a-z) or underscore (_)
    - Contain only letters, digits (0-9), and underscores
    - Not be empty or whitespace-only
    - Not exceed 64 characters (reasonable limit for portability)
    """
    if not identifier or not isinstance(identifier, str):
        raise ValueError("SQL identifier must be a non-empty string")

    identifier = identifier.strip()
    if not identifier:
        raise ValueError("SQL identifier cannot be whitespace-only")

    if len(identifier) > 64:
        raise ValueError(f"SQL identifier too long (max 64 chars): '{identifier}'")

    # Check for valid SQL identifier pattern: ^[A-Za-z_][A-Za-z0-9_]*$
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", identifier):
        raise ValueError(
            f"Invalid SQL identifier '{identifier}'. Must start with letter or underscore, "
            f"and contain only letters, digits, and underscores."
        )

    # Check against SQLite reserved words (common subset)
    reserved_words = {
        "abort",
        "action",
        "add",
        "after",
        "all",
        "alter",
        "analyze",
        "and",
        "as",
        "asc",
        "attach",
        "autoincrement",
        "before",
        "begin",
        "between",
        "by",
        "cascade",
        "case",
        "cast",
        "check",
        "collate",
        "column",
        "commit",
        "conflict",
        "constraint",
        "create",
        "cross",
        "current_date",
        "current_time",
        "current_timestamp",
        "database",
        "default",
        "deferrable",
        "deferred",
        "delete",
        "desc",
        "detach",
        "distinct",
        "drop",
        "each",
        "else",
        "end",
        "escape",
        "except",
        "exclusive",
        "exists",
        "explain",
        "fail",
        "for",
        "foreign",
        "from",
        "full",
        "glob",
        "group",
        "having",
        "if",
        "ignore",
        "immediate",
        "in",
        "index",
        "indexed",
        "initially",
        "inner",
        "insert",
        "instead",
        "intersect",
        "into",
        "is",
        "isnull",
        "join",
        "key",
        "left",
        "like",
        "limit",
        "match",
        "natural",
        "no",
        "not",
        "notnull",
        "null",
        "of",
        "offset",
        "on",
        "or",
        "order",
        "outer",
        "plan",
        "pragma",
        "primary",
        "query",
        "raise",
        "recursive",
        "references",
        "regexp",
        "reindex",
        "release",
        "rename",
        "replace",
        "restrict",
        "right",
        "rollback",
        "row",
        "savepoint",
        "select",
        "set",
        "table",
        "temp",
        "temporary",
        "then",
        "to",
        "transaction",
        "trigger",
        "union",
        "unique",
        "update",
        "using",
        "vacuum",
        "values",
        "view",
        "virtual",
        "when",
        "where",
        "with",
        "without",
    }

    if identifier.lower() in reserved_words:
        raise ValueError(f"SQL identifier '{identifier}' is a reserved word")


def quote_sql_identifier(identifier: str) -> str:
    """
    Validate and quote a SQL identifier for safe use in DDL operations.

    This function first validates the identifier, then wraps it in double quotes
    with proper escaping for defense in depth.

    Parameters
    ----------
    identifier : str
        The identifier to validate and quote

    Returns
    -------
    str
        The safely quoted identifier (e.g., '"field_name"')

    Raises
    ------
    ValueError
        If the identifier is not safe for SQL DDL operations

    Examples
    --------
    >>> quote_sql_identifier("my_field")
    '"my_field"'
    """
    validate_sql_identifier(identifier)
    # Escape any embedded double quotes (shouldn't happen due to validation, but defense in depth)
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


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
    - Reserved column names (`id`, `content`, `content_hash`, `created_at`, `updated_at`) cannot be used.
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
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        doc_faiss_id INTEGER
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
        section_id INTEGER REFERENCES sections(id) ON DELETE SET NULL,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        UNIQUE(document_id, chunk_index)
    )"""

    BASE_METADATA_SCHEMA = """CREATE TABLE IF NOT EXISTS metadata_schema (
        field_name TEXT PRIMARY KEY,
        field_type TEXT NOT NULL CHECK(field_type IN ('text', 'integer', 'real', 'boolean', 'date', 'json')),
        indexed BOOLEAN DEFAULT FALSE,
        required BOOLEAN DEFAULT FALSE,
        default_value TEXT,
        embedding_enabled BOOLEAN DEFAULT FALSE,
        fts_enabled BOOLEAN DEFAULT FALSE
    )"""

    BASE_COLUMN_EMBEDDINGS_SCHEMA = """CREATE TABLE IF NOT EXISTS column_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL,
        field_name TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        faiss_id INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        FOREIGN KEY (field_name) REFERENCES metadata_schema(field_name),
        UNIQUE(document_id, field_name, chunk_index)
    )"""

    BASE_SECTIONS_SCHEMA = """CREATE TABLE IF NOT EXISTS sections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id TEXT NOT NULL,
        section_index INTEGER NOT NULL,
        heading TEXT,
        heading_level INTEGER,
        start_pos INTEGER NOT NULL,
        end_pos INTEGER NOT NULL,
        start_line INTEGER,
        end_line INTEGER,
        content_hash TEXT NOT NULL,
        metadata JSON,
        faiss_id INTEGER,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        UNIQUE(document_id, section_index)
    )"""

    BASE_MIGRATION_LOG_SCHEMA = """CREATE TABLE IF NOT EXISTS migration_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT NOT NULL,
        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        rollback_script TEXT,
        checksum TEXT
    )"""

    BASE_BACKUP_LOG_SCHEMA = """CREATE TABLE IF NOT EXISTS backup_log (
        id TEXT PRIMARY KEY,
        backup_type TEXT CHECK(backup_type IN ('full', 'incremental')) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        database_version TEXT NOT NULL,
        file_path TEXT NOT NULL,
        checksum TEXT NOT NULL,
        parent_backup_id TEXT REFERENCES backup_log(id),
        metadata TEXT,
        size_bytes INTEGER,
        compression_algorithm TEXT
    )"""

    BASE_SCHEMA = {
        "documents": BASE_DOCUMENTS_SCHEMA,
        "chunks": BASE_CHUNKS_SCHEMA,
        "sections": BASE_SECTIONS_SCHEMA,
        "metadata_schema": BASE_METADATA_SCHEMA,
        "column_embeddings": BASE_COLUMN_EMBEDDINGS_SCHEMA,
        "migration_log": BASE_MIGRATION_LOG_SCHEMA,
        "backup_log": BASE_BACKUP_LOG_SCHEMA,
        "config": """CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""",
    }

    # Updated base indexes to include content_hash, column_embeddings, migration_log, backup_log, and sections
    BASE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id)",
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)",
        "CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_sections_faiss_id ON sections(faiss_id)",
        "CREATE INDEX IF NOT EXISTS idx_column_embeddings_doc_field ON column_embeddings(document_id, field_name)",
        "CREATE INDEX IF NOT EXISTS idx_column_embeddings_faiss ON column_embeddings(faiss_id)",
        "CREATE INDEX IF NOT EXISTS idx_migration_log_version ON migration_log(version)",
        "CREATE INDEX IF NOT EXISTS idx_migration_log_applied_at ON migration_log(applied_at)",
        "CREATE INDEX IF NOT EXISTS idx_backup_log_created_at ON backup_log(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_backup_log_type ON backup_log(backup_type)",
        "CREATE INDEX IF NOT EXISTS idx_backup_log_parent ON backup_log(parent_backup_id)",
    ]

    BASE_COLUMNS = ["id", "content", "content_hash", "created_at", "updated_at"]

    def __init__(self, db_path: Union[str, Path], read_write_lock: "ReadWriteLock"):
        self.db_path = normalize_db_path(db_path)
        self.metadata_fields: Dict[str, MetadataField] = {}
        self._read_write_lock: ReadWriteLock = read_write_lock
        self._sync_executor = SyncDatabaseExecutor()
        self._async_executor = AsyncDatabaseExecutor()

    def _check_trigram_tokenizer_availability(self, conn: sqlite3.Connection) -> bool:
        """
        Check if SQLite FTS5 trigram tokenizer is available.

        The trigram tokenizer requires compile-time SQLite extension support
        and may not be available in all SQLite installations.

        Parameters
        ----------
        conn : sqlite3.Connection
            SQLite database connection

        Returns
        -------
        bool
            True if trigram tokenizer is available
        """
        try:
            # Test creation of a temporary FTS table with trigram tokenizer
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.trigram_test " "USING fts5(content, tokenize='trigram')"
            )
            conn.execute("DROP TABLE temp.trigram_test")
            return True
        except sqlite3.OperationalError as e:
            if "no such tokenizer" in str(e).lower():
                logger.warning("FTS5 trigram tokenizer not available, falling back to default tokenizer")
                return False
            raise  # Re-raise other operational errors
        except Exception:
            return False

    async def _check_trigram_tokenizer_availability_async(self, conn) -> bool:
        """
        Check if SQLite FTS5 trigram tokenizer is available (async version).

        The trigram tokenizer requires compile-time SQLite extension support
        and may not be available in all SQLite installations.

        Parameters
        ----------
        conn : aiosqlite.Connection
            Async SQLite database connection

        Returns
        -------
        bool
            True if trigram tokenizer is available
        """
        try:
            # Test creation of a temporary FTS table with trigram tokenizer
            await conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.trigram_test " "USING fts5(content, tokenize='trigram')"
            )
            await conn.execute("DROP TABLE temp.trigram_test")
            return True
        except Exception as e:
            # aiosqlite wraps exceptions differently, check string representation
            if "no such tokenizer" in str(e).lower():
                logger.warning("FTS5 trigram tokenizer not available, falling back to default tokenizer")
                return False
            # For other errors, also return False to fall back gracefully
            return False

    def _get_sqlite_version(self, conn: sqlite3.Connection) -> tuple[int, ...]:
        """
        Get SQLite version as a tuple for comparison.

        Parameters
        ----------
        conn : sqlite3.Connection
            SQLite database connection

        Returns
        -------
        tuple
            SQLite version as (major, minor, patch) tuple
        """
        row = conn.execute("SELECT sqlite_version()").fetchone()
        assert row is not None
        version_string: str = row[0]
        return tuple(map(int, version_string.split(".")))

    def _supports_drop_column(self, conn: sqlite3.Connection) -> bool:
        """
        Check if SQLite version supports DROP COLUMN (requires 3.35+).

        Parameters
        ----------
        conn : sqlite3.Connection
            SQLite database connection

        Returns
        -------
        bool
            True if DROP COLUMN is supported
        """
        version = self._get_sqlite_version(conn)
        return version >= (3, 35, 0)

    def _rebuild_table_for_column_drop(self, conn: sqlite3.Connection, field_name: str) -> None:
        """
        Rebuild documents table without the specified column using SQLite-compatible operations.

        Parameters
        ----------
        conn : sqlite3.Connection
            SQLite database connection
        field_name : str
            Name of the column to drop

        Raises
        ------
        Exception
            If table rebuild fails
        """
        # Get current table schema
        cursor = conn.execute("PRAGMA table_info(documents)")
        current_columns = [row[1] for row in cursor.fetchall() if row[1] != field_name]

        # Build new table SQL
        column_definitions = []
        for col_name in current_columns:
            if col_name == "id":
                column_definitions.append("id TEXT PRIMARY KEY")
            elif col_name == "content":
                column_definitions.append("content TEXT NOT NULL")
            elif col_name == "content_hash":
                column_definitions.append("content_hash TEXT NOT NULL")
            elif col_name in ["created_at", "updated_at"]:
                column_definitions.append(f"{col_name} TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            else:
                # Metadata column - get its type from metadata_schema
                try:
                    metadata_cursor = conn.execute(
                        "SELECT field_type FROM metadata_schema WHERE field_name = ?", (col_name,)
                    )
                    result = metadata_cursor.fetchone()
                    if result:
                        field_type = result[0]
                        sqlite_type_map = {
                            "text": "TEXT",
                            "integer": "INTEGER",
                            "real": "REAL",
                            "boolean": "BOOLEAN",
                            "date": "TEXT",
                            "json": "TEXT",
                        }
                        sql_type = sqlite_type_map.get(field_type, "TEXT")
                        column_definitions.append(f"{col_name} {sql_type}")
                    else:
                        # Fallback if metadata not found
                        column_definitions.append(f"{col_name} TEXT")
                except Exception:
                    # Fallback if query fails
                    column_definitions.append(f"{col_name} TEXT")

        # Create new table with a temporary name
        new_table_sql = f"""
        CREATE TABLE documents_new (
            {', '.join(column_definitions)}
        )
        """

        conn.execute(new_table_sql)

        # Copy data from old table to new table
        columns_str = ", ".join(current_columns)
        conn.execute(f"""
        INSERT INTO documents_new ({columns_str})
        SELECT {columns_str} FROM documents
        """)

        # Drop old table and rename new table
        conn.execute("DROP TABLE documents")
        conn.execute("ALTER TABLE documents_new RENAME TO documents")

        # Recreate indexes (excluding the one for the dropped column)
        for index_sql in self.BASE_INDEXES:
            if "documents" in index_sql and field_name not in index_sql:
                try:
                    conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to recreate index: {e}")

        # Recreate any custom indexes for remaining metadata fields
        cursor = conn.execute("SELECT field_name FROM metadata_schema WHERE indexed = 1")
        for (indexed_field,) in cursor.fetchall():
            if indexed_field in current_columns:
                try:
                    quoted_field = quote_sql_identifier(indexed_field)
                    index_name = f"idx_documents_{indexed_field}"
                    conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON documents({quoted_field})")
                except Exception as e:
                    logger.warning(f"Failed to recreate index for {indexed_field}: {e}")

    def _rebuild_table_for_column_type_change(
        self, conn: sqlite3.Connection, field_name: str, new_sqlite_type: str
    ) -> None:
        """
        Rebuild documents table with a changed column type using SQLite-compatible operations.

        Parameters
        ----------
        conn : sqlite3.Connection
            SQLite database connection
        field_name : str
            Name of the column to change type for
        new_sqlite_type : str
            New SQLite type (e.g., 'TEXT', 'INTEGER', 'REAL')

        Raises
        ------
        Exception
            If table rebuild fails
        """
        # Get current table schema
        cursor = conn.execute("PRAGMA table_info(documents)")
        current_columns = [(row[1], row[2]) for row in cursor.fetchall()]

        # Build new table SQL with updated type
        column_definitions = []
        for col_name, col_type in current_columns:
            if col_name == field_name:
                # Use the new type
                column_definitions.append(f"{col_name} {new_sqlite_type}")
            elif col_name == "id":
                column_definitions.append("id TEXT PRIMARY KEY")
            elif col_name == "content":
                column_definitions.append("content TEXT NOT NULL")
            elif col_name == "content_hash":
                column_definitions.append("content_hash TEXT NOT NULL")
            elif col_name in ["created_at", "updated_at"]:
                column_definitions.append(f"{col_name} TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
            else:
                # Keep existing type
                column_definitions.append(f"{col_name} {col_type}")

        # Create new table with a temporary name
        new_table_sql = f"""
        CREATE TABLE documents_new (
            {', '.join(column_definitions)}
        )
        """

        conn.execute(new_table_sql)

        # Copy data from old table to new table with type conversion
        columns_str = ", ".join([col[0] for col in current_columns])
        conn.execute(f"""
        INSERT INTO documents_new ({columns_str})
        SELECT {columns_str} FROM documents
        """)

        # Drop old table and rename new table
        conn.execute("DROP TABLE documents")
        conn.execute("ALTER TABLE documents_new RENAME TO documents")

        # Recreate all indexes
        for index_sql in self.BASE_INDEXES:
            if "documents" in index_sql:
                try:
                    conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to recreate index: {e}")

        # Recreate custom indexes for metadata fields
        cursor = conn.execute("SELECT field_name FROM metadata_schema WHERE indexed = 1")
        for (indexed_field,) in cursor.fetchall():
            try:
                quoted_field = quote_sql_identifier(indexed_field)
                index_name = f"idx_documents_{indexed_field}"
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON documents({quoted_field})")
            except Exception as e:
                logger.warning(f"Failed to recreate index for {indexed_field}: {e}")

    def initialize(self, metadata_schema: Optional[Dict[str, MetadataField]] = None, db_connection=None):
        """Initialize database schema"""
        with self._read_write_lock.write_lock():
            if db_connection is None:
                db_connection = sqlite3.connect(
                    self.db_path, uri=is_sqlite_uri(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES
                )
            with db_connection as conn:
                self._core_initialize_sync(metadata_schema, conn)
                conn.commit()

    def _setup_metadata_schema(self, conn: sqlite3.Connection, schema: Dict[str, MetadataField]):
        """Set up metadata schema and add columns to documents table"""
        return self._core_setup_metadata_schema_sync(conn, schema)

    def _validate_metadata_field_name(self, field_name: str):
        """Validate metadata field name (pure business logic)"""
        try:
            validate_sql_identifier(field_name)
        except ValueError as e:
            raise ValueError(f"Cannot add unsafe field name '{field_name}': {str(e)}") from e

    def _get_sqlite_type_mapping(self, field_def: MetadataField) -> str:
        """Get SQLite type for metadata field (pure business logic)"""
        assert isinstance(field_def.type, MetadataFieldType)
        sqlite_type_map = {
            MetadataFieldType.TEXT: "TEXT",
            MetadataFieldType.INTEGER: "INTEGER",
            MetadataFieldType.REAL: "REAL",
            MetadataFieldType.BOOLEAN: "BOOLEAN",
            MetadataFieldType.DATE: "TEXT",  # Store as ISO string
            MetadataFieldType.JSON: "TEXT",  # Store as JSON string
        }
        return sqlite_type_map[field_def.type]

    def _build_default_clause(self, field_def: MetadataField) -> str:
        """
        Build DEFAULT clause for column (pure business logic).

        For security, TEXT/DATE/JSON types avoid DEFAULT in DDL entirely.
        Instead, defaults are populated via immediate UPDATE after column creation.
        This prevents SQL injection from unescaped default values.
        """
        assert isinstance(field_def.type, MetadataFieldType)
        if field_def.default_value is None:
            return ""

        if field_def.type in (MetadataFieldType.TEXT, MetadataFieldType.DATE, MetadataFieldType.JSON):
            # Skip DEFAULT in DDL for TEXT/DATE/JSON types to avoid SQL injection
            # These will be populated via _populate_field_defaults() instead
            return ""
        elif field_def.type == MetadataFieldType.BOOLEAN:
            # Boolean: convert to 0/1 for SQLite
            return f" DEFAULT {1 if field_def.default_value else 0}"
        else:
            # INTEGER, REAL: safe for direct interpolation
            return f" DEFAULT {field_def.default_value}"

    def _build_fts_triggers(self, field_name: str, fts_table_name: str) -> List[str]:
        """Build FTS trigger SQL statements (pure business logic)"""
        return [
            f"""
            CREATE TRIGGER IF NOT EXISTS fts_{field_name}_insert
            AFTER INSERT ON documents
            WHEN NEW.{field_name} IS NOT NULL
            BEGIN
                INSERT INTO {fts_table_name}(document_id, content)
                VALUES (NEW.id, NEW.{field_name});
            END
            """,
            f"""
            CREATE TRIGGER IF NOT EXISTS fts_{field_name}_update
            AFTER UPDATE OF {field_name} ON documents
            WHEN NEW.{field_name} IS NOT NULL
            BEGIN
                DELETE FROM {fts_table_name} WHERE document_id = NEW.id;
                INSERT INTO {fts_table_name}(document_id, content)
                VALUES (NEW.id, NEW.{field_name});
            END
            """,
            f"""
            CREATE TRIGGER IF NOT EXISTS fts_{field_name}_delete
            AFTER DELETE ON documents
            BEGIN
                DELETE FROM {fts_table_name} WHERE document_id = OLD.id;
            END
            """,
        ]

    def _migrate_hierarchical_columns(self, conn: sqlite3.Connection) -> None:
        """Add hierarchical embedding columns to existing databases if missing."""
        # Add section_id to chunks table if missing
        try:
            cursor = conn.execute("PRAGMA table_info(chunks)")
            chunk_columns = {row[1] for row in cursor.fetchall()}
            if "section_id" not in chunk_columns:
                conn.execute(
                    "ALTER TABLE chunks ADD COLUMN section_id INTEGER " "REFERENCES sections(id) ON DELETE SET NULL"
                )
                logger.info("Added section_id column to chunks table")
        except Exception as e:
            logger.debug(f"section_id migration check: {e}")

        # Add doc_faiss_id to documents table if missing
        try:
            cursor = conn.execute("PRAGMA table_info(documents)")
            doc_columns = {row[1] for row in cursor.fetchall()}
            if "doc_faiss_id" not in doc_columns:
                conn.execute("ALTER TABLE documents ADD COLUMN doc_faiss_id INTEGER")
                logger.info("Added doc_faiss_id column to documents table")
        except Exception as e:
            logger.debug(f"doc_faiss_id migration check: {e}")

    async def _migrate_hierarchical_columns_async(self, conn) -> None:
        """Add hierarchical embedding columns to existing databases if missing (async)."""
        try:
            cursor = await conn.execute("PRAGMA table_info(chunks)")
            rows = await cursor.fetchall()
            chunk_columns = {row[1] for row in rows}
            if "section_id" not in chunk_columns:
                await conn.execute(
                    "ALTER TABLE chunks ADD COLUMN section_id INTEGER " "REFERENCES sections(id) ON DELETE SET NULL"
                )
                logger.info("Added section_id column to chunks table")
        except Exception as e:
            logger.debug(f"section_id migration check: {e}")

        try:
            cursor = await conn.execute("PRAGMA table_info(documents)")
            rows = await cursor.fetchall()
            doc_columns = {row[1] for row in rows}
            if "doc_faiss_id" not in doc_columns:
                await conn.execute("ALTER TABLE documents ADD COLUMN doc_faiss_id INTEGER")
                logger.info("Added doc_faiss_id column to documents table")
        except Exception as e:
            logger.debug(f"doc_faiss_id migration check: {e}")

    def _core_initialize_sync(self, metadata_schema: Optional[Dict[str, MetadataField]], conn: sqlite3.Connection):
        """Core logic for initializing database schema (sync version)"""
        # Enable foreign keys
        self._sync_executor.execute(conn, "PRAGMA foreign_keys = ON")

        # Create base tables
        for _, ddl in self.BASE_SCHEMA.items():
            self._sync_executor.execute(conn, ddl)

        # Migrate existing databases to add hierarchical columns
        self._migrate_hierarchical_columns(conn)

        # Create base indexes (uses IF NOT EXISTS so safe for existing DBs)
        for index_ddl in self.BASE_INDEXES:
            self._sync_executor.execute(conn, index_ddl)

        # Set up metadata schema if provided
        if metadata_schema:
            self._setup_metadata_schema(conn, metadata_schema)

        # Initialize version tracking for new databases
        self._initialize_version_tracking(conn)

    async def _core_initialize_async(self, metadata_schema: Optional[Dict[str, MetadataField]], conn):
        """Core logic for initializing database schema (async version)"""
        # Enable foreign keys
        await self._async_executor.execute(conn, "PRAGMA foreign_keys = ON")

        # Create base tables
        for _, ddl in self.BASE_SCHEMA.items():
            await self._async_executor.execute(conn, ddl)

        # Migrate existing databases to add hierarchical columns
        await self._migrate_hierarchical_columns_async(conn)

        # Create base indexes
        for index_ddl in self.BASE_INDEXES:
            await self._async_executor.execute(conn, index_ddl)

        # Set up metadata schema if provided
        if metadata_schema:
            await self._setup_metadata_schema_async(conn, metadata_schema)

    def _core_setup_metadata_schema_sync(self, conn: sqlite3.Connection, schema: Dict[str, MetadataField]):
        """Core logic for setting up metadata schema and adding columns to documents table (sync version)"""
        # Validate that no metadata field names conflict with reserved columns
        for field_name in schema.keys():
            # Validate SQL identifier safety
            try:
                validate_sql_identifier(field_name)
            except ValueError as e:
                raise ValueError(f"Invalid metadata field name '{field_name}': {str(e)}") from e

            if field_name.lower() in self.BASE_COLUMNS:
                raise ValueError(
                    f"Metadata field name '{field_name}' conflicts with reserved column name. "
                    f"Reserved columns are: {', '.join(sorted(self.BASE_COLUMNS))}"
                )

        for field_name, field_def in schema.items():
            if isinstance(field_def, str):
                field_def = MetadataField(MetadataFieldType(field_def), False, required=False)
            elif isinstance(field_def, tuple):
                if len(field_def) == 2:
                    field_type, should_index = field_def
                    required = False
                    default_value = None
                elif len(field_def) == 3:
                    field_type, should_index, required = field_def
                    default_value = None
                elif len(field_def) == 4:
                    field_type, should_index, required, default_value = field_def
                else:
                    raise ValueError(
                        f"Schema definition tuple must be 2-4 items: "
                        f"(field_type, should_index[, required, default_value]). Found: {len(field_def)}"
                    )
                field_def = MetadataField(
                    MetadataFieldType(field_type), indexed=should_index, required=required, default_value=default_value
                )

            # Store schema definition
            assert isinstance(field_def.type, MetadataFieldType)
            self._sync_executor.execute(
                conn,
                """INSERT OR REPLACE INTO metadata_schema
                (field_name, field_type, indexed, required, default_value, embedding_enabled, fts_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    field_name,
                    field_def.type.value,
                    field_def.indexed,
                    field_def.required,
                    json.dumps(field_def.default_value) if field_def.default_value is not None else None,
                    field_def.embedding_enabled,
                    field_def.fts_enabled,
                ),
            )

            # Add column to documents table
            self._add_metadata_column(conn, field_name, field_def)

        self.metadata_fields = schema

    async def _core_setup_metadata_schema_async(self, conn, schema: Dict[str, MetadataField]):
        """Core logic for setting up metadata schema and adding columns to documents table (async version)"""
        # Validate that no metadata field names conflict with reserved columns
        for field_name in schema.keys():
            # Validate SQL identifier safety
            try:
                validate_sql_identifier(field_name)
            except ValueError as e:
                raise ValueError(f"Invalid metadata field name '{field_name}': {str(e)}") from e

            if field_name.lower() in self.BASE_COLUMNS:
                raise ValueError(
                    f"Metadata field name '{field_name}' conflicts with reserved column name. "
                    f"Reserved columns are: {', '.join(sorted(self.BASE_COLUMNS))}"
                )

        for field_name, field_def in schema.items():
            if isinstance(field_def, str):
                field_def = MetadataField(MetadataFieldType(field_def), False, required=False)
            elif isinstance(field_def, tuple):
                if len(field_def) == 2:
                    field_type, should_index = field_def
                    required = False
                    default_value = None
                elif len(field_def) == 3:
                    field_type, should_index, required = field_def
                    default_value = None
                elif len(field_def) == 4:
                    field_type, should_index, required, default_value = field_def
                else:
                    raise ValueError(
                        f"Schema definition tuple must be 2-4 items: "
                        f"(field_type, should_index[, required, default_value]). Found: {len(field_def)}"
                    )
                field_def = MetadataField(
                    MetadataFieldType(field_type), indexed=should_index, required=required, default_value=default_value
                )

            # Store schema definition (fixed to include embedding_enabled and fts_enabled)
            assert isinstance(field_def.type, MetadataFieldType)
            await self._async_executor.execute(
                conn,
                """INSERT OR REPLACE INTO metadata_schema
                (field_name, field_type, indexed, required, default_value, embedding_enabled, fts_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    field_name,
                    field_def.type.value,
                    field_def.indexed,
                    field_def.required,
                    json.dumps(field_def.default_value) if field_def.default_value is not None else None,
                    field_def.embedding_enabled,
                    field_def.fts_enabled,
                ),
            )

            # Add column to documents table
            await self._add_metadata_column_async(conn, field_name, field_def)

        self.metadata_fields = schema

    def _core_load_metadata_schema_sync(self, conn: sqlite3.Connection) -> Dict[str, MetadataField]:
        """Core logic for loading metadata schema from database (sync version)"""
        cursor = self._sync_executor.execute(conn, "SELECT * FROM metadata_schema")
        schema = {}

        for row in self._sync_executor.fetchall(cursor):
            # Handle both old schema (5 columns) and new schema (7 columns)
            if len(row) == 5:
                # Old schema - no embedding_enabled or fts_enabled
                field_name, field_type, indexed, required, default_value = row
                embedding_enabled = False
                fts_enabled = False
            else:
                # New schema with embedding_enabled and fts_enabled
                field_name, field_type, indexed, required, default_value, embedding_enabled, fts_enabled = row

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
                default_value=parsed_default,
                embedding_enabled=bool(embedding_enabled),
                fts_enabled=bool(fts_enabled),
            )

        self.metadata_fields = schema
        return schema

    async def _core_load_metadata_schema_async(self, conn) -> Dict[str, MetadataField]:
        """Core logic for loading metadata schema from database (async version)"""
        cursor = await self._async_executor.execute(conn, "SELECT * FROM metadata_schema")
        schema = {}

        rows = await self._async_executor.fetchall(cursor)
        for row in rows:
            # Handle both old schema (5 columns) and new schema (7 columns) - fixed bug
            if len(row) == 5:
                # Old schema - no embedding_enabled or fts_enabled
                field_name, field_type, indexed, required, default_value = row
                embedding_enabled = False
                fts_enabled = False
            else:
                # New schema with embedding_enabled and fts_enabled
                field_name, field_type, indexed, required, default_value, embedding_enabled, fts_enabled = row

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
                default_value=parsed_default,
                embedding_enabled=bool(embedding_enabled),
                fts_enabled=bool(fts_enabled),
            )

        self.metadata_fields = schema
        return schema

    def _add_metadata_column(self, conn: sqlite3.Connection, field_name: str, field_def: MetadataField):
        """Add a metadata column to the documents table"""
        # Business logic validation
        self._validate_metadata_field_name(field_name)
        sqlite_type = self._get_sqlite_type_mapping(field_def)

        # Check if column already exists
        cursor = conn.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if field_name not in existing_columns:
            # Build and execute column addition with quoted identifier
            quoted_field = quote_sql_identifier(field_name)
            default_clause = self._build_default_clause(field_def)
            ddl = f"ALTER TABLE documents ADD COLUMN {quoted_field} {sqlite_type}{default_clause}"
            conn.execute(ddl)

            logger.info(f"Added new column: {field_name} {sqlite_type}{default_clause}")

            # Populate default values if specified (especially for TEXT/DATE/JSON types that skip DDL defaults)
            if field_def.default_value is not None:
                populated_info = self._populate_field_defaults(conn, field_name, field_def)
                if populated_info["rows_updated"] > 0:
                    logger.info(
                        f"Populated default values for {populated_info['rows_updated']} "
                        f"existing documents in column '{field_name}'"
                    )

            # Create index if requested
            if field_def.indexed:
                index_name = f"idx_documents_{field_name}"
                conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON documents({quoted_field})")

            # Create FTS table if requested
            if field_def.fts_enabled and field_def.type == MetadataFieldType.TEXT:
                # FTS table names are also validated via field_name validation above
                fts_table_name = f"fts_{field_name}"

                # Check tokenizer availability and build FTS SQL
                if self._check_trigram_tokenizer_availability(conn):
                    tokenizer_clause = "tokenize='trigram'"
                    logger.debug(f"Using trigram tokenizer for FTS table: {fts_table_name}")
                else:
                    tokenizer_clause = ""
                    logger.debug(f"Using default tokenizer for FTS table: {fts_table_name}")

                create_fts_sql = f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table_name}
                    USING fts5(document_id, content{', ' + tokenizer_clause if tokenizer_clause else ''})
                """
                conn.execute(create_fts_sql)
                logger.info(f"Created FTS table: {fts_table_name}")

                # Create triggers using shared business logic
                triggers = self._build_fts_triggers(field_name, fts_table_name)
                for trigger_sql in triggers:
                    conn.execute(trigger_sql)

    def _ensure_enhanced_metadata_schema(self, db_connection):
        """
        Ensure metadata_schema table has embedding_enabled and fts_enabled columns.
        This handles migration from older database versions.
        """
        with db_connection as conn:
            # Check if the columns exist
            cursor = conn.execute("PRAGMA table_info(metadata_schema)")
            columns = {row[1] for row in cursor.fetchall()}

            if "embedding_enabled" not in columns:
                logger.info("Migrating metadata_schema table to add embedding_enabled column")
                conn.execute("ALTER TABLE metadata_schema ADD COLUMN embedding_enabled BOOLEAN DEFAULT FALSE")

            if "fts_enabled" not in columns:
                logger.info("Migrating metadata_schema table to add fts_enabled column")
                conn.execute("ALTER TABLE metadata_schema ADD COLUMN fts_enabled BOOLEAN DEFAULT FALSE")

                # Auto-enable FTS for indexed TEXT fields
                conn.execute("""
                    UPDATE metadata_schema
                    SET fts_enabled = TRUE
                    WHERE field_type = 'text' AND indexed = TRUE
                """)

            # Ensure column_embeddings table exists
            conn.execute(self.BASE_COLUMN_EMBEDDINGS_SCHEMA)

            # Create indexes for column_embeddings if not already present
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_column_embeddings_doc_field ON "
                "column_embeddings(document_id, field_name)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_column_embeddings_faiss ON column_embeddings(faiss_id)")

            conn.commit()

    def _initialize_version_tracking(self, conn: sqlite3.Connection):
        """
        Initialize version tracking for new databases.

        Sets up PRAGMA user_version and records initial state in migration_log.
        For existing databases, checks and updates version tracking as needed.
        """
        try:
            version_manager = VersionManager(self.db_path)

            # Check if this is a new database or needs version initialization
            current_version = version_manager.get_database_version(conn)

            if current_version == DatabaseVersion("0.0.0"):
                # New database - initialize with current schema version
                logger.info("Initializing version tracking for new database")
                version_manager.initialize_version_tracking(conn)
            else:
                # Existing database - ensure version tracking is up to date
                logger.debug(f"Database version: {current_version}")

                # Check if migration_log table exists (might be an older database)
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='migration_log'
                """)
                if not cursor.fetchone():
                    # Old database without migration_log - record current state
                    logger.info("Adding migration tracking to existing database")
                    version_manager.record_migration(
                        str(current_version), rollback_script=None, checksum=None, conn=conn
                    )

        except Exception as e:
            logger.warning(f"Could not initialize version tracking: {e}")
            # Don't fail database initialization for version tracking issues

    def load_metadata_schema(self, db_connection=None) -> Dict[str, MetadataField]:
        """Load metadata schema from database"""
        if db_connection is None:
            db_connection = sqlite3.connect(
                self.db_path, uri=is_sqlite_uri(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES
            )

        # First ensure the schema is up to date
        self._ensure_enhanced_metadata_schema(db_connection)

        with db_connection as conn:
            return self._core_load_metadata_schema_sync(conn)

    def update_metadata_schema(
        self,
        new_schema: Dict[str, MetadataField],
        db_connection=None,
        drop_columns: bool = False,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Update the metadata schema, adding new fields and updating existing ones

        This enhanced version supports column remapping to rename existing columns
        and transfer their data. The processing order is:
        1. Create new columns (including remapping targets)
        2. Transfer data from old columns to new columns
        3. Remove old columns that are no longer needed

        Parameters
        ----------
        new_schema : Dict[str, MetadataField]
            The new metadata schema to apply
        db_connection : sqlite3.Connection, optional
            Database connection to use
        drop_columns : bool, default=False
            Whether to actually drop columns that are no longer in the schema.
            If False, columns are kept but removed from schema for safety.
        column_mapping : Dict[str, str], optional
            Optionally provide a mapping of old column names to new column names.
            Format: {'old_column_name': 'new_column_name'}
            Data will be transferred from old columns to new columns.

        Returns
        -------
        Dict[str, Any]
            Summary of changes made including added, removed, modified fields,
            and column remapping operations
        """
        with self._read_write_lock.write_lock():
            if db_connection is None:
                db_connection = sqlite3.connect(
                    self.db_path, uri=is_sqlite_uri(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES
                )

            changes: Dict[str, Any] = {
                "added_fields": [],
                "removed_fields": [],
                "modified_fields": [],
                "populated_defaults": [],
                "dropped_columns": [],
                "remapped_columns": [],
                "warnings": [],
                "errors": [],
            }

            # Load current schema
            current_schema = self.load_metadata_schema(db_connection)

            # Validate column mapping if provided
            if column_mapping:
                self._validate_column_mapping(column_mapping, current_schema, new_schema, self.BASE_COLUMNS)

            with db_connection as conn:
                try:
                    # Validate new schema field names and required fields
                    for field_name, field_def in new_schema.items():
                        # Validate SQL identifier safety
                        try:
                            validate_sql_identifier(field_name)
                        except ValueError as e:
                            raise ValueError(f"Invalid metadata field name '{field_name}': {str(e)}") from e

                        if field_name.lower() in self.BASE_COLUMNS:
                            raise ValueError(
                                f"Metadata field name '{field_name}' conflicts with reserved column name. "
                                f"Reserved columns are: {', '.join(sorted(self.BASE_COLUMNS))}"
                            )

                        # Validate required fields have defaults if they're new
                        if field_def.required and field_name not in current_schema and field_def.default_value is None:
                            # Check if we have documents that would need this field
                            count_row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                            assert count_row is not None
                            doc_count = count_row[0]
                            if doc_count > 0:
                                raise ValueError(
                                    f"Required field '{field_name}' must have a default_value when added to "
                                    f"existing database with documents"
                                )

                    # STEP 1: Handle new and modified fields (CREATE new columns first)
                    for field_name, field_def in new_schema.items():
                        # Normalize field definition
                        if isinstance(field_def, str):
                            field_def = MetadataField(MetadataFieldType(field_def), False, required=False)
                        elif isinstance(field_def, tuple):
                            if len(field_def) == 2:
                                field_type, should_index = field_def
                                required = False
                                default_value = None
                            elif len(field_def) == 3:
                                field_type, should_index, required = field_def
                                default_value = None
                            elif len(field_def) == 4:
                                field_type, should_index, required, default_value = field_def
                            else:
                                raise ValueError(
                                    f"Schema definition tuple must be 2-4 items: "
                                    f"(field_type, should_index[, required, default_value]). Found: {len(field_def)}"
                                )
                            field_def = MetadataField(
                                MetadataFieldType(field_type),
                                indexed=should_index,
                                required=required,
                                default_value=default_value,
                            )

                        if field_name not in current_schema:
                            # New field - add it
                            try:
                                # Store schema definition
                                assert isinstance(field_def.type, MetadataFieldType)
                                conn.execute(
                                    """INSERT OR REPLACE INTO metadata_schema
                                    (field_name, field_type, indexed, required, default_value)
                                    VALUES (?, ?, ?, ?, ?)
                                """,
                                    (
                                        field_name,
                                        field_def.type.value,
                                        field_def.indexed,
                                        field_def.required,
                                        field_def.default_value,
                                    ),
                                )

                                # Add column to documents table
                                self._add_metadata_column(conn, field_name, field_def)

                                changes["added_fields"].append(field_name)

                                # Populate default values if specified and we have existing documents
                                if field_def.default_value is not None:
                                    populated_info = self._populate_field_defaults(conn, field_name, field_def)
                                    if populated_info["rows_updated"] > 0:
                                        changes["populated_defaults"].append(populated_info)

                                # Add warning for nullable new fields on existing documents
                                count_row = conn.execute("SELECT COUNT(*) FROM documents").fetchone()
                                assert count_row is not None
                                doc_count = count_row[0]
                                if doc_count > 0 and field_def.default_value is None:
                                    changes["warnings"].append(
                                        f"New field '{field_name}' added to database with existing documents. "
                                        f"Existing documents will have NULL values."
                                    )

                            except Exception as e:
                                changes["errors"].append(f"Failed to add field '{field_name}': {str(e)}")

                        else:
                            # Existing field - check if it needs updates
                            current_field = current_schema[field_name]
                            field_changed = False
                            change_details: Dict[str, Any] = {}

                            # Check if any properties changed
                            assert isinstance(current_field.type, MetadataFieldType)
                            assert isinstance(field_def.type, MetadataFieldType)
                            if current_field.type != field_def.type:
                                change_details["type"] = {"old": current_field.type.value, "new": field_def.type.value}
                                field_changed = True

                            if current_field.indexed != field_def.indexed:
                                change_details["indexed"] = {"old": current_field.indexed, "new": field_def.indexed}
                                field_changed = True

                            if current_field.required != field_def.required:
                                change_details["required"] = {"old": current_field.required, "new": field_def.required}
                                field_changed = True

                            if current_field.default_value != field_def.default_value:
                                change_details["default_value"] = {
                                    "old": current_field.default_value,
                                    "new": field_def.default_value,
                                }
                                field_changed = True

                            if field_changed:
                                try:
                                    # Update schema definition
                                    conn.execute(
                                        """UPDATE metadata_schema
                                        SET field_type = ?, indexed = ?, required = ?, default_value = ?
                                        WHERE field_name = ?
                                    """,
                                        (
                                            field_def.type.value,
                                            field_def.indexed,
                                            field_def.required,
                                            field_def.default_value,
                                            field_name,
                                        ),
                                    )

                                    # Handle index changes
                                    if "indexed" in change_details:
                                        quoted_field = quote_sql_identifier(field_name)
                                        if field_def.indexed:
                                            # Add index
                                            index_name = f"idx_documents_{field_name}"
                                            try:
                                                conn.execute(
                                                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                                                    f"ON documents({quoted_field})"
                                                )
                                            except Exception as e:
                                                changes["errors"].append(
                                                    f"Failed to create index on '{field_name}': {str(e)}"
                                                )
                                        else:
                                            # Remove index (if it exists)
                                            index_name = f"idx_documents_{field_name}"
                                            try:
                                                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                                            except Exception as e:
                                                changes["errors"].append(
                                                    f"Failed to drop index on '{field_name}': {str(e)}"
                                                )

                                    if "type" in change_details:
                                        old_type_str = change_details["type"]["old"]
                                        new_type_str = change_details["type"]["new"]

                                        try:
                                            # Map MetadataFieldType to SQLite types
                                            sqlite_type_map = {
                                                "text": "TEXT",
                                                "integer": "INTEGER",
                                                "real": "REAL",
                                                "boolean": "BOOLEAN",
                                                "date": "TEXT",
                                                "json": "TEXT",
                                            }

                                            new_sqlite_type = sqlite_type_map.get(new_type_str, "TEXT")

                                            # Perform the type change using table rebuild
                                            self._rebuild_table_for_column_type_change(
                                                conn, field_name, new_sqlite_type
                                            )

                                            logger.info(
                                                f"Changed column '{field_name}' data type from {old_type_str} "
                                                f"to {new_type_str}"
                                            )

                                        except Exception as e:
                                            changes["errors"].append(
                                                f"Failed to change column type for '{field_name}': {str(e)}"
                                            )
                                            changes["warnings"].append(
                                                f"Type change failed for '{field_name}', "
                                                f"schema updated but column type unchanged"
                                            )

                                    changes["modified_fields"].append(
                                        {"field_name": field_name, "changes": change_details}
                                    )

                                except Exception as e:
                                    changes["errors"].append(f"Failed to modify field '{field_name}': {str(e)}")

                    # STEP 2: Perform column remapping (AFTER new columns are created)
                    if column_mapping:
                        remapping_changes = self._perform_column_remapping(
                            conn, column_mapping, current_schema, new_schema
                        )
                        changes["remapped_columns"] = remapping_changes["remapped_columns"]
                        changes["warnings"].extend(remapping_changes["warnings"])
                        changes["errors"].extend(remapping_changes["errors"])

                    # STEP 3: Handle removed fields (after remapping)
                    for field_name in current_schema:
                        if field_name not in new_schema:
                            try:
                                # Remove from schema table (even if it was remapped)
                                conn.execute("DELETE FROM metadata_schema WHERE field_name = ?", (field_name,))
                                changes["removed_fields"].append(field_name)

                                logger.info(f"Removed {field_name} from metadata_schema")

                                # Skip dropping the actual column if it was remapped
                                if column_mapping and field_name in column_mapping:
                                    logger.info(
                                        f"Skipping column drop for '{field_name}' as it was remapped to "
                                        f"'{column_mapping[field_name]}'"
                                    )
                                    continue

                                # Optionally drop the actual column
                                if drop_columns:
                                    try:
                                        # First check if column actually exists
                                        cursor = conn.execute("PRAGMA table_info(documents)")
                                        existing_columns = {row[1] for row in cursor.fetchall()}

                                        if field_name in existing_columns:
                                            # Validate field_name for safe use in DDL
                                            validate_sql_identifier(field_name)
                                            # First drop any FTS triggers and tables
                                            drop_field: Optional[MetadataField] = current_schema.get(field_name)
                                            if drop_field and getattr(drop_field, "fts_enabled", False):
                                                fts_table_name = f"fts_{field_name}"
                                                try:
                                                    # Drop FTS triggers (names derived from validated field_name)
                                                    conn.execute(f"DROP TRIGGER IF EXISTS fts_{field_name}_insert")
                                                    conn.execute(f"DROP TRIGGER IF EXISTS fts_{field_name}_update")
                                                    conn.execute(f"DROP TRIGGER IF EXISTS fts_{field_name}_delete")
                                                    # Drop FTS table (table name derived from validated field_name)
                                                    conn.execute(f"DROP TABLE IF EXISTS {fts_table_name}")
                                                    logger.debug(f"Dropped FTS triggers and table for '{field_name}'")
                                                except Exception as e:
                                                    logger.warning(
                                                        f"Failed to drop FTS components for '{field_name}': {e}"
                                                    )

                                            # Drop any indexes on this column
                                            index_name = f"idx_documents_{field_name}"
                                            try:
                                                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                                                logger.debug(f"Dropped index {index_name} before dropping column")
                                            except Exception:
                                                pass  # Index might not exist, that's okay

                                            # Now drop the column - use native DROP COLUMN if supported,
                                            # otherwise rebuild table
                                            quoted_field = quote_sql_identifier(field_name)
                                            if self._supports_drop_column(conn):
                                                conn.execute(f"ALTER TABLE documents DROP COLUMN {quoted_field}")
                                                logger.info(f"Dropped column '{field_name}' using native DROP COLUMN")
                                            else:
                                                self._rebuild_table_for_column_drop(conn, field_name)
                                                logger.info(f"Dropped column '{field_name}' using table rebuild")

                                            changes["dropped_columns"].append(field_name)
                                        else:
                                            changes["warnings"].append(
                                                f"Column '{field_name}' not found in table, skipping drop"
                                            )

                                    except Exception as e:
                                        changes["errors"].append(f"Failed to drop column '{field_name}': {str(e)}")
                                        changes["warnings"].append(
                                            f"Column '{field_name}' could not be dropped but was removed from schema. "
                                            f"Column data may still exist in the table."
                                        )
                            except Exception as e:
                                changes["errors"].append(f"Failed to remove field '{field_name}' from schema: {str(e)}")

                    conn.commit()

                    # Update in-memory schema
                    self.metadata_fields = new_schema.copy()

                except Exception as e:
                    conn.rollback()
                    changes["errors"].append(f"Schema update failed: {str(e)}")
                    raise

            return changes

    def _validate_column_mapping(
        self,
        column_mapping: Dict[str, str],
        current_schema: Dict[str, MetadataField],
        new_schema: Dict[str, MetadataField],
        reserved_columns: list,
    ) -> None:
        """Validate the column mapping configuration"""
        for old_col, new_col in column_mapping.items():
            # Check that old column exists
            if old_col not in current_schema:
                raise ValueError(f"Cannot remap column '{old_col}': column does not exist in current schema")

            # Check that new column name is not reserved
            if new_col.lower() in reserved_columns:
                raise ValueError(f"Cannot remap to '{new_col}': it conflicts with reserved column names")

            # Check that new column is defined in new schema
            if new_col not in new_schema:
                raise ValueError(f"Column mapping specifies '{new_col}' but it's not defined in new_schema")

            # Validate that the types are compatible
            old_field = current_schema[old_col]
            new_field = new_schema[new_col]
            assert isinstance(old_field.type, MetadataFieldType)
            assert isinstance(new_field.type, MetadataFieldType)
            if not self._are_types_compatible(old_field.type, new_field.type):
                raise ValueError(
                    f"Cannot remap '{old_col}' (type: {old_field.type.value}) to "
                    f"'{new_col}' (type: {new_field.type.value}): incompatible types"
                )

    def _are_types_compatible(self, old_type: MetadataFieldType, new_type: MetadataFieldType) -> bool:
        """Check if two metadata field types are compatible for data transfer"""
        # Same types are always compatible
        if old_type == new_type:
            return True

        # TEXT is compatible with most types (can convert)
        if old_type == MetadataFieldType.TEXT:
            return True

        # Some numeric conversions are safe
        if old_type == MetadataFieldType.INTEGER and new_type == MetadataFieldType.REAL:
            return True

        # Boolean to integer/real is safe
        if old_type == MetadataFieldType.BOOLEAN and new_type in (MetadataFieldType.INTEGER, MetadataFieldType.REAL):
            return True

        # JSON can be converted to TEXT
        if old_type == MetadataFieldType.JSON and new_type == MetadataFieldType.TEXT:
            return True

        return False

    def _perform_column_remapping(
        self,
        conn: sqlite3.Connection,
        column_mapping: Dict[str, str],
        current_schema: Dict[str, MetadataField],
        new_schema: Dict[str, MetadataField],
    ) -> Dict[str, Any]:
        """Perform the actual column remapping operations"""
        remap_changes: Dict[str, Any] = {"remapped_columns": [], "warnings": [], "errors": []}

        for old_col, new_col in column_mapping.items():
            try:
                # The new column should already exist (created in Step 1)
                # Just verify it exists
                cursor = conn.execute("PRAGMA table_info(documents)")
                existing_columns = {row[1] for row in cursor.fetchall()}

                if new_col not in existing_columns:
                    raise ValueError(
                        f"Target column '{new_col}' does not exist. It should have been created in the new schema."
                    )

                # Transfer data from old column to new column
                old_field_def = current_schema[old_col]
                new_field_def = new_schema[new_col]

                rows_transferred = self._transfer_column_data(conn, old_col, new_col, old_field_def, new_field_def)

                logger.info(f"Remapped column {old_col} to {new_col}")

                remap_changes["remapped_columns"].append(
                    {"old_column": old_col, "new_column": new_col, "rows_transferred": rows_transferred}
                )

            except Exception as e:
                remap_changes["errors"].append(f"Failed to remap column '{old_col}' to '{new_col}': {str(e)}")

        return remap_changes

    def _transfer_column_data(
        self, conn: sqlite3.Connection, old_col: str, new_col: str, old_field: MetadataField, new_field: MetadataField
    ) -> int:
        """Transfer data from old column to new column with type conversion"""

        # Check if both columns exist in the documents table
        cursor = conn.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        if old_col not in existing_columns:
            raise ValueError(f"Source column '{old_col}' does not exist in documents table")

        if new_col not in existing_columns:
            raise ValueError(f"Target column '{new_col}' does not exist in documents table")

        # Get the data transfer SQL based on type compatibility
        assert isinstance(old_field.type, MetadataFieldType)
        assert isinstance(new_field.type, MetadataFieldType)
        transfer_sql = self._get_transfer_sql(old_col, new_col, old_field.type, new_field.type)

        # Execute the transfer
        cursor = conn.execute(transfer_sql)
        return cursor.rowcount

    @staticmethod
    def _get_transfer_sql(old_col: str, new_col: str, old_type: MetadataFieldType, new_type: MetadataFieldType) -> str:
        """Generate SQL for transferring data between columns with type conversion.

        Validates and quotes column identifiers for SQL safety.
        """
        # Validate and quote column identifiers
        quoted_new = quote_sql_identifier(new_col)
        quoted_old = quote_sql_identifier(old_col)

        if old_type == new_type:
            # Direct copy for same types
            return f"UPDATE documents SET {quoted_new} = {quoted_old} WHERE {quoted_old} IS NOT NULL"

        elif old_type == MetadataFieldType.TEXT:
            # TEXT to other types - direct copy (SQLite will handle conversion)
            return f"UPDATE documents SET {quoted_new} = {quoted_old} WHERE {quoted_old} IS NOT NULL"

        elif old_type == MetadataFieldType.INTEGER and new_type == MetadataFieldType.REAL:
            # Integer to real - direct copy
            return f"UPDATE documents SET {quoted_new} = CAST({quoted_old} AS REAL) WHERE {quoted_old} IS NOT NULL"

        elif old_type == MetadataFieldType.BOOLEAN and new_type == MetadataFieldType.INTEGER:
            # Boolean to integer
            return f"UPDATE documents SET {quoted_new} = CAST({quoted_old} AS INTEGER) WHERE {quoted_old} IS NOT NULL"

        elif old_type == MetadataFieldType.BOOLEAN and new_type == MetadataFieldType.REAL:
            # Boolean to real
            return f"UPDATE documents SET {quoted_new} = CAST({quoted_old} AS REAL) WHERE {quoted_old} IS NOT NULL"

        elif old_type == MetadataFieldType.JSON and new_type == MetadataFieldType.TEXT:
            # JSON to text - direct copy (it's already stored as text in SQLite)
            return f"UPDATE documents SET {quoted_new} = {quoted_old} WHERE {quoted_old} IS NOT NULL"

        else:
            # Default case - try direct copy and let SQLite handle it
            return f"UPDATE documents SET {quoted_new} = {quoted_old} WHERE {quoted_old} IS NOT NULL"

    @staticmethod
    def _populate_field_defaults(conn: sqlite3.Connection, field_name: str, field_def: MetadataField) -> Dict[str, Any]:
        """Populate default values for a new field in existing documents"""
        if field_def.default_value is None:
            return {"field_name": field_name, "rows_updated": 0}

        # Prepare the default value based on field type
        if field_def.type == MetadataFieldType.JSON:
            default_value = json.dumps(field_def.default_value)
        elif field_def.type in (MetadataFieldType.TEXT, MetadataFieldType.DATE):
            default_value = str(field_def.default_value)
        else:
            default_value = field_def.default_value

        # Update existing documents with the default value (use quoted identifier)
        quoted_field = quote_sql_identifier(field_name)
        cursor = conn.execute(f"UPDATE documents SET {quoted_field} = ? WHERE {quoted_field} IS NULL", (default_value,))

        return {"field_name": field_name, "rows_updated": cursor.rowcount, "default_value": field_def.default_value}

    # Async methods for DatabaseSchema
    async def initialize_async(
        self, metadata_schema: Optional[Dict[str, MetadataField]] = None, db_connection=None
    ) -> None:
        """Initialize database schema asynchronously"""

        # Determine if we need to manage the connection lifecycle
        owns_connection = db_connection is None

        if owns_connection:
            db_connection = await aiosqlite.connect(
                self.db_path, uri=is_sqlite_uri(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES
            )

        try:
            await self._core_initialize_async(metadata_schema, db_connection)
            await db_connection.commit()
        finally:
            if owns_connection and db_connection:
                await db_connection.close()

    async def _setup_metadata_schema_async(self, conn, schema: Dict[str, MetadataField]) -> None:
        """Set up metadata schema and add columns to documents table asynchronously"""
        await self._core_setup_metadata_schema_async(conn, schema)

    async def _add_metadata_column_async(self, conn, field_name: str, field_def: MetadataField) -> None:
        """Add a metadata column to the documents table asynchronously"""
        # Business logic validation using shared helpers
        self._validate_metadata_field_name(field_name)
        sqlite_type = self._get_sqlite_type_mapping(field_def)

        # Check if column already exists
        cursor = await conn.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in await cursor.fetchall()}

        if field_name not in existing_columns:
            # Build and execute column addition with quoted identifier
            quoted_field = quote_sql_identifier(field_name)
            default_clause = self._build_default_clause(field_def)
            ddl = f"ALTER TABLE documents ADD COLUMN {quoted_field} {sqlite_type}{default_clause}"
            await conn.execute(ddl)

            logger.info(f"Added new column: {field_name} {sqlite_type}{default_clause}")

            # Populate default values if specified (especially for TEXT/DATE/JSON types that skip DDL defaults)
            if field_def.default_value is not None:
                populated_info = await self._populate_field_defaults_async(conn, field_name, field_def)
                if populated_info and populated_info["rows_updated"] > 0:
                    logger.info(
                        f"Populated default values for {populated_info['rows_updated']} "
                        f"existing documents in column '{field_name}'"
                    )

            # Create index if requested
            if field_def.indexed:
                index_name = f"idx_documents_{field_name}"
                await conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON documents({quoted_field})")

            # Create FTS table if requested
            if field_def.fts_enabled and field_def.type == MetadataFieldType.TEXT:
                # FTS table names are validated via field_name validation above
                fts_table_name = f"fts_{field_name}"

                # Check tokenizer availability and build FTS SQL
                if await self._check_trigram_tokenizer_availability_async(conn):
                    tokenizer_clause = "tokenize='trigram'"
                    logger.debug(f"Using trigram tokenizer for FTS table: {fts_table_name}")
                else:
                    tokenizer_clause = ""
                    logger.debug(f"Using default tokenizer for FTS table: {fts_table_name}")

                create_fts_sql = f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table_name}
                    USING fts5(document_id, content{', ' + tokenizer_clause if tokenizer_clause else ''})
                """
                await conn.execute(create_fts_sql)
                logger.info(f"Created FTS table: {fts_table_name}")

                # Create triggers using shared business logic
                triggers = self._build_fts_triggers(field_name, fts_table_name)
                for trigger_sql in triggers:
                    await conn.execute(trigger_sql)

    async def load_metadata_schema_async(
        self, db_connection: Optional[aiosqlite.Connection] = None
    ) -> Dict[str, MetadataField]:
        """Load metadata schema from database asynchronously"""

        # Determine if we need to manage the connection lifecycle
        owns_connection = db_connection is None

        if owns_connection:
            db_connection = await aiosqlite.connect(
                self.db_path, uri=is_sqlite_uri(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES
            )

        conn = db_connection
        try:
            return await self._core_load_metadata_schema_async(conn)
        finally:
            if owns_connection and db_connection:
                await db_connection.close()

    async def update_metadata_schema_async(
        self,
        new_schema: Dict[str, MetadataField],
        db_connection: Optional[aiosqlite.Connection] = None,
        drop_columns: bool = False,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Update the metadata schema asynchronously, adding new fields and updating existing ones

        This enhanced version supports column remapping to rename existing columns
        and transfer their data. The processing order is:
        1. Create new columns (including remapping targets)
        2. Transfer data from old columns to new columns
        3. Remove old columns that are no longer needed

        Parameters
        ----------
        new_schema : Dict[str, MetadataField]
            The new metadata schema to apply
        db_connection : aiosqlite.Connection, optional
            Database connection to use
        drop_columns : bool, default=False
            Whether to actually drop columns that are no longer in the schema.
            If False, columns are kept but removed from schema for safety.
        column_mapping : Dict[str, str], optional
            Optionally provide a mapping of old column names to new column names.
            Format: {'old_column_name': 'new_column_name'}
            Data will be transferred from old columns to new columns.

        Returns
        -------
        Dict[str, Any]
            Summary of changes made including added, removed, modified fields,
            and column remapping operations
        """

        # Determine if we need to manage the connection lifecycle
        owns_connection = db_connection is None

        if owns_connection:
            db_connection = await aiosqlite.connect(
                self.db_path, uri=is_sqlite_uri(self.db_path), detect_types=sqlite3.PARSE_DECLTYPES
            )

        assert db_connection is not None

        changes: Dict[str, Any] = {
            "added_fields": [],
            "removed_fields": [],
            "modified_fields": [],
            "populated_defaults": [],
            "dropped_columns": [],
            "remapped_columns": [],
            "warnings": [],
            "errors": [],
        }

        try:
            # Load current schema
            current_schema = await self.load_metadata_schema_async(db_connection)

            # Validate column mapping if provided
            if column_mapping:
                validation_errors = await self._validate_column_mapping_async(
                    current_schema, new_schema, column_mapping
                )
                if validation_errors:
                    changes["errors"].extend(validation_errors)
                    return changes
            await db_connection.execute("BEGIN TRANSACTION")

            # Process column remapping first
            if column_mapping:
                remapped = await self._perform_column_remapping_async(
                    db_connection, column_mapping, new_schema, current_schema
                )
                changes["remapped_columns"].extend(remapped)

                # Process added and modified fields
                for field_name, field_def in new_schema.items():
                    if field_name not in current_schema:
                        # Add new field
                        await self._add_metadata_column_async(db_connection, field_name, field_def)
                        changes["added_fields"].append(field_name)

                        # Populate defaults if needed
                        populated = await self._populate_field_defaults_async(db_connection, field_name, field_def)
                        if populated:
                            changes["populated_defaults"].append(populated)
                    else:
                        # Check for modifications
                        old_def = current_schema[field_name]
                        modifications: List[Dict[str, Any]] = []

                        # Check type change
                        assert isinstance(old_def.type, MetadataFieldType)
                        assert isinstance(field_def.type, MetadataFieldType)
                        if old_def.type != field_def.type:
                            if not self._are_types_compatible(old_def.type, field_def.type):
                                changes["warnings"].append(
                                    f"Type change for '{field_name}' from {old_def.type.value} "
                                    f"to {field_def.type.value} may cause data loss"
                                )
                            modifications.append(
                                {"change": "type", "from": old_def.type.value, "to": field_def.type.value}
                            )

                        # Check index change
                        if old_def.indexed != field_def.indexed:
                            quoted_field = quote_sql_identifier(field_name)
                            index_name = f"idx_documents_{field_name}"
                            if field_def.indexed:
                                await db_connection.execute(
                                    f"CREATE INDEX IF NOT EXISTS {index_name} ON documents({quoted_field})"
                                )
                                modifications.append({"change": "index", "action": "added"})
                            else:
                                await db_connection.execute(f"DROP INDEX IF EXISTS {index_name}")
                                modifications.append({"change": "index", "action": "removed"})

                        # Check required change
                        if old_def.required != field_def.required:
                            modifications.append(
                                {"change": "required", "from": old_def.required, "to": field_def.required}
                            )

                            # If making field required, populate defaults for NULLs
                            if field_def.required and not old_def.required:
                                populated = await self._populate_field_defaults_async(
                                    db_connection, field_name, field_def, old_def
                                )
                                if populated:
                                    changes["populated_defaults"].append(populated)

                        # Check default value change
                        if old_def.default_value != field_def.default_value:
                            modifications.append(
                                {
                                    "change": "default_value",
                                    "from": old_def.default_value,
                                    "to": field_def.default_value,
                                }
                            )

                        if modifications:
                            changes["modified_fields"].append(
                                {"field_name": field_name, "modifications": modifications}
                            )

                # Process removed fields
                removed_fields = set(current_schema.keys()) - set(new_schema.keys())

                # Don't mark remapped source columns as removed
                if column_mapping:
                    removed_fields -= set(column_mapping.keys())

                for field_name in removed_fields:
                    changes["removed_fields"].append(field_name)
                    if drop_columns:
                        # SQLite doesn't support DROP COLUMN in older versions
                        # This is a placeholder - would need table recreation for full support
                        changes["warnings"].append(
                            f"Column '{field_name}' marked for removal but SQLite "
                            "requires table recreation for column drops"
                        )

                    # Remove from metadata_schema table
                    await db_connection.execute("DELETE FROM metadata_schema WHERE field_name = ?", (field_name,))

                # Update metadata_schema table for all fields in new schema
                for field_name, field_def in new_schema.items():
                    assert isinstance(field_def.type, MetadataFieldType)
                    await db_connection.execute(
                        """
                        INSERT OR REPLACE INTO metadata_schema
                        (field_name, field_type, indexed, required, default_value)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            field_name,
                            field_def.type.value,
                            field_def.indexed,
                            field_def.required,
                            json.dumps(field_def.default_value) if field_def.default_value is not None else None,
                        ),
                    )

                await db_connection.commit()

            # Update in-memory schema
            self.metadata_fields = new_schema.copy()

        except Exception as e:
            try:
                await db_connection.execute("ROLLBACK")
            except Exception:
                pass  # Ignore rollback errors
            changes["errors"].append(f"Schema update failed: {str(e)}")
            raise e
        finally:
            if owns_connection and db_connection:
                await db_connection.close()

        return changes

    async def _validate_column_mapping_async(
        self,
        current_schema: Dict[str, MetadataField],
        new_schema: Dict[str, MetadataField],
        column_mapping: Dict[str, str],
    ) -> List[str]:
        """Validate column mapping asynchronously"""
        errors = []

        for old_name, new_name in column_mapping.items():
            # Check source column exists
            if old_name not in current_schema:
                errors.append(f"Source column '{old_name}' does not exist")
                continue

            # Check target is in new schema
            if new_name not in new_schema:
                errors.append(f"Target column '{new_name}' is not in new schema")
                continue

            # Check type compatibility
            old_type = current_schema[old_name].type
            new_type = new_schema[new_name].type
            assert isinstance(old_type, MetadataFieldType)
            assert isinstance(new_type, MetadataFieldType)
            if not self._are_types_compatible(old_type, new_type):
                errors.append(
                    f"Incompatible types for mapping '{old_name}' ({old_type.value}) "
                    f"to '{new_name}' ({new_type.value})"
                )

        return errors

    async def _perform_column_remapping_async(
        self,
        conn: aiosqlite.Connection,
        column_mapping: Dict[str, str],
        new_schema: Dict[str, MetadataField],
        current_schema: Dict[str, MetadataField],
    ) -> List[Dict[str, Any]]:
        """Perform column remapping operations asynchronously"""
        remapped = []

        for old_name, new_name in column_mapping.items():
            # Transfer data from old column to new column
            field_def = new_schema[new_name]
            old_field_def = current_schema[old_name]
            # Add the new column if it doesn't exist
            await self._add_metadata_column_async(conn, new_name, field_def)

            # Transfer the data
            rows_affected = await self._transfer_column_data_async(conn, old_name, new_name, field_def, old_field_def)

            remapped.append({"from": old_name, "to": new_name, "rows_affected": rows_affected})

            # Remove old column from metadata_schema
            await conn.execute("DELETE FROM metadata_schema WHERE field_name = ?", (old_name,))

        return remapped

    async def _transfer_column_data_async(
        self,
        conn: aiosqlite.Connection,
        old_name: str,
        new_name: str,
        field_def: MetadataField,
        old_field_def: MetadataField,
    ) -> int:
        """Transfer data from old column to new column asynchronously"""
        # Get the appropriate SQL for data transfer
        assert isinstance(field_def.type, MetadataFieldType)
        assert isinstance(old_field_def.type, MetadataFieldType)
        transfer_sql = self._get_transfer_sql(old_name, new_name, field_def.type, old_field_def.type)

        # Execute the transfer
        cursor = await conn.execute(transfer_sql)
        return int(cursor.rowcount)

    @staticmethod
    async def _populate_field_defaults_async(
        conn: aiosqlite.Connection,
        field_name: str,
        field_def: MetadataField,
        old_field_def: Optional[MetadataField] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Populate default values for a field where NULL values exist asynchronously.

        This handles both new fields and fields being made required.
        """
        if field_def.default_value is None:
            return None

        # Validate and quote field name for SQL safety
        quoted_field = quote_sql_identifier(field_name)

        # Build WHERE clause with quoted identifier
        where_clause = f"{quoted_field} IS NULL"

        # Don't populate defaults if the field already had a default
        # (existing NULLs were intentional)
        if old_field_def and old_field_def.default_value is not None:
            return None

        # Check how many rows would be affected
        cursor = await conn.execute(f"SELECT COUNT(*) FROM documents WHERE {where_clause}")
        row = await cursor.fetchone()
        assert row is not None
        null_count = row[0]

        if null_count == 0:
            return None

        # Prepare the default value for SQL
        if field_def.type == MetadataFieldType.JSON:
            sql_value = json.dumps(field_def.default_value)
        elif field_def.type in (MetadataFieldType.TEXT, MetadataFieldType.DATE):
            sql_value = str(field_def.default_value)
        else:
            sql_value = field_def.default_value

        # Update the NULL values (use quoted identifier)
        await conn.execute(f"UPDATE documents SET {quoted_field} = ? WHERE {where_clause}", (sql_value,))

        return {"field_name": field_name, "rows_updated": null_count, "default_value": field_def.default_value}


def get_common_metadata_schemas(
    schema: Optional[str] = None,
) -> dict[str, dict[str, MetadataField]] | dict[str, MetadataField]:
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
            "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True, fts_enabled=True),
            "author": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
            "date": MetadataField(type=MetadataFieldType.DATE, indexed=True),
            "tags": MetadataField(type=MetadataFieldType.JSON),
            "category": MetadataField(type=MetadataFieldType.TEXT, indexed=True),
        },
        "research_papers": {
            "title": MetadataField(type=MetadataFieldType.TEXT, indexed=True, embedding_enabled=True, fts_enabled=True),
            "authors": MetadataField(type=MetadataFieldType.JSON, indexed=False),
            "abstract": MetadataField(
                type=MetadataFieldType.TEXT, indexed=True, embedding_enabled=True, fts_enabled=True
            ),
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
        },
    }

    if not schema:
        return schemas
    else:
        if schema not in schemas:
            raise KeyError(
                f"Schema `{schema}` was not found in predefined schema templates. Available options: "
                f"{", ".join(schemas.keys())}"
            )
        return schemas[schema]
