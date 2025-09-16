# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# ${DIR_PATH}/${FILE_NAME}
import json
import sqlite3
from pathlib import Path
from typing import Union, Dict, Optional, Any, List
import logging
import aiosqlite

from localvectordb.core import MetadataFieldType, MetadataField
from localvectordb._pools import ReadWriteLock
from localvectordb.versioning import VersionManager, DatabaseVersion


logger = logging.getLogger(__name__)

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
        "metadata_schema": BASE_METADATA_SCHEMA,
        "column_embeddings": BASE_COLUMN_EMBEDDINGS_SCHEMA,
        "migration_log": BASE_MIGRATION_LOG_SCHEMA,
        "backup_log": BASE_BACKUP_LOG_SCHEMA,
        "config": """CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )"""
    }

    # Updated base indexes to include content_hash, column_embeddings, migration_log, and backup_log
    BASE_INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_faiss_id ON chunks(faiss_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)",  # New index
        "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash)",
        "CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_column_embeddings_doc_field ON column_embeddings(document_id, field_name)",
        "CREATE INDEX IF NOT EXISTS idx_column_embeddings_faiss ON column_embeddings(faiss_id)",
        "CREATE INDEX IF NOT EXISTS idx_migration_log_version ON migration_log(version)",
        "CREATE INDEX IF NOT EXISTS idx_migration_log_applied_at ON migration_log(applied_at)",
        "CREATE INDEX IF NOT EXISTS idx_backup_log_created_at ON backup_log(created_at)",
        "CREATE INDEX IF NOT EXISTS idx_backup_log_type ON backup_log(backup_type)",
        "CREATE INDEX IF NOT EXISTS idx_backup_log_parent ON backup_log(parent_backup_id)"
    ]

    BASE_COLUMNS = [
        "id", "content", "content_hash", "created_at", "updated_at"
    ]

    def __init__(self, db_path: Union[str, Path], read_write_lock: "ReadWriteLock"):
        self.db_path = Path(db_path)
        self.metadata_fields: Dict[str, MetadataField] = {}
        self._read_write_lock: ReadWriteLock = read_write_lock

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
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.trigram_test "
                "USING fts5(content, tokenize='trigram')"
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
                "CREATE VIRTUAL TABLE IF NOT EXISTS temp.trigram_test "
                "USING fts5(content, tokenize='trigram')"
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

    def _get_sqlite_version(self, conn: sqlite3.Connection) -> tuple:
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
        version_string = conn.execute("SELECT sqlite_version()").fetchone()[0]
        return tuple(map(int, version_string.split('.')))

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
            if col_name == 'id':
                column_definitions.append('id TEXT PRIMARY KEY')
            elif col_name == 'content':
                column_definitions.append('content TEXT NOT NULL')
            elif col_name == 'content_hash':
                column_definitions.append('content_hash TEXT NOT NULL')
            elif col_name in ['created_at', 'updated_at']:
                column_definitions.append(f'{col_name} TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            else:
                # Metadata column - get its type from metadata_schema
                try:
                    metadata_cursor = conn.execute(
                        "SELECT field_type FROM metadata_schema WHERE field_name = ?", 
                        (col_name,)
                    )
                    result = metadata_cursor.fetchone()
                    if result:
                        field_type = result[0]
                        sqlite_type_map = {
                            'text': 'TEXT',
                            'integer': 'INTEGER', 
                            'real': 'REAL',
                            'boolean': 'BOOLEAN',
                            'date': 'TEXT',
                            'json': 'TEXT'
                        }
                        sql_type = sqlite_type_map.get(field_type, 'TEXT')
                        column_definitions.append(f'{col_name} {sql_type}')
                    else:
                        # Fallback if metadata not found
                        column_definitions.append(f'{col_name} TEXT')
                except Exception:
                    # Fallback if query fails
                    column_definitions.append(f'{col_name} TEXT')
        
        # Create new table with a temporary name
        new_table_sql = f"""
        CREATE TABLE documents_new (
            {', '.join(column_definitions)}
        )
        """
        
        conn.execute(new_table_sql)
        
        # Copy data from old table to new table
        columns_str = ', '.join(current_columns)
        conn.execute(f"""
        INSERT INTO documents_new ({columns_str})
        SELECT {columns_str} FROM documents
        """)
        
        # Drop old table and rename new table
        conn.execute("DROP TABLE documents")
        conn.execute("ALTER TABLE documents_new RENAME TO documents")
        
        # Recreate indexes (excluding the one for the dropped column)
        for index_sql in self.BASE_INDEXES:
            if 'documents' in index_sql and field_name not in index_sql:
                try:
                    conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to recreate index: {e}")
        
        # Recreate any custom indexes for remaining metadata fields
        cursor = conn.execute("SELECT field_name FROM metadata_schema WHERE indexed = 1")
        for (indexed_field,) in cursor.fetchall():
            if indexed_field in current_columns:
                try:
                    index_name = f'idx_documents_{indexed_field}'
                    conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({indexed_field})')
                except Exception as e:
                    logger.warning(f"Failed to recreate index for {indexed_field}: {e}")

    def _rebuild_table_for_column_type_change(self, conn: sqlite3.Connection, field_name: str, new_sqlite_type: str) -> None:
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
                column_definitions.append(f'{col_name} {new_sqlite_type}')
            elif col_name == 'id':
                column_definitions.append('id TEXT PRIMARY KEY')
            elif col_name == 'content':
                column_definitions.append('content TEXT NOT NULL')
            elif col_name == 'content_hash':
                column_definitions.append('content_hash TEXT NOT NULL')
            elif col_name in ['created_at', 'updated_at']:
                column_definitions.append(f'{col_name} TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
            else:
                # Keep existing type
                column_definitions.append(f'{col_name} {col_type}')
        
        # Create new table with a temporary name
        new_table_sql = f"""
        CREATE TABLE documents_new (
            {', '.join(column_definitions)}
        )
        """
        
        conn.execute(new_table_sql)
        
        # Copy data from old table to new table with type conversion
        columns_str = ', '.join([col[0] for col in current_columns])
        conn.execute(f"""
        INSERT INTO documents_new ({columns_str})
        SELECT {columns_str} FROM documents
        """)
        
        # Drop old table and rename new table  
        conn.execute("DROP TABLE documents")
        conn.execute("ALTER TABLE documents_new RENAME TO documents")
        
        # Recreate all indexes
        for index_sql in self.BASE_INDEXES:
            if 'documents' in index_sql:
                try:
                    conn.execute(index_sql)
                except Exception as e:
                    logger.warning(f"Failed to recreate index: {e}")
        
        # Recreate custom indexes for metadata fields
        cursor = conn.execute("SELECT field_name FROM metadata_schema WHERE indexed = 1")
        for (indexed_field,) in cursor.fetchall():
            try:
                index_name = f'idx_documents_{indexed_field}'
                conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({indexed_field})')
            except Exception as e:
                logger.warning(f"Failed to recreate index for {indexed_field}: {e}")

    def initialize(self, metadata_schema: Optional[Dict[str, MetadataField]] = None, db_connection = None):
        """Initialize database schema"""
        with self._read_write_lock.write_lock():
            if db_connection is None:
                db_connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
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

                # Initialize version tracking for new databases
                self._initialize_version_tracking(conn)

                conn.commit()

    def _setup_metadata_schema(self, conn: sqlite3.Connection, schema: Dict[str, MetadataField]):
        """Set up metadata schema and add columns to documents table"""

        # Validate that no metadata field names conflict with reserved columns
        for field_name in schema.keys():
            if field_name.lower() in self.BASE_COLUMNS:
                raise ValueError(
                    f"Metadata field name '{field_name}' conflicts with reserved column name. "
                    f"Reserved columns are: {", ".join(sorted(self.BASE_COLUMNS))}"
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
                        f"(field_type, should_index[, required, default_value]). Found: {len(field_def)}")
                field_def = MetadataField(MetadataFieldType(field_type), indexed=should_index,
                                          required=required, default_value=default_value)

            # Store schema definition
            conn.execute("""INSERT OR REPLACE INTO metadata_schema 
                (field_name, field_type, indexed, required, default_value, embedding_enabled, fts_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                field_name,
                field_def.type.value,
                field_def.indexed,
                field_def.required,
                json.dumps(field_def.default_value) if field_def.default_value is not None else None,
                field_def.embedding_enabled,
                field_def.fts_enabled
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

            logger.info(f"Added new column: {field_name} {sqlite_type}{default_clause}")
            # Create index if requested
            if field_def.indexed:
                index_name = f'idx_documents_{field_name}'
                conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({field_name})')

            # Create FTS table if requested
            if field_def.fts_enabled and field_def.type == MetadataFieldType.TEXT:
                fts_table_name = f'fts_{field_name}'
                # Check if trigram tokenizer is available, fall back to default if not
                if self._check_trigram_tokenizer_availability(conn):
                    tokenizer_clause = "tokenize='trigram'"
                    logger.debug(f"Using trigram tokenizer for FTS table: {fts_table_name}")
                else:
                    tokenizer_clause = ""  # Use default tokenizer
                    logger.debug(f"Using default tokenizer for FTS table: {fts_table_name}")
                
                conn.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table_name} 
                    USING fts5(document_id, content{', ' + tokenizer_clause if tokenizer_clause else ''})
                ''')
                logger.info(f"Created FTS table: {fts_table_name}")

                # Create triggers to keep FTS in sync
                conn.execute(f'''
                    CREATE TRIGGER IF NOT EXISTS fts_{field_name}_insert 
                    AFTER INSERT ON documents
                    WHEN NEW.{field_name} IS NOT NULL
                    BEGIN
                        INSERT INTO {fts_table_name}(document_id, content) 
                        VALUES (NEW.id, NEW.{field_name});
                    END
                ''')

                conn.execute(f'''
                    CREATE TRIGGER IF NOT EXISTS fts_{field_name}_update 
                    AFTER UPDATE OF {field_name} ON documents
                    WHEN NEW.{field_name} IS NOT NULL
                    BEGIN
                        DELETE FROM {fts_table_name} WHERE document_id = NEW.id;
                        INSERT INTO {fts_table_name}(document_id, content) 
                        VALUES (NEW.id, NEW.{field_name});
                    END
                ''')

                conn.execute(f'''
                    CREATE TRIGGER IF NOT EXISTS fts_{field_name}_delete 
                    AFTER DELETE ON documents
                    BEGIN
                        DELETE FROM {fts_table_name} WHERE document_id = OLD.id;
                    END
                ''')

    def _ensure_enhanced_metadata_schema(self, db_connection):
        """
        Ensure metadata_schema table has embedding_enabled and fts_enabled columns.
        This handles migration from older database versions.
        """
        with db_connection as conn:
            # Check if the columns exist
            cursor = conn.execute("PRAGMA table_info(metadata_schema)")
            columns = {row[1] for row in cursor.fetchall()}

            if 'embedding_enabled' not in columns:
                logger.info("Migrating metadata_schema table to add embedding_enabled column")
                conn.execute("ALTER TABLE metadata_schema ADD COLUMN embedding_enabled BOOLEAN DEFAULT FALSE")

            if 'fts_enabled' not in columns:
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_column_embeddings_doc_field ON column_embeddings(document_id, field_name)")
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
                        str(current_version),
                        rollback_script=None,
                        checksum=None,
                        conn=conn
                    )

        except Exception as e:
            logger.warning(f"Could not initialize version tracking: {e}")
            # Don't fail database initialization for version tracking issues

    def load_metadata_schema(self, db_connection=None) -> Dict[str, MetadataField]:
        """Load metadata schema from database"""
        if db_connection is None:
            db_connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        # First ensure the schema is up to date
        self._ensure_enhanced_metadata_schema(db_connection)

        with db_connection as conn:
            cursor = conn.execute("SELECT * FROM metadata_schema")
            schema = {}

            for row in cursor.fetchall():
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
                    fts_enabled=bool(fts_enabled)
                )

            self.metadata_fields = schema
            return schema

    def update_metadata_schema(
            self,
            new_schema: Dict[str, MetadataField],
            db_connection=None,
            drop_columns: bool = False,
            column_mapping: Optional[Dict[str, str]] = None
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
                db_connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

            changes = {
                'added_fields': [],
                'removed_fields': [],
                'modified_fields': [],
                'populated_defaults': [],
                'dropped_columns': [],
                'remapped_columns': [],
                'warnings': [],
                'errors': []
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
                        if field_name.lower() in self.BASE_COLUMNS:
                            raise ValueError(
                                f"Metadata field name '{field_name}' conflicts with reserved column name. "
                                f"Reserved columns are: {', '.join(sorted(self.BASE_COLUMNS))}"
                            )

                        # Validate required fields have defaults if they're new
                        if (field_def.required and
                                field_name not in current_schema and
                                field_def.default_value is None):
                            # Check if we have documents that would need this field
                            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
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
                                    f"(field_type, should_index[, required, default_value]). Found: {len(field_def)}")
                            field_def = MetadataField(MetadataFieldType(field_type), indexed=should_index,
                                                      required=required, default_value=default_value)

                        if field_name not in current_schema:
                            # New field - add it
                            try:
                                # Store schema definition
                                conn.execute("""INSERT OR REPLACE INTO metadata_schema 
                                    (field_name, field_type, indexed, required, default_value)
                                    VALUES (?, ?, ?, ?, ?)
                                """, (
                                    field_name,
                                    field_def.type.value,
                                    field_def.indexed,
                                    field_def.required,
                                    field_def.default_value
                                ))

                                # Add column to documents table
                                self._add_metadata_column(conn, field_name, field_def)

                                changes['added_fields'].append(field_name)

                                # Populate default values if specified and we have existing documents
                                if field_def.default_value is not None:
                                    populated_info = self._populate_field_defaults(conn, field_name, field_def)
                                    if populated_info['rows_updated'] > 0:
                                        changes['populated_defaults'].append(populated_info)

                                # Add warning for nullable new fields on existing documents
                                doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                                if doc_count > 0 and field_def.default_value is None:
                                    changes['warnings'].append(
                                        f"New field '{field_name}' added to database with existing documents. "
                                        f"Existing documents will have NULL values."
                                    )

                            except Exception as e:
                                changes['errors'].append(f"Failed to add field '{field_name}': {str(e)}")

                        else:
                            # Existing field - check if it needs updates
                            current_field = current_schema[field_name]
                            field_changed = False
                            change_details = {}

                            # Check if any properties changed
                            if current_field.type != field_def.type:
                                change_details['type'] = {'old': current_field.type.value, 'new': field_def.type.value}
                                field_changed = True

                            if current_field.indexed != field_def.indexed:
                                change_details['indexed'] = {'old': current_field.indexed, 'new': field_def.indexed}
                                field_changed = True

                            if current_field.required != field_def.required:
                                change_details['required'] = {'old': current_field.required, 'new': field_def.required}
                                field_changed = True

                            if current_field.default_value != field_def.default_value:
                                change_details['default_value'] = {
                                    'old': current_field.default_value, 'new': field_def.default_value
                                }
                                field_changed = True

                            if field_changed:
                                try:
                                    # Update schema definition
                                    conn.execute("""UPDATE metadata_schema 
                                        SET field_type = ?, indexed = ?, required = ?, default_value = ?
                                        WHERE field_name = ?
                                    """, (
                                        field_def.type.value,
                                        field_def.indexed,
                                        field_def.required,
                                        field_def.default_value,
                                        field_name
                                    ))

                                    # Handle index changes
                                    if 'indexed' in change_details:
                                        if field_def.indexed:
                                            # Add index
                                            index_name = f'idx_documents_{field_name}'
                                            try:
                                                conn.execute(
                                                    f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({field_name})')
                                            except Exception as e:
                                                changes['errors'].append(
                                                    f"Failed to create index on '{field_name}': {str(e)}")
                                        else:
                                            # Remove index (if it exists)
                                            index_name = f'idx_documents_{field_name}'
                                            try:
                                                conn.execute(f'DROP INDEX IF EXISTS {index_name}')
                                            except Exception as e:
                                                changes['errors'].append(
                                                    f"Failed to drop index on '{field_name}': {str(e)}")

                                    if 'type' in change_details:
                                        old_type_str = change_details['type']['old']
                                        new_type_str = change_details['type']['new']

                                        try:
                                            # Map MetadataFieldType to SQLite types
                                            sqlite_type_map = {
                                                'text': 'TEXT',
                                                'integer': 'INTEGER',
                                                'real': 'REAL',
                                                'boolean': 'BOOLEAN',
                                                'date': 'TEXT',
                                                'json': 'TEXT'
                                            }

                                            new_sqlite_type = sqlite_type_map.get(new_type_str, 'TEXT')

                                            # Perform the type change using table rebuild
                                            self._rebuild_table_for_column_type_change(conn, field_name, new_sqlite_type)

                                            logger.info(f"Changed column '{field_name}' data type from {old_type_str} "
                                                        f"to {new_type_str}")

                                        except Exception as e:
                                            changes['errors'].append(
                                                f"Failed to change column type for '{field_name}': {str(e)}")
                                            changes['warnings'].append(
                                                f"Type change failed for '{field_name}', schema updated but column type unchanged")

                                    changes['modified_fields'].append({
                                        'field_name': field_name,
                                        'changes': change_details
                                    })

                                except Exception as e:
                                    changes['errors'].append(f"Failed to modify field '{field_name}': {str(e)}")

                    # STEP 2: Perform column remapping (AFTER new columns are created)
                    if column_mapping:
                        remapping_changes = self._perform_column_remapping(
                            conn, column_mapping, current_schema, new_schema
                        )
                        changes['remapped_columns'] = remapping_changes['remapped_columns']
                        changes['warnings'].extend(remapping_changes['warnings'])
                        changes['errors'].extend(remapping_changes['errors'])

                    # STEP 3: Handle removed fields (after remapping)
                    for field_name in current_schema:
                        if field_name not in new_schema:
                            try:
                                # Remove from schema table (even if it was remapped)
                                conn.execute("DELETE FROM metadata_schema WHERE field_name = ?", (field_name,))
                                changes['removed_fields'].append(field_name)

                                logger.info(f"Removed {field_name} from metadata_schema")

                                # Skip dropping the actual column if it was remapped
                                if column_mapping and field_name in column_mapping:
                                    logger.info(f"Skipping column drop for '{field_name}' as it was remapped to '{column_mapping[field_name]}'")
                                    continue

                                # Optionally drop the actual column
                                if drop_columns:
                                    try:
                                        # First check if column actually exists
                                        cursor = conn.execute("PRAGMA table_info(documents)")
                                        existing_columns = {row[1] for row in cursor.fetchall()}

                                        if field_name in existing_columns:
                                            # First drop any FTS triggers and tables
                                            current_field = current_schema.get(field_name)
                                            if current_field and getattr(current_field, 'fts_enabled', False):
                                                fts_table_name = f'fts_{field_name}'
                                                try:
                                                    # Drop FTS triggers
                                                    conn.execute(f'DROP TRIGGER IF EXISTS fts_{field_name}_insert')
                                                    conn.execute(f'DROP TRIGGER IF EXISTS fts_{field_name}_update')
                                                    conn.execute(f'DROP TRIGGER IF EXISTS fts_{field_name}_delete')
                                                    # Drop FTS table
                                                    conn.execute(f'DROP TABLE IF EXISTS {fts_table_name}')
                                                    logger.debug(f"Dropped FTS triggers and table for '{field_name}'")
                                                except Exception as e:
                                                    logger.warning(f"Failed to drop FTS components for '{field_name}': {e}")

                                            # Drop any indexes on this column
                                            index_name = f'idx_documents_{field_name}'
                                            try:
                                                conn.execute(f'DROP INDEX IF EXISTS {index_name}')
                                                logger.debug(f"Dropped index {index_name} before dropping column")
                                            except Exception:
                                                pass  # Index might not exist, that's okay

                                            # Now drop the column - use native DROP COLUMN if supported, otherwise rebuild table
                                            if self._supports_drop_column(conn):
                                                conn.execute(f'ALTER TABLE documents DROP COLUMN {field_name}')
                                                logger.info(f"Dropped column '{field_name}' using native DROP COLUMN")
                                            else:
                                                self._rebuild_table_for_column_drop(conn, field_name)
                                                logger.info(f"Dropped column '{field_name}' using table rebuild")
                                            
                                            changes['dropped_columns'].append(field_name)
                                        else:
                                            changes['warnings'].append(
                                                f"Column '{field_name}' not found in table, skipping drop")

                                    except Exception as e:
                                        changes['errors'].append(f"Failed to drop column '{field_name}': {str(e)}")
                                        changes['warnings'].append(
                                            f"Column '{field_name}' could not be dropped but was removed from schema. "
                                            f"Column data may still exist in the table."
                                        )
                            except Exception as e:
                                changes['errors'].append(f"Failed to remove field '{field_name}' from schema: {str(e)}")

                    conn.commit()

                    # Update in-memory schema
                    self.metadata_fields = new_schema.copy()

                except Exception as e:
                    conn.rollback()
                    changes['errors'].append(f"Schema update failed: {str(e)}")
                    raise

            return changes

    def _validate_column_mapping(
            self,
            column_mapping: Dict[str, str],
            current_schema: Dict[str, MetadataField],
            new_schema: Dict[str, MetadataField],
            reserved_columns: list
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
        if old_type == MetadataFieldType.BOOLEAN and new_type in (MetadataFieldType.INTEGER,
                                                                  MetadataFieldType.REAL):
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
            new_schema: Dict[str, MetadataField]
            ) -> Dict[str, Any]:
        """Perform the actual column remapping operations"""
        remap_changes = {
            'remapped_columns': [],
            'warnings': [],
            'errors': []
        }

        for old_col, new_col in column_mapping.items():
            try:
                # The new column should already exist (created in Step 1)
                # Just verify it exists
                cursor = conn.execute("PRAGMA table_info(documents)")
                existing_columns = {row[1] for row in cursor.fetchall()}

                if new_col not in existing_columns:
                    raise ValueError(
                        f"Target column '{new_col}' does not exist. It should have been created in the new schema.")

                # Transfer data from old column to new column
                old_field_def = current_schema[old_col]
                new_field_def = new_schema[new_col]

                rows_transferred = self._transfer_column_data(conn, old_col, new_col,
                                                              old_field_def, new_field_def)

                logger.info(f"Remapped column {old_col} to {new_col}")

                remap_changes['remapped_columns'].append({
                    'old_column': old_col,
                    'new_column': new_col,
                    'rows_transferred': rows_transferred
                })

            except Exception as e:
                remap_changes['errors'].append(
                    f"Failed to remap column '{old_col}' to '{new_col}': {str(e)}"
                )

        return remap_changes

    def _transfer_column_data(
            self,
            conn: sqlite3.Connection,
            old_col: str,
            new_col: str,
            old_field: MetadataField,
            new_field: MetadataField
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
        transfer_sql = self._get_transfer_sql(old_col, new_col, old_field.type, new_field.type)

        # Execute the transfer
        cursor = conn.execute(transfer_sql)
        return cursor.rowcount

    @staticmethod
    def _get_transfer_sql(
            old_col: str,
            new_col: str,
            old_type: MetadataFieldType,
            new_type: MetadataFieldType
            ) -> str:
        """Generate SQL for transferring data between columns with type conversion.

        **Warning:** input must be pre-validated before using this function.
        """

        if old_type == new_type:
            # Direct copy for same types
            return f"UPDATE documents SET {new_col} = {old_col} WHERE {old_col} IS NOT NULL"

        elif old_type == MetadataFieldType.TEXT:
            # TEXT to other types - direct copy (SQLite will handle conversion)
            return f"UPDATE documents SET {new_col} = {old_col} WHERE {old_col} IS NOT NULL"

        elif old_type == MetadataFieldType.INTEGER and new_type == MetadataFieldType.REAL:
            # Integer to real - direct copy
            return f"UPDATE documents SET {new_col} = CAST({old_col} AS REAL) WHERE {old_col} IS NOT NULL"

        elif old_type == MetadataFieldType.BOOLEAN and new_type == MetadataFieldType.INTEGER:
            # Boolean to integer
            return f"UPDATE documents SET {new_col} = CAST({old_col} AS INTEGER) WHERE {old_col} IS NOT NULL"

        elif old_type == MetadataFieldType.BOOLEAN and new_type == MetadataFieldType.REAL:
            # Boolean to real
            return f"UPDATE documents SET {new_col} = CAST({old_col} AS REAL) WHERE {old_col} IS NOT NULL"

        elif old_type == MetadataFieldType.JSON and new_type == MetadataFieldType.TEXT:
            # JSON to text - direct copy (it's already stored as text in SQLite)
            return f"UPDATE documents SET {new_col} = {old_col} WHERE {old_col} IS NOT NULL"

        else:
            # Default case - try direct copy and let SQLite handle it
            return f"UPDATE documents SET {new_col} = {old_col} WHERE {old_col} IS NOT NULL"

    @staticmethod
    def _populate_field_defaults(
            conn: sqlite3.Connection,
            field_name: str,
            field_def: MetadataField
            ) -> Dict[str, Any]:
        """Populate default values for a new field in existing documents"""
        if field_def.default_value is None:
            return {'field_name': field_name, 'rows_updated': 0}

        # Prepare the default value based on field type
        if field_def.type == MetadataFieldType.JSON:
            default_value = json.dumps(field_def.default_value)
        elif field_def.type in (MetadataFieldType.TEXT, MetadataFieldType.DATE):
            default_value = str(field_def.default_value)
        else:
            default_value = field_def.default_value

        # Update existing documents with the default value
        cursor = conn.execute(
            f"UPDATE documents SET {field_name} = ? WHERE {field_name} IS NULL",
            (default_value,)
        )

        return {
            'field_name': field_name,
            'rows_updated': cursor.rowcount,
            'default_value': field_def.default_value
        }

    # Async methods for DatabaseSchema
    async def initialize_async(self, metadata_schema: Optional[Dict[str, MetadataField]] = None, db_connection = None) -> None:
        """Initialize database schema asynchronously"""

        # Determine if we need to manage the connection lifecycle
        owns_connection = db_connection is None

        if owns_connection:
            db_connection = await aiosqlite.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        try:
            # Enable foreign keys
            await db_connection.execute("PRAGMA foreign_keys = ON")

            # Create base tables
            for table_name, ddl in self.BASE_SCHEMA.items():
                await db_connection.execute(ddl)

            # Create base indexes
            for index_ddl in self.BASE_INDEXES:
                await db_connection.execute(index_ddl)

            # Set up metadata schema if provided
            if metadata_schema:
                await self._setup_metadata_schema_async(db_connection, metadata_schema)

            await db_connection.commit()
        finally:
            if owns_connection and db_connection:
                await db_connection.close()

    async def _setup_metadata_schema_async(self, conn, schema: Dict[str, MetadataField]) -> None:
        """Set up metadata schema and add columns to documents table asynchronously"""

        # Validate that no metadata field names conflict with reserved columns
        for field_name in schema.keys():
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
                        f"(field_type, should_index[, required, default_value]). Found: {len(field_def)}")
                field_def = MetadataField(MetadataFieldType(field_type), indexed=should_index,
                                          required=required, default_value=default_value)

            # Store schema definition
            await conn.execute("""INSERT OR REPLACE INTO metadata_schema 
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
            await self._add_metadata_column_async(conn, field_name, field_def)

        self.metadata_fields = schema

    async def _add_metadata_column_async(self, conn, field_name: str, field_def: MetadataField) -> None:
        """Add a metadata column to the documents table asynchronously"""
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
        cursor = await conn.execute("PRAGMA table_info(documents)")
        existing_columns = {row[1] for row in await cursor.fetchall()}

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
            await conn.execute(ddl)

            logger.info(f"Added new column: {field_name} {sqlite_type}{default_clause}")
            # Create index if requested
            if field_def.indexed:
                index_name = f'idx_documents_{field_name}'
                await conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({field_name})')

            # Create FTS table if requested
            if field_def.fts_enabled and field_def.type == MetadataFieldType.TEXT:
                fts_table_name = f'fts_{field_name}'
                # Check if trigram tokenizer is available, fall back to default if not
                if await self._check_trigram_tokenizer_availability_async(conn):
                    tokenizer_clause = "tokenize='trigram'"
                    logger.debug(f"Using trigram tokenizer for FTS table: {fts_table_name}")
                else:
                    tokenizer_clause = ""  # Use default tokenizer
                    logger.debug(f"Using default tokenizer for FTS table: {fts_table_name}")
                
                await conn.execute(f'''
                    CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table_name} 
                    USING fts5(document_id, content{', ' + tokenizer_clause if tokenizer_clause else ''})
                ''')
                logger.info(f"Created FTS table: {fts_table_name}")

                # Create triggers to keep FTS in sync
                await conn.execute(f'''
                    CREATE TRIGGER IF NOT EXISTS fts_{field_name}_insert 
                    AFTER INSERT ON documents
                    WHEN NEW.{field_name} IS NOT NULL
                    BEGIN
                        INSERT INTO {fts_table_name}(document_id, content) 
                        VALUES (NEW.id, NEW.{field_name});
                    END
                ''')

                await conn.execute(f'''
                    CREATE TRIGGER IF NOT EXISTS fts_{field_name}_update 
                    AFTER UPDATE OF {field_name} ON documents
                    WHEN NEW.{field_name} IS NOT NULL
                    BEGIN
                        DELETE FROM {fts_table_name} WHERE document_id = NEW.id;
                        INSERT INTO {fts_table_name}(document_id, content) 
                        VALUES (NEW.id, NEW.{field_name});
                    END
                ''')

                await conn.execute(f'''
                    CREATE TRIGGER IF NOT EXISTS fts_{field_name}_delete 
                    AFTER DELETE ON documents
                    BEGIN
                        DELETE FROM {fts_table_name} WHERE document_id = OLD.id;
                    END
                ''')

    async def load_metadata_schema_async(self, db_connection: Optional[aiosqlite.Connection] = None) -> Dict[str, MetadataField]:
        """Load metadata schema from database asynchronously"""

        # Determine if we need to manage the connection lifecycle
        owns_connection = db_connection is None

        if owns_connection:
            db_connection = await aiosqlite.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        conn = db_connection
        try:
            cursor = await conn.execute("SELECT * FROM metadata_schema")
            schema = {}

            rows = await cursor.fetchall()
            for row in rows:
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
        finally:
            if owns_connection and db_connection:
                await db_connection.close()

    async def update_metadata_schema_async(
            self,
            new_schema: Dict[str, MetadataField],
            db_connection: Optional[aiosqlite.Connection] = None,
            drop_columns: bool = False,
            column_mapping: Optional[Dict[str, str]] = None
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
            db_connection = await aiosqlite.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)

        changes = {
            'added_fields': [],
            'removed_fields': [],
            'modified_fields': [],
            'populated_defaults': [],
            'dropped_columns': [],
            'remapped_columns': [],
            'warnings': [],
            'errors': []
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
                    changes['errors'].extend(validation_errors)
                    return changes
            await db_connection.execute("BEGIN TRANSACTION")

            # Process column remapping first
            if column_mapping:
                remapped = await self._perform_column_remapping_async(
                    db_connection, column_mapping, new_schema, current_schema
                )
                changes['remapped_columns'].extend(remapped)

                # Process added and modified fields
                for field_name, field_def in new_schema.items():
                    if field_name not in current_schema:
                        # Add new field
                        await self._add_metadata_column_async(db_connection, field_name, field_def)
                        changes['added_fields'].append(field_name)

                        # Populate defaults if needed
                        populated = await self._populate_field_defaults_async(
                            db_connection, field_name, field_def
                        )
                        if populated:
                            changes['populated_defaults'].append(populated)
                    else:
                        # Check for modifications
                        old_def = current_schema[field_name]
                        modifications = []

                        # Check type change
                        if old_def.type != field_def.type:
                            if not self._are_types_compatible(old_def.type, field_def.type):
                                changes['warnings'].append(
                                    f"Type change for '{field_name}' from {old_def.type.value} "
                                    f"to {field_def.type.value} may cause data loss"
                                )
                            modifications.append({
                                'change': 'type',
                                'from': old_def.type.value,
                                'to': field_def.type.value
                            })

                        # Check index change
                        if old_def.indexed != field_def.indexed:
                            index_name = f'idx_documents_{field_name}'
                            if field_def.indexed:
                                await db_connection.execute(
                                    f'CREATE INDEX IF NOT EXISTS {index_name} ON documents({field_name})'
                                )
                                modifications.append({'change': 'index', 'action': 'added'})
                            else:
                                await db_connection.execute(f'DROP INDEX IF EXISTS {index_name}')
                                modifications.append({'change': 'index', 'action': 'removed'})

                        # Check required change
                        if old_def.required != field_def.required:
                            modifications.append({
                                'change': 'required',
                                'from': old_def.required,
                                'to': field_def.required
                            })

                            # If making field required, populate defaults for NULLs
                            if field_def.required and not old_def.required:
                                populated = await self._populate_field_defaults_async(
                                    db_connection, field_name, field_def, old_def
                                )
                                if populated:
                                    changes['populated_defaults'].append(populated)

                        # Check default value change
                        if old_def.default_value != field_def.default_value:
                            modifications.append({
                                'change': 'default_value',
                                'from': old_def.default_value,
                                'to': field_def.default_value
                            })

                        if modifications:
                            changes['modified_fields'].append({
                                'field_name': field_name,
                                'modifications': modifications
                            })

                # Process removed fields
                removed_fields = set(current_schema.keys()) - set(new_schema.keys())

                # Don't mark remapped source columns as removed
                if column_mapping:
                    removed_fields -= set(column_mapping.keys())

                for field_name in removed_fields:
                    changes['removed_fields'].append(field_name)
                    if drop_columns:
                        # SQLite doesn't support DROP COLUMN in older versions
                        # This is a placeholder - would need table recreation for full support
                        changes['warnings'].append(
                            f"Column '{field_name}' marked for removal but SQLite "
                            "requires table recreation for column drops"
                        )

                    # Remove from metadata_schema table
                    await db_connection.execute(
                        "DELETE FROM metadata_schema WHERE field_name = ?", (field_name,)
                    )

                # Update metadata_schema table for all fields in new schema
                for field_name, field_def in new_schema.items():
                    await db_connection.execute("""
                        INSERT OR REPLACE INTO metadata_schema 
                        (field_name, field_type, indexed, required, default_value)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        field_name,
                        field_def.type.value,
                        field_def.indexed,
                        field_def.required,
                        json.dumps(field_def.default_value) if field_def.default_value is not None else None
                    ))

                await db_connection.commit()

            # Update in-memory schema
            self.metadata_fields = new_schema.copy()

        except Exception as e:
            try:
                await db_connection.execute("ROLLBACK")
            except:
                pass  # Ignore rollback errors
            changes['errors'].append(f"Schema update failed: {str(e)}")
            raise
        finally:
            if owns_connection and db_connection:
                await db_connection.close()

        return changes

    async def _validate_column_mapping_async(
            self,
            current_schema: Dict[str, MetadataField],
            new_schema: Dict[str, MetadataField],
            column_mapping: Dict[str, str]
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
            current_schema: Dict[str, MetadataField]
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

            remapped.append({
                'from': old_name,
                'to': new_name,
                'rows_affected': rows_affected
            })

            # Remove old column from metadata_schema
            await conn.execute("DELETE FROM metadata_schema WHERE field_name = ?", (old_name,))

        return remapped

    async def _transfer_column_data_async(
            self,
            conn: aiosqlite.Connection,
            old_name: str,
            new_name: str,
            field_def: MetadataField,
            old_field_def: MetadataField
    ) -> int:
        """Transfer data from old column to new column asynchronously"""
        # Get the appropriate SQL for data transfer
        transfer_sql = self._get_transfer_sql(old_name, new_name, field_def.type, old_field_def.type)

        # Execute the transfer
        cursor = await conn.execute(transfer_sql)
        return cursor.rowcount

    @staticmethod
    async def _populate_field_defaults_async(
            conn: aiosqlite.Connection,
            field_name: str,
            field_def: MetadataField,
            old_field_def: Optional[MetadataField] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Populate default values for a field where NULL values exist asynchronously.

        This handles both new fields and fields being made required.
        """
        if field_def.default_value is None:
            return None

        # Build WHERE clause
        where_clause = f"{field_name} IS NULL"

        # Don't populate defaults if the field already had a default
        # (existing NULLs were intentional)
        if old_field_def and old_field_def.default_value is not None:
            return None

        # Check how many rows would be affected
        cursor = await conn.execute(f"SELECT COUNT(*) FROM documents WHERE {where_clause}")
        row = await cursor.fetchone()
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

        # Update the NULL values
        await conn.execute(
            f"UPDATE documents SET {field_name} = ? WHERE {where_clause}",
            (sql_value,)
        )

        return {
            'field_name': field_name,
            'rows_updated': null_count,
            'default_value': field_def.default_value
        }


def get_common_metadata_schemas(
        schema: Optional[str] = None
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
            "abstract": MetadataField(type=MetadataFieldType.TEXT, indexed=True, embedding_enabled=True, fts_enabled=True),
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