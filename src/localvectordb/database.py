# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/database.py
"""
LocalVectorDB v1.0

This module contains the main LocalVectorDB v1.0 implementation with:
- Document-first API that hides chunking complexity
- Direct SQLite implementation
- Unified query interface with normalized scoring
- Position-tracking chunking for perfect reconstruction
- Structured metadata with indexed columns
- Plugin-based embedding providers
"""

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any

import faiss
import numpy as np

from localvectordb.chunking import ChunkerFactory
from localvectordb.core import (
    DatabaseSchema, ConnectionPool, Document, Chunk, QueryResult,
    MetadataField, MetadataFieldType, ChunkPosition, get_common_metadata_schemas
)
from localvectordb.embeddings import EmbeddingRegistry
from localvectordb.exceptions import DatabaseNotFoundError, DuplicateDocumentIDError, DatabaseError
from localvectordb.utils import get_system_version

logger = logging.getLogger(__name__)


class LocalVectorDB:
    """
    Document-first vector database with SQLite + FAISS + embeddings

    This is the main interface for LocalVectorDB v1.0, designed around documents
    rather than chunks. All chunking is handled internally.

    Parameters
    ----------
    name : str
        Database name (used for file naming)
    base_path : str, optional
        Directory to store database files, by default ".lvdb"
    metadata_schema : str | Dict[str, MetadataField], optional
        Schema definition for metadata fields
    doc_id_pattern : str, optional
        Pattern for auto-generating document IDs, by default "doc_{idx}"
    chunk_id_pattern : str, optional
        Pattern for chunk IDs, by default "{doc_id}:{chunk_idx}"
    embedding_provider : str, optional
        Embedding provider name, by default "ollama"
    embedding_model : str, optional
        Embedding model name, by default "nomic-embed-text"
    embedding_config : Dict[str, Any], optional
        Configuration for embedding provider
    chunking_method : str, optional
        Chunking method, by default "sentences"
    chunk_size : int, optional
        Maximum tokens per chunk, by default 500
    chunk_overlap : int, optional
        Overlap between chunks, by default 1
    enable_gpu : bool, optional
        Whether to use GPU for FAISS, by default False
    enable_fts : bool, optional
        Whether to enable full-text search, by default True
    create_if_not_exists: bool, default = True
        If False, raises DatabaseNotFoundError if the database doesn't exist.
    """

    def __init__(
            self,
            name: str,
            base_path: Union[str, Path] = ".lvdb",
            *,
            # Metadata schema
            metadata_schema: Optional[Dict[str, MetadataField]] = None,

            # ID generation patterns
            doc_id_pattern: str = "doc_{idx}",
            chunk_id_pattern: str = "{doc_id}:{chunk_idx}",

            # Embedding configuration
            embedding_provider: str = "ollama",
            embedding_model: str = "nomic-embed-text",
            embedding_config: Optional[Dict[str, Any]] = None,

            # Chunking configuration
            chunking_method: str = "sentences",
            chunk_size: int = 500,
            chunk_overlap: int = 1,

            # Performance settings
            enable_gpu: bool = False,
            enable_fts: bool = True,
            connection_pool_size: int = 10,

            # Other
            create_if_not_exists: bool = True,
    ):
        self.name = name
        self.is_memory_only = (name == ":memory:" or base_path == ":memory:")

        if self.is_memory_only:
            self.base_path = None
            self.db_path = ":memory:"
            self.index_path = None
            logger.info("Creating in-memory database")
        else:
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Database files
            self.db_path = self.base_path / f"{name}.sqlite"
            self.index_path = self.base_path / f"{name}.faiss"

            if not create_if_not_exists and not self.db_path.exists():
                raise DatabaseNotFoundError(f"Database: {name} in {base_path} could not be found.")

        # Configuration
        if isinstance(metadata_schema, str):
            self.metadata_schema = get_common_metadata_schemas(metadata_schema)
        else:
            self.metadata_schema = metadata_schema or {}
        self.doc_id_pattern = doc_id_pattern
        self.chunk_id_pattern = chunk_id_pattern

        # Chunking setup
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = ChunkerFactory.create_chunker(
            chunking_method, chunk_size, chunk_overlap
        )

        # Initialize embedding provider
        embedding_config = embedding_config or {}
        self.embedding_provider = EmbeddingRegistry.create_provider(
            embedding_provider, embedding_model, **embedding_config
        )

        # Validate embedding model
        if not self.embedding_provider.validate_model():
            raise ValueError(f"Embedding model '{embedding_model}' is not available")

        self.embedding_dimension = self.embedding_provider.get_dimension()

        # Database setup
        self.schema = DatabaseSchema(self.db_path)
        self.connection_pool = ConnectionPool(self.db_path, connection_pool_size)

        # Initialize schema
        self.schema.initialize(self.metadata_schema, db_connection=self.connection_pool.get_connection())

        # Load existing metadata schema if database already exists
        if not self.is_memory_only and self.db_path.exists():
            existing_schema = self.schema.load_metadata_schema(db_connection=self.connection_pool.get_connection())
            self.metadata_schema.update(existing_schema)

        # FTS setup
        self.fts_enabled = False
        if enable_fts:
            self._init_fts()

        # FAISS index setup
        self._init_faiss_index(enable_gpu)

        # Threading
        self._lock = threading.RLock()

        # State
        # self._closed = False
        self._next_doc_id = self._load_next_doc_id()

        # Save configuration
        self._save_config()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, 'connection_pool') and not self.connection_pool.closed:
                self.close()
        except Exception:
            pass  # Ignore errors during cleanup

    @property
    def closed(self):
        return self.connection_pool.closed

    def _check_fts5_availability(self) -> bool:
        """Check if FTS5 is available in SQLite"""
        try:
            with self.connection_pool.get_connection() as conn:
                # Try to create a temporary FTS5 table
                conn.execute("CREATE VIRTUAL TABLE temp.fts5_test USING fts5(content)")
                conn.execute("DROP TABLE temp.fts5_test")
                return True
        except sqlite3.OperationalError:
            logger.warning("SQLite FTS5 extension not available. Keyword search will be disabled.")
            return False
        except Exception as e:
            logger.error(f"Error checking FTS5 availability: {e}")
            return False

    def _init_fts(self):
        """Initialize Full-Text Search (FTS5) if available"""
        if not self._check_fts5_availability():
            self.fts_enabled = False
            return

        try:
            with self.connection_pool.get_connection() as conn:
                # Create FTS virtual table for documents
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                        id,
                        content,
                        content='documents',
                        content_rowid='rowid'
                    )
                """)

                # Create FTS virtual table for chunks
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        document_id,
                        content,
                        content='chunks',
                        content_rowid='id'
                    )
                """)

                # Create triggers for documents FTS
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                        INSERT INTO documents_fts(rowid, id, content) 
                        VALUES (new.rowid, new.id, new.content);
                    END
                """)

                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                        DELETE FROM documents_fts WHERE rowid = old.rowid;
                    END
                """)

                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                        DELETE FROM documents_fts WHERE rowid = old.rowid;
                        INSERT INTO documents_fts(rowid, id, content) 
                        VALUES (new.rowid, new.id, new.content);
                    END
                """)

                # Create triggers for chunks FTS
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, document_id, content) 
                        VALUES (new.id, new.document_id, new.content);
                    END
                """)

                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.id;
                    END
                """)

                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.id;
                        INSERT INTO chunks_fts(rowid, document_id, content) 
                        VALUES (new.id, new.document_id, new.content);
                    END
                """)

                conn.commit()
                self.fts_enabled = True
                logger.info("FTS5 initialized successfully")

        except Exception as e:
            logger.error(f"Error setting up FTS5: {e}")
            self.fts_enabled = False

    def _sanitize_fts_query(self, query: str) -> str:
        """
        Sanitize query for FTS5 while preserving useful search capabilities

        This method tries to balance safety with search effectiveness by:
        1. Supporting exact phrase matching with quotes
        2. Using AND logic by default (all terms must match)
        3. Safely handling FTS5 operators
        4. Falling back to safe term-by-term search for complex queries
        """
        if not query or not query.strip():
            return ""

        query = query.strip()

        # If the entire query is already quoted, treat as exact phrase
        if query.startswith('"') and query.endswith('"') and query.count('"') == 2:
            # Validate the phrase doesn't contain FTS5 special chars that could break things
            inner_query = query[1:-1]
            if self._is_safe_phrase(inner_query):
                return query
            else:
                # Fall back to safe handling
                return f'"{self._clean_term(inner_query)}"'

        # Check if query contains quotes for phrase matching
        if '"' in query:
            return self._handle_phrase_query(query)

        # Check if query contains basic boolean operators
        if any(op in query.upper() for op in [' AND ', ' OR ', ' NOT ']):
            return self._handle_boolean_query(query)

        # Simple multi-term query - default to AND behavior for better relevance
        terms = query.split()
        if len(terms) == 1:
            # Single term - clean and return
            clean_term = self._clean_term(terms[0])
            return f'"{clean_term}"' if clean_term else ""
        else:
            # Multiple terms - use AND logic (all terms must be present)
            clean_terms = []
            for term in terms:
                clean_term = self._clean_term(term)
                if clean_term:
                    clean_terms.append(f'"{clean_term}"')

            return " AND ".join(clean_terms) if clean_terms else ""

    @staticmethod
    def _is_safe_phrase(phrase: str) -> bool:
        """Check if a phrase is safe to use in FTS5 without additional escaping"""
        # Avoid phrases with FTS5 special characters that could cause issues
        dangerous_chars = ['*', ':', '^', '(', ')', '[', ']', '{', '}']
        return not any(char in phrase for char in dangerous_chars)

    @staticmethod
    def _clean_term(term: str) -> str:
        """Clean a single term for safe FTS5 usage"""
        # Remove FTS5 special characters but preserve basic word characters
        # Keep unicode word characters, numbers, hyphens, apostrophes
        clean_term = re.sub(r'[^\w\s\'-]', '', term, flags=re.UNICODE).strip()
        return clean_term

    def _handle_phrase_query(self, query: str) -> str:
        """Handle queries that contain quoted phrases"""
        # Split on quotes to separate phrases from individual terms
        parts = []
        in_quote = False
        current_part = ""

        i = 0
        while i < len(query):
            char = query[i]
            if char == '"':
                if in_quote:
                    # End of phrase
                    if current_part.strip():
                        clean_phrase = self._clean_term(current_part)
                        if clean_phrase:
                            parts.append(f'"{clean_phrase}"')
                    current_part = ""
                    in_quote = False
                else:
                    # Start of phrase - first process any pending non-quoted content
                    if current_part.strip():
                        # Split into terms and add as AND
                        terms = current_part.split()
                        for term in terms:
                            clean_term = self._clean_term(term)
                            if clean_term:
                                parts.append(f'"{clean_term}"')
                    current_part = ""
                    in_quote = True
            else:
                current_part += char
            i += 1

        # Handle any remaining content
        if current_part.strip():
            if in_quote:
                # Unclosed quote - treat as phrase anyway
                clean_phrase = self._clean_term(current_part)
                if clean_phrase:
                    parts.append(f'"{clean_phrase}"')
            else:
                # Regular terms
                terms = current_part.split()
                for term in terms:
                    clean_term = self._clean_term(term)
                    if clean_term:
                        parts.append(f'"{clean_term}"')

        return " AND ".join(parts) if parts else ""

    def _handle_boolean_query(self, query: str) -> str:
        """Handle queries with AND/OR/NOT operators"""
        # For safety, we'll parse basic boolean queries but fall back to term-by-term
        # if the query is too complex

        # Replace boolean operators with standardized versions
        normalized = query.upper()
        normalized = re.sub(r'\bAND\b', ' AND ', normalized)
        normalized = re.sub(r'\bOR\b', ' OR ', normalized)
        normalized = re.sub(r'\bNOT\b', ' NOT ', normalized)

        # Split by operators while preserving them
        tokens = re.split(r'(\s+(?:AND|OR|NOT)\s+)', normalized)

        # Clean each non-operator token
        cleaned_tokens = []
        for token in tokens:
            token = token.strip()
            if token in ['AND', 'OR', 'NOT']:
                cleaned_tokens.append(token)
            elif token:
                # Regular term - clean it
                clean_term = self._clean_term(token)
                if clean_term:
                    cleaned_tokens.append(f'"{clean_term}"')

        # Validate the structure (operators should be between terms)
        if self._is_valid_boolean_structure(cleaned_tokens):
            return " ".join(cleaned_tokens)
        else:
            # Fall back to simple AND of all terms
            terms = re.split(r'\s+(?:AND|OR|NOT)\s+', query, flags=re.IGNORECASE)
            clean_terms = []
            for term in terms:
                clean_term = self._clean_term(term.strip())
                if clean_term:
                    clean_terms.append(f'"{clean_term}"')
            return " AND ".join(clean_terms) if clean_terms else ""

    def _is_valid_boolean_structure(self, tokens: List[str]) -> bool:
        """Check if boolean query structure is valid"""
        if not tokens:
            return False

        # Should start and end with terms, not operators
        if tokens[0] in ['AND', 'OR', 'NOT'] or tokens[-1] in ['AND', 'OR']:
            return False

        # Operators and terms should alternate (roughly)
        operator_count = sum(1 for token in tokens if token in ['AND', 'OR', 'NOT'])
        term_count = len(tokens) - operator_count

        # Should have roughly one fewer operator than terms
        return operator_count <= term_count

    def _init_faiss_index(self, enable_gpu: bool):
        """Initialize FAISS index with ID mapping support"""
        if self.index_path and self.index_path.exists():
            # Load existing index
            loaded_index = faiss.read_index(str(self.index_path))

            # Check if it's already an IndexIDMap
            if hasattr(loaded_index, 'id_map'):
                self.index = loaded_index
                logger.info(f"Loaded existing FAISS IndexIDMap with {self.index.ntotal} vectors")
            else:
                raise DatabaseError("Expected FAISS index to have `id_map` attribute. Invalid faiss index!")

        else:
            # Create new index with ID mapping
            base_index = faiss.IndexFlatL2(self.embedding_dimension)
            self.index = faiss.IndexIDMap(base_index)
            logger.info(f"Created new FAISS IndexIDMap with dimension {self.embedding_dimension}")

        # GPU setup
        if enable_gpu and faiss.get_num_gpus() > 0:
            # Note: GPU indices may not support all IndexIDMap operations
            try:
                self.index = faiss.index_cpu_to_all_gpus(self.index)
                logger.info("Moved FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Could not move IndexIDMap to GPU: {e}")
        elif enable_gpu:
            logger.warning("GPU requested but no GPUs available")

    def _load_next_doc_id(self) -> int:
        """Load the next document ID counter"""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute('SELECT value FROM config WHERE key = ?', ('next_doc_id',))
            row = cursor.fetchone()
            return int(row['value']) if row else 1

    def _save_next_doc_id(self):
        """Save the next document ID counter"""
        with self.connection_pool.get_connection() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
                ('next_doc_id', str(self._next_doc_id))
            )
            conn.commit()

    def _save_config(self):
        """Save database configuration"""
        config = {
            'embedding_provider': self.embedding_provider.provider_name,
            'embedding_model': self.embedding_provider.model,
            'embedding_dimension': self.embedding_dimension,
            'chunking_method': self.chunking_method,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'doc_id_pattern': self.doc_id_pattern,
            'chunk_id_pattern': self.chunk_id_pattern,
            'fts_enabled': str(self.fts_enabled),
            'version': get_system_version()
        }

        with self.connection_pool.get_connection() as conn:
            for key, value in config.items():
                conn.execute(
                    'INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
                    (key, str(value))
                )
            conn.commit()

    def _generate_doc_id(self) -> str:
        """Generate a new document ID"""
        doc_id = self.doc_id_pattern.format(idx=self._next_doc_id)
        self._next_doc_id += 1
        return doc_id

    def _generate_chunk_id(self, doc_id: str, chunk_idx: int) -> str:
        """Generate a chunk ID"""
        return self.chunk_id_pattern.format(doc_id=doc_id, chunk_idx=chunk_idx)

    def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100
    ) -> List[str]:
        """
        Insert or update documents in the database

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing, by default 100

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        with self._lock:
            # Normalize inputs
            if isinstance(documents, str):
                documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

            # Handle metadata
            if metadata is None:
                metadata = [{}] * len(documents)
            elif len(metadata) != len(documents):
                raise ValueError("Number of metadata entries must match number of documents")

            # Handle IDs
            if ids is None:
                ids = [self._generate_doc_id() for _ in documents]
            elif len(ids) != len(documents):
                raise ValueError("Number of IDs must match number of documents")

            # Validate metadata against schema
            self._validate_metadata_batch(metadata)

            # Process in batches
            result_ids = []
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadata[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                batch_result = self._upsert_batch(batch_docs, batch_meta, batch_ids)
                result_ids.extend(batch_result)

            # Save state
            self._save_next_doc_id()
            self.save()

            return result_ids

    def _validate_metadata_batch(self, metadata_batch: List[Dict[str, Any]]):
        """Validate metadata against schema"""
        for metadata in metadata_batch:
            self._validate_metadata(metadata)

    def _validate_metadata(self, metadata: Dict[str, Any]):
        """Validate a single metadata dict against schema"""
        for field_name, value in metadata.items():
            if field_name in self.metadata_schema:
                field_def = self.metadata_schema[field_name]
                # Type validation would go here
                # For now, we'll trust the user
                pass

        # Check required fields
        for field_name, field_def in self.metadata_schema.items():
            if field_def.required and field_name not in metadata:
                if field_def.default_value is not None:
                    metadata[field_name] = field_def.default_value
                else:
                    raise ValueError(f"Required metadata field '{field_name}' is missing")

    def _upsert_batch(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str]
    ) -> List[str]:
        """Upsert a batch of documents"""

        # Check which documents need updates
        docs_to_process = []
        embeddings_needed = []

        with self.connection_pool.get_connection() as conn:
            for doc_text, metadata, doc_id in zip(documents, metadata_batch, ids):
                # Check if document exists and needs update
                cursor = conn.execute(
                    'SELECT content_hash FROM documents WHERE id = ?', (doc_id,)
                )
                row = cursor.fetchone()

                new_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()

                if row is None or row['content_hash'] != new_hash:
                    # Document is new or content has changed
                    docs_to_process.append((doc_text, metadata, doc_id, new_hash))
                    embeddings_needed.append(doc_text)

        if not docs_to_process:
            return ids  # No changes needed

        # Generate chunks for documents that need processing
        all_chunks = []
        chunk_texts = []

        for doc_text, metadata, doc_id, content_hash in docs_to_process:
            chunks = self.chunker.chunk(doc_text)

            # Assign FAISS IDs (will be updated after adding to index)
            for chunk in chunks:
                chunk.faiss_id = None

            all_chunks.append((doc_id, chunks))
            chunk_texts.extend([chunk.content for chunk in chunks])

        # Generate embeddings for all chunks
        if chunk_texts:
            embeddings = self.embedding_provider.embed_sync(chunk_texts)
        else:
            embeddings = np.array([]).reshape(0, self.embedding_dimension)

        # Store everything in database
        with self.connection_pool.get_connection() as conn:
            try:
                # Start transaction
                conn.execute('BEGIN')

                embedding_idx = 0

                for doc_text, metadata, doc_id, content_hash in docs_to_process:
                    # Get old FAISS IDs before deletion
                    old_faiss_ids = []
                    cursor = conn.execute(
                        'SELECT faiss_id FROM chunks WHERE document_id = ? AND faiss_id IS NOT NULL',
                        (doc_id,)
                    )
                    old_faiss_ids = [row['faiss_id'] for row in cursor.fetchall()]

                    # Delete existing document and chunks
                    conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))

                    # Remove old vectors from FAISS index
                    if old_faiss_ids and hasattr(self.index, 'remove_ids'):
                        try:
                            ids_array = np.array(old_faiss_ids, dtype=np.int64)
                            self.index.remove_ids(ids_array)
                        except Exception as e:
                            logger.warning(f"Failed to remove old vectors from FAISS: {e}")

                    # Insert/update document
                    self._insert_document(conn, doc_id, doc_text, content_hash, metadata)

                    # Get chunks for this document
                    doc_chunks = next(chunks for did, chunks in all_chunks if did == doc_id)

                    # Add chunk embeddings to FAISS index
                    doc_embeddings = embeddings[embedding_idx:embedding_idx + len(doc_chunks)]
                    if len(doc_embeddings) > 0:
                        # Generate new FAISS IDs
                        # Use sequential IDs starting from current index size
                        start_faiss_id = self.index.ntotal
                        new_faiss_ids = np.arange(
                            start_faiss_id,
                            start_faiss_id + len(doc_chunks),
                            dtype=np.int64
                        )
                        self.index.add_with_ids(doc_embeddings, new_faiss_ids)

                        # Update chunk FAISS IDs
                        for i, chunk in enumerate(doc_chunks):
                            chunk.faiss_id = int(new_faiss_ids[i])

                    # Insert chunks
                    for chunk in doc_chunks:
                        self._insert_chunk(conn, doc_id, chunk)

                    embedding_idx += len(doc_chunks)

                # Commit transaction
                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

        return ids

    def _insert_document(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        content: str,
        content_hash: str,
        metadata: Dict[str, Any]
    ):
        """Insert a document with metadata"""

        # Build dynamic INSERT statement based on metadata schema
        columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
        values = [doc_id, content, content_hash, datetime.now(), datetime.now()]
        placeholders = ['?'] * len(columns)

        # Add metadata columns
        for field_name, value in metadata.items():
            if field_name in self.metadata_schema:
                columns.append(field_name)

                # Convert value based on field type
                field_def = self.metadata_schema[field_name]
                if field_def.type == MetadataFieldType.JSON:
                    values.append(json.dumps(value))
                elif field_def.type == MetadataFieldType.DATE:
                    # Assume value is a datetime or ISO string
                    if isinstance(value, datetime):
                        values.append(value.isoformat())
                    else:
                        values.append(str(value))
                else:
                    values.append(value)

                placeholders.append('?')

        sql = f"INSERT OR REPLACE INTO documents ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
        conn.execute(sql, values)

    def _insert_chunk(self, conn: sqlite3.Connection, doc_id: str, chunk: Chunk):
        """Insert a chunk"""
        conn.execute('''
            INSERT INTO chunks 
            (document_id, chunk_index, content, content_hash, start_pos, end_pos, start_line, 
            start_col, end_line, end_col, tokens, faiss_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id,
            chunk.index,
            chunk.content,
            chunk.content_hash,
            chunk.position.start,
            chunk.position.end,
            chunk.position.line,
            chunk.position.column,
            chunk.position.end_line,
            chunk.position.end_column,
            chunk.tokens,
            chunk.faiss_id
        ))

    def _get_existing_chunks(self, conn: sqlite3.Connection, doc_id: str) -> Dict[str, Chunk]:
        """Get existing chunks for a document, keyed by content hash"""
        cursor = conn.execute('''
            SELECT chunk_index, content, content_hash, start_pos, end_pos, start_line, start_col, tokens, faiss_id
            FROM chunks 
            WHERE document_id = ?
            ORDER BY chunk_index
        ''', (doc_id,))

        chunks_by_hash = {}
        for row in cursor.fetchall():
            position = ChunkPosition(
                start=row['start_pos'],
                end=row['end_pos'],
                line=row['start_line'],
                column=row['start_col']
            )

            chunk = Chunk(
                content=row['content'],
                position=position,
                tokens=row['tokens'],
                index=row['chunk_index'],
                faiss_id=row['faiss_id'],
                content_hash=row['content_hash']
            )
            chunks_by_hash[chunk.content_hash] = chunk

        return chunks_by_hash

    def insert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            errors: Literal["ignore", "raise"] = "raise",
            similarity_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Insert new documents into the database

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing, by default 100
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        similarity_threshold : Optional[float]
            If provided, skip chunks that are too similar to existing chunks.
            Value should be between 0-1 where 1.0 means identical.
            Setting to 0.95 makes sure you wouldn't overpopulate the database with similar information
            repeatedly (e.g. headings or boilerplate language) which could be discarded.

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        ValueError
            If errors="raise" and duplicate document IDs are found
        """
        with self._lock:
            # Normalize inputs
            if isinstance(documents, str):
                documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

            # Handle metadata
            if metadata is None:
                metadata = [{}] * len(documents)
            elif len(metadata) != len(documents):
                raise ValueError("Number of metadata entries must match number of documents")

            # Handle IDs
            if ids is None:
                ids = [self._generate_doc_id() for _ in documents]
            elif len(ids) != len(documents):
                raise ValueError("Number of IDs must match number of documents")

            # Validate metadata against schema
            self._validate_metadata_batch(metadata)

            # Check for existing document IDs
            existing_ids = set()
            with self.connection_pool.get_connection() as conn:
                if ids:
                    placeholders = ','.join(['?'] * len(ids))
                    cursor = conn.execute(f'SELECT id FROM documents WHERE id IN ({placeholders})', ids)
                    existing_ids = {row['id'] for row in cursor.fetchall()}

            # Handle ID conflicts
            docs_to_insert = []
            for doc, meta, doc_id in zip(documents, metadata, ids):
                if doc_id in existing_ids:
                    if errors == "raise":
                        raise DuplicateDocumentIDError(f"Document with ID '{doc_id}' already exists")
                    elif errors == "ignore":
                        logger.info(f"Skipping existing document ID: {doc_id}")
                        continue
                docs_to_insert.append((doc, meta, doc_id))

            if not docs_to_insert:
                return []  # No documents to insert

            # Process in batches
            result_ids = []
            for i in range(0, len(docs_to_insert), batch_size):
                batch = docs_to_insert[i:i + batch_size]
                batch_docs = [item[0] for item in batch]
                batch_meta = [item[1] for item in batch]
                batch_ids = [item[2] for item in batch]

                batch_result = self._insert_batch(batch_docs, batch_meta, batch_ids, similarity_threshold)
                result_ids.extend(batch_result)

            # Save state
            self._save_next_doc_id()
            self.save()

            return result_ids

    def _insert_batch(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            similarity_threshold: Optional[float] = None
    ) -> List[str]:
        """Insert a batch of documents with optional similarity filtering and chunk content hashing"""

        # Generate chunks for all documents
        all_chunks = []
        chunk_texts = []
        doc_chunk_mapping = []  # Track which chunks belong to which document

        for doc_text, metadata, doc_id in zip(documents, metadata_batch, ids):
            content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
            chunks = self.chunker.chunk(doc_text)

            # Track document info
            doc_info = (doc_text, metadata, doc_id, content_hash)

            for chunk in chunks:
                chunk.faiss_id = None
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
                doc_chunk_mapping.append(doc_info)

        if not chunk_texts:
            return []

        # Generate embeddings for all chunks
        embeddings = self.embedding_provider.embed_sync(chunk_texts)

        # Filter chunks by similarity if threshold provided
        chunks_to_keep = []
        embeddings_to_keep = []
        # docs_with_chunks = set()

        if similarity_threshold is not None and self.index.ntotal > 0:
            # Convert similarity threshold to distance threshold
            # similarity = 1 / (1 + distance), so distance = (1/similarity) - 1
            distance_threshold = (1.0 / max(similarity_threshold, 0.001)) - 1.0

            # Build a set of existing chunk hashes for faster lookup
            existing_chunk_hashes = set()
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute('SELECT DISTINCT content_hash FROM chunks')
                existing_chunk_hashes = {row['content_hash'] for row in cursor.fetchall()}

            for i, (chunk, embedding, doc_info) in enumerate(zip(all_chunks, embeddings, doc_chunk_mapping)):
                # First check if we already have a chunk with identical content
                if chunk.content_hash in existing_chunk_hashes:
                    logger.debug(f"Skipping chunk with identical content (hash: {chunk.content_hash[:8]}...)")
                    continue

                # Search for similar chunks in existing index
                distances, indices = self.index.search(embedding.reshape(1, -1), k=1)

                # Check if closest match is too similar
                if indices[0][0] != -1 and distances[0][0] < distance_threshold:
                    # Skip this chunk - too similar to existing content
                    logger.debug(f"Skipping similar chunk (distance: {distances[0][0]:.4f})")
                    continue

                chunks_to_keep.append(chunk)
                embeddings_to_keep.append(embedding)
                # docs_with_chunks.add(doc_info[2])  # doc_id
        else:
            # Keep all chunks
            chunks_to_keep = all_chunks
            embeddings_to_keep = embeddings
            # docs_with_chunks = {doc_info[2] for doc_info in doc_chunk_mapping}

        if not chunks_to_keep:
            logger.info("No chunks to insert after similarity filtering")
            return []

        # Group chunks by document
        doc_chunks_map = {}
        for chunk, embedding, doc_info in zip(chunks_to_keep, embeddings_to_keep, doc_chunk_mapping):
            doc_id = doc_info[2]
            if doc_id not in doc_chunks_map:
                doc_chunks_map[doc_id] = {
                    'doc_info': doc_info,
                    'chunks': [],
                    'embeddings': []
                }
            doc_chunks_map[doc_id]['chunks'].append(chunk)
            doc_chunks_map[doc_id]['embeddings'].append(embedding)

        # Insert documents and chunks
        inserted_ids = []

        with self.connection_pool.get_connection() as conn:
            try:
                conn.execute('BEGIN')

                for doc_id, doc_data in doc_chunks_map.items():
                    doc_text, metadata, doc_id, content_hash = doc_data['doc_info']
                    chunks = doc_data['chunks']
                    doc_embeddings = np.array(doc_data['embeddings'])

                    # Insert document
                    self._insert_document(conn, doc_id, doc_text, content_hash, metadata)

                    # Add chunk embeddings to FAISS index with explicit IDs
                    if len(doc_embeddings) > 0:
                        # Generate sequential FAISS IDs starting from current index size
                        start_faiss_id = self.index.ntotal
                        new_faiss_ids = np.arange(
                            start_faiss_id,
                            start_faiss_id + len(chunks),
                            dtype=np.int64
                        )
                        self.index.add_with_ids(doc_embeddings, new_faiss_ids)

                        # Update chunk FAISS IDs
                        for i, chunk in enumerate(chunks):
                            chunk.faiss_id = int(new_faiss_ids[i])


                    # Insert chunks with their FAISS IDs
                    for chunk in chunks:
                        self._insert_chunk(conn, doc_id, chunk)

                    inserted_ids.append(doc_id)

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

        logger.info(f"Successfully inserted {len(inserted_ids)} documents with "
                    f"{sum(len(doc_data['chunks']) for doc_data in doc_chunks_map.values())} chunks")

        return inserted_ids

    # TODO: how are we gonna get chunks? Allow the id to have a chunk indicator.
    def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document], None]:
        """
        Retrieve documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to retrieve

        Returns
        -------
        Union[Document, List[Document], None]
            Retrieved document(s) or None if not found
        """
        single_id = isinstance(ids, str)
        if single_id:
            ids = [ids]

        with self.connection_pool.get_connection() as conn:
            # Build query to get documents with metadata
            metadata_columns = list(self.metadata_schema.keys())
            if metadata_columns:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
            else:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']

            placeholders = ','.join(['?'] * len(ids))
            sql = f"SELECT {', '.join(columns)} FROM documents WHERE id IN ({placeholders})"

            cursor = conn.execute(sql, ids)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                # Extract metadata
                metadata = {}
                for col_name in metadata_columns:
                    if col_name in row.keys():
                        value = row[col_name]
                        if col_name in self.metadata_schema:
                            field_def = self.metadata_schema[col_name]
                            if field_def.type == MetadataFieldType.JSON and value:
                                value = json.loads(value)
                        metadata[col_name] = value

                doc = Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                    content_hash=row['content_hash']
                )
                documents.append(doc)

        if single_id:
            return documents[0] if documents else None
        return documents

    # TODO: needs to check chunks
    def exists(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """
        Check if documents exist

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to check

        Returns
        -------
        Union[bool, List[bool]]
            Existence status for each ID
        """
        single_id = isinstance(ids, str)
        if single_id:
            ids = [ids]

        with self.connection_pool.get_connection() as conn:
            placeholders = ','.join(['?'] * len(ids))
            cursor = conn.execute(f'SELECT id FROM documents WHERE id IN ({placeholders})', ids)
            existing_ids = {row['id'] for row in cursor.fetchall()}

        results = [doc_id in existing_ids for doc_id in ids]

        if single_id:
            return results[0]
        return results

    def delete(self, ids: Union[str, List[str]]) -> int:
        """
        Delete documents

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to delete

        Returns
        -------
        int
            Number of documents deleted
        """
        with self._lock:
            if isinstance(ids, str):
                ids = [ids]

            # Get FAISS IDs for chunks to remove
            faiss_ids_to_remove = []

            with self.connection_pool.get_connection() as conn:
                placeholders = ','.join(['?'] * len(ids))
                cursor = conn.execute(
                    f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                    ids
                )
                faiss_ids_to_remove = [row['faiss_id'] for row in cursor.fetchall()]

                # Delete documents (cascades to chunks)
                cursor = conn.execute(f'DELETE FROM documents WHERE id IN ({placeholders})', ids)
                deleted_count = cursor.rowcount
                conn.commit()

            # Remove from FAISS index
            if faiss_ids_to_remove and hasattr(self.index, 'remove_ids'):
                try:
                    # Convert to numpy array of int64
                    ids_array = np.array(faiss_ids_to_remove, dtype=np.int64)
                    self.index.remove_ids(ids_array)
                    logger.info(f"Removed {len(faiss_ids_to_remove)} vectors from FAISS index")
                except Exception as e:
                    logger.error(f"Failed to remove vectors from FAISS index: {e}")
            elif faiss_ids_to_remove:
                logger.warning(f"FAISS index doesn't support removal, {len(faiss_ids_to_remove)} vectors orphaned")

            return deleted_count

    def update(
            self,
            doc_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document's content and/or metadata

        Parameters
        ----------
        doc_id : str
            Document ID to update
        content : Optional[str]
            New content (if None, content is not updated)
        metadata : Optional[Dict[str, Any]]
            New metadata (merged with existing)

        Returns
        -------
        bool
            True if document was updated, False if not found
        """
        with self._lock:
            # Get existing document
            existing_doc = self.get(doc_id)
            if not existing_doc:
                return False

            # Update content if provided
            if content is not None:
                # Check if content actually changed
                new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                if new_hash != existing_doc.content_hash:
                    # Content changed, use upsert to handle chunking/embedding
                    updated_metadata = existing_doc.metadata.copy()
                    if metadata:
                        updated_metadata.update(metadata)

                    self.upsert([content], [updated_metadata], [doc_id])
                    return True

            # Update metadata only
            if metadata:
                updated_metadata = existing_doc.metadata.copy()
                updated_metadata.update(metadata)

                self._validate_metadata(updated_metadata)

                with self.connection_pool.get_connection() as conn:
                    # Build UPDATE statement for metadata
                    set_clauses = ['updated_at = ?']
                    values = [datetime.now()]

                    for field_name, value in updated_metadata.items():
                        if field_name in self.metadata_schema:
                            set_clauses.append(f'{field_name} = ?')

                            field_def = self.metadata_schema[field_name]
                            if field_def.type == MetadataFieldType.JSON:
                                values.append(json.dumps(value))
                            elif field_def.type == MetadataFieldType.DATE:
                                if isinstance(value, datetime):
                                    values.append(value.isoformat())
                                else:
                                    values.append(str(value))
                            else:
                                values.append(value)

                    values.append(doc_id)
                    sql = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?"
                    conn.execute(sql, values)
                    conn.commit()

                return True

            return False

    def query(
            self,
            query: str,
            *,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,  # For hybrid search
    ) -> List[QueryResult]:
        """
        Unified query interface for all search types

        Parameters
        ----------
        query : str
            Query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        if search_type == 'vector':
            return self._vector_search(query, return_type, k, score_threshold, filters)
        elif search_type == 'keyword':
            return self._keyword_search(query, return_type, k, score_threshold, filters)
        elif search_type == 'hybrid':
            return self._hybrid_search(query, return_type, k, score_threshold, filters, vector_weight)
        else:
            raise ValueError(f"Unknown search type: {search_type}")

    def _vector_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]]
    ) -> List[QueryResult]:
        """Perform vector similarity search"""

        # Generate query embedding
        query_embedding = self.embedding_provider.embed_sync([query])

        # For document-level results, we need to search more chunks initially
        # since we'll be deduplicating by document ID
        search_k = k * 3 if return_type == 'documents' else k * 2

        with self._lock:
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, search_k)

        # Get chunk information
        chunk_results = []

        with self.connection_pool.get_connection() as conn:
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue

                # Get chunk info
                cursor = conn.execute('''
                    SELECT c.*, d.id as doc_id, d.content as doc_content
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id  
                    WHERE c.faiss_id = ?
                ''', (int(idx),))

                row = cursor.fetchone()
                if not row:
                    continue

                # Convert distance to normalized score (0-1, higher=better)
                # L2 distance is always >= 0, we'll use a simple conversion
                score = max(0.0, 1.0 / (1.0 + float(dist)))

                if score < score_threshold:
                    continue

                # Get document metadata
                doc_metadata = self._get_document_metadata(conn, row['doc_id'])

                # Apply metadata filters early
                if filters and not self._matches_filters(doc_metadata, filters):
                    continue

                # Create chunk position
                position = ChunkPosition(
                    start=row['start_pos'],
                    end=row['end_pos'],
                    line=row['start_line'],
                    column=row['start_col'],
                    end_line=row['end_line'],
                    end_column=row['end_col']
                )

                if return_type == 'chunks':
                    result = QueryResult(
                        id=f"{row['document_id']}:{row['chunk_index']}",
                        score=score,
                        type='chunk',
                        content=row['content'],
                        metadata=doc_metadata,
                        document_id=row['doc_id'],
                        position=position
                    )
                    chunk_results.append(result)
                else:  # documents - collect for deduplication
                    chunk_results.append({
                        'doc_id': row['doc_id'],
                        'score': score,
                        'doc_content': row['doc_content'],
                        'metadata': doc_metadata,
                        'chunk_content': row['content'],
                        'position': position
                    })

        if return_type == 'chunks':
            # Sort by score and limit for chunks
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]
        else:
            # For documents, deduplicate and keep best score per document
            doc_map = {}
            for chunk_data in chunk_results:
                doc_id = chunk_data['doc_id']
                if doc_id not in doc_map or chunk_data['score'] > doc_map[doc_id]['score']:
                    doc_map[doc_id] = chunk_data

            # Convert to QueryResult objects
            document_results = []
            for doc_data in doc_map.values():
                result = QueryResult(
                    id=doc_data['doc_id'],
                    score=doc_data['score'],
                    type='document',
                    content=doc_data['doc_content'],
                    metadata=doc_data['metadata']
                )
                document_results.append(result)

            # Sort by score and limit
            document_results.sort(key=lambda x: x.score, reverse=True)
            return document_results[:k]

    def _keyword_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]]
    ) -> List[QueryResult]:
        """Perform keyword search using FTS5"""
        if not self.fts_enabled:
            logger.warning("FTS not available, returning empty results")
            return []

        # Sanitize and prepare query for FTS5
        sanitized_query = self._sanitize_fts_query(query)
        if not sanitized_query:
            return []

        results = []

        with self.connection_pool.get_connection() as conn:
            if return_type == 'documents':
                # Search documents directly - no deduplication needed
                cursor = conn.execute('''
                    SELECT d.*, rank
                    FROM documents_fts, documents d
                    WHERE documents_fts.rowid = d.rowid
                    AND documents_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (sanitized_query, k * 2))  # Get extra for filtering


                for row in cursor.fetchall():
                    # Convert FTS5 rank to normalized score (0-1, higher=better)
                    # FTS5 rank is negative (lower is better), convert to positive score
                    # fts_rank = abs(float(row['rank']))
                    # score = max(0.0, 1.0 / (1.0 + fts_rank))

                    score = 1.0 - min(1.0, math.exp(float(row['rank'])))

                    if score < score_threshold:
                        continue

                    # Get document metadata
                    doc_metadata = self._get_document_metadata(conn, row['id'])

                    # Apply metadata filters
                    if filters and not self._matches_filters(doc_metadata, filters):
                        continue

                    result = QueryResult(
                        id=row['id'],
                        score=score,
                        type='document',
                        content=row['content'],
                        metadata=doc_metadata
                    )
                    results.append(result)

            else:  # chunks
                # Search chunks - get extra for potential deduplication if needed
                cursor = conn.execute('''
                    SELECT c.*, d.id as doc_id, d.content as doc_content, rank
                    FROM chunks_fts, chunks c, documents d
                    WHERE chunks_fts.rowid = c.id
                    AND c.document_id = d.id
                    AND chunks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (sanitized_query, k * 2))  # Get extra for filtering

                for row in cursor.fetchall():
                    # Convert FTS5 rank to normalized score
                    # fts_rank = abs(float(row['rank']))
                    # score = max(0.0, 1.0 / (1.0 + fts_rank))
                    score = 1.0 - min(1.0, math.exp(float(row['rank'])))

                    if score < score_threshold:
                        continue

                    # Create chunk position
                    position = ChunkPosition(
                        start=row['start_pos'],
                        end=row['end_pos'],
                        line=row['start_line'],
                        column=row['start_col'],
                        end_line=row['end_line'],
                        end_column=row['end_col']
                    )

                    # Get document metadata
                    doc_metadata = self._get_document_metadata(conn, row['doc_id'])

                    # Apply metadata filters
                    if filters and not self._matches_filters(doc_metadata, filters):
                        continue

                    result = QueryResult(
                        id=f"{row['document_id']}:{row['chunk_index']}",
                        score=score,
                        type='chunk',
                        content=row['content'],
                        metadata=doc_metadata,
                        document_id=row['doc_id'],
                        position=position
                    )
                    results.append(result)

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def _hybrid_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            vector_weight: float
    ) -> List[QueryResult]:
        """Perform hybrid search combining vector and keyword"""
        if not self.fts_enabled:
            logger.info("FTS not available, falling back to vector search")
            return self._vector_search(query, return_type, k, score_threshold, filters)

        # Get more results than requested for better reranking
        # Increase the multiplier to account for potential deduplication
        search_k = min(k * 4, 100)

        # Perform both searches
        vector_results = self._vector_search(query, return_type, search_k, 0.0, filters)
        keyword_results = self._keyword_search(query, return_type, search_k, 0.0, filters)

        # If either returns no results, return the other
        if not vector_results:
            return keyword_results[:k]
        if not keyword_results:
            return vector_results[:k]

        # Combine results with weighted scoring
        return self._combine_search_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            k=k,
            score_threshold=score_threshold
        )

    @staticmethod
    def _combine_search_results(
            vector_results: List[QueryResult],
            keyword_results: List[QueryResult],
            vector_weight: float,
            k: int,
            score_threshold: float
    ) -> List[QueryResult]:
        """Combine and rank results from vector and keyword searches"""
        # Create a map of document/chunk ID to its result data
        combined_results = {}

        # Process vector results
        for result in vector_results:
            combined_results[result.id] = {
                "result": result,
                "vector_score": result.score,
                "keyword_score": 0.0
            }

        # Process keyword results
        for result in keyword_results:
            if result.id in combined_results:
                combined_results[result.id]["keyword_score"] = result.score
            else:
                combined_results[result.id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "keyword_score": result.score
                }

        # Calculate combined score for each result
        final_results = []
        for result_data in combined_results.values():
            # Weighted average of scores
            final_score = (
                    vector_weight * result_data["vector_score"] +
                    (1.0 - vector_weight) * result_data["keyword_score"]
            )

            if final_score >= score_threshold:
                # Update the result with the hybrid score
                result = result_data["result"]
                result.score = final_score
                final_results.append(result)

        # Sort by final score (higher is better) and limit
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]

    def _get_document_metadata(self, conn: sqlite3.Connection, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a document"""
        metadata_columns = list(self.metadata_schema.keys())
        if not metadata_columns:
            return {}

        cursor = conn.execute(
            f"SELECT {', '.join(metadata_columns)} FROM documents WHERE id = ?",
            (doc_id,)
        )
        row = cursor.fetchone()
        if not row:
            return {}

        metadata = {}
        for col_name in metadata_columns:
            if col_name in row.keys():
                value = row[col_name]
                if value is not None and col_name in self.metadata_schema:
                    field_def = self.metadata_schema[col_name]
                    if field_def.type == MetadataFieldType.JSON:
                        value = json.loads(value)
                metadata[col_name] = value

        return metadata


    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        # Simple implementation - would be expanded for complex filters
        for key, expected_value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != expected_value:
                return False
        return True

    def filter(
            self,
            where: Optional[Dict[str, Any]] = None,
            sql: Optional[str] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List[Document]:
        """
        Filter documents using SQL-like queries

        Parameters
        ----------
        where : Optional[Dict[str, Any]]
            Simple filter conditions
        sql : Optional[str]
            Raw SQL WHERE clause
        order_by : Optional[str]
            ORDER BY clause
        limit : Optional[int]
            Maximum number of results
        offset : int
            Number of results to skip

        Returns
        -------
        List[Document]
            Filtered documents
        """
        # Build SQL query
        metadata_columns = list(self.metadata_schema.keys())
        if metadata_columns:
            columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
        else:
            columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']

        query_parts = [f"SELECT {', '.join(columns)} FROM documents"]
        params = []

        # Add WHERE clause
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, (str, int, float, bool)):
                    conditions.append(f"{key} = ?")
                    params.append(value)
                elif isinstance(value, dict):
                    operator = next(k for k in value)
                    condition = value[operator]
                    if isinstance(condition, (list, tuple)):
                        if operator.lower() == "between":
                            clause = f"{key} BETWEEN ? AND ?"
                            if len(condition) != 2:
                                raise ValueError(f"'between' operator requires 2 operands. Found: {condition}")
                        else:
                            clause = f"{key} {operator} (" + ",".join("?" * len(condition)) + ")"
                        params += condition
                    else:
                        clause = f"{key} {operator} ?"
                        params.append(condition)
                    conditions.append(clause)

            query_parts.append(f"WHERE {' AND '.join(conditions)}")
        elif sql:
            query_parts.append(f"WHERE {sql}")

        # Add ORDER BY
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")

        # Add LIMIT/OFFSET
        if limit:
            query_parts.append(f"LIMIT {limit}")
        if offset:
            query_parts.append(f"OFFSET {offset}")

        # Execute query
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(' '.join(query_parts), params)
            rows = cursor.fetchall()

            documents = []
            for row in rows:
                # Extract metadata
                metadata = {}
                for col_name in metadata_columns:
                    if col_name in row.keys():
                        value = row[col_name]
                        if value is not None and col_name in self.metadata_schema:
                            field_def = self.metadata_schema[col_name]
                            if field_def.type == MetadataFieldType.JSON:
                                value = json.loads(value)
                        metadata[col_name] = value

                doc = Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=metadata,
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                    content_hash=row['content_hash']
                )
                documents.append(doc)

        return documents

    def save(self):
        """Save the FAISS index to disk"""
        with self._lock:
            if not self.is_memory_only and hasattr(self.index, 'ntotal') and self.index.ntotal > 0:
                # If using GPU, move to CPU for saving
                if hasattr(self.index, 'index') and hasattr(self.index.index, 'device'):  # GPU index wrapper
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, str(self.index_path))
                else:
                    faiss.write_index(self.index, str(self.index_path))

    def close(self):
        """Close the database"""
        # if not self._closed:
        self.save()
        self.connection_pool.close_all()
        # self._closed = True


    @property
    def stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.connection_pool.get_connection() as conn:
            # Document count
            doc_count = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]

            # Chunk count
            chunk_count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]

            # Index info
            index_size = self.index.ntotal if hasattr(self.index, 'ntotal') else 0

            return {
                'documents': doc_count,
                'chunks': chunk_count,
                'index_vectors': index_size,
                'embedding_dimension': self.embedding_dimension,
                'embedding_provider': self.embedding_provider.provider_name,
                'embedding_model': self.embedding_provider.model,
                'chunking_method': self.chunking_method,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'fts_enabled': self.fts_enabled
            }