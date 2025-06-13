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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any, Tuple

import faiss
import numpy as np

from localvectordb._filters import FilterQueryBuilder, matches_metadata_filter
from localvectordb.chunking import ChunkerFactory
from localvectordb.core import (
    DatabaseSchema, ConnectionPool, Document, Chunk, QueryResult,
    MetadataField, MetadataFieldType, ChunkPosition, get_common_metadata_schemas, ReadWriteLock, BaseVectorDB
)
from localvectordb.embeddings import EmbeddingRegistry, EmbeddingProvider
from localvectordb.exceptions import DatabaseNotFoundError, DuplicateDocumentIDError, DatabaseError, MetadataFilterError
from localvectordb.utils import get_system_version

logger = logging.getLogger(__name__)


class LocalVectorDB(BaseVectorDB):
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

            # Embedding configuration
            embedding_provider: str = "ollama",
            embedding_model: str = "nomic-embed-text",
            embedding_config: Optional[Dict[str, Any]] = None,

            # Chunking configuration
            chunking_method: str = "sentences",
            chunk_size: int = 500,
            chunk_overlap: int = 1,

            # Index type
            faiss_index_type: Literal["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"] = "IndexFlatL2",
            faiss_index_hnsw_flat_neighbors: int = None,  # Only used for IndexHNSWFlat
            faiss_index_lsh_bits: int = None,

            # Performance settings
            enable_gpu: bool = False,
            enable_fts: bool = True,
            connection_pool_size: int = 10,

            # Other
            create_if_not_exists: bool = True,
    ):
        super().__init__()
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
            self._metadata_schema = get_common_metadata_schemas(metadata_schema)
        else:
            self._metadata_schema = metadata_schema or {}
        self.doc_id_pattern = doc_id_pattern

        # Chunking setup
        self._chunking_method = chunking_method
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.chunker = ChunkerFactory.create_chunker(
            chunking_method, chunk_size, chunk_overlap
        )

        # Initialize embedding provider
        embedding_config = embedding_config or {}
        self._embedding_provider = EmbeddingRegistry.create_provider(
            embedding_provider, embedding_model, **embedding_config
        )

        # Validate embedding model
        if not self._embedding_provider.validate_model():
            raise ValueError(f"Embedding model '{embedding_model}' is not available")

        self._embedding_dimension = self._embedding_provider.get_dimension()

        # Database setup
        self.schema = DatabaseSchema(self.db_path)
        self.connection_pool = ConnectionPool(self.db_path, connection_pool_size)

        # Initialize schema
        self.schema.initialize(self._metadata_schema, db_connection=self.connection_pool.get_connection())

        # Load existing metadata schema if database already exists
        if not self.is_memory_only and self.db_path.exists():
            existing_schema = self.schema.load_metadata_schema(db_connection=self.connection_pool.get_connection())
            self._metadata_schema.update(existing_schema)

        # FTS setup
        self._fts_enabled = False
        if enable_fts:
            self._init_fts()

        # FAISS index setup
        self._init_faiss_index(enable_gpu, faiss_index_type, faiss_index_hnsw_flat_neighbors, faiss_index_lsh_bits)

        # Threading
        self._lock = threading.RLock()
        self._read_write_lock = ReadWriteLock()

        # State
        # self._closed = False
        self._next_doc_id = self._load_next_doc_id()

        # Save configuration
        self._save_config()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    @property
    def embedding_provider(self) -> Union[str, EmbeddingProvider]:
        return self._embedding_provider

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @property
    def chunking_method(self) -> str:
        return self._chunking_method

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    @property
    def embedding_model(self) -> str:
        return self.embedding_provider.model

    @property
    def metadata_schema(self) -> dict[str, MetadataField]:
        return self._metadata_schema.copy()

    @property
    def closed(self):
        return self.connection_pool.closed

    def ping(self):
        return not self.closed

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
            self._fts_enabled = False
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
                self._fts_enabled = True
                logger.info("FTS5 initialized successfully")

        except Exception as e:
            logger.error(f"Error setting up FTS5: {e}")
            self._fts_enabled = False

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

    @staticmethod
    def _is_valid_boolean_structure(tokens: List[str]) -> bool:
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

    def _init_faiss_index(self,
                          enable_gpu: bool,
                          faiss_index_type,
                          faiss_index_hnsw_flat_neighbors: int | None,
                          faiss_index_lsh_bits: int | None):
        """Initialize FAISS index with ID mapping support"""
        if self.index_path and self.index_path.exists():

            try:
                # Load existing index
                loaded_index = faiss.read_index(str(self.index_path))
            except RuntimeError as e:
                raise DatabaseError(f"Error loading faiss index: {str(e)}")

            # Check if it's already an IndexIDMap
            if hasattr(loaded_index, 'id_map'):
                self.index = loaded_index
                logger.info(f"Loaded existing FAISS IndexIDMap with {self.index.ntotal} vectors")
            else:
                raise DatabaseError("Expected FAISS index to have `id_map` attribute. Invalid faiss index!")

        else:
            if faiss_index_type == "IndexFlatL2":
                base_index = faiss.IndexFlatL2(self.embedding_dimension)
            elif faiss_index_type == "IndexFlatIP":
                base_index = faiss.IndexFlatIP(self.embedding_dimension)
            elif faiss_index_type == "IndexHNSWFlat":
                base_index = faiss.IndexHNSWFlat(self.embedding_dimension,
                                                 faiss_index_hnsw_flat_neighbors or 16)
            elif faiss_index_type == "IndexLSH":
                base_index = faiss.IndexLSH(self.embedding_dimension,
                                            faiss_index_lsh_bits or self.embedding_dimension * 2)
            else:
                raise ValueError("Invalid faiss index for LocalVectorDB. "
                                 "Must be one of: IndexFlatL2, IndexFlatIP, IndexHNSWFlat, IndexLSH")
            # Create new index with ID mapping
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

    def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None
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
        similarity_threshold : float, optional
            If provided, skip chunks that are too similar to existing chunks.
            Value should be between 0-1 where 1.0 means identical.
            Setting to 0.95 makes sure you wouldn't overpopulate the database with semantically similar information
            repeatedly (e.g. headings or boilerplate language) which could be discarded.


        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        with self._read_write_lock.write_lock():
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

            # Handle if they included some ids but not others.
            ids = [(self._generate_doc_id() if i is None else i) for i in ids]

            # Validate metadata against schema
            self._validate_metadata_batch(metadata)

            # Process in batches
            result_ids = []
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadata[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                batch_result = self._upsert_batch(batch_docs, batch_meta, batch_ids,
                                                  similarity_threshold=similarity_threshold,
                                                  embedding_batch_size=batch_size)
                result_ids.extend(batch_result)

            # Save state
            self._save_next_doc_id()
            self._save_internal()

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

    def _generate_embeddings_chunked(
            self,
            texts: List[str],
            batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings in manageable batches to prevent memory issues

        Parameters
        ----------
        texts : List[str]
            List of text strings to embed
        batch_size : int
            Number of texts to process at once

        Returns
        -------
        np.ndarray
            Array of embeddings with shape (len(texts), embedding_dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dimension)

            # Pre-allocate the final array
        total_embeddings = len(texts)
        final_embeddings = np.empty(
            (total_embeddings, self.embedding_dimension),
            dtype=np.float32
        )

        current_idx = 0
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedding_provider.embed_sync(batch)

            # Write directly to final array
            current_batch_size = len(batch_embeddings)
            final_embeddings[current_idx:current_idx + current_batch_size] = batch_embeddings
            current_idx += current_batch_size

            # Optional: Clear batch from memory immediately
            del batch_embeddings

        return final_embeddings

    def _filter_similar_chunks_vectorized(
            self,
            embeddings: np.ndarray,
            chunks: List[Chunk],
            doc_chunk_mapping: List[Tuple],
            similarity_threshold: float
    ) -> Tuple[List[Chunk], np.ndarray, List[Tuple]]:
        """
        Vectorized similarity filtering using single FAISS search and numpy operations

        Parameters
        ----------
        embeddings : np.ndarray
            Array of embeddings to check
        chunks : List[Chunk]
            List of chunks corresponding to embeddings
        doc_chunk_mapping : List[Tuple]
            Document info for each chunk
        similarity_threshold : float
            Similarity threshold (0-1, higher=more similar)

        Returns
        -------
        Tuple[List[Chunk], np.ndarray, List[Tuple]]
            Filtered chunks, embeddings, and doc mappings
        """
        if len(embeddings) == 0 or self.index.ntotal == 0:
            return chunks, embeddings, doc_chunk_mapping

        # Build set of existing chunk hashes for fast lookup
        existing_chunk_hashes = set()
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute('SELECT DISTINCT content_hash FROM chunks')
            existing_chunk_hashes = {row['content_hash'] for row in cursor.fetchall()}

        # Filter by content hash first (exact duplicates)
        hash_mask = np.array([
            chunk.content_hash not in existing_chunk_hashes
            for chunk in chunks
        ])

        if not hash_mask.any():
            logger.debug("All chunks filtered out by content hash")
            return [], np.array([]).reshape(0, self.embedding_dimension), []

        # Apply hash filter
        filtered_chunks = [chunks[i] for i in range(len(chunks)) if hash_mask[i]]
        filtered_embeddings = embeddings[hash_mask]
        filtered_mappings = [doc_chunk_mapping[i] for i in range(len(doc_chunk_mapping)) if hash_mask[i]]

        # Skip similarity check if no existing vectors or threshold is None/0
        if self.index.ntotal == 0 or similarity_threshold is None or similarity_threshold <= 0:
            return filtered_chunks, filtered_embeddings, filtered_mappings

        # Convert similarity threshold to distance threshold
        # similarity = 1 / (1 + distance), so distance = (1/similarity) - 1
        distance_threshold = (1.0 / max(similarity_threshold, 0.001)) - 1.0

        # Single batch FAISS search - much faster than individual searches
        distances, indices = self.index.search(filtered_embeddings, k=1)

        # Vectorized numpy operations for filtering
        valid_matches = indices[:, 0] != -1
        too_similar = (distances[:, 0] < distance_threshold) & valid_matches

        # Boolean indexing to keep non-similar chunks
        keep_mask = ~too_similar

        final_chunks = [filtered_chunks[i] for i in range(len(filtered_chunks)) if keep_mask[i]]
        final_embeddings = filtered_embeddings[keep_mask]
        final_mappings = [filtered_mappings[i] for i in range(len(filtered_mappings)) if keep_mask[i]]

        logger.debug(f"Similarity filtering: {len(chunks)} → {len(final_chunks)} chunks "
                     f"(removed {len(chunks) - len(final_chunks)} similar/duplicate)")

        return final_chunks, final_embeddings, final_mappings

    def _insert_documents_bulk(
            self,
            conn: sqlite3.Connection,
            documents_data: List[Tuple[str, str, str, Dict[str, Any]]]
    ) -> None:
        """
        Insert multiple documents using bulk operations

        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection
        documents_data : List[Tuple[str, str, str, Dict[str, Any]]]
            List of (doc_id, content, content_hash, metadata) tuples
        """
        if not documents_data:
            return

        # Build dynamic INSERT statement based on metadata schema
        base_columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
        metadata_columns = list(self.metadata_schema.keys())
        all_columns = base_columns + metadata_columns

        placeholders = ['?'] * len(all_columns)
        sql = f"INSERT OR REPLACE INTO documents ({', '.join(all_columns)}) VALUES ({', '.join(placeholders)})"

        # Prepare bulk data
        bulk_data = []
        current_time = datetime.now()

        for doc_id, content, content_hash, metadata in documents_data:
            row_data = [doc_id, content, content_hash, current_time, current_time]

            # Add metadata values in schema order
            for field_name in metadata_columns:
                value = metadata.get(field_name)

                if value is not None and field_name in self.metadata_schema:
                    field_def = self.metadata_schema[field_name]
                    if field_def.type == MetadataFieldType.JSON:
                        value = json.dumps(value)
                    elif field_def.type == MetadataFieldType.DATE:
                        if isinstance(value, datetime):
                            value = value.isoformat()
                        else:
                            value = str(value)

                row_data.append(value)

            bulk_data.append(tuple(row_data))

        # Execute bulk insert
        conn.executemany(sql, bulk_data)

    def _insert_chunks_bulk(
            self,
            conn: sqlite3.Connection,
            chunks_data: List[Tuple[str, Chunk]]
    ) -> None:
        """
        Insert multiple chunks using bulk operations

        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection
        chunks_data : List[Tuple[str, Chunk]]
            List of (doc_id, chunk) tuples
        """
        if not chunks_data:
            return

        # Prepare bulk data
        bulk_data = []
        for doc_id, chunk in chunks_data:
            bulk_data.append((
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

        # Execute bulk insert
        conn.executemany('''
            INSERT INTO chunks 
            (document_id, chunk_index, content, content_hash, start_pos, end_pos, start_line, 
            start_col, end_line, end_col, tokens, faiss_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', bulk_data)

    def _add_vectors_to_faiss_bulk(
            self,
            embeddings: np.ndarray,
            chunks: List[Chunk]
    ) -> None:
        """
        Add multiple vectors to FAISS index efficiently

        Parameters
        ----------
        embeddings : np.ndarray
            Array of embeddings to add
        chunks : List[Chunk]
            Corresponding chunks (will be updated with FAISS IDs)
        """
        if len(embeddings) == 0:
            return

        # Generate sequential FAISS IDs starting from current index size
        start_faiss_id = self.index.ntotal
        new_faiss_ids = np.arange(
            start_faiss_id,
            start_faiss_id + len(embeddings),
            dtype=np.int64
        )

        # Add all vectors at once
        self.index.add_with_ids(embeddings, new_faiss_ids)

        # Update chunk FAISS IDs
        for i, chunk in enumerate(chunks):
            chunk.faiss_id = int(new_faiss_ids[i])

    def _remove_old_vectors_bulk(self, faiss_ids: List[int]) -> None:
        """
        Remove multiple vectors from FAISS index efficiently

        Parameters
        ----------
        faiss_ids : List[int]
            List of FAISS IDs to remove
        """
        if not faiss_ids or not hasattr(self.index, 'remove_ids'):
            return

        try:
            ids_array = np.array(faiss_ids, dtype=np.int64)
            self.index.remove_ids(ids_array)
            logger.debug(f"Removed {len(faiss_ids)} vectors from FAISS index")
        except Exception as e:
            logger.warning(f"Failed to remove vectors from FAISS: {e}")

    def _upsert_batch(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            similarity_threshold: Optional[float] = None,
            embedding_batch_size: int = 100
    ) -> List[str]:
        """
        Optimized batch upsert with chunk-level content hash comparison

        This method avoids re-embedding chunks that haven't changed, improving efficiency
        for documents with minor updates.

        Parameters
        ----------
        documents : List[str]
            List of document texts
        metadata_batch : List[Dict[str, Any]]
            List of metadata dicts
        ids : List[str]
            List of document IDs
        similarity_threshold : Optional[float]
            Similarity threshold for chunk filtering
        embedding_batch_size : int
            Batch size for embedding generation

        Returns
        -------
        List[str]
            List of processed document IDs
        """
        # Check which documents need updates
        docs_to_process = []


        with self.connection_pool.get_connection() as conn:
            for doc_text, metadata, doc_id in zip(documents, metadata_batch, ids):
                cursor = conn.execute(
                    'SELECT content_hash FROM documents WHERE id = ?', (doc_id,)
                )
                row = cursor.fetchone()

                new_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()

                if row is None or row['content_hash'] != new_hash:
                    docs_to_process.append((doc_text, metadata, doc_id, new_hash))

        if not docs_to_process:
            return ids  # No changes needed

        # Get all existing chunks for all documents in one connection
        existing_chunks_by_doc = {}
        doc_ids = [doc[2] for doc in docs_to_process]

        if doc_ids:
            with self.connection_pool.get_connection() as conn:
                placeholders = ','.join(['?'] * len(doc_ids))
                cursor = conn.execute(f'''
                    SELECT document_id, chunk_index, content_hash, faiss_id 
                    FROM chunks 
                    WHERE document_id IN ({placeholders})
                ''', doc_ids)

                for row in cursor.fetchall():
                    doc_id = row['document_id']
                    if doc_id not in existing_chunks_by_doc:
                        existing_chunks_by_doc[doc_id] = {}
                    existing_chunks_by_doc[doc_id][row['chunk_index']] = {
                        'content_hash': row['content_hash'],
                        'faiss_id': row['faiss_id']
                    }

        # Process each document
        documents_data = []  # Document data for bulk insert
        all_chunks_for_db = []  # Chunks for bulk insert
        all_old_faiss_ids = []  # FAISS IDs to remove

        for doc_text, metadata, doc_id, content_hash in docs_to_process:
            # Get existing chunks for this document
            existing_chunks = existing_chunks_by_doc.get(doc_id, {})

            # Generate new chunks
            new_chunks = self.chunker.chunk(doc_text)

            # Identify unchanged chunks vs. chunks needing embedding
            unchanged_chunks = []
            chunks_to_embed = []
            chunk_texts = []

            for chunk in new_chunks:
                existing = existing_chunks.get(chunk.index)

                if existing and existing['content_hash'] == chunk.content_hash and existing['faiss_id'] is not None:
                    # Chunk hasn't changed, reuse FAISS ID
                    chunk.faiss_id = existing['faiss_id']
                    unchanged_chunks.append(chunk)
                else:
                    # New or changed chunk, needs embedding
                    chunks_to_embed.append(chunk)
                    chunk_texts.append(chunk.content)

            # Generate embeddings only for chunks that need it
            if chunks_to_embed:
                # Generate embeddings
                embeddings = self._generate_embeddings_chunked(
                    chunk_texts,
                    batch_size=embedding_batch_size
                )

                # Apply similarity filtering if requested
                if similarity_threshold is not None and similarity_threshold > 0:
                    doc_info = (doc_text, metadata, doc_id, content_hash)
                    doc_chunk_mapping = [doc_info] * len(chunks_to_embed)

                    filtered_chunks, filtered_embeddings, _ = self._filter_similar_chunks_vectorized(
                        embeddings, chunks_to_embed, doc_chunk_mapping, similarity_threshold
                    )

                    chunks_to_embed = filtered_chunks
                    embeddings = filtered_embeddings

                # Add vectors to FAISS
                if len(chunks_to_embed) > 0 and embeddings.size > 0:
                    self._add_vectors_to_faiss_bulk(embeddings, chunks_to_embed)

            # Combine unchanged and new chunks
            all_chunks = unchanged_chunks + chunks_to_embed

            # Sort chunks by index for consistency
            all_chunks.sort(key=lambda x: x.index)

            # Identify FAISS IDs to remove (those from existing chunks that are not being reused)
            used_faiss_ids = {chunk.faiss_id for chunk in unchanged_chunks if chunk.faiss_id is not None}
            existing_faiss_ids = {c['faiss_id'] for c in existing_chunks.values() if c['faiss_id'] is not None}
            faiss_ids_to_remove = list(existing_faiss_ids - used_faiss_ids)
            all_old_faiss_ids.extend(faiss_ids_to_remove)

            # Add document data for bulk insert
            documents_data.append((doc_id, doc_text, content_hash, metadata))

            # Add chunk data for bulk insert
            for chunk in all_chunks:
                all_chunks_for_db.append((doc_id, chunk))

        # Remove old FAISS IDs that are no longer used
        if all_old_faiss_ids:
            self._remove_old_vectors_bulk(all_old_faiss_ids)

        # Database transaction to update documents and chunks
        with self.connection_pool.get_connection() as conn:
            try:
                conn.execute('BEGIN')

                # Delete existing documents and chunks
                for doc_id in doc_ids:
                    conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))

                # Bulk insert documents
                self._insert_documents_bulk(conn, documents_data)

                # Bulk insert chunks
                self._insert_chunks_bulk(conn, all_chunks_for_db)

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

        return ids

    def _upsert_with_precomputed_embeddings(
            self,
            documents: List[str],
            metadata_list: List[Dict[str, Any]],
            ids_list: List[str],
            chunks: List[Chunk],
            embeddings: np.ndarray,
            doc_chunk_mapping: List[int],
            similarity_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Upsert documents using precomputed chunks and embeddings

        This helper method is used when chunking and embedding generation have
        already been completed, particularly useful for async implementations.

        Parameters
        ----------
        documents : List[str]
            List of document texts
        metadata_list : List[Dict[str, Any]]
            List of metadata dicts
        ids_list : List[str]
            List of document IDs
        chunks : List[Chunk]
            Precomputed chunks for all documents
        embeddings : np.ndarray
            Precomputed embeddings corresponding to chunks
        doc_chunk_mapping : List[int]
            Mapping from chunk index to document index in documents/metadata_list/ids_list
        similarity_threshold : Optional[float]
            Similarity threshold for filtering (0-1, higher=more similar)

        Returns
        -------
        List[str]
            List of processed document IDs
        """
        # Validate input
        if not (len(chunks) == len(embeddings) == len(doc_chunk_mapping)):
            raise ValueError("Number of chunks must match number of embeddings and doc_chunk_mapping")

        # Build a map of document IDs to document content hashes
        doc_hash_map = {}
        for doc_text, doc_id in zip(documents, ids_list):
            doc_hash_map[doc_id] = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()

        # Get existing document hashes to detect changes
        existing_doc_hashes = {}
        with self.connection_pool.get_connection() as conn:
            placeholders = ','.join(['?'] * len(ids_list))
            if placeholders:  # Only execute if there are documents
                cursor = conn.execute(
                    f'SELECT id, content_hash FROM documents WHERE id IN ({placeholders})',
                    ids_list
                )
                existing_doc_hashes = {row['id']: row['content_hash'] for row in cursor.fetchall()}

        # Filter to only documents that need updating
        docs_to_update_indices = set()
        docs_to_update = []

        for i, (doc_id, doc_hash) in enumerate(zip(ids_list, [doc_hash_map[id] for id in ids_list])):
            if doc_id not in existing_doc_hashes or existing_doc_hashes[doc_id] != doc_hash:
                docs_to_update_indices.add(i)
                docs_to_update.append(doc_id)

        if not docs_to_update:
            return ids_list  # No changes needed

        # Organize chunks and embeddings by document
        doc_chunks_map = {i: [] for i in docs_to_update_indices}
        doc_embeddings_map = {i: [] for i in docs_to_update_indices}
        doc_indices_map = {i: [] for i in docs_to_update_indices}

        for i, (chunk, doc_idx) in enumerate(zip(chunks, doc_chunk_mapping)):
            if doc_idx in docs_to_update_indices:
                doc_chunks_map[doc_idx].append(chunk)
                doc_embeddings_map[doc_idx].append(embeddings[i])
                doc_indices_map[doc_idx].append(i)

        # Apply similarity filtering if requested
        filtered_chunks_by_doc = {}
        filtered_embeddings_by_doc = {}

        for doc_idx in docs_to_update_indices:
            doc_chunks = doc_chunks_map.get(doc_idx, [])
            doc_embeddings = np.array(doc_embeddings_map.get(doc_idx, []))

            # Apply similarity filtering if requested
            if similarity_threshold is not None and similarity_threshold > 0 and len(doc_chunks) > 0:
                # Create doc_info tuple for compatibility with existing filtering method
                doc_id = ids_list[doc_idx]
                doc_info = (documents[doc_idx], metadata_list[doc_idx], doc_id, doc_hash_map[doc_id])
                doc_chunk_info_mapping = [doc_info] * len(doc_chunks)

                filtered_c, filtered_e, _ = self._filter_similar_chunks_vectorized(
                    doc_embeddings, doc_chunks, doc_chunk_info_mapping, similarity_threshold
                )

                filtered_chunks_by_doc[doc_idx] = filtered_c
                filtered_embeddings_by_doc[doc_idx] = filtered_e
            else:
                filtered_chunks_by_doc[doc_idx] = doc_chunks
                filtered_embeddings_by_doc[doc_idx] = doc_embeddings

        # Get FAISS IDs to remove
        old_faiss_ids = []
        with self.connection_pool.get_connection() as conn:
            for doc_id in docs_to_update:
                cursor = conn.execute(
                    'SELECT faiss_id FROM chunks WHERE document_id = ? AND faiss_id IS NOT NULL',
                    (doc_id,)
                )
                old_faiss_ids.extend(row['faiss_id'] for row in cursor.fetchall())

        # Remove old vectors from FAISS
        if old_faiss_ids:
            self._remove_old_vectors_bulk(old_faiss_ids)

        # Add new vectors to FAISS by document
        for doc_idx in docs_to_update_indices:
            doc_chunks = filtered_chunks_by_doc.get(doc_idx, [])
            doc_embeddings = filtered_embeddings_by_doc.get(doc_idx, np.array([]))

            if len(doc_chunks) > 0 and doc_embeddings.size > 0:
                self._add_vectors_to_faiss_bulk(doc_embeddings, doc_chunks)

        # Prepare data for bulk database operations
        documents_data = []
        chunks_data = []

        for doc_idx in docs_to_update_indices:
            doc_id = ids_list[doc_idx]
            documents_data.append((
                doc_id,
                documents[doc_idx],
                doc_hash_map[doc_id],
                metadata_list[doc_idx]
            ))

            # Add chunks for this document
            for chunk in filtered_chunks_by_doc.get(doc_idx, []):
                chunks_data.append((doc_id, chunk))

        # Database transaction
        with self.connection_pool.get_connection() as conn:
            try:
                conn.execute('BEGIN')

                # Delete existing documents and their chunks
                for doc_id in docs_to_update:
                    conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))

                # Bulk insert documents and chunks
                self._insert_documents_bulk(conn, documents_data)
                self._insert_chunks_bulk(conn, chunks_data)

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

        return ids_list

    def _insert_batch(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            similarity_threshold: Optional[float] = None,
            embedding_batch_size: int = 100
    ) -> List[str]:
        """
        Optimized batch insert with vectorized operations

        Parameters
        ----------
        documents : List[str]
            List of document texts
        metadata_batch : List[Dict[str, Any]]
            List of metadata dicts
        ids : List[str]
            List of document IDs
        similarity_threshold : Optional[float]
            Similarity threshold for chunk filtering
        embedding_batch_size : int
            Batch size for embedding generation

        Returns
        -------
        List[str]
            List of successfully inserted document IDs
        """
        # Generate chunks for all documents
        all_chunks = []
        chunk_texts = []
        doc_chunk_mapping = []

        for doc_text, metadata, doc_id in zip(documents, metadata_batch, ids):
            content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
            chunks = self.chunker.chunk(doc_text)
            doc_info = (doc_text, metadata, doc_id, content_hash)

            for chunk in chunks:
                chunk.faiss_id = None
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
                doc_chunk_mapping.append(doc_info)

        if not chunk_texts:
            return []

        # Generate embeddings in batches
        embeddings = self._generate_embeddings_chunked(
            chunk_texts,
            batch_size=embedding_batch_size
        )

        # Apply similarity filtering if requested
        if similarity_threshold is not None and similarity_threshold > 0:
            all_chunks, embeddings, doc_chunk_mapping = self._filter_similar_chunks_vectorized(
                embeddings, all_chunks, doc_chunk_mapping, similarity_threshold
            )

        if len(all_chunks) == 0:
            logger.info("No chunks to insert after similarity filtering")
            return []

        # Group chunks and embeddings by document
        doc_chunks_map = {}
        embedding_idx = 0

        for chunk, doc_info in zip(all_chunks, doc_chunk_mapping):
            doc_id = doc_info[2]
            if doc_id not in doc_chunks_map:
                doc_chunks_map[doc_id] = {
                    'doc_info': doc_info,
                    'chunks': [],
                    'embeddings': []
                }
            doc_chunks_map[doc_id]['chunks'].append(chunk)
            if embedding_idx < len(embeddings):
                doc_chunks_map[doc_id]['embeddings'].append(embeddings[embedding_idx])
                embedding_idx += 1

        # Database transaction with bulk operations
        inserted_ids = []

        with self.connection_pool.get_connection() as conn:
            try:
                conn.execute('BEGIN')

                # Prepare bulk data
                documents_data = []
                all_chunks_for_faiss = []
                all_embeddings_for_faiss = []
                all_chunks_for_db = []

                for doc_id, doc_data in doc_chunks_map.items():
                    doc_text, metadata, doc_id, content_hash = doc_data['doc_info']
                    chunks = doc_data['chunks']
                    doc_embeddings = doc_data['embeddings']

                    # Prepare document data
                    documents_data.append((doc_id, doc_text, content_hash, metadata))

                    # Collect chunks and embeddings for bulk FAISS operation
                    all_chunks_for_faiss.extend(chunks)
                    all_embeddings_for_faiss.extend(doc_embeddings)

                    # Prepare chunk data for database
                    for chunk in chunks:
                        all_chunks_for_db.append((doc_id, chunk))

                    inserted_ids.append(doc_id)

                # Bulk insert documents
                self._insert_documents_bulk(conn, documents_data)

                # Bulk add vectors to FAISS
                if all_embeddings_for_faiss:
                    embeddings_array = np.array(all_embeddings_for_faiss)
                    self._add_vectors_to_faiss_bulk(embeddings_array, all_chunks_for_faiss)

                # Bulk insert chunks
                self._insert_chunks_bulk(conn, all_chunks_for_db)

                conn.commit()

            except Exception as e:
                conn.rollback()
                raise e

        logger.info(f"Successfully inserted {len(inserted_ids)} documents with "
                    f"{sum(len(doc_data['chunks']) for doc_data in doc_chunks_map.values())} chunks")

        return inserted_ids

    def insert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
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
        similarity_threshold : Optional[float]
            If provided, skip chunks that are too similar to existing chunks.
            Value should be between 0-1 where 1.0 means identical.
            Setting to 0.95 makes sure you wouldn't overpopulate the database with similar information
            repeatedly (e.g. headings or boilerplate language) which could be discarded.
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"


        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        ValueError
            If errors="raise" and duplicate document IDs are found
        """
        with self._read_write_lock.write_lock():
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
            self._save_internal()

            return result_ids

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

        with self._read_write_lock.read_lock():
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

                # TODO: we don't check to see if any are missing
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
        with self._read_write_lock.write_lock():
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
        with self._read_write_lock.write_lock():
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
            return_type: Literal['documents', 'chunks', 'context'] = 'documents',  # Add 'context'
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,  # For hybrid search
            # NEW PARAMETERS:
            context_window: int = 2,
            semantic_dedup_threshold: Optional[float] = None,
            document_scoring_method: Literal[
                "best", "average", "worst", "weighted_average", "frequency_boost"] = "frequency_boost"
    ) -> List[QueryResult]:
        """
        Unified query interface for all search types

        Parameters
        ----------
        query : str
            Query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks', 'context']
            Whether to return full documents, individual chunks, or chunks with context
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        context_window : int
            Number of chunks before and after to include when return_type='context'
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar)
        document_scoring_method : str
            Method for aggregating chunk scores into document scores

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        with self._read_write_lock.read_lock():
            if search_type == 'vector':
                return self._vector_search(query, return_type, k, score_threshold, filters,
                                           context_window, semantic_dedup_threshold, document_scoring_method)
            elif search_type == 'keyword':
                return self._keyword_search(query, return_type, k, score_threshold, filters,
                                            context_window, semantic_dedup_threshold, document_scoring_method)
            elif search_type == 'hybrid':
                return self._hybrid_search(query, return_type, k, score_threshold, filters, vector_weight,
                                           context_window, semantic_dedup_threshold, document_scoring_method)
            else:
                raise ValueError(f"Unknown search type: {search_type}")

    def _vector_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: str
    ) -> List[QueryResult]:
        """Perform vector similarity search with enhanced processing"""

        # Generate query embedding
        query_embedding = self.embedding_provider.embed_sync([query])

        # Search more chunks initially since we might deduplicate and need enough for final k
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == 'documents' else k * 2)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, initial_k)

        # Get chunk information - ALWAYS get chunks first
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

        # Apply semantic deduplication if requested
        if semantic_dedup_threshold is not None:
            chunk_results = self._apply_semantic_deduplication(chunk_results, semantic_dedup_threshold)

        # Process based on return type
        if return_type == 'context':
            # Add context window and return
            final_results = self._add_context_window(chunk_results, context_window)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]

        elif return_type == 'documents':
            # Aggregate to document level
            document_results = self._aggregate_document_scores_with_method(
                chunk_results, document_scoring_method
            )
            return document_results[:k]

        else:  # return_type == 'chunks'
            # Sort by score and limit
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]

    def _keyword_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: str
    ) -> List[QueryResult]:
        """Perform keyword search using FTS5 with enhanced processing"""
        if not self.fts_enabled:
            logger.warning("FTS not available, returning empty results")
            return []

        # Sanitize and prepare query for FTS5
        sanitized_query = self._sanitize_fts_query(query)
        if not sanitized_query:
            return []

        # Always get chunks first, then process based on return_type
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == 'documents' else k * 2)

        chunk_results = []

        with self.connection_pool.get_connection() as conn:
            # Search chunks - get extra for potential deduplication if needed
            cursor = conn.execute('''
                SELECT c.*, d.id as doc_id, d.content as doc_content, rank
                FROM chunks_fts, chunks c, documents d
                WHERE chunks_fts.rowid = c.id
                AND c.document_id = d.id
                AND chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (sanitized_query, initial_k))

            for row in cursor.fetchall():
                # Convert FTS5 rank to normalized score
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
                chunk_results.append(result)

        # Apply same processing pipeline as vector search
        if semantic_dedup_threshold is not None:
            chunk_results = self._apply_semantic_deduplication(chunk_results, semantic_dedup_threshold)

        if return_type == 'context':
            final_results = self._add_context_window(chunk_results, context_window)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == 'documents':
            document_results = self._aggregate_document_scores_with_method(
                chunk_results, document_scoring_method
            )
            return document_results[:k]
        else:  # chunks
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]

    def _hybrid_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            vector_weight: float,
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: str
    ) -> List[QueryResult]:
        """Perform hybrid search combining vector and keyword with enhanced processing"""
        if not self.fts_enabled:
            logger.info("FTS not available, falling back to vector search")
            return self._vector_search(query, return_type, k, score_threshold, filters,
                                       context_window, semantic_dedup_threshold, document_scoring_method)

        # Get more results than requested for better reranking
        search_k = min(k * 4, 100)

        # Perform both searches - always get chunks
        vector_results = self._vector_search(query, 'chunks', search_k, 0.0, filters,
                                             0, None, "best")  # No processing yet
        keyword_results = self._keyword_search(query, 'chunks', search_k, 0.0, filters,
                                               0, None, "best")  # No processing yet

        # If either returns no results, return the other
        if not vector_results:
            return keyword_results[:k]
        if not keyword_results:
            return vector_results[:k]

        # Combine results with weighted scoring
        combined_results = self._combine_search_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            k=search_k,  # Don't limit yet
            score_threshold=0.0  # Don't filter yet
        )

        # Apply same processing pipeline
        if semantic_dedup_threshold is not None:
            combined_results = self._apply_semantic_deduplication(combined_results, semantic_dedup_threshold)

        # Filter by score threshold now
        combined_results = [r for r in combined_results if r.score >= score_threshold]

        if return_type == 'context':
            final_results = self._add_context_window(combined_results, context_window)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == 'documents':
            document_results = self._aggregate_document_scores_with_method(
                combined_results, document_scoring_method
            )
            return document_results[:k]
        else:  # chunks
            combined_results.sort(key=lambda x: x.score, reverse=True)
            return combined_results[:k]

    def _search_with_embedding(
            self,
            query: str,
            query_embedding: np.ndarray,
            *,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks', 'context'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,
            context_window: int = 2,
            semantic_dedup_threshold: Optional[float] = None,
            document_scoring_method: Literal[
                "best", "average", "worst", "weighted_average", "frequency_boost"] = "frequency_boost"
    ) -> List[QueryResult]:
        """
        Execute search with a precomputed query embedding

        This helper method allows async implementations to generate the embedding
        asynchronously and then pass it to this method for searching.

        Parameters
        ----------
        query : str
            Original query text (used for keyword search in hybrid mode)
        query_embedding : np.ndarray
            Precomputed embedding for the query
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks', 'context']
            Whether to return full documents, individual chunks, or chunks with context
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        context_window : int
            Number of chunks before and after to include when return_type='context'
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar)
        document_scoring_method : str
            Method for aggregating chunk scores into document scores

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        # Handle different search types
        if search_type == 'vector':
            results = self._vector_search_with_embedding(
                query_embedding, return_type, k, score_threshold, filters,
                context_window, semantic_dedup_threshold, document_scoring_method
            )
        elif search_type == 'keyword':
            # Keyword search doesn't use embeddings, use standard method
            results = self._keyword_search(
                query, return_type, k, score_threshold, filters,
                context_window, semantic_dedup_threshold, document_scoring_method
            )
        elif search_type == 'hybrid':
            results = self._hybrid_search_with_embedding(
                query, query_embedding, return_type, k, score_threshold, filters, vector_weight,
                context_window, semantic_dedup_threshold, document_scoring_method
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        return results

    def _vector_search_with_embedding(
            self,
            query_embedding: np.ndarray,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: str
    ) -> List[QueryResult]:
        """Perform vector similarity search with a precomputed embedding"""
        # Search more chunks initially since we might deduplicate and need enough for final k
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == 'documents' else k * 2)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, initial_k)

        # Get chunk information - ALWAYS get chunks first
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

        # Apply semantic deduplication if requested
        if semantic_dedup_threshold is not None:
            chunk_results = self._apply_semantic_deduplication(chunk_results, semantic_dedup_threshold)

        # Process based on return type
        if return_type == 'context':
            # Add context window and return
            final_results = self._add_context_window(chunk_results, context_window)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == 'documents':
            # Aggregate to document level
            document_results = self._aggregate_document_scores_with_method(
                chunk_results, document_scoring_method
            )
            return document_results[:k]
        else:  # return_type == 'chunks'
            # Sort by score and limit
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]

    def _hybrid_search_with_embedding(
            self,
            query: str,
            query_embedding: np.ndarray,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            vector_weight: float,
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: str
    ) -> List[QueryResult]:
        """Perform hybrid search with a precomputed embedding"""
        if not self.fts_enabled:
            # Fall back to vector search if FTS is not available
            return self._vector_search_with_embedding(
                query_embedding, return_type, k, score_threshold, filters,
                context_window, semantic_dedup_threshold, document_scoring_method
            )

        # Get more results than requested for better reranking
        search_k = min(k * 4, 100)

        # Perform vector search with precomputed embedding
        vector_results = self._vector_search_with_embedding(
            query_embedding, 'chunks', search_k, 0.0, filters, 0, None, "best"
        )

        # Perform keyword search
        keyword_results = self._keyword_search(
            query, 'chunks', search_k, 0.0, filters, 0, None, "best"
        )

        # If either returns no results, return the other
        if not vector_results:
            return keyword_results[:k]
        if not keyword_results:
            return vector_results[:k]

        # Combine results with weighted scoring
        combined_results = self._combine_search_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            k=search_k,  # Don't limit yet
            score_threshold=0.0  # Don't filter yet
        )

        # Apply semantic deduplication if requested
        if semantic_dedup_threshold is not None:
            combined_results = self._apply_semantic_deduplication(combined_results, semantic_dedup_threshold)

        # Filter by score threshold now
        combined_results = [r for r in combined_results if r.score >= score_threshold]

        # Process based on return type
        if return_type == 'context':
            final_results = self._add_context_window(combined_results, context_window)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == 'documents':
            document_results = self._aggregate_document_scores_with_method(
                combined_results, document_scoring_method
            )
            return document_results[:k]
        else:  # chunks
            combined_results.sort(key=lambda x: x.score, reverse=True)
            return combined_results[:k]

    def _apply_semantic_deduplication(
            self,
            results: List[QueryResult],
            threshold: float
    ) -> List[QueryResult]:
        """
        Apply semantic deduplication to search results using FAISS index embeddings.

        Parameters
        ----------
        results : List[QueryResult]
            Initial search results to deduplicate
        threshold : float
            Similarity threshold (0-1, higher=more similar). Chunks above this threshold are considered duplicates.

        Returns
        -------
        List[QueryResult]
            Deduplicated results with highest-scored chunk from each similar group
        """
        if not results or threshold is None or threshold <= 0:
            return results

        # Extract FAISS IDs from chunk results
        faiss_ids = []
        valid_results = []

        # TODO: can we get all the rows at once? This is silly to get them one at a time.
        for result in results:
            if result.type == 'chunk':
                # For chunk results, we need to get the FAISS ID from the database
                with self.connection_pool.get_connection() as conn:
                    cursor = conn.execute(
                        'SELECT faiss_id FROM chunks WHERE document_id = ? AND chunk_index = ?',
                        (result.document_id, self._extract_chunk_index_from_id(result.id))
                    )
                    row = cursor.fetchone()
                    if row and row['faiss_id'] is not None:
                        faiss_ids.append(row['faiss_id'])
                        valid_results.append(result)
            else:
                # For document results, we can't easily deduplicate without chunk info
                valid_results.append(result)

        if len(faiss_ids) < 2:
            return results  # Not enough chunks to deduplicate

        try:
            # Reconstruct embeddings from FAISS index
            embeddings = np.array([
                self.index.reconstruct(int(faiss_id)) for faiss_id in faiss_ids
            ])

            # Calculate pairwise cosine similarities
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized_embeddings = embeddings / norms
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

            # Find groups of similar chunks
            processed = set()
            keep_indices = []

            for i in range(len(valid_results)):
                if i in processed:
                    continue

                # Find all chunks similar to this one
                similar_indices = np.where(similarity_matrix[i] >= threshold)[0]

                # Among similar chunks, keep the one with highest score
                best_idx = i
                best_score = valid_results[i].score

                for j in similar_indices:
                    if j != i and valid_results[j].score > best_score:
                        best_idx = j
                        best_score = valid_results[j].score
                    processed.add(j)

                keep_indices.append(best_idx)
                processed.add(i)

            # Return deduplicated results
            deduplicated = [valid_results[i] for i in sorted(set(keep_indices))]

            logger.debug(f"Semantic deduplication: {len(results)} → {len(deduplicated)} results "
                         f"(removed {len(results) - len(deduplicated)} similar chunks)")

            return deduplicated

        except Exception as e:
            logger.warning(f"Semantic deduplication failed: {e}")
            return results

    @staticmethod
    def _extract_chunk_index_from_id(chunk_id: str) -> int:
        """Extract chunk index from chunk ID format 'doc_id:chunk_index'"""
        try:
            return int(chunk_id.split(':')[-1])
        except (ValueError, IndexError, TypeError):
            return 0

    def _add_context_window(
            self,
            results: List[QueryResult],
            context_window: int
    ) -> List[QueryResult]:
        """
        Add context window around found chunks by including surrounding chunks.

        Parameters
        ----------
        results : List[QueryResult]
            Chunk-level search results to add context to
        context_window : int
            Number of chunks to include before and after each found chunk

        Returns
        -------
        List[QueryResult]
            Results with expanded content including context chunks
        """
        if context_window <= 0 or not results:
            return results

        context_results = []

        with self.connection_pool.get_connection() as conn:
            for result in results:
                if result.type != 'chunk':
                    # For document results, just pass through
                    context_results.append(result)
                    continue

                doc_id = result.document_id
                chunk_index = self._extract_chunk_index_from_id(result.id)

                # Get surrounding chunks
                start_index = max(0, chunk_index - context_window)
                end_index = chunk_index + context_window + 1  # +1 because range is exclusive

                cursor = conn.execute('''
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col
                    FROM chunks 
                    WHERE document_id = ? AND chunk_index >= ? AND chunk_index < ?
                    ORDER BY chunk_index
                ''', (doc_id, start_index, end_index))

                context_chunks = cursor.fetchall()

                if not context_chunks:
                    # No context found, use original result
                    context_results.append(result)
                    continue

                # Combine chunks into single content
                combined_content = []
                min_start_pos = float('inf')
                max_end_pos = 0
                min_start_line = float('inf')
                min_start_col = float('inf')
                max_end_line = 0
                max_end_col = 0

                found_original = False

                for chunk_row in context_chunks:
                    combined_content.append(chunk_row['content'])

                    # Track if we found our original chunk
                    if chunk_row['chunk_index'] == chunk_index:
                        found_original = True

                    # Update position boundaries
                    min_start_pos = min(min_start_pos, chunk_row['start_pos'])
                    max_end_pos = max(max_end_pos, chunk_row['end_pos'])
                    min_start_line = min(min_start_line, chunk_row['start_line'])
                    max_end_line = max(max_end_line, chunk_row['end_line'])

                    if chunk_row['start_line'] == min_start_line:
                        min_start_col = min(min_start_col, chunk_row['start_col'])
                    if chunk_row['end_line'] == max_end_line:
                        max_end_col = max(max_end_col, chunk_row['end_col'])

                # Create new position spanning the entire context
                context_position = ChunkPosition(
                    start=int(min_start_pos),
                    end=int(max_end_pos),
                    line=int(min_start_line),
                    column=int(min_start_col),
                    end_line=int(max_end_line),
                    end_column=int(max_end_col)
                )

                # Create new result with combined content
                # Use a separator to distinguish chunks (could be configurable)
                separator = "\n\n---\n\n"
                combined_text = separator.join(combined_content)

                context_result = QueryResult(
                    id=f"{doc_id}:context:{chunk_index}",
                    score=result.score,  # Keep original score
                    type='chunk',  # Still a chunk type, but with context
                    content=combined_text,
                    metadata=result.metadata.copy(),
                    document_id=doc_id,
                    position=context_position
                )

                # Add metadata about context
                context_result.metadata['_context_window'] = context_window
                context_result.metadata['_original_chunk_index'] = chunk_index
                context_result.metadata['_context_chunk_count'] = len(context_chunks)
                context_result.metadata['_context_start_index'] = start_index
                context_result.metadata['_context_end_index'] = end_index - 1

                context_results.append(context_result)

        return context_results

    def _aggregate_document_scores_with_method(
            self,
            chunk_results: List[QueryResult],
            method: Literal["best", "average", "worst", "weighted_average", "frequency_boost"] = "frequency_boost"
    ) -> List[QueryResult]:
        """
        Aggregate chunk results into document results with enhanced scoring.

        Parameters
        ----------
        chunk_results : List[QueryResult]
            Chunk-level search results to aggregate by document
        method : str
            Scoring method: "best", "average", "worst", "weighted_average", or "frequency_boost"

        Returns
        -------
        List[QueryResult]
            Document-level results with aggregated scores
        """
        if not chunk_results:
            return []

        # Group chunks by document
        doc_groups = defaultdict(list)
        for result in chunk_results:
            doc_id = result.document_id if result.type == 'chunk' else result.id
            doc_groups[doc_id].append(result)

        document_results = []

        with self.connection_pool.get_connection() as conn:
            for doc_id, chunks in doc_groups.items():
                # Calculate aggregated score based on method
                scores = [chunk.score for chunk in chunks]

                if method == "best":
                    final_score = max(scores)
                elif method == "worst":
                    final_score = min(scores)
                elif method == "average":
                    final_score = sum(scores) / len(scores)
                elif method == "weighted_average":
                    # Weight by normalized scores
                    weights = np.array(scores)
                    weights = weights / weights.sum() if weights.sum() > 0 else weights
                    final_score = np.average(scores, weights=weights)
                elif method == "frequency_boost":
                    best_score = max(scores)
                    chunk_count = len(chunks)
                    # Frequency boost: best_score × log(1 + chunk_count)
                    frequency_multiplier = math.log(1 + chunk_count)
                    final_score = min(1.0, best_score * frequency_multiplier)  # Cap at 1.0
                else:
                    final_score = max(scores)  # Default to best

                # Get document content and metadata
                cursor = conn.execute(
                    'SELECT content FROM documents WHERE id = ?', (doc_id,)
                )
                doc_row = cursor.fetchone()

                if not doc_row:
                    continue

                # Get document metadata
                doc_metadata = self._get_document_metadata(conn, doc_id)

                # Add aggregation metadata
                doc_metadata['_aggregation_method'] = method
                doc_metadata['_chunk_count'] = len(chunks)
                doc_metadata['_best_chunk_score'] = max(scores)
                doc_metadata['_average_chunk_score'] = sum(scores) / len(scores)

                if method == "frequency_boost":
                    doc_metadata['_frequency_multiplier'] = math.log(1 + len(chunks))
                    doc_metadata['_original_best_score'] = max(scores)

                # Create document result
                doc_result = QueryResult(
                    id=doc_id,
                    score=final_score,
                    type='document',
                    content=doc_row['content'],
                    metadata=doc_metadata
                )

                document_results.append(doc_result)

        # Sort by final score
        document_results.sort(key=lambda x: x.score, reverse=True)

        return document_results


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

    def filter(
            self,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List[Document]:
        """
        Filter documents using enhanced metadata filtering

        This method supports advanced MongoDB-style filtering with operators
        like $gt, $lt, $contains, $exists, etc. The SQL injection vulnerability
        has been removed by eliminating raw SQL support.

        Parameters
        ----------
        where : Optional[Dict[str, Any]]
            Filter conditions using either simple format or MongoDB-style operators.

            Simple format::

                {"author": "John Doe", "year": 2023}

            Advanced format with operators::

                {
                    "author": {"$eq": "John Doe"},
                    "year": {"$gte": 2020, "$lte": 2024},
                    "tags": {"$contains": "python"},
                    "rating": {"$in": [4, 5]},
                    "$and": [
                        {"category": "tech"},
                        {"$or": [{"lang": "en"}, {"lang": "es"}]}
                    ]
                }

            Supported operators:

                - Comparison: $eq, $ne, $gt, $lt, $gte, $lte, $in, $nin
                - String: $like, $ilike, $contains, $startswith, $endswith
                - Existence: $exists, $not_exists
                - Type: $type
                - Logical: $and, $or, $not
                - JSON: $contains, $not_contains (for JSON fields)

        order_by : Optional[str]
            ORDER BY clause (field name with optional ASC/DESC)
            Examples: "created_at DESC", "author ASC", "rating"
        limit : Optional[int]
            Maximum number of results
        offset : int
            Number of results to skip

        Returns
        -------
        List[Document]
            Filtered documents

        Examples
        --------
        Simple filtering::

            # Simple equality
            docs = db.filter(where={"author": "John Doe"})

            # Multiple conditions (AND)
            docs = db.filter(where={"author": "John Doe", "year": 2023})

        Advanced filtering::

            # Range queries
            docs = db.filter(where={
                "year": {"$gte": 2020, "$lte": 2024},
                "rating": {"$gt": 4.0}
            })

            # String operations
            docs = db.filter(where={
                "title": {"$contains": "python"},
                "author": {"$startswith": "Dr."}
            })

            # List operations
            docs = db.filter(where={
                "category": {"$in": ["tech", "science"]},
                "tags": {"$contains": "tutorial"}
            })

            # Logical operations
            docs = db.filter(where={
                "$and": [
                    {"year": {"$gte": 2020}},
                    {"$or": [
                        {"author": "John Doe"},
                        {"author": "Jane Smith"}
                    ]}
                ]
            })

            # Existence checks
            docs = db.filter(where={
                "optional_field": {"$exists": False},
                "required_field": {"$exists": True}
            })

        Notes
        -----
        - All queries are converted to safe parameterized SQL
        - Field names are validated against the metadata schema
        - JSON fields support special operations like $contains
        """

        # Build SQL query
        metadata_columns = list(self.metadata_schema.keys())
        if metadata_columns:
            columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
        else:
            columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']

        query_parts = [f"SELECT {', '.join(columns)} FROM documents"]
        params = []

        # Build WHERE clause using new filter system
        if where:
            try:
                filter_builder = FilterQueryBuilder(self.metadata_schema)
                where_clause, filter_params = filter_builder.build_where_clause(where)
                if where_clause:
                    query_parts.append(f"WHERE {where_clause}")
                    params.extend(filter_params)
            except Exception as e:
                raise MetadataFilterError(f"Error building filter query: {str(e)}")

        # Validate and add ORDER BY clause
        if order_by:
            # Parse order_by to validate field names
            order_parts = order_by.strip().split()
            if len(order_parts) > 2:
                raise MetadataFilterError("Invalid ORDER BY clause format")

            field_name = order_parts[0]
            direction = order_parts[1].upper() if len(order_parts) == 2 else 'ASC'

            if direction not in ('ASC', 'DESC'):
                raise MetadataFilterError("ORDER BY direction must be ASC or DESC")

            # Validate field name
            if field_name not in columns:
                raise MetadataFilterError(f"Cannot order by field '{field_name}' - not in schema")

            query_parts.append(f"ORDER BY {field_name} {direction}")

        # Add LIMIT/OFFSET
        if limit:
            if not isinstance(limit, int) or limit <= 0:
                raise MetadataFilterError("Limit must be a positive integer")
            query_parts.append(f"LIMIT {limit}")

        if offset:
            if not isinstance(offset, int) or offset < 0:
                raise MetadataFilterError("Offset must be a non-negative integer")
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
                                try:
                                    value = json.loads(value)
                                except (json.JSONDecodeError, TypeError):
                                    pass  # Keep as string if not valid JSON
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

    @staticmethod
    def _matches_filters(metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """
        Check if metadata matches filter criteria (in-memory validation)

        This method now uses the enhanced filtering system for consistent
        behavior between SQL and in-memory filtering.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Document metadata to check
        filters : Dict[str, Any]
            Filter criteria in MongoDB-style format

        Returns
        -------
        bool
            True if metadata matches all filter criteria
        """
        if not filters:
            return True

        return matches_metadata_filter(metadata, filters)

    def update_metadata_schema(
            self,
            new_schema: Union[str, Dict[str, MetadataField]],
            drop_columns: bool = False,
            column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Update the metadata schema for the database

        This method allows you to add new metadata fields, modify existing ones,
        or remove fields from the schema. Existing document data is preserved.

        Parameters
        ----------
        new_schema : Union[str, Dict[str, MetadataField]]
            The new metadata schema to apply. Can be:
            - str: Schema name from common schemas (e.g., 'research_papers')
            - Dict[str, MetadataField]: Complete field definitions
            - Dict[str, str]: Simple type-only definitions (e.g., {'field': 'text'})
            - Dict[str, tuple]: Tuple definitions (type, indexed) or (type, indexed, required)
        drop_columns : bool, default=False
            Whether to actually drop columns that are no longer in the schema.
            If False, columns are kept but removed from schema for safety.
        column_mapping : dict, optional
            Optionally provide a mapping dict with old-column (key) -> new-column (value)

        Returns
        -------
        Dict[str, Any]
            Summary of changes made including:
            - added_fields: List of newly added field names
            - removed_fields: List of removed field names
            - modified_fields: List of modified fields with change details
            - populated_defaults: List of fields where default values were populated
            - dropped_columns: List of actually dropped columns (if drop_columns=True)
            - warnings: List of warnings about potential issues
            - errors: List of any errors encountered

        Examples
        --------
        Add new metadata fields::

            new_schema = {
                'category': MetadataField(type=MetadataFieldType.TEXT, indexed=True),
                'priority': MetadataField(type=MetadataFieldType.INTEGER, default_value=0),
                'tags': MetadataField(type=MetadataFieldType.JSON)
            }

            changes = db.update_metadata_schema(new_schema)
            print(f"Added fields: {changes['added_fields']}")

        Use shorthand syntax::

            new_schema = {
                'category': 'text',  # Simple type
                'priority': ('integer', False, True),  # (type, indexed, required)
                'rating': ('real', True)  # (type, indexed)
            }

            changes = db.update_metadata_schema(new_schema)

        Apply a common schema::

            changes = db.update_metadata_schema('research_papers')

        Notes
        -----
        - Field names cannot conflict with reserved columns: id, content, content_hash, created_at, updated_at
        - Removed fields are removed from the schema but columns are kept for data safety
        - Type changes are recorded but don't modify existing data (SQLite limitation)
        - Index changes are applied immediately
        - Changes are applied in a transaction and rolled back on error
        """
        with self._lock:
            # Handle special cases
            if isinstance(new_schema, str):
                # Load from common schemas
                new_schema = get_common_metadata_schemas(new_schema)
            elif isinstance(new_schema, dict):
                # Convert any shorthand definitions
                normalized_schema = {}
                for field_name, field_def in new_schema.items():
                    if isinstance(field_def, str):
                        normalized_schema[field_name] = MetadataField(MetadataFieldType(field_def))
                    elif isinstance(field_def, tuple):
                        if len(field_def) == 2:
                            field_type, indexed = field_def
                            normalized_schema[field_name] = MetadataField(
                                MetadataFieldType(field_type), indexed=indexed
                            )
                        elif len(field_def) == 3:
                            field_type, indexed, required = field_def
                            normalized_schema[field_name] = MetadataField(
                                MetadataFieldType(field_type), indexed=indexed, required=required
                            )
                        else:
                            raise ValueError(f"Tuple definition for '{field_name}' must have 2 or 3 elements")
                    elif isinstance(field_def, MetadataField):
                        normalized_schema[field_name] = field_def
                    else:
                        raise ValueError(f"Invalid field definition for '{field_name}': {type(field_def)}")

                new_schema = normalized_schema

            if not isinstance(new_schema, dict):
                raise ValueError("new_schema must be a dictionary, string (schema name), or Dict[str, MetadataField]")

            # Validate new schema
            for field_name, field_def in new_schema.items():
                if not isinstance(field_name, str) or not field_name.strip():
                    raise ValueError("Metadata field names must be non-empty strings")
                if not isinstance(field_def, MetadataField):
                    raise ValueError(f"Field definition for '{field_name}' must be a MetadataField instance")

            try:
                # Apply schema changes
                with self.connection_pool.get_connection() as conn:
                    changes = self.schema.update_metadata_schema(new_schema, conn, drop_columns, column_mapping)

                # Update in-memory schema
                self._metadata_schema = new_schema.copy()

                # Log the changes
                logger.info(f"Updated metadata schema for database '{self.name}'")
                if changes['added_fields']:
                    logger.info(f"Added fields: {changes['added_fields']}")
                if changes['removed_fields']:
                    logger.info(f"Removed fields: {changes['removed_fields']}")
                if changes['modified_fields']:
                    modified_names = [f['field_name'] for f in changes['modified_fields']]
                    logger.info(f"Modified fields: {modified_names}")
                if changes['populated_defaults']:
                    populated_info = [(p['field_name'], p['rows_updated']) for p in changes['populated_defaults']]
                    logger.info(f"Populated default values: {populated_info}")
                if changes['warnings']:
                    for warning in changes['warnings']:
                        logger.warning(f"Schema update warning: {warning}")
                if changes['errors']:
                    logger.error(f"Errors during schema update: {changes['errors']}")

                # Save the database to persist changes
                self.save()

                return changes

            except Exception as e:
                logger.error(f"Failed to update metadata schema: {e}")
                raise DatabaseError(f"Schema update failed: {str(e)}")

    def get_metadata_schema_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current metadata schema

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - fields: Dict of field definitions
            - field_count: Number of fields
            - indexed_fields: List of indexed field names
            - required_fields: List of required field names
            - field_types: Summary of field types used
        """
        info = {
            'fields': {},
            'field_count': len(self.metadata_schema),
            'indexed_fields': [],
            'required_fields': [],
            'field_types': {}
        }

        for field_name, field_def in self.metadata_schema.items():
            info['fields'][field_name] = {
                'type': field_def.type.value,
                'indexed': field_def.indexed,
                'required': field_def.required,
                'default_value': field_def.default_value
            }

            if field_def.indexed:
                info['indexed_fields'].append(field_name)
            if field_def.required:
                info['required_fields'].append(field_name)

            field_type = field_def.type.value
            info['field_types'][field_type] = info['field_types'].get(field_type, 0) + 1

        return info

    def _save_internal(self):
        """For saving the database to disk. Doesn't acquire lock."""
        if not self.is_memory_only and hasattr(self.index, 'ntotal') and self.index.ntotal > 0:
            # If using GPU, move to CPU for saving
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'device'):  # GPU index wrapper
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(self.index_path))
            else:
                faiss.write_index(self.index, str(self.index_path))

    def save(self):
        """Save the FAISS index to disk"""
        with self._read_write_lock.write_lock():
            self._save_internal()

    def close(self):
        """Close the database"""
        self.save()
        self.connection_pool.close_all()

    def get_stats(self) -> Dict[str, Any]:
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
