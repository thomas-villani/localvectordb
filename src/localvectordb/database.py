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
- Async/sync interface
- Direct SQLite implementation
- Unified query interface with normalized scoring
- Position-tracking chunking for perfect reconstruction
- Structured metadata with indexed columns
- Plugin-based embedding providers
"""
import asyncio
import hashlib
import json
import logging
import math
import queue
import sqlite3
import statistics
import threading
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import aiosqlite
import faiss
import numpy as np

from localvectordb._filters import FilterQueryBuilder, FTSQuerySanitization, matches_metadata_filter
from localvectordb.chunking import ChunkerFactory, PositionTrackingChunker
from localvectordb.extractors import ExtractorRegistry
from localvectordb.core import (
    BaseVectorDB,
    Chunk,
    ChunkPosition,
    Document,
    DocumentScoringMethod,
    MetadataField,
    MetadataFieldType,
    QueryResult,
)
from localvectordb._pools import ConnectionPool, AsyncConnectionPool, ReadWriteLock
from localvectordb._schema import DatabaseSchema, get_common_metadata_schemas
from localvectordb.embeddings import EmbeddingProvider, EmbeddingRegistry
from localvectordb.exceptions import (
    DatabaseError,
    DatabaseNotFoundError,
    DocumentNotFoundError,
    DuplicateDocumentIDError,
    MetadataFilterError,
)
from localvectordb.query_builder import QueryBuilder
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
            chunking_method: Union[str, PositionTrackingChunker] = "sentences",
            chunk_size: int = 500,
            chunk_overlap: int = 1,

            # Index type
            faiss_index_type: Literal["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"] = "IndexFlatL2",
            faiss_index_hnsw_flat_neighbors: Optional[int] = None,  # Only used for IndexHNSWFlat
            faiss_index_lsh_bits: Optional[int] = None,

            # Performance settings
            enable_gpu: bool = False,
            enable_fts: bool = True,
            connection_pool_size: int = 10,

            # Other
            create_if_not_exists: bool = True,
    ):
        super().__init__()
        self.name = name
        self._original_memory_request = (name == ":memory:" or base_path == ":memory:")
        
        if self._original_memory_request:
            # For in-memory databases, use SQLite shared cache to allow 
            # multiple connections (sync and async) to access the same database
            unique_id = str(uuid.uuid4()).replace("-", "")[:8]
            self.db_path = f"file:memdb_{unique_id}?mode=memory&cache=shared"
            self.base_path = None
            self.index_path = None
            logger.info(f"Creating in-memory database with shared cache: {self.db_path}")
        else:
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)

            # Database files
            self.db_path = str(self.base_path / f"{name}.sqlite")
            self.index_path = self.base_path / f"{name}.faiss"

            if not create_if_not_exists and not Path(self.db_path).exists():
                raise DatabaseNotFoundError(f"Database: {name} in {base_path} could not be found.")

        # Configuration
        if isinstance(metadata_schema, str):
            self._metadata_schema = get_common_metadata_schemas(metadata_schema)
        else:
            self._metadata_schema: Dict[str, MetadataField] = metadata_schema or {}
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

        # TODO: store in db config.
        # TODO: Where is config loaded?

        # Validate embedding model
        if not self._embedding_provider.validate_model():
            raise ValueError(f"Embedding model '{embedding_model}' is not available")

        # TODO: allow toggle auto detect vs. manual input
        self._embedding_dimension = self._embedding_provider.get_dimension()

        # Threading
        self._read_write_lock = ReadWriteLock()

        # Database setup
        self.schema = DatabaseSchema(self.db_path, self._read_write_lock)

        self.connection_pool = ConnectionPool(self.db_path, connection_pool_size)
        self.async_connection_pool: Optional[AsyncConnectionPool] = None
        self.async_max_connections = connection_pool_size or 10
        self._async_schema_initialized = False  # Track if async schema has been initialized for memory DBs

        with self.connection_pool.get_connection() as conn:
            # Initialize schema
            self.schema.initialize(self._metadata_schema, db_connection=conn)

            # Load existing metadata schema if database already exists
            if not self.is_memory_only and Path(self.db_path).exists():
                existing_schema = self.schema.load_metadata_schema(db_connection=conn)
                self._metadata_schema.update(existing_schema)

        # FTS setup
        self._fts_enabled = False
        if enable_fts:
            self._init_fts()

        # FAISS index setup
        self._init_faiss_index(enable_gpu, faiss_index_type, faiss_index_hnsw_flat_neighbors, faiss_index_lsh_bits)

        # State
        self._next_doc_id = self._load_next_doc_id()

        # Save configuration
        self._save_config()

    def __enter__(self) -> 'LocalVectorDB':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def embedding_provider(self) -> EmbeddingProvider:
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
    def closed(self) -> bool:
        return self.connection_pool.closed
    
    @property 
    def is_memory_only(self) -> bool:
        """Returns True if this database uses in-memory storage (including shared cache)"""
        return self._original_memory_request

    def ping(self) -> bool:
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

    def _init_fts(self) -> None:
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

    # TODO: add optional parameter to control metadata validation strictness
    # TODO: add a global batch size parameter for the database so it doesn't need to be included in each call
    def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            queue_size: int = 3
    ) -> List[str]:
        """
        Insert or update documents in the database with pipeline processing

        This enhanced version uses a 3-stage pipeline to overlap chunking,
        embedding generation, and database operations for 2-3x better throughput.

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
            Skip adding chunks that are more similar than this value
        queue_size : int, default=3
            Number of items allowed on the queue for the pipeline (to control memory usage)

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        # Use pipeline for multiple documents, simple processing for single docs
        # if (isinstance(documents, str)) or len(documents) == 1:
        #     return self._upsert_simple(documents, metadata, ids, batch_size, similarity_threshold)

        with self._read_write_lock.write_lock():
            # Normalize inputs (reuse existing logic)
            if isinstance(documents, str):
                documents = [documents]
            if isinstance(metadata, dict):
                metadata = [metadata]
            if isinstance(ids, str):
                ids = [ids]

            # Handle metadata and IDs (reuse existing logic)
            if metadata is None:
                metadata = [{}] * len(documents)
            elif len(metadata) != len(documents):
                raise ValueError("Number of metadata entries must match number of documents")

            if ids is None:
                ids = [self._generate_doc_id() for _ in documents]
            elif len(ids) != len(documents):
                raise ValueError("Number of IDs must match number of documents")

            ids = [(self._generate_doc_id() if i is None else i) for i in ids]

            # Validate metadata (reuse existing logic)
            self._validate_metadata_batch(metadata)

            # Process with pipeline
            result_ids = self._process_with_pipeline(
                documents, metadata, ids, batch_size, similarity_threshold, queue_size, mode="upsert"
            )

            # Save state (reuse existing logic)
            self._save_next_doc_id()
            self._save_internal()

            return result_ids

    def upsert_from_file(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            queue_size: int = 3,
            extractor_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Insert or update documents from files using file extraction.

        Uses the ExtractorRegistry to automatically extract text from files based on
        file extension and MIME type, then calls the regular upsert method.

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and upsert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : Optional[float]
            Skip adding chunks that are more similar than this value
        queue_size : int, default=3
            Number of items allowed on the queue for the pipeline
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were upserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file and no fallback is available
        """
        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Extract text from files
        documents = []
        merged_metadata = []
        final_ids = []
        extractor_kwargs = extractor_kwargs or {}

        for i, file_path in enumerate(file_paths):
            # Check file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read file content
            file_content = file_path.read_bytes()
            filename = file_path.name

            # Extract text using ExtractorRegistry
            extraction_result = ExtractorRegistry.extract_text(
                file_content, filename, **extractor_kwargs
            )

            if not extraction_result.success:
                raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")

            documents.append(extraction_result.text)

            # Merge metadata
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)

            # Generate ID if not provided
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                # Use filename without extension as ID
                doc_id = file_path.stem
            final_ids.append(doc_id)

        # Call regular upsert method
        return self.upsert(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            queue_size=queue_size
        )

    def upsert_from_chunks(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            queue_size: int = 3
    ) -> List[str]:
        """
        Insert or update documents from pre-chunked data with pipeline processing.
        
        This method allows you to directly provide chunks for documents, bypassing the 
        chunking step and enabling more efficient processing of pre-processed documents.
        
        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks. Chunks can be either:
            - List[Chunk]: Full Chunk objects with position information
            - List[str]: Simple strings that will be converted to Chunk objects
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata. If None, empty metadata 
            is used for all documents.
        batch_size : int, default=100
            Number of embeddings to generate at once
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks
        queue_size : int, default=3
            Number of items allowed on the queue for the pipeline (to control memory usage)
            
        Returns
        -------
        List[str]
            List of document IDs that were processed
            
        Raises
        ------
        ValueError
            If chunk data is invalid or metadata doesn't match schema
        """
        with self._read_write_lock.write_lock():
            # Validate input
            if not chunks_by_document:
                return []
            
            # Normalize metadata
            if metadata is None:
                metadata = {}
            
            # Ensure all documents have metadata (even if empty)
            metadata_batch = {}
            for doc_id in chunks_by_document.keys():
                metadata_batch[doc_id] = metadata.get(doc_id, {})
            
            # Validate metadata against schema
            self._validate_metadata_batch(list(metadata_batch.values()))
            
            # Normalize chunks for all documents
            normalized_chunks_by_document = {}
            for doc_id, chunks in chunks_by_document.items():
                normalized_chunks = self._normalize_chunks(chunks, doc_id)
                if normalized_chunks:  # Only include documents with valid chunks
                    normalized_chunks_by_document[doc_id] = normalized_chunks
                
            if not normalized_chunks_by_document:
                return []
            
            # Process with chunk-based pipeline
            result_ids = self._process_from_chunks_pipeline(
                normalized_chunks_by_document, 
                metadata_batch, 
                batch_size, 
                similarity_threshold, 
                queue_size, 
                mode="upsert"
            )
            
            # Save state
            self._save_next_doc_id()
            self._save_internal()
            
            return result_ids

    def _validate_metadata_batch(self, metadata_batch: List[Dict[str, Any]]):
        """Validate metadata against schema"""
        for metadata in metadata_batch:
            for field_name, value in metadata.items():
                if field_name in self.metadata_schema:
                    field_def = self.metadata_schema[field_name]
                    if value and not isinstance(value, field_def.type.valid_types()):
                        raise ValueError(f"Metadata field '{field_name}' is type {field_def.type.name}. "
                                         f"Found: {type(value)}")

            # Check required fields
            for field_name, field_def in self.metadata_schema.items():
                if field_def.required and field_name not in metadata:
                    if field_def.default_value is not None:
                        metadata[field_name] = field_def.default_value
                    else:
                        raise ValueError(f"Required metadata field '{field_name}' is missing")

    def _normalize_chunks(self, chunks: Union[List[Chunk], List[str]], doc_id: str) -> List[Chunk]:
        """
        Convert mixed chunk input to standardized Chunk objects.
        
        Parameters
        ----------
        chunks : Union[List[Chunk], List[str]]
            List of chunks as either Chunk objects or strings
        doc_id : str
            Document ID for context in error messages
            
        Returns
        -------
        List[Chunk]
            List of normalized Chunk objects with proper content_hash and indexes
        """
        if not chunks:
            return []
            
        normalized_chunks = []
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, Chunk):
                # Ensure existing Chunk has content_hash
                if chunk.content_hash is None:
                    chunk.content_hash = chunk.calculate_content_hash()
                    
                # Ensure chunk index matches position in list
                if chunk.index != i:
                    logger.warning(f"Chunk index mismatch in document {doc_id}: "
                                 f"expected {i}, got {chunk.index}. Correcting index.")
                    chunk.index = i
                    
                normalized_chunks.append(chunk)
                
            elif isinstance(chunk, str):
                # Convert string to Chunk object
                from localvectordb.core import ChunkPosition
                
                # Create minimal position info (since we don't have original document)
                position = ChunkPosition(
                    start=0,  # Unknown position in original document
                    end=len(chunk),
                    line=1,   # Assume single line for string chunks
                    column=1,
                    end_line=1,
                    end_column=len(chunk) + 1
                )
                
                chunk_obj = Chunk(
                    content=chunk,
                    position=position,
                    tokens=self.chunker.count_tokens(chunk),
                    index=i,
                    faiss_id=None,
                    content_hash=None  # Will be auto-calculated in __post_init__
                )
                
                normalized_chunks.append(chunk_obj)
                
            else:
                raise ValueError(f"Invalid chunk type in document {doc_id} at index {i}: "
                               f"expected Chunk or str, got {type(chunk)}")
        
        return normalized_chunks

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
        valid_matches = (indices[:, 0] != -1)
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
            documents_data: List[Tuple[str, str, str, Dict[str, Any]]],
            mode: Literal["insert", "replace"] = "replace"
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
        base_columns = self.schema.BASE_COLUMNS.copy()
        metadata_columns = list(self.metadata_schema.keys())
        all_columns = base_columns + metadata_columns

        placeholders = ['?'] * len(all_columns)

        sql_verb = "INSERT OR REPLACE" if mode == "replace" else "INSERT"
        sql = f"{sql_verb} INTO documents ({', '.join(all_columns)}) VALUES ({', '.join(placeholders)})"

        # Prepare bulk data
        bulk_data = []
        current_time = datetime.now(UTC)

        for doc_id, content, content_hash, metadata in documents_data:
            row_data = [doc_id, content, content_hash, current_time, current_time]

            # Add metadata values in schema order
            for field_name in metadata_columns:
                value = metadata.get(field_name)
                row_data.append(value)

            bulk_data.append(tuple(row_data))

        # Execute bulk insert
        conn.executemany(sql, bulk_data)

    @staticmethod
    def _insert_chunks_bulk(
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

    def insert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            queue_size: int = 3
    ) -> List[str]:
        """
        Insert new documents into the database with pipeline processing

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
            Skip chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        queue_size : int, default=3
            Number of items allowed on the queue for the pipeline (to control memory usage)

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
        """
        # For single documents or when pipeline is disabled, use simple processing
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
            for doc, meta, doc_id in zip(documents, metadata, ids, strict=False):
                if doc_id in existing_ids:
                    if errors == "raise":
                        raise DuplicateDocumentIDError(f"Document with ID '{doc_id}' already exists")
                    elif errors == "ignore":
                        logger.info(f"Skipping existing document ID: {doc_id}")
                        continue
                docs_to_insert.append((doc, meta, doc_id))

            if not docs_to_insert:
                return []  # No documents to insert

            # Extract separate lists for pipeline processing
            docs_to_process = [item[0] for item in docs_to_insert]
            meta_to_process = [item[1] for item in docs_to_insert]
            ids_to_process = [item[2] for item in docs_to_insert]

            # Process with pipeline (reuse upsert pipeline logic but without overwriting)
            result_ids = self._process_with_pipeline(
                docs_to_process, meta_to_process, ids_to_process,
                batch_size, similarity_threshold, queue_size, mode="insert"
            )

            # Save state
            self._save_next_doc_id()
            self._save_internal()

            return result_ids

    def insert_from_file(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            queue_size: int = 3,
            extractor_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Insert new documents from files using file extraction.

        Uses the ExtractorRegistry to automatically extract text from files based on
        file extension and MIME type, then calls the regular insert method.

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and insert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing, by default 100
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        queue_size : int, default=3
            Number of items allowed on the queue for the pipeline
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file and no fallback is available
        DuplicateDocumentIDError
            If errors="raise" and document ID conflicts occur
        """
        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Extract text from files
        documents = []
        merged_metadata = []
        final_ids = []
        extractor_kwargs = extractor_kwargs or {}

        for i, file_path in enumerate(file_paths):
            # Check file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            # Read file content
            file_content = file_path.read_bytes()
            filename = file_path.name

            # Extract text using ExtractorRegistry
            extraction_result = ExtractorRegistry.extract_text(
                file_content, filename, **extractor_kwargs
            )

            if not extraction_result.success:
                raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")

            documents.append(extraction_result.text)

            # Merge metadata
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)

            # Generate ID if not provided
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                # Use filename without extension as ID
                doc_id = file_path.stem
            final_ids.append(doc_id)

        # Call regular insert method
        return self.insert(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            errors=errors,
            queue_size=queue_size
        )

    def insert_from_chunks(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            queue_size: int = 3
    ) -> List[str]:
        """
        Insert documents from pre-chunked data with conflict handling.
        
        Similar to upsert_from_chunks but fails on duplicate document IDs unless
        configured to ignore them.
        
        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks. Chunks can be either:
            - List[Chunk]: Full Chunk objects with position information
            - List[str]: Simple strings that will be converted to Chunk objects
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata. If None, empty metadata 
            is used for all documents.
        batch_size : int, default=100
            Number of embeddings to generate at once
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"], default="raise"
            How to handle document ID conflicts:
            - "raise": Raise DuplicateDocumentIDError
            - "ignore": Skip existing documents and continue
        queue_size : int, default=3
            Number of items allowed on the queue for the pipeline (to control memory usage)
            
        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
            
        Raises
        ------
        DuplicateDocumentIDError
            If a document ID already exists and errors="raise"
        ValueError
            If chunk data is invalid or metadata doesn't match schema
        """
        with self._read_write_lock.write_lock():
            # Validate input
            if not chunks_by_document:
                return []
            
            # Normalize metadata
            if metadata is None:
                metadata = {}
            
            # Check for existing document IDs
            doc_ids = list(chunks_by_document.keys())
            existing_ids = set()
            with self.connection_pool.get_connection() as conn:
                if doc_ids:
                    placeholders = ','.join(['?'] * len(doc_ids))
                    cursor = conn.execute(f'SELECT id FROM documents WHERE id IN ({placeholders})', doc_ids)
                    existing_ids = {row['id'] for row in cursor.fetchall()}
            
            # Handle ID conflicts
            chunks_to_insert = {}
            metadata_to_insert = {}
            
            for doc_id, chunks in chunks_by_document.items():
                if doc_id in existing_ids:
                    if errors == "raise":
                        raise DuplicateDocumentIDError(f"Document with ID '{doc_id}' already exists")
                    elif errors == "ignore":
                        logger.info(f"Skipping existing document ID: {doc_id}")
                        continue
                
                chunks_to_insert[doc_id] = chunks
                metadata_to_insert[doc_id] = metadata.get(doc_id, {})
            
            if not chunks_to_insert:
                return []  # No documents to insert
            
            # Validate metadata against schema
            self._validate_metadata_batch(list(metadata_to_insert.values()))
            
            # Normalize chunks for all documents
            normalized_chunks_by_document = {}
            for doc_id, chunks in chunks_to_insert.items():
                normalized_chunks = self._normalize_chunks(chunks, doc_id)
                if normalized_chunks:  # Only include documents with valid chunks
                    normalized_chunks_by_document[doc_id] = normalized_chunks
            
            if not normalized_chunks_by_document:
                return []
            
            # Process with chunk-based pipeline
            result_ids = self._process_from_chunks_pipeline(
                normalized_chunks_by_document,
                metadata_to_insert,
                batch_size,
                similarity_threshold,
                queue_size,
                mode="insert"
            )
            
            # Save state
            self._save_next_doc_id()
            self._save_internal()
            
            return result_ids

    def _process_with_pipeline(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            batch_size: int,
            similarity_threshold: Optional[float],
            queue_size: int = 3,
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> List[str]:
        """
        Enhanced pipeline implementation with chunk hash optimization

        3-stage pipeline:
        1. Chunking thread (+ existing chunk lookup)
        2. Embedding thread (only for changed chunks)
        3. Database thread
        """
        # PRE-PROCESSING: Fetch existing chunks for all documents
        existing_chunks_by_doc = self._fetch_existing_chunks_batch(ids)

        # Queues for pipeline coordination
        chunk_queue = queue.Queue(maxsize=queue_size)
        embedding_queue = queue.Queue(maxsize=queue_size)
        result_queue = queue.Queue()

        # Completion tracking
        total_docs = len(documents)

        def chunking_worker():
            """Stage 1: Document chunking + existing chunk comparison"""
            try:
                for i, (doc_text, metadata, doc_id) in enumerate(zip(documents, metadata_batch, ids, strict=False)):
                    # Generate new chunks
                    content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
                    chunks = self.chunker.chunk(doc_text)

                    # Get existing chunks for this document
                    existing_chunks = existing_chunks_by_doc.get(doc_id, {})

                    # Categorize chunks: unchanged vs needs_embedding
                    unchanged_chunks = []
                    chunks_needing_embedding = []
                    chunk_texts_for_embedding = []

                    # Track which existing chunks are being reused
                    reused_chunk_indices = set()

                    for chunk in chunks:
                        existing_chunk = existing_chunks.get(chunk.index)

                        if (existing_chunk and
                                existing_chunk['content_hash'] == chunk.content_hash and
                                existing_chunk['faiss_id'] is not None):

                            # Chunk unchanged - reuse existing FAISS ID
                            chunk.faiss_id = existing_chunk['faiss_id']
                            unchanged_chunks.append(chunk)
                            reused_chunk_indices.add(chunk.index)
                            logger.debug(f"Reusing chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")

                        else:
                            # Chunk changed/new - needs embedding
                            chunks_needing_embedding.append(chunk)
                            chunk_texts_for_embedding.append(chunk.content)
                            logger.debug(
                                f"Re-embedding chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")

                    # Calculate what needs to be removed (existing chunks not being reused)
                    chunk_indices_to_remove = []
                    faiss_ids_to_remove = []

                    for chunk_index, chunk_info in existing_chunks.items():
                        if chunk_index not in reused_chunk_indices:
                            chunk_indices_to_remove.append(chunk_index)
                            if chunk_info['faiss_id'] is not None:
                                faiss_ids_to_remove.append(chunk_info['faiss_id'])

                    chunk_data = {
                        'doc_index': i,
                        'doc_id': doc_id,
                        'doc_text': doc_text,
                        'content_hash': content_hash,
                        'metadata': metadata,
                        'unchanged_chunks': unchanged_chunks,
                        'chunks_needing_embedding': chunks_needing_embedding,
                        'chunk_texts_for_embedding': chunk_texts_for_embedding,
                        'chunk_indices_to_remove': chunk_indices_to_remove,
                        'faiss_ids_to_remove': faiss_ids_to_remove
                    }

                    chunk_queue.put(chunk_data)

                # Signal completion
                chunk_queue.put(None)

            except Exception as e:
                logger.error(f"Chunking worker error: {e}")
                chunk_queue.put(None)
                raise

        def embedding_worker():
            """Stage 2: Embedding generation (only for chunks that need it)"""
            try:
                # Get metadata fields that need embeddings once
                embedding_enabled_fields = self._get_embedding_enabled_fields()

                while True:
                    chunk_data = chunk_queue.get()
                    if chunk_data is None:  # Completion signal
                        embedding_queue.put(None)
                        break

                    # Generate embeddings only for chunks that need them
                    chunk_texts = chunk_data['chunk_texts_for_embedding']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']

                    if chunk_texts:
                        logger.debug(f"Generating embeddings for {len(chunk_texts)} chunks in {chunk_data['doc_id']}")
                        embeddings = self.embedding_provider.embed_sync(chunk_texts, batch_size)
                        chunk_data['new_embeddings'] = embeddings

                        # Assign FAISS IDs to new chunks (will be updated with actual IDs in database worker)
                        for chunk in chunks_needing_embedding:
                            chunk.faiss_id = None  # Will be set when added to FAISS

                    else:
                        chunk_data['new_embeddings'] = np.array([]).reshape(0, self.embedding_dimension)
                        logger.debug(f"No new embeddings needed for {chunk_data['doc_id']}")

                    # Generate metadata embeddings if needed
                    if embedding_enabled_fields:
                        metadata = chunk_data['metadata']
                        field_embeddings = self._generate_metadata_embeddings(
                            metadata, embedding_enabled_fields, batch_size
                        )
                        chunk_data['field_embeddings'] = field_embeddings
                    else:
                        chunk_data['field_embeddings'] = {}

                    embedding_queue.put(chunk_data)
                    chunk_queue.task_done()

            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
                embedding_queue.put(None)
                raise

        def database_worker():
            """Stage 3: Database operations with unchanged + new chunks"""
            try:
                while True:
                    chunk_data = embedding_queue.get()
                    if chunk_data is None:  # Completion signal
                        result_queue.put(None)
                        break

                    # Combine unchanged and new chunks
                    unchanged_chunks = chunk_data['unchanged_chunks']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    new_embeddings = chunk_data['new_embeddings']
                    field_embeddings = chunk_data['field_embeddings']

                    # Apply similarity filtering only to new chunks if requested
                    if similarity_threshold is not None and len(chunks_needing_embedding) > 0:
                        doc_info = (chunk_data['doc_text'], chunk_data['metadata'],
                                    chunk_data['doc_id'], chunk_data['content_hash'])
                        doc_chunk_mapping = [doc_info] * len(chunks_needing_embedding)

                        filtered_chunks, filtered_embeddings, _ = self._filter_similar_chunks_vectorized(
                            new_embeddings, chunks_needing_embedding, doc_chunk_mapping, similarity_threshold
                        )
                        chunks_needing_embedding = filtered_chunks
                        new_embeddings = filtered_embeddings

                    # Combine all chunks for final processing
                    all_chunks = unchanged_chunks + chunks_needing_embedding

                    # Database operations
                    if len(all_chunks) > 0 or mode == "upsert":  # Always process upserts to update metadata
                        documents_data = [(chunk_data['doc_id'], chunk_data['doc_text'],
                                           chunk_data['content_hash'], chunk_data['metadata'])]
                        chunks_data = [(chunk_data['doc_id'], chunk) for chunk in all_chunks]

                        with self.connection_pool.get_connection() as conn:
                            conn.execute('BEGIN')
                            try:
                                if mode == "upsert":
                                    # Remove old data that's not being reused (including metadata embeddings)
                                    self._remove_metadata_embeddings(conn, chunk_data['doc_id'])
                                    self._remove_old_chunks_batch(
                                        chunk_data['doc_id'],
                                        chunk_data['chunk_indices_to_remove'],
                                        chunk_data['faiss_ids_to_remove']
                                    )

                                self._insert_documents_bulk(conn, documents_data, mode=mode)
                                # Add only new embeddings to FAISS
                                if new_embeddings.size > 0:
                                    self._add_vectors_to_faiss_bulk(new_embeddings, chunks_needing_embedding)

                                self._insert_chunks_bulk(conn, chunks_data)

                                # Store metadata field embeddings
                                if field_embeddings:
                                    self._store_metadata_embeddings(conn, chunk_data['doc_id'], field_embeddings)

                                conn.commit()

                                logger.debug(f"Processed {chunk_data['doc_id']}: "
                                             f"{len(unchanged_chunks)} reused, "
                                             f"{len(chunks_needing_embedding)} new chunks, "
                                             f"{len(field_embeddings)} metadata fields embedded")

                            except Exception:
                                conn.rollback()
                                raise

                    result_queue.put(chunk_data['doc_id'])
                    embedding_queue.task_done()

            except Exception as e:
                logger.error(f"Database worker error: {e}")
                result_queue.put(None)
                raise

        # Start workers
        workers = [
            threading.Thread(target=chunking_worker, name="ChunkingWorker"),
            threading.Thread(target=embedding_worker, name="EmbeddingWorker"),
            threading.Thread(target=database_worker, name="DatabaseWorker")
        ]

        for worker in workers:
            worker.start()

        # Collect results
        processed_ids = []
        try:
            while len(processed_ids) < total_docs:
                result = result_queue.get()
                if result is None:  # Completion signal
                    break
                processed_ids.append(result)

        except Exception as e:
            logger.error(f"Error collecting results: {e}")
            raise
        finally:
            # Wait for all workers to complete
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not complete in time")

        return processed_ids

    def _process_from_chunks_pipeline(
            self,
            chunks_by_document: Dict[str, List[Chunk]],
            metadata_batch: Dict[str, Dict[str, Any]],
            batch_size: int,
            similarity_threshold: Optional[float],
            queue_size: int = 3,
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> List[str]:
        """
        Enhanced pipeline implementation for pre-chunked documents with chunk hash optimization.
        
        Similar to _process_with_pipeline but skips the chunking stage since chunks are provided.
        
        2-stage pipeline:
        1. Embedding thread (only for changed chunks)  
        2. Database thread
        
        Parameters
        ----------
        chunks_by_document : Dict[str, List[Chunk]]
            Pre-chunked documents indexed by document ID
        metadata_batch : Dict[str, Dict[str, Any]]
            Metadata indexed by document ID
        batch_size : int
            Batch size for embedding generation
        similarity_threshold : Optional[float]
            Similarity threshold for filtering duplicate chunks
        queue_size : int, default=3
            Queue size for pipeline coordination
        mode : Literal["upsert", "insert"], default="upsert"
            Operation mode
            
        Returns
        -------
        List[str]
            List of processed document IDs
        """
        # PRE-PROCESSING: Fetch existing chunks for all documents
        doc_ids = list(chunks_by_document.keys())
        existing_chunks_by_doc = self._fetch_existing_chunks_batch(doc_ids)

        # Queues for pipeline coordination
        embedding_queue = queue.Queue(maxsize=queue_size)
        result_queue = queue.Queue()

        # Completion tracking
        total_docs = len(doc_ids)

        def chunk_comparison_worker():
            """Stage 1: Compare provided chunks with existing chunks"""
            try:
                for doc_id, chunks in chunks_by_document.items():
                    metadata = metadata_batch.get(doc_id, {})
                    
                    # Get existing chunks for this document
                    existing_chunks = existing_chunks_by_doc.get(doc_id, {})
                    
                    # Categorize chunks: unchanged vs needs_embedding
                    unchanged_chunks = []
                    chunks_needing_embedding = []
                    chunk_texts_for_embedding = []
                    
                    # Track which existing chunks are being reused
                    reused_chunk_indices = set()
                    
                    for chunk in chunks:
                        existing_chunk = existing_chunks.get(chunk.index)
                        
                        if (existing_chunk and
                                existing_chunk['content_hash'] == chunk.content_hash and
                                existing_chunk['faiss_id'] is not None):
                            
                            # Chunk unchanged - reuse existing FAISS ID
                            chunk.faiss_id = existing_chunk['faiss_id']
                            unchanged_chunks.append(chunk)
                            reused_chunk_indices.add(chunk.index)
                            logger.debug(f"Reusing chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")
                            
                        else:
                            # Chunk changed/new - needs embedding
                            chunks_needing_embedding.append(chunk)
                            chunk_texts_for_embedding.append(chunk.content)
                            logger.debug(f"Re-embedding chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")
                    
                    # Calculate what needs to be removed (existing chunks not being reused)
                    chunk_indices_to_remove = []
                    faiss_ids_to_remove = []
                    
                    for chunk_index, chunk_info in existing_chunks.items():
                        if chunk_index not in reused_chunk_indices:
                            chunk_indices_to_remove.append(chunk_index)
                            if chunk_info['faiss_id'] is not None:
                                faiss_ids_to_remove.append(chunk_info['faiss_id'])
                    
                    # Reconstruct document text from chunks for metadata purposes
                    doc_text = "\n".join([chunk.content for chunk in chunks])
                    content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
                    
                    chunk_data = {
                        'doc_id': doc_id,
                        'doc_text': doc_text,
                        'content_hash': content_hash,
                        'metadata': metadata,
                        'unchanged_chunks': unchanged_chunks,
                        'chunks_needing_embedding': chunks_needing_embedding,
                        'chunk_texts_for_embedding': chunk_texts_for_embedding,
                        'chunk_indices_to_remove': chunk_indices_to_remove,
                        'faiss_ids_to_remove': faiss_ids_to_remove
                    }
                    
                    embedding_queue.put(chunk_data)
                
                # Signal completion
                embedding_queue.put(None)
                
            except Exception as e:
                logger.error(f"Chunk comparison worker error: {e}")
                embedding_queue.put(None)
                raise
        
        def embedding_worker():
            """Stage 2: Embedding generation (only for chunks that need it)"""
            try:
                # Get metadata fields that need embeddings once
                embedding_enabled_fields = self._get_embedding_enabled_fields()
                
                while True:
                    chunk_data = embedding_queue.get()
                    if chunk_data is None:  # Completion signal
                        result_queue.put(None)
                        break
                    
                    # Generate embeddings only for chunks that need them
                    chunk_texts = chunk_data['chunk_texts_for_embedding']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    
                    if chunk_texts:
                        logger.debug(f"Generating embeddings for {len(chunk_texts)} chunks in {chunk_data['doc_id']}")
                        embeddings = self.embedding_provider.embed_sync(chunk_texts, batch_size)
                        chunk_data['new_embeddings'] = embeddings
                        
                        # Assign FAISS IDs to new chunks (will be updated with actual IDs in database worker)
                        for chunk in chunks_needing_embedding:
                            chunk.faiss_id = None  # Will be set when added to FAISS
                    
                    else:
                        chunk_data['new_embeddings'] = np.array([]).reshape(0, self.embedding_dimension)
                        logger.debug(f"No new embeddings needed for {chunk_data['doc_id']}")
                    
                    # Generate metadata embeddings if needed
                    if embedding_enabled_fields:
                        metadata = chunk_data['metadata']
                        field_embeddings = self._generate_metadata_embeddings(
                            metadata, embedding_enabled_fields, batch_size
                        )
                        chunk_data['field_embeddings'] = field_embeddings
                    else:
                        chunk_data['field_embeddings'] = {}
                    
                    result_queue.put(chunk_data)
                    embedding_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Embedding worker error: {e}")
                result_queue.put(None)
                raise
        
        def database_worker():
            """Stage 3: Database operations with unchanged + new chunks"""
            try:
                processed_ids = []
                
                while len(processed_ids) < total_docs:
                    chunk_data = result_queue.get()
                    if chunk_data is None:  # Completion signal
                        break
                    
                    # Combine unchanged and new chunks
                    unchanged_chunks = chunk_data['unchanged_chunks']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    new_embeddings = chunk_data['new_embeddings']
                    field_embeddings = chunk_data['field_embeddings']
                    
                    # Apply similarity filtering only to new chunks if requested
                    if similarity_threshold is not None and len(chunks_needing_embedding) > 0:
                        doc_info = (chunk_data['doc_text'], chunk_data['metadata'],
                                    chunk_data['doc_id'], chunk_data['content_hash'])
                        doc_chunk_mapping = [doc_info] * len(chunks_needing_embedding)
                        
                        filtered_chunks, filtered_embeddings, _ = self._filter_similar_chunks_vectorized(
                            new_embeddings, chunks_needing_embedding, doc_chunk_mapping, similarity_threshold
                        )
                        chunks_needing_embedding = filtered_chunks
                        new_embeddings = filtered_embeddings
                    
                    # Combine all chunks for final processing
                    all_chunks = unchanged_chunks + chunks_needing_embedding
                    
                    # Database operations
                    if len(all_chunks) > 0 or mode == "upsert":  # Always process upserts to update metadata
                        documents_data = [(chunk_data['doc_id'], chunk_data['doc_text'],
                                           chunk_data['content_hash'], chunk_data['metadata'])]
                        chunks_data = [(chunk_data['doc_id'], chunk) for chunk in all_chunks]
                        
                        with self.connection_pool.get_connection() as conn:
                            conn.execute('BEGIN')
                            try:
                                if mode == "upsert":
                                    # Remove old data that's not being reused (including metadata embeddings)
                                    self._remove_metadata_embeddings(conn, chunk_data['doc_id'])
                                    self._remove_old_chunks_batch(
                                        chunk_data['doc_id'],
                                        chunk_data['chunk_indices_to_remove'],
                                        chunk_data['faiss_ids_to_remove']
                                    )
                                
                                self._insert_documents_bulk(conn, documents_data, mode=mode)
                                # Add only new embeddings to FAISS
                                if new_embeddings.size > 0:
                                    self._add_vectors_to_faiss_bulk(new_embeddings, chunks_needing_embedding)
                                
                                self._insert_chunks_bulk(conn, chunks_data)
                                
                                # Store metadata field embeddings
                                if field_embeddings:
                                    self._store_metadata_embeddings(conn, chunk_data['doc_id'], field_embeddings)
                                
                                conn.commit()
                                
                                logger.debug(f"Processed {chunk_data['doc_id']}: "
                                             f"{len(unchanged_chunks)} reused, "
                                             f"{len(chunks_needing_embedding)} new chunks, "
                                             f"{len(field_embeddings)} metadata fields embedded")
                                
                            except Exception:
                                conn.rollback()
                                raise
                    
                    processed_ids.append(chunk_data['doc_id'])
                    result_queue.task_done()
                
                return processed_ids
                
            except Exception as e:
                logger.error(f"Database worker error: {e}")
                raise
        
        # Start workers
        workers = [
            threading.Thread(target=chunk_comparison_worker, name="ChunkComparisonWorker"),
            threading.Thread(target=embedding_worker, name="EmbeddingWorker")
        ]
        
        for worker in workers:
            worker.start()
        
        # Run database worker in main thread to get return value
        try:
            processed_ids = database_worker()
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise
        finally:
            # Ensure all workers complete
            for worker in workers:
                worker.join(timeout=30)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} did not complete in time")
        
        return processed_ids

    def _fetch_existing_chunks_batch(self, doc_ids: List[str]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Efficiently fetch existing chunk data for multiple documents

        Returns
        -------
        Dict[str, Dict[int, Dict[str, Any]]]
            Nested dict: {doc_id: {chunk_index: {content_hash, faiss_id}}}
        """
        if not doc_ids:
            return {}

        existing_chunks_by_doc = {}

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

        logger.debug(f"Fetched existing chunks for {len(existing_chunks_by_doc)} documents")
        return existing_chunks_by_doc

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

    def _remove_old_chunks_batch(
            self,
            doc_id: str,
            chunk_indices_to_remove: List[int],
            faiss_ids_to_remove: List[int]
    ) -> None:
        """
        Remove only the chunks and FAISS vectors that are being replaced

        For upsert operations, we:
        1. Keep the document record (will be updated by INSERT OR REPLACE)
        2. Only remove chunks that are being replaced
        3. Preserve unchanged chunks and their FAISS vectors

        Parameters
        ----------
        doc_id : str
            Document ID being processed
        chunk_indices_to_remove : List[int]
            Chunk indices that need to be removed
        faiss_ids_to_remove : List[int]
            FAISS IDs that need to be removed
        """
        # Remove specific chunks from database (but keep document record)
        if chunk_indices_to_remove:
            with self.connection_pool.get_connection() as conn:
                # Delete only the chunks that are being replaced
                placeholders = ','.join(['?'] * len(chunk_indices_to_remove))
                conn.execute(
                    f'DELETE FROM chunks WHERE document_id = ? AND chunk_index IN ({placeholders})',
                    [doc_id] + chunk_indices_to_remove
                )
                conn.commit()

        # Remove unused vectors from FAISS
        if faiss_ids_to_remove:
            self._remove_old_vectors_bulk(faiss_ids_to_remove)
            logger.debug(f"Removed {len(chunk_indices_to_remove)} old chunks and "
                         f"{len(faiss_ids_to_remove)} FAISS vectors for {doc_id}")


    def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document]]:
        """
        Retrieve documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to retrieve

        Returns
        -------
        Union[Document, List[Document]]
            Retrieved document(s)

        Raises
        ------
        DocumentNotFoundError
            If any requested documents are not found
        """
        single_id = isinstance(ids, str)
        requested_ids = [ids] if single_id else ids

        with self._read_write_lock.read_lock():
            with self.connection_pool.get_connection() as conn:
                # Build query to get documents with metadata
                metadata_columns = list(self.metadata_schema.keys())
                if metadata_columns:
                    columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
                else:
                    columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']

                placeholders = ','.join(['?'] * len(requested_ids))
                sql = f"SELECT {', '.join(columns)} FROM documents WHERE id IN ({placeholders})"

                cursor = conn.execute(sql, requested_ids)
                rows = cursor.fetchall()

                # Check for missing documents
                found_ids = {row['id'] for row in rows}
                missing_ids = [doc_id for doc_id in requested_ids if doc_id not in found_ids]

                if missing_ids:
                    if single_id:
                        raise DocumentNotFoundError(f"Document not found: {missing_ids[0]}", missing_ids[0])
                    else:
                        raise DocumentNotFoundError(
                            f"Documents not found: {', '.join(missing_ids)}",
                            missing_ids
                        )

                # Build document objects
                documents = []
                for row in rows:
                    # Extract metadata
                    metadata = {}
                    for col_name in metadata_columns:
                        if col_name in row.keys():
                            metadata[col_name] = row[col_name]
                            # if col_name in self.metadata_schema:
                            #     field_def = self.metadata_schema[col_name]
                            #     if field_def.type == MetadataFieldType.JSON and value:
                            #         value = json.loads(value)
                            # metadata[col_name] = value

                    doc = Document(
                        id=row['id'],
                        content=row['content'],
                        metadata=metadata,
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        content_hash=row['content_hash']
                    )
                    documents.append(doc)

            # Ensure documents are returned in the same order as requested
            id_to_doc = {doc.id: doc for doc in documents}
            ordered_documents = [id_to_doc[doc_id] for doc_id in requested_ids]

            if single_id:
                return ordered_documents[0]
            return ordered_documents


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

            # Get FAISS IDs for chunks and metadata to remove
            faiss_ids_to_remove = []

            with self.connection_pool.get_connection() as conn:
                placeholders = ','.join(['?'] * len(ids))

                # Get chunk FAISS IDs
                cursor = conn.execute(
                    f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                    ids
                )
                faiss_ids_to_remove.extend([row['faiss_id'] for row in cursor.fetchall()])

                # Get metadata field FAISS IDs
                cursor = conn.execute(
                    f'SELECT faiss_id FROM column_embeddings WHERE document_id IN ({placeholders})',
                    ids
                )
                faiss_ids_to_remove.extend([row['faiss_id'] for row in cursor.fetchall()])

                # Delete documents (cascades to chunks and column_embeddings)
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

                self._validate_metadata_batch([updated_metadata])

                # Check if any embedding-enabled metadata fields have changed
                changed_embedding_fields = self._get_changed_embedding_fields(
                    existing_doc.metadata, updated_metadata
                )

                with self.connection_pool.get_connection() as conn:
                    conn.execute('BEGIN')
                    try:
                        # Remove old metadata embeddings if any fields changed
                        if changed_embedding_fields:
                            self._remove_metadata_embeddings(conn, doc_id)
                            
                            # Generate new metadata embeddings for changed fields
                            new_field_embeddings = self._generate_metadata_embeddings(
                                updated_metadata, changed_embedding_fields, batch_size=100
                            )
                            
                            # Store new metadata embeddings
                            if new_field_embeddings:
                                self._store_metadata_embeddings(conn, doc_id, new_field_embeddings)
                                logger.debug(f"Updated embeddings for {len(new_field_embeddings)} metadata fields in document {doc_id}")

                        # Build UPDATE statement for metadata
                        set_clauses = ['updated_at = ?']
                        values = [datetime.now(UTC)]

                        for field_name, value in updated_metadata.items():
                            if field_name in self.metadata_schema:
                                set_clauses.append(f'{field_name} = ?')
                                values.append(value)

                        values.append(doc_id)
                        sql = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?"
                        conn.execute(sql, values)
                        conn.commit()
                        
                        logger.debug(f"Updated metadata for document {doc_id}")
                        
                    except Exception:
                        conn.rollback()
                        raise

                return True

            return False

    def query_builder(self):
        return QueryBuilder(self)

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
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
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
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters to pass to the scoring method function.
            - frequency_boost
                frequency_bias : 0.0 - 1.0, default = 0.3
                    The ratio of the frequency multiplier to apply. Higher favors documents with more matching chunks
            - harmonic_mean
                max_chunks : int, default = 5
                    The number of top-scoring chunks to include to calculate the score
                coverage_threshold : 0.0 - 1.0, default = 0.7
                    The score threshold, above which chunks are considered "high-quality" and give an
                    additional bonus to the score.
            - diminishing_returns
                decay_factor : float, default = 0.8
                    The decay of the cumulative score of multiple chunks from the same document
            - statistical
                best_weight : float, default = 0.6
                    The weight of the best scoring chunk in the total score
                mean_weight : float, default = 0.2
                    The weight of the mean chunk score in the total score
                consistency_weight : float, default = 0.1
                    The weight applied based on how low the variance in the chunk scores is
                coverage_weight : float, default = 0.1
                    The weight applied for how many chunks are retrieved
            - robust_mean
                outlier_threshold : float, default = 2.0
                    The z-score threshold to identifier outliers
                position_decay : float, default = 0.9
                    The penalization for the rank of the chunk on its score
            - percentile
                primary_percentile : float, default = 0.9
                    The first percentile of chunks to sample for the overall document score
                secondary_percentile : float, default = 0.7
                    The lower percentile of chunks to sample for the overall document score
                primary_weight : float, default = 0.7
                    The weight to apply to the primary percentile result

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        with self._read_write_lock.read_lock():
            if search_type == 'vector':
                return self._vector_search(query, return_type, k, score_threshold, filters,
                                           context_window, semantic_dedup_threshold, document_scoring_method,
                                           document_scoring_options)
            elif search_type == 'keyword':
                return self._keyword_search(query, return_type, k, score_threshold, filters,
                                            context_window, semantic_dedup_threshold, document_scoring_method,
                                            document_scoring_options)
            elif search_type == 'hybrid':
                return self._hybrid_search(query, return_type, k, score_threshold, filters, vector_weight,
                                           context_window, semantic_dedup_threshold, document_scoring_method,
                                           document_scoring_options)
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
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """Perform vector similarity search with enhanced processing"""

        # Generate query embedding
        query_embeddings = self.embedding_provider.embed_sync([query])
        query_embedding = np.array(query_embeddings[0])  # Get single embedding and convert to numpy array
        
        # Reshape for FAISS (expects 2D array: n_queries x embedding_dim)
        query_embedding = query_embedding.reshape(1, -1)

        # Search more chunks initially since we might deduplicate and need enough for final k
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == 'documents' else k * 2)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, initial_k)

        # Filter valid indices and calculate scores first
        valid_results = []
        valid_faiss_ids = []

        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx == -1:  # Invalid index
                continue

            # Convert distance to normalized score (0-1, higher=better)
            score = (max(0.0, 1.0 / (1.0 + float(dist))) - 0.5 / 0.5)

            if score < score_threshold:
                continue

            valid_results.append((int(idx), score))
            valid_faiss_ids.append(int(idx))

        if not valid_faiss_ids:
            return []

        # Batch fetch ALL chunk and document info in a single query
        chunk_results = []

        with self.connection_pool.get_connection() as conn:
            # Single query to get all chunk and document info
            placeholders = ','.join(['?'] * len(valid_faiss_ids))
            cursor = conn.execute(f'''
                SELECT c.*, d.id as doc_id, d.content as doc_content
                FROM chunks c
                JOIN documents d ON c.document_id = d.id  
                WHERE c.faiss_id IN ({placeholders})
            ''', valid_faiss_ids)

            # Create a mapping from faiss_id to row data
            faiss_id_to_row = {}
            doc_ids_to_fetch = set()

            for row in cursor.fetchall():
                faiss_id_to_row[row['faiss_id']] = row
                doc_ids_to_fetch.add(row['doc_id'])

            # Batch fetch all metadata at once
            doc_metadata_batch = self._get_documents_metadata_batch(conn, list(doc_ids_to_fetch))

            # Create results using the batched data
            for faiss_id, score in valid_results:
                row = faiss_id_to_row.get(faiss_id)
                if not row:
                    continue  # Skip if chunk not found

                doc_metadata = doc_metadata_batch.get(row['doc_id'], {})

                # Apply metadata filters early
                if filters and not matches_metadata_filter(doc_metadata, filters):
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
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            # Semantic deduplication needs sorted
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
                chunk_results, document_scoring_method, document_scoring_options
            )
            return document_results[:k]

        else:  # return_type == 'chunks'
            # Sort by score and limit
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            return chunk_results[:k]

    def _get_documents_metadata_batch(self, conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for multiple documents in a single query"""
        if not doc_ids or not self.metadata_schema:
            return {doc_id: {} for doc_id in doc_ids}

        metadata_columns = list(self.metadata_schema.keys())
        placeholders = ','.join(['?'] * len(doc_ids))

        cursor = conn.execute(
            f"SELECT id, {', '.join(metadata_columns)} FROM documents WHERE id IN ({placeholders})",
            doc_ids
        )

        result = {}
        for row in cursor.fetchall():
            doc_id = row['id']
            result[doc_id] = {col_name: row[col_name] for col_name in metadata_columns}

        # Ensure all requested doc_ids have entries (even if empty)
        for doc_id in doc_ids:
            if doc_id not in result:
                result[doc_id] = {}

        return result

    def _keyword_search(
            self,
            query: str,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """Perform keyword search using FTS5 with enhanced processing"""
        if not self.fts_enabled:
            logger.warning("FTS not available, returning empty results")
            return []

        # Sanitize and prepare query for FTS5
        sanitized_query = FTSQuerySanitization.sanitize_fts_query(query)
        if not sanitized_query:
            return []

        # Always get chunks first, then process based on return_type
        initial_k = k * 4 if semantic_dedup_threshold else (k * 3 if return_type == 'documents' else k * 2)

        chunk_results = []

        with self.connection_pool.get_connection() as conn:
            # First pass: Get all matching chunk IDs and scores from FTS
            cursor = conn.execute('''
                SELECT rowid, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (sanitized_query, initial_k))

            # Collect valid chunk IDs and scores
            valid_chunk_data = []
            valid_chunk_ids = []

            for row in cursor.fetchall():
                # Convert FTS5 rank to normalized score
                score = 1.0 - min(1.0, math.exp(float(row['rank'])))

                if score < score_threshold:
                    continue

                chunk_id = row['rowid']
                valid_chunk_data.append((chunk_id, score))
                valid_chunk_ids.append(chunk_id)

            if not valid_chunk_ids:
                return []

            # Batch fetch all chunk and document info in a single query
            placeholders = ','.join(['?'] * len(valid_chunk_ids))
            cursor = conn.execute(f'''
                SELECT c.*, d.id as doc_id, d.content as doc_content
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.id IN ({placeholders})
            ''', valid_chunk_ids)

            # Create mappings
            chunk_id_to_row = {}
            doc_ids_to_fetch = set()

            for row in cursor.fetchall():
                chunk_id_to_row[row['id']] = row
                doc_ids_to_fetch.add(row['doc_id'])

            # Batch fetch all metadata at once
            doc_metadata_batch = self._get_documents_metadata_batch(conn, list(doc_ids_to_fetch))

            # Create results using the batched data
            for chunk_id, score in valid_chunk_data:
                row = chunk_id_to_row.get(chunk_id)
                if not row:
                    continue  # Skip if chunk not found

                doc_metadata = doc_metadata_batch.get(row['doc_id'], {})

                # Apply metadata filters early
                if filters and not matches_metadata_filter(doc_metadata, filters):
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

        # Apply same processing pipeline as vector search
        if semantic_dedup_threshold is not None:
            chunk_results.sort(key=lambda x: x.score, reverse=True)
            chunk_results = self._apply_semantic_deduplication(chunk_results, semantic_dedup_threshold)

        if return_type == 'context':
            final_results = self._add_context_window(chunk_results, context_window)
            final_results.sort(key=lambda x: x.score, reverse=True)
            return final_results[:k]
        elif return_type == 'documents':
            document_results = self._aggregate_document_scores_with_method(
                chunk_results, document_scoring_method, document_scoring_options
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
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """Perform hybrid search combining vector and keyword with enhanced processing"""
        if not self.fts_enabled:
            logger.info("FTS not available, falling back to vector search")
            return self._vector_search(query, return_type, k, score_threshold, filters,
                                       context_window, semantic_dedup_threshold, document_scoring_method,
                                       document_scoring_options)

        # Get more results than requested for better reranking
        search_k = min(k * 4, 100)

        # Perform both searches - always get chunks
        vector_results = self._vector_search(query, 'chunks', search_k, 0.0, filters,
                                             0, None)  # No processing yet
        keyword_results = self._keyword_search(query, 'chunks', search_k, 0.0, filters,
                                               0, None)  # No processing yet

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
                combined_results, document_scoring_method, document_scoring_options
            )
            return document_results[:k]
        else:  # chunks
            combined_results.sort(key=lambda x: x.score, reverse=True)
            return combined_results[:k]

    def get_chunk_embeddings(self, chunk_ids: str | List[str]) -> np.ndarray:
        """Returns embeddings for chunks given by `chunk_ids`"""
        single_id = isinstance(chunk_ids, str)
        if single_id:
            chunk_ids = [chunk_ids]

        chunk_list = []
        for cid in chunk_ids:
            doc_id, chunk_idx = self._split_chunk_id(cid)
            if chunk_idx == -1:
                raise ValueError(f"Expected chunk ids (e.g. doc_1:1), found: {cid}")
            chunk_list.append((doc_id, chunk_idx))

        placeholders = ",".join(["(?,?)"] * len(chunk_list))
        query_str = f"""SELECT faiss_id, document_id, chunk_index FROM chunks WHERE (document_id, chunk_index) IN ({placeholders})"""
        params = [item for pair in chunk_list for item in pair]
        with self._read_write_lock.read_lock():
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute(query_str, params)
                rows = cursor.fetchall()
                faiss_ids = [row["faiss_id"] for row in rows]

            return self._reconstruct_embeddings_batch(faiss_ids)

    def _reconstruct_embeddings_batch(self, faiss_ids: List[int]) -> np.ndarray:
        """Batch reconstruct embeddings handling IndexIDMap"""
        if not faiss_ids:
            return np.array([]).reshape(0, self.embedding_dimension)

        # Convert FAISS IDs to underlying index positions
        faiss_ids_array = np.array(faiss_ids, dtype=np.int64)

        # Get the mapping from FAISS IDs to internal indices
        # IndexIDMap stores id_map which maps external IDs to internal positions
        internal_indices = []
        for fid in faiss_ids_array:
            # Find the internal index for this FAISS ID
            # The id_map contains the mapping
            internal_idx = -1
            for i in range(self.index.ntotal):
                if self.index.id_map.at(i) == fid:
                    internal_idx = i
                    break
            if internal_idx != -1:
                internal_indices.append(internal_idx)

        if not internal_indices:
            return np.array([]).reshape(0, self.embedding_dimension)

        # Use the underlying index for batch reconstruction
        internal_indices_array = np.array(internal_indices, dtype=np.int64)
        embeddings = self.index.index.reconstruct_batch(internal_indices_array)

        return embeddings

    def _apply_semantic_deduplication(
            self,
            results: List[QueryResult],
            threshold: float
    ) -> List[QueryResult]:
        """
        Apply semantic deduplication to search results using FAISS index embeddings.

        Optimized version that minimizes database calls and uses batch FAISS operations.

        Parameters
        ----------
        results : List[QueryResult]
            Initial search results to deduplicate - MUST BE SORTED with highest score first
        threshold : float
            Similarity threshold (0-1, higher=more similar). Chunks above this threshold are considered duplicates.

        Returns
        -------
        List[QueryResult]
            Deduplicated results with highest-scored chunk from each similar group
        """
        if not results or threshold is None or threshold <= 0:
            return results

        # Separate chunk results from other types (only chunks have embeddings)
        chunk_results = [r for r in results if r.type == 'chunk']

        if len(chunk_results) <= 1:
            return results  # No deduplication needed

        # Step 1: Batch retrieve all FAISS IDs in a single SQL query
        chunk_identifiers = [(r.document_id, self._extract_chunk_index_from_id(r.id)) for r in chunk_results]

        with self.connection_pool.get_connection() as conn:
            # Create parameterized query for all chunks at once
            placeholders = ','.join(['(?,?)'] * len(chunk_identifiers))
            query = f'''
                SELECT document_id, chunk_index, faiss_id 
                FROM chunks 
                WHERE (document_id, chunk_index) IN ({placeholders})
            '''

            # Flatten the list of tuples for query parameters
            params = [item for pair in chunk_identifiers for item in pair]
            cursor = conn.execute(query, params)
            faiss_id_mapping = {
                (row['document_id'], row['chunk_index']): row['faiss_id']
                for row in cursor.fetchall()
            }

        # Step 2: Extract FAISS IDs and create mapping to results
        faiss_ids = []
        result_mapping = {}  # faiss_id -> QueryResult

        for result in chunk_results:
            doc_id, chunk_idx = self._split_chunk_id(result.id)
            faiss_id = faiss_id_mapping.get((doc_id, chunk_idx))

            if faiss_id is not None:
                faiss_ids.append(faiss_id)
                result_mapping[faiss_id] = result

        # Step 3: Batch retrieve embeddings from FAISS index
        embeddings_matrix = self._reconstruct_embeddings_batch(faiss_ids)

        # Step 4: Compute pairwise similarities using vectorized operations
        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        normalized_embeddings = embeddings_matrix / np.maximum(norms, 1e-8)

        # Compute similarity matrix (upper triangular to avoid duplicates)
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Step 5: Identify duplicates using pure numpy operations
        # Since results are sorted by score (descending), score[i] >= score[j] when i < j
        # Create boolean mask for similarities above threshold (excluding diagonal)
        similar_pairs = similarity_matrix >= threshold
        np.fill_diagonal(similar_pairs, False)  # Exclude self-similarity

        # Step 6: Vectorized duplicate identification
        # Since we're sorted by score, we only need the upper triangular part
        # A chunk j should be removed if it's similar to any earlier (higher-scoring) chunk i where i < j
        upper_triangular_similarities = np.triu(similar_pairs, k=1)

        # For each chunk j, check if any earlier chunk i is similar (column-wise check)
        should_remove = np.any(upper_triangular_similarities, axis=0)

        # Step 7: Build final results using boolean indexing
        keep_mask = ~should_remove
        final_chunk_results = [result_mapping[faiss_ids[i]] for i in range(len(faiss_ids)) if keep_mask[i]]

        # Log deduplication statistics
        original_count = len(chunk_results)
        final_count = len(final_chunk_results)
        removed_count = original_count - final_count

        logger.debug(f"Semantic deduplication: {original_count} → {final_count} chunks "
                     f"({removed_count} removed)")

        # Combine deduplicated chunks with non-chunk results
        return final_chunk_results

    @staticmethod
    def _split_chunk_id(chunk_id: str) -> tuple[str, int]:
        try:
            parts = chunk_id.rsplit(':', maxsplit=1)
            chunk_idx = int(parts[-1])
            doc_id = parts[0]
            return doc_id, chunk_idx
        except (ValueError, IndexError, TypeError):
            return chunk_id, -1

    @staticmethod
    def _extract_chunk_index_from_id(chunk_id: str) -> int:
        """Extract chunk index from chunk ID format 'doc_id:chunk_index'"""
        _, chunk_idx = LocalVectorDB._split_chunk_id(chunk_id)
        return chunk_idx

    def _add_context_window(
            self,
            results: List[QueryResult],
            context_window: int
    ) -> List[QueryResult]:
        """
        Add context window around found chunks by including surrounding chunks.
        Optimized to batch database queries by document and merge overlapping ranges.
        """
        if context_window <= 0 or not results:
            return results

        context_results = []

        # Group results by document and collect chunk indices
        doc_chunk_requests = defaultdict(list)

        for result in results:
            if result.type != 'chunk':
                # For document results, just pass through
                context_results.append(result)
                continue

            doc_id = result.document_id
            chunk_index = self._extract_chunk_index_from_id(result.id)
            doc_chunk_requests[doc_id].append((chunk_index, result))

        # Process each document
        with self.connection_pool.get_connection() as conn:
            for doc_id, chunk_requests in doc_chunk_requests.items():
                # Calculate ranges needed for all chunks in this document
                ranges_needed = []
                for chunk_index, result in chunk_requests:
                    start_index = max(0, chunk_index - context_window)
                    end_index = chunk_index + context_window
                    ranges_needed.append((start_index, end_index, chunk_index, result))

                # Merge overlapping ranges to minimize data fetched
                merged_ranges = self._merge_overlapping_ranges(ranges_needed)

                # Fetch all needed chunks for this document in a single query
                all_chunk_indices = set()
                for start, end, _, _ in merged_ranges:
                    all_chunk_indices.update(range(start, end + 1))

                if not all_chunk_indices:
                    continue

                # Single query to get all chunks we need for this document
                placeholders = ','.join(['?'] * len(all_chunk_indices))
                cursor = conn.execute(f'''
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col
                    FROM chunks 
                    WHERE document_id = ? AND chunk_index IN ({placeholders})
                    ORDER BY chunk_index
                ''', [doc_id] + list(all_chunk_indices))

                # Create lookup map
                chunks_by_index = {row['chunk_index']: row for row in cursor.fetchall()}

                # Process each original result for this document
                for chunk_index, result in chunk_requests:
                    # Get context chunks for this specific result
                    context_chunks = []
                    start_context = max(0, chunk_index - context_window)
                    end_context = chunk_index + context_window

                    for i in range(start_context, end_context + 1):
                        if i in chunks_by_index:
                            context_chunks.append(chunks_by_index[i])

                    if not context_chunks:
                        # No context found, use original result
                        context_results.append(result)
                        continue

                    # Combine chunks into single content (same logic as original)
                    combined_content = []
                    min_start_pos = float('inf')
                    max_end_pos = 0
                    min_start_line = float('inf')
                    min_start_col = float('inf')
                    max_end_line = 0
                    max_end_col = 0

                    for chunk_row in context_chunks:
                        combined_content.append(chunk_row['content'])

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
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(combined_content)

                    context_result = QueryResult(
                        id=f"{doc_id}:context:{chunk_index}",
                        score=result.score,  # Keep original score
                        type="context",
                        content=combined_text,
                        metadata=result.metadata.copy(),
                        document_id=doc_id,
                        position=context_position
                    )

                    # Add metadata about context
                    context_result.metadata['_context_window'] = context_window
                    context_result.metadata['_original_chunk_index'] = chunk_index
                    context_result.metadata['_context_chunk_count'] = len(context_chunks)
                    context_result.metadata['_context_start_index'] = start_context
                    context_result.metadata['_context_end_index'] = end_context

                    context_results.append(context_result)

        return context_results

    @staticmethod
    def _merge_overlapping_ranges(ranges_needed: List[Tuple[int, int, int, Any]]) -> List[Tuple[int, int, int, Any]]:
        """
        Merge overlapping ranges to minimize database queries.

        For example, if we need chunks [5-10] and [8-15], we can fetch [5-15] once
        instead of making separate queries.

        Parameters
        ----------
        ranges_needed : List[Tuple[int, int, int, Any]]
            List of (start_index, end_index, original_chunk_index, result_object)

        Returns
        -------
        List[Tuple[int, int, int, Any]]
            Merged ranges - note that multiple original requests might map to one merged range
        """
        if not ranges_needed:
            return []

        # Sort by start index
        sorted_ranges = sorted(ranges_needed, key=lambda x: x[0])

        merged = []
        current_start, current_end, first_chunk, first_result = sorted_ranges[0]

        for start, end, chunk_idx, result in sorted_ranges[1:]:
            if start <= current_end + 1:  # +1 to merge adjacent ranges
                # Overlapping or adjacent, extend current range
                current_end = max(current_end, end)
            else:
                # No overlap, save current range and start new one
                merged.append((current_start, current_end, first_chunk, first_result))
                current_start, current_end, first_chunk, first_result = start, end, chunk_idx, result

        # Add the last range
        merged.append((current_start, current_end, first_chunk, first_result))

        return merged

    def query_multi_column(
            self,
            query: str,
            *,
            columns: Optional[List[str]] = None,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """
        Query across multiple columns (main content + embedding-enabled metadata fields)
        
        Parameters
        ----------
        query : str
            Query text
        columns : Optional[List[str]]
            Specific columns to search. If None, searches all embedding-enabled fields
            plus main content. Use 'content' for main document content.
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters for the scoring method
            
        Returns
        -------
        List[QueryResult]
            Search results with column attribution
        """
        with self._read_write_lock.read_lock():
            # Determine which columns to search
            embedding_enabled_fields = self._get_embedding_enabled_fields()

            if columns is None:
                # Search all embedding-enabled fields plus main content
                search_columns = ['content'] + list(embedding_enabled_fields.keys())
            else:
                # Validate requested columns
                search_columns = []
                for col in columns:
                    if col == 'content':
                        search_columns.append(col)
                    elif col in embedding_enabled_fields:
                        search_columns.append(col)
                    else:
                        logger.warning(f"Column '{col}' is not embedding-enabled, skipping")

                if not search_columns:
                    logger.warning("No valid columns specified for search")
                    return []

            all_results = []

            # Search main content if requested
            if 'content' in search_columns:
                content_results = self.query(
                    query=query,
                    search_type=search_type,
                    return_type='chunks',  # Always get chunks for multi-column
                    k=k * 2,  # Get more results to allow for proper ranking
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    document_scoring_method=document_scoring_method,
                    document_scoring_options=document_scoring_options
                )

                # Add column attribution
                for result in content_results:
                    result.metadata = result.metadata or {}
                    result.metadata['_search_column'] = 'content'
                    all_results.append(result)

            # Search metadata fields
            metadata_columns = [col for col in search_columns if col != 'content']
            if metadata_columns and search_type in ['vector', 'hybrid']:
                for field_name in metadata_columns:
                    field_results = self._search_metadata_field(
                        query=query,
                        field_name=field_name,
                        k=k * 2,
                        score_threshold=score_threshold,
                        filters=filters
                    )

                    # Add column attribution
                    for result in field_results:
                        result.metadata = result.metadata or {}
                        result.metadata['_search_column'] = field_name
                        all_results.append(result)

            # Sort all results by score and limit
            all_results.sort(key=lambda x: x.score, reverse=True)
            limited_results = all_results[:k]

            if return_type == 'documents':
                # Aggregate chunks into documents
                return self._aggregate_document_scores_with_method(
                    limited_results,
                    document_scoring_method,
                    document_scoring_options
                )
            else:
                return limited_results

    def _search_metadata_field(
            self,
            query: str,
            field_name: str,
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]]
    ) -> List[QueryResult]:
        """
        Search a specific metadata field's embeddings
        
        Parameters
        ----------
        query : str
            Query text
        field_name : str
            Name of metadata field to search
        k : int
            Maximum results to return
        score_threshold : float
            Minimum score threshold
        filters : Optional[Dict[str, Any]]
            Metadata filters
            
        Returns
        -------
        List[QueryResult]
            Search results for this field
        """
        # Generate query embedding
        if hasattr(self.embedding_provider, 'embed_query'):
            query_embedding = self.embedding_provider.embed_query(query)
        else:
            # Fallback to embed_sync for single query
            query_embedding = self.embedding_provider.embed_sync([query])[0]

        with self.connection_pool.get_connection() as conn:
            # Get all metadata field embeddings
            cursor = conn.execute("""
                SELECT ce.faiss_id, ce.document_id, ce.chunk_index, d.content, d.created_at, d.updated_at
                FROM column_embeddings ce
                JOIN documents d ON ce.document_id = d.id
                WHERE ce.field_name = ?
            """, (field_name,))

            field_embedding_data = cursor.fetchall()

            if not field_embedding_data:
                return []

            # Extract FAISS IDs for this field
            faiss_ids = [row['faiss_id'] for row in field_embedding_data]

            # Search in FAISS index
            if not faiss_ids:
                return []

            # Get embeddings for these FAISS IDs
            field_embeddings = self._reconstruct_embeddings_batch(faiss_ids)

            if field_embeddings.size == 0:
                return []

            # Compute similarities
            query_embedding_2d = query_embedding.reshape(1, -1)
            similarities = np.dot(field_embeddings, query_embedding_2d.T).flatten()

            # Convert to scores (higher is better)
            scores = (similarities + 1) / 2  # Normalize to 0-1

            # Filter by score threshold
            valid_indices = np.where(scores >= score_threshold)[0]

            if len(valid_indices) == 0:
                return []

            # Sort by score and limit
            sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]][:k]

            results = []
            for idx in sorted_indices:
                row_data = field_embedding_data[idx]

                # Get document metadata
                doc_metadata = self._get_document_metadata(conn, row_data['document_id'])

                # Create result
                result = QueryResult(
                    id=f"{row_data['document_id']}:meta:{field_name}:{row_data['chunk_index']}",
                    content=str(doc_metadata.get(field_name, "")),
                    score=float(scores[idx]),
                    document_id=row_data['document_id'],
                    metadata=doc_metadata,
                    type='chunk'
                )
                results.append(result)

            # Apply metadata filters if provided
            if filters:
                results = [r for r in results if matches_metadata_filter(r.metadata, filters)]

            return results

    def _get_document_metadata(self, conn: sqlite3.Connection, document_id: str) -> Dict[str, Any]:
        """Get metadata for a single document"""
        if not self.metadata_schema:
            return {}

        columns = ['id'] + list(self.metadata_schema.keys())
        cursor = conn.execute(f"SELECT {', '.join(columns)} FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()

        if not row:
            return {}

        return {col: row[col] for col in columns[1:]}  # Exclude 'id'

    def _aggregate_document_scores_with_method(
            self,
            chunk_results: List[QueryResult],
            method: DocumentScoringMethod = "frequency_boost",
            method_options: dict = None
    ) -> List[QueryResult]:
        """
        Aggregate chunk results into document results with enhanced scoring.

        Parameters
        ----------
        chunk_results : List[QueryResult]
            Chunk-level search results to aggregate by document
        method : DocumentScoringMethod
            Scoring method to aggregate score at the document-level.

        Returns
        -------
        List[QueryResult]
            Document-level results with aggregated scores
        """
        if not chunk_results:
            return []
        method_options = method_options or {}

        # Group chunks by document
        doc_groups = defaultdict(list)
        for result in chunk_results:
            doc_id = result.document_id if result.type == 'chunk' else result.id
            doc_groups[doc_id].append(result)

        # Get all unique document IDs
        all_doc_ids = list(doc_groups.keys())

        if not all_doc_ids:
            return []

        # Batch fetch all document content and metadata in single queries
        with self.connection_pool.get_connection() as conn:
            # Batch fetch document content
            placeholders = ','.join(['?'] * len(all_doc_ids))
            cursor = conn.execute(f'''
                SELECT id, content 
                FROM documents 
                WHERE id IN ({placeholders})
            ''', all_doc_ids)

            doc_content_map = {row['id']: row['content'] for row in cursor.fetchall()}

            # Batch fetch all metadata at once
            doc_metadata_batch = self._get_documents_metadata_batch(conn, all_doc_ids)

        return self._compute_document_scores(method, method_options, doc_groups, doc_content_map, doc_metadata_batch)


    @staticmethod
    def _compute_document_scores(method, method_options, doc_groups, doc_content_map, doc_metadata_batch):
        document_results = []

        for doc_id, chunks in doc_groups.items():
            # Check if we have the document content
            doc_content = doc_content_map.get(doc_id)
            if not doc_content:
                continue  # Skip if document not found

            # Calculate aggregated score based on method
            scores = [chunk.score for chunk in chunks]
            method_metadata = {}

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
                method_metadata["weights"] = weights.tolist()
                final_score = np.average(scores, weights=weights)
            elif method == "frequency_boost":
                best_score = max(scores)

                # Handle edge case where all scores are 0
                if best_score == 0:
                    quality_weights = [1.0 for _ in scores]  # Equal weights if all scores are 0
                else:
                    quality_weights = [score / best_score for score in scores]
                effective_chunk_count = sum(quality_weights)
                frequency_multiplier = (1.0 + (math.log2(2 + effective_chunk_count) - 1)
                                        * method_options.get("frequency_bias", 0.3))

                method_metadata["effective_chunk_count"] = effective_chunk_count
                method_metadata["frequency_multiplier"] = frequency_multiplier
                final_score = min(1.0, best_score * frequency_multiplier)
            elif method == "harmonic_mean":
                max_chunks_for_harmonic = method_options.get("max_chunks", 5)
                coverage_threshold = method_options.get("coverage_threshold", 0.7)
                sorted_scores = sorted(scores, reverse=True)
                # Use top chunk scores for harmonic mean
                top_scores = sorted_scores[:max_chunks_for_harmonic]
                harmonic_mean = len(top_scores) / sum(1 / max(score, 0.001) for score in top_scores)

                # Coverage bonus: percentage of chunks above threshold
                high_quality_chunks = sum(1 for score in scores if score >= coverage_threshold)
                coverage_ratio = high_quality_chunks / len(scores)
                coverage_bonus = 1.0 + (coverage_ratio * 0.2)  # Max 20% bonus

                method_metadata["harmonic_mean"] = harmonic_mean
                method_metadata["coverage_ratio"] = coverage_ratio

                final_score = min(1.0, harmonic_mean * coverage_bonus)
            elif method == "diminishing_returns":
                sorted_scores = sorted(scores, reverse=True)
                decay_factor = method_options.get("decay_factor", 0.8)
                total_score = 0.0
                weight = 1.0

                for score in sorted_scores:
                    total_score += score * weight
                    weight *= decay_factor  # Each subsequent chunk has less impact

                # Normalize by the theoretical maximum to keep scores in [0,1] range
                max_possible = sum(decay_factor ** i for i in range(len(scores)))
                method_metadata["max_possible"] = max_possible

                final_score = min(1.0, total_score / max_possible)
            elif method == "statistical":
                if len(scores) == 1:
                    final_score = scores[0]
                else:
                    best_weight = method_options.get("best_weight", 0.6)
                    mean_weight = method_options.get("mean_weight", 0.2)
                    consistency_weight = method_options.get("consistency_weight", 0.1)
                    coverage_weight = method_options.get("coverage_weight", 0.1)
                    best_score = max(scores)
                    mean_score = statistics.mean(scores)

                    # Calculate consistency (inverse of coefficient of variation)
                    std_dev = statistics.stdev(scores)
                    consistency = 1.0 - min(1.0, std_dev / mean_score) if mean_score > 0 else 0.0

                    # Percentage of chunks above median
                    median_score = statistics.median(scores)
                    above_median_ratio = sum(1 for score in scores if score >= median_score) / len(scores)

                    method_metadata["standard_deviation"] = std_dev
                    method_metadata["median"] = median_score
                    method_metadata["above_median_ratio"] = above_median_ratio

                    # Combine metrics with specified weights
                    final_score = (
                            best_score * best_weight +
                            mean_score * mean_weight +
                            consistency * consistency_weight +
                            above_median_ratio * coverage_weight
                    )

                    final_score = min(1.0, final_score)
            elif method == "robust_mean":
                if len(scores) == 1:
                    final_score = scores[0]
                else:
                    outlier_threshold = method_options.get("outlier_threshold", 2.0)
                    position_decay = method_options.get("position_decay", 0.9)
                    # Remove outliers using z-score
                    mean_score = statistics.mean(scores)
                    std_score = statistics.stdev(scores) if len(scores) > 1 else 0

                    if std_score > 0:
                        filtered_scores = [
                            score for score in scores
                            if abs(score - mean_score) <= outlier_threshold * std_score
                        ]
                    else:
                        filtered_scores = scores

                    if not filtered_scores:
                        filtered_scores = scores  # Fallback if all scores were outliers

                    # Sort and apply position-based weighting
                    sorted_scores = sorted(filtered_scores, reverse=True)
                    weights = [position_decay ** i for i in range(len(sorted_scores))]

                    weighted_sum = sum(score * weight for score, weight in zip(sorted_scores, weights, strict=False))
                    weight_sum = sum(weights)

                    method_metadata["standard_deviation"] = std_score
                    method_metadata["weighted_sum"] = weighted_sum
                    method_metadata["weights"] = weights

                    final_score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
            elif method == "percentile":
                if len(scores) == 1:
                    final_score = scores[0]
                else:
                    primary_percentile = method_options.get("primary_percentile", 0.9)
                    secondary_percentile = method_options.get("secondary_percentile", 0.7)
                    primary_weight = method_options.get("primary_weight", 0.7)
                    # Calculate percentiles
                    primary_score = np.percentile(scores, primary_percentile * 100)
                    secondary_score = np.percentile(scores, secondary_percentile * 100)
                    method_metadata["primary_score"] = primary_score
                    method_metadata["secondary_score"] = secondary_score
                    method_metadata["primary_percentile"] = primary_percentile
                    method_metadata["secondary_percentile"] = secondary_percentile
                    method_metadata["primary_wieght"] = primary_weight
                    # Weighted combination
                    final_score = primary_score * primary_weight + secondary_score * (1 - primary_weight)
            elif method == "geometric_mean":
                stabilization_factor = 0.1
                stabilized_scores = [score + stabilization_factor for score in scores]

                # Calculate geometric mean
                product = 1.0
                for score in stabilized_scores:
                    product *= score

                geometric_mean = product ** (1.0 / len(stabilized_scores))

                # Remove stabilization factor
                final_score = max(0.0, geometric_mean - stabilization_factor)
            else:
                final_score = max(scores)  # Default to best

            # Get document metadata
            doc_metadata = doc_metadata_batch.get(doc_id, {})

            # Add aggregation metadata
            method_metadata['_aggregation_method'] = method
            method_metadata['_chunk_count'] = len(chunks)
            method_metadata['_best_chunk_score'] = max(scores)
            method_metadata['_average_chunk_score'] = sum(scores) / len(scores)
            doc_metadata["_scoring"] = method_metadata

            # Create document result
            doc_result = QueryResult(
                id=doc_id,
                score=final_score,
                type='document',
                content=doc_content,
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

        filter_builder = None

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

        # Build secure ORDER BY clause
        if order_by:
            try:
                # Create or reuse filter_builder for ORDER BY validation
                if filter_builder is None:
                    filter_builder = FilterQueryBuilder(self.metadata_schema)

                # Build secure ORDER BY clause with proper validation and quoting
                valid_columns = set(columns)  # Use the actual columns from this query
                order_by_clause = filter_builder.build_order_by_clause(order_by, valid_columns)
                query_parts.append(order_by_clause)
            except Exception as e:
                raise MetadataFilterError(f"Error building ORDER BY clause: {str(e)}")

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
                metadata = {col_name: row[col_name] for col_name in metadata_columns}
                doc = Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=metadata,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    content_hash=row['content_hash']
                )
                documents.append(doc)

        return documents

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
        with self._read_write_lock.write_lock():
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

    async def update_metadata_schema_async(
            self,
            new_schema: Union[str, Dict[str, MetadataField]],
            drop_columns: bool = False,
            column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Update the metadata schema for the database asynchronously

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

            changes = await db.update_metadata_schema_async(new_schema)
            print(f"Added fields: {changes['added_fields']}")

        Use shorthand syntax::

            new_schema = {
                'category': 'text',  # Simple type
                'priority': ('integer', False, True),  # (type, indexed, required)
                'rating': ('real', True)  # (type, indexed)
            }

            changes = await db.update_metadata_schema_async(new_schema)

        Apply a common schema::

            changes = await db.update_metadata_schema_async('research_papers')

        Notes
        -----
        - Field names cannot conflict with reserved columns: id, content, content_hash, created_at, updated_at
        - Removed fields are removed from the schema but columns are kept for data safety
        - Type changes are recorded but don't modify existing data (SQLite limitation)
        - Index changes are applied immediately
        - Changes are applied in a transaction and rolled back on error
        """
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
            # Ensure async connection pool is initialized
            if self.async_connection_pool is None:
                self.async_connection_pool = AsyncConnectionPool(self.db_path, self.async_max_connections)

            # Apply schema changes using async connection
            async with self.async_connection_pool.get_connection_context() as conn:
                changes = await self.schema.update_metadata_schema_async(new_schema, conn, drop_columns, column_mapping)

            # Update in-memory schema
            self._metadata_schema = new_schema.copy()

            # Log the changes
            logger.info(f"Updated metadata schema for database '{self.name}' (async)")
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

            return changes

        except Exception as e:
            logger.error(f"Failed to update metadata schema (async): {e}")
            raise DatabaseError(f"Schema update failed: {str(e)}")

    async def get_metadata_schema_info_async(self) -> Dict[str, Any]:
        """
        Get detailed information about the current metadata schema asynchronously

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

        if hasattr(self, 'async_connection_pool') and self.async_connection_pool is not None:
            try:
                # Only run if there's an active event loop
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    # Schedule cleanup
                    asyncio.create_task(self.close_async())
                else:
                    asyncio.run(self.close_async())
            except RuntimeError:
                # No event loop running, use asyncio.run
                asyncio.run(self.close_async())
            except Exception as e:
                logger.warning(f"Error closing async resources: {e}")

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

    def count(
            self,
            filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Async count documents matching filter criteria

        Parameters
        ----------
        filters : Optional[Dict[str, Any]]
            Filter criteria using MongoDB-style syntax, by default None

        Returns
        -------
        int
            Number of documents matching the criteria
        """
        if filters:
            # Build filter query
            filter_builder = FilterQueryBuilder(self.metadata_schema)
            where_clause, params = filter_builder.build_where_clause(filters)
            sql = f"SELECT COUNT(*) as count FROM documents WHERE {where_clause}"
        else:
            sql = "SELECT COUNT(*) as count FROM documents"
            params = []

        with self.connection_pool.get_connection_context() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            return row['count'] if row else 0


    #############################
    # Metadata Embedding Methods #
    #############################

    def _get_embedding_enabled_fields(self) -> Dict[str, MetadataField]:
        """Get all metadata fields that have embedding_enabled=True"""
        return {
            field_name: field_def
            for field_name, field_def in self.metadata_schema.items()
            if field_def.embedding_enabled
        }

    def _get_changed_embedding_fields(
        self, 
        old_metadata: Dict[str, Any], 
        new_metadata: Dict[str, Any]
    ) -> Dict[str, MetadataField]:
        """
        Identify embedding-enabled metadata fields that have changed values.
        
        Parameters
        ----------
        old_metadata : Dict[str, Any]
            Original metadata values
        new_metadata : Dict[str, Any]
            New metadata values
            
        Returns
        -------
        Dict[str, MetadataField]
            Dictionary of changed embedding-enabled fields and their definitions
        """
        embedding_enabled_fields = self._get_embedding_enabled_fields()
        changed_fields = {}
        
        for field_name, field_def in embedding_enabled_fields.items():
            old_value = old_metadata.get(field_name)
            new_value = new_metadata.get(field_name)
            
            # Check if field value actually changed
            if old_value != new_value:
                # Only include if new value is not None/empty
                if new_value is not None and str(new_value).strip():
                    changed_fields[field_name] = field_def
                elif old_value is not None and str(old_value).strip():
                    # Old value existed but new is None/empty - still need to update
                    changed_fields[field_name] = field_def
                    
        return changed_fields

    def _track_column_embedding(
        self,
        conn: sqlite3.Connection,
        document_id: str,
        field_name: str,
        chunk_index: int,
        faiss_id: int
    ) -> None:
        """Track a metadata field embedding in the column_embeddings table"""
        conn.execute("""
            INSERT OR REPLACE INTO column_embeddings 
            (document_id, field_name, chunk_index, faiss_id)
            VALUES (?, ?, ?, ?)
        """, (document_id, field_name, chunk_index, faiss_id))

    def _generate_metadata_embeddings(
        self,
        metadata: Dict[str, Any],
        embedding_enabled_fields: Dict[str, MetadataField],
        batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for metadata fields that have embedding_enabled=True
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Document metadata
        embedding_enabled_fields : Dict[str, MetadataField]
            Fields that need embeddings
        batch_size : int
            Batch size for embedding generation
            
        Returns
        -------
        Dict[str, np.ndarray]
            Field name to embeddings mapping
        """
        field_embeddings = {}

        for field_name, field_def in embedding_enabled_fields.items():
            field_value = metadata.get(field_name)

            if field_value is None:
                continue

            # Convert value to text for embedding
            if field_def.type == MetadataFieldType.JSON:
                text_value = json.dumps(field_value)
            else:
                text_value = str(field_value)

            # TODO: probably remove the chunking?
            # Chunk the field value if it's long
            field_chunks = self.chunker.chunk(text_value)

            if field_chunks:
                # Generate embeddings for all chunks
                chunk_texts = [chunk.content for chunk in field_chunks]
                embeddings = self.embedding_provider.embed_sync(chunk_texts, batch_size)
                field_embeddings[field_name] = embeddings

        return field_embeddings

    async def _generate_metadata_embeddings_async(
        self,
        metadata: Dict[str, Any],
        embedding_enabled_fields: Dict[str, MetadataField],
        batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Async version of generating embeddings for metadata fields that have embedding_enabled=True
        
        Parameters
        ----------
        metadata : Dict[str, Any]
            Document metadata
        embedding_enabled_fields : Dict[str, MetadataField]
            Fields that need embeddings
        batch_size : int
            Batch size for embedding generation
            
        Returns
        -------
        Dict[str, np.ndarray]
            Field name to embeddings array mapping
        """
        field_embeddings = {}
        
        for field_name, field_def in embedding_enabled_fields.items():
            if field_name not in metadata:
                continue
                
            field_value = metadata[field_name]
            
            # Handle different field types
            field_chunks = []
            if field_def.type == MetadataFieldType.TEXT:
                if isinstance(field_value, str) and field_value.strip():
                    # Chunk the text field (simplified chunking for metadata)
                    text_chunks = self.chunker.chunk(field_value)
                    field_chunks.extend(text_chunks)
            elif field_def.type == MetadataFieldType.JSON:
                if field_value:
                    # Convert JSON to text and chunk
                    text_value = str(field_value)
                    text_chunks = self.chunker.chunk(text_value)
                    field_chunks.extend(text_chunks)
            
            if field_chunks:
                # Generate embeddings for all chunks asynchronously
                chunk_texts = [chunk.content for chunk in field_chunks]
                embeddings = await self.embedding_provider.embed_batch(chunk_texts, batch_size)
                field_embeddings[field_name] = embeddings

        return field_embeddings

    async def _remove_metadata_embeddings_async(
        self,
        conn: aiosqlite.Connection,
        document_id: str
    ) -> None:
        """
        Async version of removing metadata field embeddings for a document
        
        Parameters
        ----------
        conn : aiosqlite.Connection
            Async database connection
        document_id : str
            Document ID to remove embeddings for
        """
        # Get FAISS IDs for metadata embeddings
        cursor = await conn.execute("""
            SELECT faiss_id FROM column_embeddings 
            WHERE document_id = ?
        """, (document_id,))
        
        rows = await cursor.fetchall()
        faiss_ids_to_remove = [row['faiss_id'] for row in rows]
        
        # Remove FAISS vectors (run in executor since it's sync)
        if faiss_ids_to_remove:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids_to_remove)
        
        # Remove database records
        await conn.execute("""
            DELETE FROM column_embeddings 
            WHERE document_id = ?
        """, (document_id,))

    async def _store_metadata_embeddings_async(
        self,
        conn: aiosqlite.Connection,
        document_id: str,
        field_embeddings: Dict[str, np.ndarray]
    ) -> None:
        """
        Async version of storing metadata field embeddings in FAISS and track in column_embeddings table
        
        Parameters
        ----------
        conn : aiosqlite.Connection
            Async database connection
        document_id : str
            Document ID
        field_embeddings : Dict[str, np.ndarray]
            Field name to embeddings mapping
        """
        for field_name, embeddings in field_embeddings.items():
            if embeddings.size == 0:
                continue
            
            # Add to FAISS index (run in executor since it's sync)
            loop = asyncio.get_event_loop()
            start_id = await loop.run_in_executor(None, self._add_vectors_to_faiss_bulk, embeddings, [])
            
            # Track in column_embeddings table
            for chunk_index in range(len(embeddings)):
                faiss_id = start_id + chunk_index
                await conn.execute("""
                    INSERT OR REPLACE INTO column_embeddings 
                    (document_id, field_name, chunk_index, faiss_id)
                    VALUES (?, ?, ?, ?)
                """, (document_id, field_name, chunk_index, faiss_id))

    def _store_metadata_embeddings(
        self,
        conn: sqlite3.Connection,
        document_id: str,
        field_embeddings: Dict[str, np.ndarray]
    ) -> None:
        """
        Store metadata field embeddings in FAISS and track in column_embeddings table
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection
        document_id : str
            Document ID
        field_embeddings : Dict[str, np.ndarray]
            Field name to embeddings mapping
        """
        for field_name, embeddings in field_embeddings.items():
            if embeddings.size == 0:
                continue

            # Add embeddings to FAISS
            start_id = self.index.ntotal
            if hasattr(self.index, 'add_with_ids'):
                # Use add_with_ids for IndexIDMap
                ids = np.arange(start_id, start_id + len(embeddings), dtype=np.int64)
                self.index.add_with_ids(embeddings, ids)
            else:
                self.index.add(embeddings)

            # Track in column_embeddings table
            for chunk_index, faiss_id in enumerate(range(start_id, self.index.ntotal)):
                self._track_column_embedding(conn, document_id, field_name, chunk_index, faiss_id)

    def _remove_metadata_embeddings(
        self,
        conn: sqlite3.Connection,
        document_id: str
    ) -> None:
        """
        Remove metadata field embeddings for a document
        
        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection
        document_id : str
            Document ID to remove embeddings for
        """
        # Get FAISS IDs for metadata embeddings
        cursor = conn.execute("""
            SELECT faiss_id FROM column_embeddings 
            WHERE document_id = ?
        """, (document_id,))

        faiss_ids = [row['faiss_id'] for row in cursor.fetchall()]

        if faiss_ids:
            # Remove from FAISS
            self._remove_old_vectors_bulk(faiss_ids)

            # Remove from tracking table
            conn.execute("""
                DELETE FROM column_embeddings 
                WHERE document_id = ?
            """, (document_id,))

    #############################
    # Async methods and helpers #
    #############################

    def _ensure_async_pool(self):
        """Lazy initialization of async connection pool"""
        if self.async_connection_pool is None:
            self.async_connection_pool = AsyncConnectionPool(
                self.db_path,
                max_connections=self.async_max_connections
            )

    async def _ensure_async_schema_initialized(self):
        """
        Fallback method to ensure schema is initialized in async connections for in-memory databases.
        
        This is a safety measure in case the shared cache approach doesn't work as expected.
        For regular databases, this is a no-op since schema is already initialized.
        """
        if not self.is_memory_only or self._async_schema_initialized:
            return
            
        try:
            # Check if tables exist by attempting to query a core table
            async with self.async_connection_pool.get_connection_context() as conn:
                await conn.execute("SELECT 1 FROM documents LIMIT 1")
            
            # If we get here, tables exist - mark as initialized
            self._async_schema_initialized = True
            
        except Exception:
            # Tables don't exist - initialize schema using async method
            logger.info("Initializing database schema for async operations in in-memory database")
            async with self.async_connection_pool.get_connection_context() as conn:
                await self.schema.initialize_async(self._metadata_schema, db_connection=conn)
            self._async_schema_initialized = True

    async def upsert_async(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2
    ) -> List[str]:
        """
        Async upsert with pipeline processing for maximum throughput

        Parameters
        ----------
        documents : Union[str, List[str]]
            Document text(s) to add
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents
        ids : Optional[Union[str, List[str]]]
            Document IDs (auto-generated if not provided)
        batch_size : int
            Batch size for processing
        similarity_threshold : Optional[float]
            Skip adding chunks that are more similar than this value
        max_concurrent_chunks : int, default=3
            Maximum concurrent chunking operations
        max_concurrent_embeddings : int, default=2
            Maximum concurrent embedding operations

        Other parameters same as upsert()

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        self._ensure_async_pool()
        await self._ensure_async_schema_initialized()

        # Input normalization (reuse sync logic)
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        if metadata is None:
            metadata = [{}] * len(documents)
        elif len(metadata) != len(documents):
            raise ValueError("Number of metadata entries must match number of documents")

        if ids is None:
            ids = [self._generate_doc_id() for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        ids = [(self._generate_doc_id() if i is None else i) for i in ids]

        # Validate metadata (reuse sync logic)
        self._validate_metadata_batch(metadata)

        # Process with async pipeline
        result_ids = await self._async_pipeline_process(
            documents, metadata, ids, batch_size, similarity_threshold,
            max_concurrent_chunks, max_concurrent_embeddings, mode="upsert"
        )

        # Save state (run in executor since it's sync)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_next_doc_id)
        await loop.run_in_executor(None, self._save_internal)

        return result_ids

    async def upsert_from_file_async(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
            extractor_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Async insert or update documents from files using file extraction.

        Uses the ExtractorRegistry to automatically extract text from files based on
        file extension and MIME type, then calls the regular upsert_async method.

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and upsert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing
        similarity_threshold : Optional[float]
            Skip adding chunks that are more similar than this value
        max_concurrent_chunks : int, default=3
            Maximum concurrent chunking operations
        max_concurrent_embeddings : int, default=2
            Maximum concurrent embedding operations
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were upserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file and no fallback is available
        """
        # File I/O operations run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Extract text from files in executor
        async def extract_file_text(file_path: Path, index: int):
            def _extract():
                # Check file exists
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Read file content
                file_content = file_path.read_bytes()
                filename = file_path.name

                # Extract text using ExtractorRegistry
                extraction_result = ExtractorRegistry.extract_text(
                    file_content, filename, **(extractor_kwargs or {})
                )

                if not extraction_result.success:
                    raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")

                return extraction_result

            return await loop.run_in_executor(None, _extract)

        # Extract all files concurrently
        extraction_tasks = [extract_file_text(file_path, i) for i, file_path in enumerate(file_paths)]
        extraction_results = await asyncio.gather(*extraction_tasks)

        # Process extraction results
        documents = []
        merged_metadata = []
        final_ids = []

        for i, (file_path, extraction_result) in enumerate(zip(file_paths, extraction_results)):
            documents.append(extraction_result.text)

            # Merge metadata
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)

            # Generate ID if not provided
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                # Use filename without extension as ID
                doc_id = file_path.stem
            final_ids.append(doc_id)

        # Call regular upsert_async method
        return await self.upsert_async(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            max_concurrent_chunks=max_concurrent_chunks,
            max_concurrent_embeddings=max_concurrent_embeddings
        )

    async def upsert_from_chunks_async(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2
    ) -> List[str]:
        """
        Async version of upsert_from_chunks - Insert or update documents from pre-chunked data.
        
        This method allows you to directly provide chunks for documents, bypassing the 
        chunking step and enabling more efficient processing of pre-processed documents.
        
        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks. Chunks can be either:
            - List[Chunk]: Full Chunk objects with position information
            - List[str]: Simple strings that will be converted to Chunk objects
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata. If None, empty metadata 
            is used for all documents.
        batch_size : int, default=None
            Number of embeddings to generate at once. If None, uses default from configuration.
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks
        max_concurrent_chunks : int, default=3
            Maximum number of concurrent chunk processing operations
        max_concurrent_embeddings : int, default=2
            Maximum number of concurrent embedding operations
            
        Returns
        -------
        List[str]
            List of document IDs that were processed
            
        Raises
        ------
        ValueError
            If chunk data is invalid or metadata doesn't match schema
        """
        self._ensure_async_pool()
        
        # Validate input
        if not chunks_by_document:
            return []
            
        # Normalize metadata
        if metadata is None:
            metadata = {}
        
        # Ensure all documents have metadata (even if empty)
        metadata_batch = {}
        for doc_id in chunks_by_document.keys():
            metadata_batch[doc_id] = metadata.get(doc_id, {})
        
        # Validate metadata against schema
        self._validate_metadata_batch(list(metadata_batch.values()))
        
        # Normalize chunks for all documents
        normalized_chunks_by_document = {}
        for doc_id, chunks in chunks_by_document.items():
            normalized_chunks = self._normalize_chunks(chunks, doc_id)
            if normalized_chunks:  # Only include documents with valid chunks
                normalized_chunks_by_document[doc_id] = normalized_chunks
            
        if not normalized_chunks_by_document:
            return []
        
        # Process with async chunk-based pipeline
        result_ids = await self._async_process_from_chunks_pipeline(
            normalized_chunks_by_document,
            metadata_batch,
            batch_size,
            similarity_threshold,
            max_concurrent_chunks,
            max_concurrent_embeddings,
            mode="upsert"
        )
        
        # Save state
        self._save_next_doc_id()
        self._save_internal()
        
        return result_ids

    async def insert_async(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2
    ) -> List[str]:
        """
        Insert new documents into the database with async pipeline

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
            Skip chunks that are too similar to existing chunks
        max_concurrent_chunks : int, default=3
            Maximum concurrent chunking operations
        max_concurrent_embeddings : int, default=2
            Maximum concurrent embedding operations

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
        """
        self._ensure_async_pool()

        # Input normalization and validation (reuse sync logic)
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        if metadata is None:
            metadata = [{}] * len(documents)
        elif len(metadata) != len(documents):
            raise ValueError("Number of metadata entries must match number of documents")

        if ids is None:
            ids = [self._generate_doc_id() for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # Validate metadata against schema
        self._validate_metadata_batch(metadata)

        # Check for existing document IDs
        existing_ids = await self._check_existing_ids_async(ids)

        # Handle ID conflicts
        docs_to_insert = []
        for doc, meta, doc_id in zip(documents, metadata, ids, strict=False):
            if doc_id in existing_ids:
                if errors == "raise":
                    raise DuplicateDocumentIDError(f"Document with ID '{doc_id}' already exists")
                elif errors == "ignore":
                    logger.info(f"Skipping existing document ID: {doc_id}")
                    continue
            docs_to_insert.append((doc, meta, doc_id))

        if not docs_to_insert:
            return []  # No documents to insert

        # Extract separate lists for pipeline processing
        docs_to_process = [item[0] for item in docs_to_insert]
        meta_to_process = [item[1] for item in docs_to_insert]
        ids_to_process = [item[2] for item in docs_to_insert]

        # Process with async pipeline
        result_ids = await self._async_pipeline_process(
            docs_to_process, meta_to_process, ids_to_process, batch_size, similarity_threshold,
            max_concurrent_chunks, max_concurrent_embeddings, mode="insert"
        )

        # Save state (run in executor since it's sync)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_next_doc_id)
        await loop.run_in_executor(None, self._save_internal)

        return result_ids

    async def insert_from_file_async(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
            extractor_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Async insert new documents from files using file extraction.

        Uses the ExtractorRegistry to automatically extract text from files based on
        file extension and MIME type, then calls the regular insert_async method.

        Parameters
        ----------
        file_paths : Union[str, Path, List[Union[str, Path]]]
            Path(s) to files to extract and insert
        metadata : Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
            Metadata for documents. Will be merged with extracted metadata.
        ids : Optional[Union[str, List[str]]]
            Document IDs. If not provided, will use filename without extension.
        batch_size : int
            Batch size for processing
        similarity_threshold : Optional[float]
            Skip chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts, by default "raise"
        max_concurrent_chunks : int, default=3
            Maximum concurrent chunking operations
        max_concurrent_embeddings : int, default=2
            Maximum concurrent embedding operations
        extractor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments passed to the extractor

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted

        Raises
        ------
        FileNotFoundError
            If any of the specified files don't exist
        ValueError
            If extraction fails for any file and no fallback is available
        DuplicateDocumentIDError
            If errors="raise" and document ID conflicts occur
        """
        # File I/O operations run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        # Normalize file paths to list
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]

        # Normalize other inputs
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]

        # Validate inputs
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        # Extract text from files in executor
        async def extract_file_text(file_path: Path, index: int):
            def _extract():
                # Check file exists
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                # Read file content
                file_content = file_path.read_bytes()
                filename = file_path.name

                # Extract text using ExtractorRegistry
                extraction_result = ExtractorRegistry.extract_text(
                    file_content, filename, **(extractor_kwargs or {})
                )

                if not extraction_result.success:
                    raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")

                return extraction_result

            return await loop.run_in_executor(None, _extract)

        # Extract all files concurrently
        extraction_tasks = [extract_file_text(file_path, i) for i, file_path in enumerate(file_paths)]
        extraction_results = await asyncio.gather(*extraction_tasks)

        # Process extraction results
        documents = []
        merged_metadata = []
        final_ids = []

        for i, (file_path, extraction_result) in enumerate(zip(file_paths, extraction_results)):
            documents.append(extraction_result.text)

            # Merge metadata
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)

            # Generate ID if not provided
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                # Use filename without extension as ID
                doc_id = file_path.stem
            final_ids.append(doc_id)

        # Call regular insert_async method
        return await self.insert_async(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            errors=errors,
            max_concurrent_chunks=max_concurrent_chunks,
            max_concurrent_embeddings=max_concurrent_embeddings
        )

    async def insert_from_chunks_async(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2
    ) -> List[str]:
        """
        Async version of insert_from_chunks - Insert documents from pre-chunked data with conflict handling.
        
        Similar to upsert_from_chunks_async but fails on duplicate document IDs unless
        configured to ignore them.
        
        Parameters
        ----------
        chunks_by_document : Dict[str, Union[List[Chunk], List[str]]]
            Dictionary mapping document IDs to their chunks. Chunks can be either:
            - List[Chunk]: Full Chunk objects with position information
            - List[str]: Simple strings that will be converted to Chunk objects
        metadata : Optional[Dict[str, Dict[str, Any]]], default=None
            Dictionary mapping document IDs to their metadata. If None, empty metadata 
            is used for all documents.
        batch_size : int, default=None
            Number of embeddings to generate at once. If None, uses default from configuration.
        similarity_threshold : Optional[float], default=None
            If provided, filters out chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"], default="raise"
            How to handle document ID conflicts:
            - "raise": Raise DuplicateDocumentIDError
            - "ignore": Skip existing documents and continue
        max_concurrent_chunks : int, default=3
            Maximum number of concurrent chunk processing operations
        max_concurrent_embeddings : int, default=2
            Maximum number of concurrent embedding operations
            
        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
            
        Raises
        ------
        DuplicateDocumentIDError
            If a document ID already exists and errors="raise"
        ValueError
            If chunk data is invalid or metadata doesn't match schema
        """
        self._ensure_async_pool()
        
        # Validate input
        if not chunks_by_document:
            return []

        # Normalize metadata
        if metadata is None:
            metadata = {}

        # Check for existing document IDs
        doc_ids = list(chunks_by_document.keys())
        existing_ids = await self._check_existing_ids_async(doc_ids)

        # Handle ID conflicts
        chunks_to_insert = {}
        metadata_to_insert = {}

        for doc_id, chunks in chunks_by_document.items():
            if doc_id in existing_ids:
                if errors == "raise":
                    raise DuplicateDocumentIDError(f"Document with ID '{doc_id}' already exists")
                elif errors == "ignore":
                    logger.info(f"Skipping existing document ID: {doc_id}")
                    continue

            chunks_to_insert[doc_id] = chunks
            metadata_to_insert[doc_id] = metadata.get(doc_id, {})

        if not chunks_to_insert:
            return []  # No documents to insert

        # Validate metadata against schema
        self._validate_metadata_batch(list(metadata_to_insert.values()))

        # Normalize chunks for all documents
        normalized_chunks_by_document = {}
        for doc_id, chunks in chunks_to_insert.items():
            normalized_chunks = self._normalize_chunks(chunks, doc_id)
            if normalized_chunks:  # Only include documents with valid chunks
                normalized_chunks_by_document[doc_id] = normalized_chunks

        if not normalized_chunks_by_document:
            return []

        # Process with async chunk-based pipeline
        result_ids = await self._async_process_from_chunks_pipeline(
            normalized_chunks_by_document,
            metadata_to_insert,
            batch_size,
            similarity_threshold,
            max_concurrent_chunks,
            max_concurrent_embeddings,
            mode="insert"
        )

        # Save state
        self._save_next_doc_id()
        self._save_internal()

        return result_ids

    async def _check_existing_ids_async(self, ids: List[str]) -> set:
        """Async helper method to check for existing document IDs"""
        if not ids:
            return set()

        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ','.join(['?'] * len(ids))
            cursor = await conn.execute(
                f'SELECT id FROM documents WHERE id IN ({placeholders})', ids
            )
            rows = await cursor.fetchall()
            return {row['id'] for row in rows}

    async def _async_pipeline_process(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            batch_size: int,
            similarity_threshold: Optional[float],
            max_concurrent_chunks: int,
            max_concurrent_embeddings: int,
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> List[str]:
        """
        Unified async pipeline implementation with chunk hash optimization

        3-stage pipeline with backpressure control:
        1. Async chunking + existing chunk comparison
        2. Async embedding generation (only for changed chunks)
        3. Async database operations
        """
        # PRE-PROCESSING: Fetch existing chunks for all documents
        existing_chunks_by_doc = await self._fetch_existing_chunks_batch_async(ids)

        # Async queues for coordination
        chunk_queue = asyncio.Queue(maxsize=5)
        embedding_queue = asyncio.Queue(maxsize=3)

        # Semaphores for concurrency control
        chunk_semaphore = asyncio.Semaphore(max_concurrent_chunks)
        embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)

        result_ids = []

        async def chunking_producer():
            """Stage 1: Async document chunking + existing chunk comparison"""
            try:
                chunk_tasks = []
                for i, (doc_text, metadata, doc_id) in enumerate(zip(documents, metadata_batch, ids, strict=False)):
                    task = asyncio.create_task(
                        self._chunk_document_with_comparison_async(
                            i, doc_id, doc_text, metadata, existing_chunks_by_doc.get(doc_id, {}), chunk_semaphore
                        )
                    )
                    chunk_tasks.append(task)

                # Wait for all chunking to complete and add to queue
                for task in asyncio.as_completed(chunk_tasks):
                    chunk_data = await task
                    await chunk_queue.put(chunk_data)

                # Signal completion
                await chunk_queue.put(None)

            except Exception as e:
                logger.error(f"Async chunking error: {e}")
                await chunk_queue.put(None)
                raise

        async def embedding_processor():
            """Stage 2: Async embedding generation (only for chunks that need it)"""
            try:
                while True:
                    chunk_data = await chunk_queue.get()
                    if chunk_data is None:  # Completion signal
                        await embedding_queue.put(None)
                        break

                    # Generate embeddings only for chunks that need them
                    async with embedding_semaphore:
                        chunk_texts_for_embedding = chunk_data['chunk_texts_for_embedding']
                        if chunk_texts_for_embedding:
                            new_embeddings = await self.embedding_provider.embed_batch(chunk_texts_for_embedding, batch_size)
                            chunk_data['new_embeddings'] = new_embeddings
                        else:
                            chunk_data['new_embeddings'] = np.array([]).reshape(0, self.embedding_dimension)

                        # Generate metadata embeddings if needed
                        embedding_enabled_fields = self._get_embedding_enabled_fields()
                        if embedding_enabled_fields:
                            metadata = chunk_data['metadata']
                            field_embeddings = await self._generate_metadata_embeddings_async(
                                metadata, embedding_enabled_fields, batch_size
                            )
                            chunk_data['field_embeddings'] = field_embeddings
                        else:
                            chunk_data['field_embeddings'] = {}

                    await embedding_queue.put(chunk_data)

            except Exception as e:
                logger.error(f"Async embedding error: {e}")
                await embedding_queue.put(None)
                raise

        async def database_processor():
            """Stage 3: Async database operations"""
            try:
                while True:
                    chunk_data = await embedding_queue.get()
                    if chunk_data is None:  # Completion signal
                        break

                    # Process document data with async database operations
                    doc_id = await self._process_document_data_async(
                        chunk_data,
                        similarity_threshold,
                        mode
                    )

                    if doc_id:
                        result_ids.append(doc_id)

            except Exception as e:
                logger.error(f"Async database error: {e}")
                raise

        # Run all stages concurrently
        await asyncio.gather(
            chunking_producer(),
            embedding_processor(),
            database_processor()
        )

        return result_ids

    async def _async_process_from_chunks_pipeline(
            self,
            chunks_by_document: Dict[str, List[Chunk]],
            metadata_batch: Dict[str, Dict[str, Any]],
            batch_size: int,
            similarity_threshold: Optional[float],
            max_concurrent_chunks: int,
            max_concurrent_embeddings: int,
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> List[str]:
        """
        Async pipeline implementation for pre-chunked documents with chunk hash optimization.
        
        Similar to _async_pipeline_process but skips the chunking stage since chunks are provided.
        
        2-stage async pipeline with backpressure control:
        1. Async chunk comparison + embedding generation (only for changed chunks)
        2. Async database operations
        
        Parameters
        ----------
        chunks_by_document : Dict[str, List[Chunk]]
            Pre-chunked documents indexed by document ID
        metadata_batch : Dict[str, Dict[str, Any]]
            Metadata indexed by document ID
        batch_size : int
            Batch size for embedding generation
        similarity_threshold : Optional[float]
            Similarity threshold for filtering duplicate chunks
        max_concurrent_chunks : int
            Maximum concurrent chunk comparison operations
        max_concurrent_embeddings : int
            Maximum concurrent embedding operations
        mode : Literal["upsert", "insert"], default="upsert"
            Operation mode
            
        Returns
        -------
        List[str]
            List of processed document IDs
        """
        # PRE-PROCESSING: Fetch existing chunks for all documents
        doc_ids = list(chunks_by_document.keys())
        existing_chunks_by_doc = await self._fetch_existing_chunks_batch_async(doc_ids)

        # Async queues for coordination
        embedding_queue = asyncio.Queue(maxsize=3)

        # Semaphores for concurrency control
        embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)

        result_ids = []

        async def chunk_comparison_producer():
            """Stage 1: Async chunk comparison and embedding preparation"""
            try:
                comparison_tasks = []
                for doc_id, chunks in chunks_by_document.items():
                    metadata = metadata_batch.get(doc_id, {})
                    task = asyncio.create_task(
                        self._compare_chunks_and_prepare_async(
                            doc_id, chunks, metadata, existing_chunks_by_doc.get(doc_id, {}), 
                            batch_size, embedding_semaphore
                        )
                    )
                    comparison_tasks.append(task)

                # Wait for all chunk comparison/embedding to complete and add to queue
                for task in asyncio.as_completed(comparison_tasks):
                    chunk_data = await task
                    if chunk_data:  # Only add valid results
                        await embedding_queue.put(chunk_data)

                # Signal completion
                await embedding_queue.put(None)

            except Exception as e:
                logger.error(f"Chunk comparison producer error: {e}")
                await embedding_queue.put(None)
                raise

        async def database_consumer():
            """Stage 2: Async database operations"""
            try:
                while True:
                    chunk_data = await embedding_queue.get()
                    if chunk_data is None:  # Completion signal
                        break

                    doc_id = await self._process_document_data_async(
                        chunk_data, similarity_threshold, mode
                    )
                    
                    if doc_id:
                        result_ids.append(doc_id)

            except Exception as e:
                logger.error(f"Database consumer error: {e}")
                raise

        # Run both stages concurrently
        await asyncio.gather(
            chunk_comparison_producer(),
            database_consumer()
        )

        return result_ids

    async def _compare_chunks_and_prepare_async(
            self,
            doc_id: str,
            chunks: List[Chunk],
            metadata: Dict[str, Any],
            existing_chunks: Dict[int, Dict[str, Any]],
            batch_size: int,
            embedding_semaphore: asyncio.Semaphore
    ) -> Optional[Dict[str, Any]]:
        """
        Async helper to compare chunks and prepare embedding data.
        
        Parameters
        ----------
        doc_id : str
            Document ID
        chunks : List[Chunk]
            List of chunks for this document
        metadata : Dict[str, Any]
            Document metadata
        existing_chunks : Dict[int, Dict[str, Any]]
            Existing chunks for comparison
        batch_size : int
            Embedding batch size
        embedding_semaphore : asyncio.Semaphore
            Semaphore for embedding concurrency control
            
        Returns
        -------
        Optional[Dict[str, Any]]
            Prepared chunk data or None if processing failed
        """
        try:
            # Categorize chunks: unchanged vs needs_embedding
            unchanged_chunks = []
            chunks_needing_embedding = []
            chunk_texts_for_embedding = []
            
            # Track which existing chunks are being reused
            reused_chunk_indices = set()
            
            for chunk in chunks:
                existing_chunk = existing_chunks.get(chunk.index)
                
                if (existing_chunk and
                        existing_chunk['content_hash'] == chunk.content_hash and
                        existing_chunk['faiss_id'] is not None):
                    
                    # Chunk unchanged - reuse existing FAISS ID
                    chunk.faiss_id = existing_chunk['faiss_id']
                    unchanged_chunks.append(chunk)
                    reused_chunk_indices.add(chunk.index)
                    logger.debug(f"Reusing chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")
                    
                else:
                    # Chunk changed/new - needs embedding
                    chunks_needing_embedding.append(chunk)
                    chunk_texts_for_embedding.append(chunk.content)
                    logger.debug(f"Re-embedding chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")
            
            # Calculate what needs to be removed
            chunk_indices_to_remove = []
            faiss_ids_to_remove = []
            
            for chunk_index, chunk_info in existing_chunks.items():
                if chunk_index not in reused_chunk_indices:
                    chunk_indices_to_remove.append(chunk_index)
                    if chunk_info['faiss_id'] is not None:
                        faiss_ids_to_remove.append(chunk_info['faiss_id'])
            
            # Generate embeddings for changed chunks
            new_embeddings = None
            if chunk_texts_for_embedding:
                async with embedding_semaphore:
                    logger.debug(f"Generating embeddings for {len(chunk_texts_for_embedding)} chunks in {doc_id}")
                    new_embeddings = await self.embedding_provider.embed_batch(chunk_texts_for_embedding, batch_size)
                    
                    # Assign FAISS IDs to new chunks (will be updated with actual IDs in database worker)
                    for chunk in chunks_needing_embedding:
                        chunk.faiss_id = None  # Will be set when added to FAISS
            else:
                new_embeddings = np.array([]).reshape(0, self.embedding_dimension)
                logger.debug(f"No new embeddings needed for {doc_id}")
            
            # Generate metadata embeddings if needed
            embedding_enabled_fields = self._get_embedding_enabled_fields()
            field_embeddings = {}
            if embedding_enabled_fields:
                field_embeddings = await self._generate_metadata_embeddings_async(
                    metadata, embedding_enabled_fields, batch_size
                )
            
            # Reconstruct document text from chunks for metadata purposes
            doc_text = "\n".join([chunk.content for chunk in chunks])
            content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
            
            return {
                'doc_id': doc_id,
                'doc_text': doc_text,
                'content_hash': content_hash,
                'metadata': metadata,
                'unchanged_chunks': unchanged_chunks,
                'chunks_needing_embedding': chunks_needing_embedding,
                'new_embeddings': new_embeddings,
                'chunk_indices_to_remove': chunk_indices_to_remove,
                'faiss_ids_to_remove': faiss_ids_to_remove,
                'field_embeddings': field_embeddings
            }
            
        except Exception as e:
            logger.error(f"Error comparing chunks for document {doc_id}: {e}")
            return None

    async def _fetch_existing_chunks_batch_async(self, doc_ids: List[str]) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Async version of fetching existing chunks for comparison

        Parameters
        ----------
        doc_ids : List[str]
            Document IDs to fetch chunks for

        Returns
        -------
        Dict[str, Dict[int, Dict[str, Any]]]
            Nested dict: {doc_id: {chunk_index: {'content_hash': str, 'faiss_id': int}}}
        """
        if not doc_ids:
            return {}

        existing_chunks_by_doc = {}

        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ','.join(['?'] * len(doc_ids))
            cursor = await conn.execute(f'''
                SELECT document_id, chunk_index, content_hash, faiss_id 
                FROM chunks 
                WHERE document_id IN ({placeholders})
            ''', doc_ids)

            async for row in cursor:
                doc_id = row['document_id']
                if doc_id not in existing_chunks_by_doc:
                    existing_chunks_by_doc[doc_id] = {}
                existing_chunks_by_doc[doc_id][row['chunk_index']] = {
                    'content_hash': row['content_hash'],
                    'faiss_id': row['faiss_id']
                }

        logger.debug(f"Fetched existing chunks for {len(existing_chunks_by_doc)} documents")
        return existing_chunks_by_doc

    async def _chunk_document_with_comparison_async(
            self,
            doc_index: int,
            doc_id: str,
            doc_text: str,
            metadata: Dict[str, Any],
            existing_chunks: Dict[int, Dict[str, Any]],
            semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """
        Async document chunking with existing chunk comparison for optimization

        Parameters
        ----------
        doc_index : int
            Document index in batch
        doc_id : str
            Document ID
        doc_text : str
            Document content
        metadata : Dict[str, Any]
            Document metadata
        existing_chunks : Dict[int, Dict[str, Any]]
            Existing chunks for this document
        semaphore : asyncio.Semaphore
            Semaphore for concurrency control

        Returns
        -------
        Dict[str, Any]
            Chunk data with unchanged vs needing embedding categorization
        """
        async with semaphore:
            # Run chunking in executor (it's CPU-bound)
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(None, self.chunker.chunk, doc_text)

            content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()

            # Categorize chunks: unchanged vs needs_embedding
            unchanged_chunks = []
            chunks_needing_embedding = []
            chunk_texts_for_embedding = []

            # Track which existing chunks are being reused
            reused_chunk_indices = set()

            for chunk in chunks:
                existing_chunk = existing_chunks.get(chunk.index)

                if (existing_chunk and
                        existing_chunk['content_hash'] == chunk.content_hash and
                        existing_chunk['faiss_id'] is not None):

                    # Chunk unchanged - reuse existing FAISS ID
                    chunk.faiss_id = existing_chunk['faiss_id']
                    unchanged_chunks.append(chunk)
                    reused_chunk_indices.add(chunk.index)
                    logger.debug(f"Reusing chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")

                else:
                    # Chunk changed/new - needs embedding
                    chunks_needing_embedding.append(chunk)
                    chunk_texts_for_embedding.append(chunk.content)
                    logger.debug(f"Re-embedding chunk {doc_id}:{chunk.index} (hash: {chunk.content_hash[:8]}...)")

            # Calculate what needs to be removed (existing chunks not being reused)
            chunk_indices_to_remove = []
            faiss_ids_to_remove = []

            for chunk_index, chunk_info in existing_chunks.items():
                if chunk_index not in reused_chunk_indices:
                    chunk_indices_to_remove.append(chunk_index)
                    if chunk_info['faiss_id'] is not None:
                        faiss_ids_to_remove.append(chunk_info['faiss_id'])

            return {
                'doc_index': doc_index,
                'doc_id': doc_id,
                'doc_text': doc_text,
                'content_hash': content_hash,
                'metadata': metadata,
                'unchanged_chunks': unchanged_chunks,
                'chunks_needing_embedding': chunks_needing_embedding,
                'chunk_texts_for_embedding': chunk_texts_for_embedding,
                'chunk_indices_to_remove': chunk_indices_to_remove,
                'faiss_ids_to_remove': faiss_ids_to_remove
            }


    async def _process_document_data_async(
            self,
            chunk_data: Dict[str, Any],
            similarity_threshold: Optional[float],
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> Optional[str]:
        """Process document data asynchronously with optimized chunk handling"""
        doc_id = chunk_data['doc_id']

        try:
            unchanged_chunks = chunk_data['unchanged_chunks']
            chunks_needing_embedding = chunk_data['chunks_needing_embedding']
            new_embeddings = chunk_data['new_embeddings']

            # Apply similarity filtering to new embeddings if needed
            if similarity_threshold is not None and len(chunks_needing_embedding) > 0:
                # Run similarity filtering in executor (uses FAISS)
                loop = asyncio.get_event_loop()
                doc_info = (chunk_data['doc_text'], chunk_data['metadata'],
                            chunk_data['doc_id'], chunk_data['content_hash'])
                doc_chunk_mapping = [doc_info] * len(chunks_needing_embedding)

                filtered_chunks, filtered_embeddings, _ = await loop.run_in_executor(
                    None,
                    self._filter_similar_chunks_vectorized,
                    new_embeddings, chunks_needing_embedding, doc_chunk_mapping, similarity_threshold
                )
                chunks_needing_embedding = filtered_chunks
                new_embeddings = filtered_embeddings

            # Combine all chunks for final processing
            all_chunks = unchanged_chunks + chunks_needing_embedding

            # Async database operations
            if len(all_chunks) > 0 or mode == "upsert":  # Always process upserts to update metadata
                documents_data = [(chunk_data['doc_id'], chunk_data['doc_text'],
                                   chunk_data['content_hash'], chunk_data['metadata'])]
                chunks_data = [(chunk_data['doc_id'], chunk) for chunk in all_chunks]

                if mode == "upsert":
                    # Remove old data that's not being reused
                    await self._remove_old_chunks_batch_async(
                        chunk_data['doc_id'],
                        chunk_data['chunk_indices_to_remove'],
                        chunk_data['faiss_ids_to_remove']
                    )

                # Insert new data using async operations
                async with self.async_connection_pool.get_connection_context() as conn:
                    try:
                        await conn.execute('BEGIN')

                        await self._insert_documents_bulk_async(conn, documents_data, mode="replace")

                        # Add only new embeddings to FAISS (unchanged chunks already have FAISS IDs)
                        if new_embeddings.size > 0:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None,
                                self._add_vectors_to_faiss_bulk,
                                new_embeddings, chunks_needing_embedding
                            )

                        await self._insert_chunks_bulk_async(conn, chunks_data)
                        await conn.commit()

                        logger.debug(
                            f"Successfully processed document {doc_id}: "
                            f"{len(unchanged_chunks)} unchanged, {len(chunks_needing_embedding)} new/changed chunks"
                        )

                    except Exception as e:
                        logger.error(f"Error in transaction for document {doc_id}: {e}")
                        await conn.rollback()
                        raise

            return doc_id

        except Exception as e:
            logger.error(f"Error processing document data for {doc_id}: {e}")
            return None

    async def _remove_old_chunks_batch_async(
            self,
            doc_id: str,
            chunk_indices_to_remove: List[int],
            faiss_ids_to_remove: List[int]
    ) -> None:
        """
        Async version of selective chunk removal

        Remove only the chunks and FAISS vectors that are being replaced.
        For upsert operations, we:
        1. Keep the document record (will be updated by INSERT OR REPLACE)
        2. Only remove chunks that are being replaced
        3. Preserve unchanged chunks and their FAISS vectors

        Parameters
        ----------
        doc_id : str
            Document ID being processed
        chunk_indices_to_remove : List[int]
            Chunk indices that need to be removed
        faiss_ids_to_remove : List[int]
            FAISS IDs that need to be removed
        """
        # Remove FAISS vectors first (still needs executor)
        if faiss_ids_to_remove:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids_to_remove)

        # Remove specific chunks from database (but keep document record)
        if chunk_indices_to_remove:
            async with self.async_connection_pool.get_connection_context() as conn:
                # Delete only the chunks that are being replaced
                placeholders = ','.join(['?'] * len(chunk_indices_to_remove))
                await conn.execute(
                    f'DELETE FROM chunks WHERE document_id = ? AND chunk_index IN ({placeholders})',
                    [doc_id] + chunk_indices_to_remove
                )

        logger.debug(
            f"Removed {len(chunk_indices_to_remove)} old chunks and "
            f"{len(faiss_ids_to_remove)} FAISS vectors for document {doc_id}"
        )

    async def get_async_stats(self) -> Dict[str, Any]:
        """Get async-specific statistics"""
        stats = {}

        if self.async_connection_pool:
            stats['async_pool'] = self.async_connection_pool.stats
        else:
            stats['async_pool'] = {'status': 'not_initialized'}

        return stats

    async def _insert_documents_bulk_async(
            self,
            conn: aiosqlite.Connection,
            documents_data: List[Tuple[str, str, str, Dict[str, Any]]],
            mode: Literal["insert", "replace"] = "replace"
    ) -> None:
        """
        Async version of _insert_documents_bulk using aiosqlite

        Parameters
        ----------
        conn : aiosqlite.Connection
            Async database connection
        documents_data : List[Tuple[str, str, str, Dict[str, Any]]]
            List of (doc_id, content, content_hash, metadata) tuples
        mode : Literal["insert", "replace"]
            Insert mode - "replace" for upserts, "insert" for inserts
        """
        if not documents_data:
            return

        # Build dynamic INSERT statement based on metadata schema
        base_columns = self.schema.BASE_COLUMNS.copy()
        metadata_columns = list(self.metadata_schema.keys())
        all_columns = base_columns + metadata_columns

        placeholders = ['?'] * len(all_columns)

        sql_verb = "INSERT OR REPLACE" if mode == "replace" else "INSERT"
        sql = f"{sql_verb} INTO documents ({', '.join(all_columns)}) VALUES ({', '.join(placeholders)})"

        # Prepare bulk data
        bulk_data = []
        current_time = datetime.now(UTC)

        for doc_id, content, content_hash, metadata in documents_data:
            row_data = [doc_id, content, content_hash, current_time, current_time]

            # Add metadata values in schema order
            for field_name in metadata_columns:
                value = metadata.get(field_name)
                row_data.append(value)

            bulk_data.append(tuple(row_data))

        # Execute bulk insert asynchronously
        await conn.executemany(sql, bulk_data)

    @staticmethod
    async def _insert_chunks_bulk_async(
            conn: aiosqlite.Connection,
            chunks_data: List[Tuple[str, Any]]  # (doc_id, chunk)
    ) -> None:
        """
        Async version of _insert_chunks_bulk using aiosqlite

        Parameters
        ----------
        conn : aiosqlite.Connection
            Async database connection
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

        # Execute bulk insert asynchronously
        await conn.executemany('''
            INSERT INTO chunks 
            (document_id, chunk_index, content, content_hash, start_pos, end_pos, start_line, 
            start_col, end_line, end_col, tokens, faiss_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', bulk_data)

    async def _remove_old_document_data_async(self, doc_ids: List[str]) -> None:
        """
        Async version of removing old document data

        Parameters
        ----------
        doc_ids : List[str]
            Document IDs to remove
        """
        if not doc_ids:
            return

        async with self.async_connection_pool.get_connection_context() as conn:
            # Get FAISS IDs to remove before deleting chunks
            placeholders = ','.join(['?'] * len(doc_ids))
            cursor = await conn.execute(
                f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                doc_ids
            )
            faiss_ids = [row['faiss_id'] for row in await cursor.fetchall()]

            # Remove from FAISS index (still needs executor)
            if faiss_ids:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids)

            # Remove from database
            await conn.execute(
                f'DELETE FROM chunks WHERE document_id IN ({placeholders})', doc_ids
            )
            await conn.execute(
                f'DELETE FROM documents WHERE id IN ({placeholders})', doc_ids
            )

    async def close_async(self):
        """Close async resources"""
        if self.async_connection_pool:
            await self.async_connection_pool.close_all()
            self.async_connection_pool = None

    async def save_async(self):
        """Saves the database"""
        # Requires calling faiss which is sync only, so just call save in the loop.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.save)

    async def filter_async(
            self,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List["Document"]:
        """
        Async filter documents by metadata criteria

        Parameters
        ----------
        where : Dict[str, Any]
            Filter criteria using MongoDB-style syntax
        order_by : Optional[str]
            SQL-style ORDER BY clause (e.g., 'created_at DESC'), by default None
        limit : Optional[int]
            Maximum number of documents to return, by default None
        offset : Optional[int]
            Number of documents to skip, by default 0

        Returns
        -------
        List[Document]
            Documents matching the filter criteria
        """
        self._ensure_async_pool()
        await self._ensure_async_schema_initialized()

        # Build the filter query using the existing FilterQueryBuilder
        filter_builder = FilterQueryBuilder(self.metadata_schema)
        where_clause, params = filter_builder.build_where_clause(where)

        # Build secure ORDER BY clause
        order_clause = ""
        if order_by:
            try:
                # Validate that order_by is a string
                if not isinstance(order_by, str):
                    raise ValueError("order_by must be a string")

                # Build valid columns set (all document columns)
                base_columns = self.schema.BASE_COLUMNS.copy()
                metadata_columns = set(self.metadata_schema.keys())
                valid_columns = set(base_columns).union(metadata_columns)

                # Use FilterQueryBuilder for secure ORDER BY construction
                order_by_clause = filter_builder.build_order_by_clause(order_by, valid_columns)
                order_clause = f" {order_by_clause}"
            except Exception as e:
                raise ValueError(f"Error building ORDER BY clause: {str(e)}")

        # Build LIMIT/OFFSET clause
        limit_clause = ""
        if limit is not None:
            limit_clause = f" LIMIT {limit}"
            if offset > 0:
                limit_clause += f" OFFSET {offset}"

        # Execute query asynchronously
        async with self.async_connection_pool.get_connection_context() as conn:
            # Build column list - same as sync version for consistency
            metadata_columns = list(self.metadata_schema.keys())
            if metadata_columns:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
            else:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
            
            # Build complete SQL query with explicit column selection
            sql_parts = [f"SELECT {', '.join(columns)} FROM documents"]
            if where_clause:
                sql_parts.append(f"WHERE {where_clause}")
            if order_clause:
                sql_parts.append(order_clause.strip())
            if limit_clause:
                sql_parts.append(limit_clause.strip())

            sql = " ".join(sql_parts)
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        # Convert rows to Document objects
        documents = []
        for row in rows:
            # Extract metadata (all columns except base columns)
            base_columns = self.schema.BASE_COLUMNS.copy()
            metadata = {}

            # Handle both aiosqlite.Row (which has .items()) and sqlite3.Row (which doesn't)
            if hasattr(row, 'items'):
                row_items = row.items()
            else:
                # For sqlite3.Row, use keys() and indexing
                row_items = [(key, row[key]) for key in row.keys()]
                
            for key, value in row_items:
                if key not in base_columns and value is not None:
                    # Parse JSON fields
                    if key in self.metadata_schema:
                        field_def = self.metadata_schema[key]
                        if field_def.type.name == 'JSON' and isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as string if parsing fails
                        # TODO: do we have to parse any other values?
                    metadata[key] = value

            document = Document(
                id=row['id'],
                content=row['content'],
                metadata=metadata,
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                content_hash=row['content_hash']
            )
            documents.append(document)

        return documents

    async def get_async(
            self,
            ids: Union[str, List[str]]
    ) -> Union[Document, List[Document], None]:
        """
        Async retrieve documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to retrieve

        Returns
        -------
        Union[Document, List[Document], None]
            Document(s) if found, None if single ID not found, empty list if no IDs found
        """
        self._ensure_async_pool()

        # Normalize input
        if isinstance(ids, str):
            single_id = True
            ids = [ids]
        else:
            single_id = False

        if not ids:
            return [] if not single_id else None

        # Query asynchronously
        async with self.async_connection_pool.get_connection_context() as conn:
            # Build column list - same as sync version for consistency
            metadata_columns = list(self.metadata_schema.keys())
            if metadata_columns:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
            else:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
                
            placeholders = ','.join(['?' for _ in ids])
            sql = f"SELECT {', '.join(columns)} FROM documents WHERE id IN ({placeholders})"
            cursor = await conn.execute(sql, ids)
            rows = await cursor.fetchall()

        # Convert to Document objects (reuse logic from filter_async)
        documents = []
        for row in rows:
            # Extract metadata
            base_columns = self.schema.BASE_COLUMNS.copy()
            metadata = {}

            # Handle both aiosqlite.Row (which has .items()) and sqlite3.Row (which doesn't)
            if hasattr(row, 'items'):
                row_items = row.items()
            else:
                # For sqlite3.Row, use keys() and indexing
                row_items = [(key, row[key]) for key in row.keys()]
                
            for key, value in row_items:
                if key not in base_columns and value is not None:
                    # Parse JSON fields
                    if key in self.metadata_schema:
                        field_def = self.metadata_schema[key]
                        if field_def.type.name == 'JSON' and isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass
                    metadata[key] = value

            document = Document(
                id=row['id'],
                content=row['content'],
                metadata=metadata,
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                content_hash=row['content_hash']
            )
            documents.append(document)

        # TODO: should raise an error if not found

        # Return results based on input type
        if single_id:
            return documents[0] if documents else None
        else:
            return documents

    async def delete_async(
            self,
            ids: Union[str, List[str]]
    ) -> int:
        """
        Async delete documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to delete

        Returns
        -------
        int
            Number of documents deleted
        """
        self._ensure_async_pool()

        # Normalize input
        if isinstance(ids, str):
            ids = [ids]

        if not ids:
            return 0

        deleted_count = 0

        async with self.async_connection_pool.get_connection_context() as conn:
            # Get FAISS IDs to remove before deleting
            placeholders = ','.join(['?' for _ in ids])
            cursor = await conn.execute(
                f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                ids
            )
            faiss_ids = [row['faiss_id'] for row in await cursor.fetchall()]

            # Remove from FAISS index (still needs executor)
            if faiss_ids:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids)

            # Delete from database
            await conn.execute('BEGIN')
            try:
                # Delete chunks first (foreign key constraint)
                cursor = await conn.execute(
                    f'DELETE FROM chunks WHERE document_id IN ({placeholders})', ids
                )

                # Delete documents
                cursor = await conn.execute(
                    f'DELETE FROM documents WHERE id IN ({placeholders})', ids
                )
                deleted_count = cursor.rowcount or 0

                await conn.commit()

            except Exception:
                await conn.rollback()
                raise

        logger.info(f"Deleted {deleted_count} documents")
        return deleted_count

    async def count_async(
            self,
            filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Async count documents matching filter criteria

        Parameters
        ----------
        filters : Optional[Dict[str, Any]]
            Filter criteria using MongoDB-style syntax, by default None

        Returns
        -------
        int
            Number of documents matching the criteria
        """
        self._ensure_async_pool()

        if filters:
            # Build filter query
            filter_builder = FilterQueryBuilder(self.metadata_schema)
            where_clause, params = filter_builder.build_where_clause(filters)
            sql = f"SELECT COUNT(*) as count FROM documents WHERE {where_clause}"
        else:
            sql = "SELECT COUNT(*) as count FROM documents"
            params = []

        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(sql, params)
            row = await cursor.fetchone()
            return row['count'] if row else 0

    async def exists_async(
            self,
            id: str
    ) -> bool:
        """
        Async check if a document exists

        Parameters
        ----------
        id : str
            Document ID to check

        Returns
        -------
        bool
            True if document exists, False otherwise
        """
        self._ensure_async_pool()

        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute("SELECT 1 FROM documents WHERE id = ? LIMIT 1", [id])
            row = await cursor.fetchone()
            return row is not None



    async def query_async(
            self,
            query: str,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'hybrid',
            return_type: Literal['documents', 'chunks', 'context'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,
            context_window: int = 2,
            semantic_dedup_threshold: Optional[float] = None,
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """
        Async query the database using vector, keyword, or hybrid search

        Parameters
        ----------
        query : str
            Search query text
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform, by default 'hybrid'
        return_type : Literal['documents', 'chunks', 'context']
            Whether to return full documents, individual chunks, or chunks with context, by default 'documents'
        k : int
            Maximum number of results to return, by default 10
        score_threshold : float
            Minimum score threshold (0-1, higher=better), by default 0.0
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply, by default None
        vector_weight : float
            Weight for vector search in hybrid mode (0-1), by default 0.7
        context_window : int
            Number of chunks before and after to include when return_type='context', by default 2
        semantic_dedup_threshold : Optional[float]
            Similarity threshold for semantic deduplication (0-1, higher=more similar), by default None
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores, by default "frequency_boost"
        document_scoring_options : dict, optional
            Parameters for the document_scoring_method (to choose overall scores for documents from chunk results)

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        self._ensure_async_pool()
        await self._ensure_async_schema_initialized()

        # For vector and hybrid search, generate embedding asynchronously
        query_embedding = None
        if search_type in ['vector', 'hybrid']:
            query_embedding = await self.embedding_provider.embed_batch([query])
            query_embedding = query_embedding[0]  # Get single embedding from batch

        # Use the new async search method
        return await self._search_with_embedding_async(
            query, query_embedding, search_type, return_type, k, score_threshold,
            filters, vector_weight, context_window, semantic_dedup_threshold, document_scoring_method,
            document_scoring_options
        )

    async def _search_with_embedding_async(
            self,
            query: str,
            query_embedding: Optional[np.ndarray],
            search_type: Literal['vector', 'keyword', 'hybrid'],
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            vector_weight: float,
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """
        Async version of _search_with_embedding with truly async database operations

        Parameters
        ----------
        query : str
            Original query text (used for keyword search in hybrid mode)
        query_embedding : Optional[np.ndarray]
            Precomputed embedding for the query (None for keyword-only search)
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
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores

        Returns
        -------
        List[QueryResult]
            Search results with normalized scores
        """
        # Handle different search types with async implementations
        if search_type == 'vector':
            results = await self._vector_search_with_embedding_async(
                query_embedding, return_type, k, score_threshold, filters,
                context_window, semantic_dedup_threshold, document_scoring_method, document_scoring_options
            )
        elif search_type == 'keyword':
            # Keyword search doesn't use embeddings
            results = await self._keyword_search_async(
                query, return_type, k, score_threshold, filters,
                context_window, semantic_dedup_threshold, document_scoring_method, document_scoring_options
            )
        elif search_type == 'hybrid':
            results = await self._hybrid_search_with_embedding_async(
                query, query_embedding, return_type, k, score_threshold, filters, vector_weight,
                context_window, semantic_dedup_threshold, document_scoring_method, document_scoring_options
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")

        return results

    async def _vector_search_with_embedding_async(
            self,
            query_embedding: np.ndarray,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: Optional[dict] = None
    ) -> List[QueryResult]:
        """Async vector similarity search with a precomputed embedding"""

        # FAISS search (still needs executor as FAISS is not async)
        loop = asyncio.get_event_loop()
        search_k = min(k * 2, 100)  # Get more results for better filtering/processing

        # Perform FAISS search in executor
        distances, indices = await loop.run_in_executor(
            None,
            lambda: self.index.search(query_embedding.reshape(1, -1), search_k)
        )

        # Convert to Python lists for easier handling
        distances = distances[0].tolist()
        indices = indices[0].tolist()

        # Filter out invalid results
        valid_results = [(dist, idx) for dist, idx in zip(distances, indices, strict=False) if idx != -1]

        if not valid_results:
            return []

        # Get chunk data from database asynchronously
        chunk_faiss_ids = [idx for _, idx in valid_results]
        chunks_data = await self._get_chunks_by_faiss_ids_async(chunk_faiss_ids)

        # Apply metadata filters if specified
        if filters:
            chunks_data = await self._apply_metadata_filters_async(chunks_data, filters)

        # Create query results with similarity scores
        query_results = []
        faiss_id_to_distance = {idx: dist for dist, idx in valid_results}

        for chunk_data in chunks_data:
            faiss_id = chunk_data['faiss_id']
            if faiss_id in faiss_id_to_distance:
                # Convert FAISS distance to similarity score (higher is better)
                distance = faiss_id_to_distance[faiss_id]
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

                if similarity >= score_threshold:
                    query_results.append(QueryResult(
                        id=chunk_data['chunk_id'],
                        score=similarity,
                        type='chunk',
                        content=chunk_data['content'],
                        metadata=chunk_data.get('metadata', {}),
                        document_id=chunk_data['document_id'],
                        position=chunk_data.get('position')
                    ))

        # Sort by score (highest first)
        query_results.sort(key=lambda x: x.score, reverse=True)

        # Apply post-processing
        if semantic_dedup_threshold is not None:
            query_results = await self._apply_semantic_deduplication_async(query_results, semantic_dedup_threshold)

        # Convert return type and aggregate scores
        final_results = await self._process_search_results_async(
            query_results, return_type, document_scoring_method, document_scoring_options, context_window
        )

        return final_results[:k]

    async def _keyword_search_async(
            self,
            query: str,
            return_type: Literal['documents', 'chunks', 'context'],
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]],
            context_window: int,
            semantic_dedup_threshold: Optional[float],
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: Optional[dict] = None
    ) -> List[QueryResult]:
        """Async keyword search using SQLite FTS5"""

        if not self.fts_enabled:
            logger.warning("FTS not enabled, returning empty results")
            return []

        # Build FTS query - escape special characters
        fts_query = query.replace("'", "''").replace('"', '""')

        # Build metadata filter clause if needed
        filter_clause = ""
        filter_params = []
        if filters:
            filter_builder = FilterQueryBuilder(self.metadata_schema)
            where_clause, params = filter_builder.build_where_clause(filters)
            filter_clause = f" AND d.rowid IN (SELECT rowid FROM documents WHERE {where_clause})"
            filter_params = params

        # Build SQL to include document metadata columns
        metadata_columns = list(self.metadata_schema.keys())
        metadata_select = ', '.join([f'd.{col}' for col in metadata_columns]) if metadata_columns else ''
        metadata_select_clause = f', {metadata_select}' if metadata_select else ''

        # Execute FTS search asynchronously
        async with self.async_connection_pool.get_connection_context() as conn:
            sql = f"""
                SELECT c.document_id, c.chunk_index, c.content, c.faiss_id,
                       c.start_pos, c.end_pos, c.start_line, c.start_col, c.end_line, c.end_col,
                       fts.rank, d.content as doc_content{metadata_select_clause}
                FROM chunks_fts fts
                JOIN chunks c ON c.rowid = fts.rowid
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ? {filter_clause}
                ORDER BY fts.rank
                LIMIT ?
            """

            params = [fts_query] + filter_params + [k * 2]
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        # Convert to QueryResult objects
        query_results = []
        for row in rows:
            # FTS rank is negative (lower is better), convert to positive similarity score
            fts_rank = row['rank']
            similarity = 1.0 / (1.0 + abs(fts_rank))  # Convert to 0-1 range

            if similarity >= score_threshold:
                # Build position object
                position = ChunkPosition(
                    start=row['start_pos'],
                    end=row['end_pos'],
                    line=row['start_line'],
                    column=row['start_col'],
                    end_line=row['end_line'],
                    end_column=row['end_col']
                )

                # Extract and parse document metadata
                metadata = {}
                for field_name in metadata_columns:
                    value = row[field_name]
                    if value is not None:
                        # Parse JSON fields
                        if field_name in self.metadata_schema:
                            field_def = self.metadata_schema[field_name]
                            if field_def.type.name == 'JSON' and isinstance(value, str):
                                try:
                                    value = json.loads(value)
                                except (json.JSONDecodeError, TypeError):
                                    pass  # Keep as string if parsing fails
                        metadata[field_name] = value

                query_results.append(QueryResult(
                    id=f"{row['document_id']}:{row['chunk_index']}",
                    score=similarity,
                    type='chunk',
                    content=row['content'],
                    metadata=metadata,  # Now properly populated!
                    document_id=row['document_id'],
                    position=position
                ))

        # Apply post-processing
        if semantic_dedup_threshold is not None:
            query_results = await self._apply_semantic_deduplication_async(query_results, semantic_dedup_threshold)

        # Convert return type and aggregate scores
        final_results = await self._process_search_results_async(
            query_results, return_type, document_scoring_method, document_scoring_options, context_window
        )

        return final_results[:k]

    async def _hybrid_search_with_embedding_async(
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
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: Optional[dict] = None
    ) -> List[QueryResult]:
        """Async hybrid search combining vector and keyword with precomputed embedding"""

        if not self.fts_enabled:
            # Fall back to vector search if FTS is not available
            return await self._vector_search_with_embedding_async(
                query_embedding, return_type, k, score_threshold, filters,
                context_window, semantic_dedup_threshold, document_scoring_method
            )

        # Get more results than requested for better reranking
        search_k = min(k * 4, 100)

        # Perform both searches concurrently
        vector_task = asyncio.create_task(
            self._vector_search_with_embedding_async(
                query_embedding, 'chunks', search_k, 0.0, filters, 0, None, "best"
            )
        )

        keyword_task = asyncio.create_task(
            self._keyword_search_async(
                query, 'chunks', search_k, 0.0, filters, 0, None, "best"
            )
        )

        # Wait for both searches to complete
        vector_results, keyword_results = await asyncio.gather(vector_task, keyword_task)

        # Combine results with weighted scoring
        combined_results = await self._combine_search_results_async(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            k=search_k,
            score_threshold=0.0
        )

        # Apply semantic deduplication if specified
        if semantic_dedup_threshold is not None:
            combined_results = await self._apply_semantic_deduplication_async(combined_results,
                                                                              semantic_dedup_threshold)

        # Filter by score threshold now
        if score_threshold > 0.0:
            combined_results = [r for r in combined_results if r.score >= score_threshold]

        # Convert return type and aggregate scores
        final_results = await self._process_search_results_async(
            combined_results, return_type, document_scoring_method, document_scoring_options, context_window
        )

        return final_results[:k]

    async def _get_chunks_by_faiss_ids_async(self, faiss_ids: List[int]) -> List[Dict[str, Any]]:
        """Get chunk data by FAISS IDs asynchronously with document metadata"""
        if not faiss_ids:
            return []

        # Build SQL to include document metadata columns
        base_columns = self.schema.BASE_COLUMNS.copy()
        metadata_columns = list(base_columns) + list(self.metadata_schema.keys())

        # Build SELECT clause with metadata columns prefixed
        chunk_columns = [
            'c.document_id', 'c.chunk_index', 'c.content', 'c.faiss_id',
            'c.start_pos', 'c.end_pos', 'c.start_line', 'c.start_col', 'c.end_line', 'c.end_col',
            'd.content as doc_content'
        ]

        # Add metadata columns with 'd.' prefix
        for col in metadata_columns:
            chunk_columns.append(f'd.{col}')

        placeholders = ','.join(['?' for _ in faiss_ids])
        sql = f"""
            SELECT {', '.join(chunk_columns)}
            FROM chunks c
            JOIN documents d ON d.id = c.document_id  
            WHERE c.faiss_id IN ({placeholders})
        """

        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(sql, faiss_ids)
            rows = await cursor.fetchall()

        # Convert to list of dicts for easier handling
        chunks_data = []
        for row in rows:
            # Build position object
            position = ChunkPosition(
                start=row['start_pos'],
                end=row['end_pos'],
                line=row['start_line'],
                column=row['start_col'],
                end_line=row['end_line'],
                end_column=row['end_col']
            )

            # Extract and parse document metadata
            metadata = {}
            for field_name in metadata_columns:
                value = row[field_name]
                if value is not None:
                    # Parse JSON fields
                    if field_name in self.metadata_schema:
                        field_def = self.metadata_schema[field_name]
                        if field_def.type.name == 'JSON' and isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as string if parsing fails
                    metadata[field_name] = value

            chunks_data.append({
                'chunk_id': f"{row['document_id']}:{row['chunk_index']}",
                'document_id': row['document_id'],
                'content': row['content'],
                'faiss_id': row['faiss_id'],
                'position': position,
                'metadata': metadata  # Now properly populated!
            })

        return chunks_data

    async def _apply_metadata_filters_async(self, chunks_data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[
        Dict[str, Any]]:
        """Apply metadata filters to chunks asynchronously using matches_metadata_filter"""
        if not filters or not chunks_data:
            return chunks_data

        # Get unique document IDs from chunks
        doc_ids = list(set(chunk['document_id'] for chunk in chunks_data))

        if not doc_ids:
            return chunks_data

        # Get metadata for just these specific documents (much more efficient)
        doc_metadata = await self._get_document_metadata_async(doc_ids)

        # Use matches_metadata_filter to check each document
        filtered_doc_ids = set()

        for doc_id, metadata in doc_metadata.items():
            if matches_metadata_filter(metadata, filters):
                filtered_doc_ids.add(doc_id)

        # Filter chunks to only include those from matching documents
        return [chunk for chunk in chunks_data if chunk['document_id'] in filtered_doc_ids]

    async def _get_document_metadata_async(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for specific documents efficiently"""
        if not doc_ids:
            return {}

        # Only select metadata columns (not content)
        base_columns = self.schema.BASE_COLUMNS.copy()
        metadata_columns = list(base_columns) + list(self.metadata_schema.keys())

        if not metadata_columns:
            # No metadata schema defined, return empty metadata for all docs
            return {doc_id: {} for doc_id in doc_ids}

        # Build SQL to select only the metadata columns we need
        sql = f"SELECT {', '.join(metadata_columns)} FROM documents WHERE id IN ({','.join(['?' for _ in doc_ids])})"

        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(sql, doc_ids)
            rows = await cursor.fetchall()

        # Convert to metadata dict
        doc_metadata = {}
        for row in rows:
            doc_id = row['id']
            metadata = {}

            # Extract and parse metadata fields
            for field_name in metadata_columns:
                value = row[field_name]
                if value is not None:
                    # Parse JSON fields
                    if field_name in self.metadata_schema:
                        field_def = self.metadata_schema[field_name]
                        if field_def.type.name == 'JSON' and isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                pass  # Keep as string if parsing fails

                metadata[field_name] = value

            doc_metadata[doc_id] = metadata

        return doc_metadata


    async def _combine_search_results_async(
            self,
            vector_results: List[QueryResult],
            keyword_results: List[QueryResult],
            vector_weight: float,
            k: int,
            score_threshold: float
    ) -> List[QueryResult]:
        """Combine vector and keyword search results asynchronously"""
        # For now, fall back to sync implementation in executor
        # The logic is primarily CPU-bound list operations
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._combine_search_results,
            vector_results, keyword_results, vector_weight, k, score_threshold
        )

    async def _process_search_results_async(
            self,
            results: List[QueryResult],
            return_type: Literal['documents', 'chunks', 'context'],
            document_scoring_method: DocumentScoringMethod,
            document_scoring_options: Optional[dict],
            context_window: int
    ) -> List[QueryResult]:
        """Process search results for return type and document scoring asynchronously"""
        if return_type == 'chunks':
            return results

        if return_type == 'documents':
            return await self._aggregate_document_scores_with_method_async(results, document_scoring_method, document_scoring_options)
        else:  # context
            return await self._add_context_window_async(results, context_window)

    async def _apply_semantic_deduplication_async(
            self,
            results: List[QueryResult],
            threshold: float
    ) -> List[QueryResult]:
        """
        Apply semantic deduplication to search results using FAISS index embeddings asynchronously.

        Optimized async version that minimizes database calls and uses batch FAISS operations.

        Parameters
        ----------
        results : List[QueryResult]
            Initial search results to deduplicate - MUST BE SORTED with highest score first
        threshold : float
            Similarity threshold (0-1, higher=more similar). Chunks above this threshold are considered duplicates.

        Returns
        -------
        List[QueryResult]
            Deduplicated results with highest-scored chunk from each similar group
        """
        if not results or threshold is None or threshold <= 0:
            return results

        # Separate chunk results from other types (only chunks have embeddings)
        chunk_results = [r for r in results if r.type == 'chunk']

        if len(chunk_results) <= 1:
            return results  # No deduplication needed

        # Step 1: Batch retrieve all FAISS IDs in a single async SQL query
        chunk_identifiers = [(r.document_id, self._extract_chunk_index_from_id(r.id)) for r in chunk_results]

        async with self.async_connection_pool.get_connection_context() as conn:
            # Create parameterized query for all chunks at once
            placeholders = ','.join(['(?,?)'] * len(chunk_identifiers))
            query = f'''
                SELECT document_id, chunk_index, faiss_id 
                FROM chunks 
                WHERE (document_id, chunk_index) IN ({placeholders})
            '''

            # Flatten the list of tuples for query parameters
            params = [item for pair in chunk_identifiers for item in pair]
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            faiss_id_mapping = {
                (row['document_id'], row['chunk_index']): row['faiss_id']
                for row in rows
            }

        # Step 2: Extract FAISS IDs and create mapping to results
        faiss_ids = []
        result_mapping = {}  # faiss_id -> QueryResult

        for result in chunk_results:
            doc_id, chunk_idx = self._split_chunk_id(result.id)
            faiss_id = faiss_id_mapping.get((doc_id, chunk_idx))

            if faiss_id is not None:
                faiss_ids.append(faiss_id)
                result_mapping[faiss_id] = result

        if not faiss_ids:
            return results

        # Step 3: Batch retrieve embeddings from FAISS index (still needs executor as FAISS is not async)
        loop = asyncio.get_event_loop()
        embeddings_matrix = await loop.run_in_executor(
            None, self._reconstruct_embeddings_batch, faiss_ids
        )

        # Step 4: Compute pairwise similarities using vectorized operations (in executor)
        def compute_similarities_and_filter():
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            normalized_embeddings = embeddings_matrix / np.maximum(norms, 1e-8)

            # Compute similarity matrix (upper triangular to avoid duplicates)
            similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

            # Step 5: Identify duplicates using pure numpy operations
            # Since results are sorted by score (descending), score[i] >= score[j] when i < j
            # Create boolean mask for similarities above threshold (excluding diagonal)
            similar_pairs = similarity_matrix >= threshold
            np.fill_diagonal(similar_pairs, False)  # Exclude self-similarity

            # Step 6: Vectorized duplicate identification
            # Since we're sorted by score, we only need the upper triangular part
            # A chunk j should be removed if it's similar to any earlier (higher-scoring) chunk i where i < j
            upper_triangular_similarities = np.triu(similar_pairs, k=1)

            # For each chunk j, check if any earlier chunk i is similar (column-wise check)
            should_remove = np.any(upper_triangular_similarities, axis=0)

            # Step 7: Build final results using boolean indexing
            keep_mask = ~should_remove
            return keep_mask

        # Run similarity computation in executor
        keep_mask = await loop.run_in_executor(None, compute_similarities_and_filter)

        final_chunk_results = [result_mapping[faiss_ids[i]] for i in range(len(faiss_ids)) if keep_mask[i]]

        # Log deduplication statistics
        original_count = len(chunk_results)
        final_count = len(final_chunk_results)
        removed_count = original_count - final_count

        logger.debug(f"Async semantic deduplication: {original_count} → {final_count} chunks "
                     f"({removed_count} removed)")

        # Combine deduplicated chunks with non-chunk results
        return final_chunk_results

    async def _aggregate_document_scores_with_method_async(
            self,
            chunk_results: List[QueryResult],
            method: DocumentScoringMethod = "frequency_boost",
            method_options: dict = None
    ) -> List[QueryResult]:
        """
        Aggregate chunk results into document results with enhanced scoring asynchronously.

        Parameters
        ----------
        chunk_results : List[QueryResult]
            Chunk-level search results to aggregate by document
        method : DocumentScoringMethod
            Scoring method to aggregate score at the document-level.
        method_options : dict, optional
            Parameters for the scoring method

        Returns
        -------
        List[QueryResult]
            Document-level results with aggregated scores
        """
        if not chunk_results:
            return []

        method_options = method_options or {}

        # Group chunks by document
        doc_groups = defaultdict(list)
        for result in chunk_results:
            doc_id = result.document_id if result.type == 'chunk' else result.id
            doc_groups[doc_id].append(result)

        # Get all unique document IDs
        all_doc_ids = list(doc_groups.keys())

        if not all_doc_ids:
            return []

        # Batch fetch all document content and metadata in single async queries
        async with self.async_connection_pool.get_connection_context() as conn:
            # Batch fetch document content
            placeholders = ','.join(['?'] * len(all_doc_ids))
            cursor = await conn.execute(f'''
                SELECT id, content 
                FROM documents 
                WHERE id IN ({placeholders})
            ''', all_doc_ids)

            doc_content_map = {}
            async for row in cursor:
                doc_content_map[row['id']] = row['content']

            # Batch fetch all metadata at once
            doc_metadata_batch = await self._get_document_metadata_async(all_doc_ids)


        # Run the computation-heavy scoring in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute_document_scores,
                                          method, method_options, doc_groups, doc_content_map,
                                          doc_metadata_batch)

    async def _add_context_window_async(
            self,
            results: List[QueryResult],
            context_window: int
    ) -> List[QueryResult]:
        """
        Add context window around found chunks by including surrounding chunks asynchronously.
        Optimized to batch database queries by document and merge overlapping ranges.
        """
        if context_window <= 0 or not results:
            return results

        context_results = []

        # Group results by document and collect chunk indices
        doc_chunk_requests = defaultdict(list)

        for result in results:
            if result.type != 'chunk':
                # For document results, just pass through
                context_results.append(result)
                continue

            doc_id = result.document_id
            chunk_index = self._extract_chunk_index_from_id(result.id)
            doc_chunk_requests[doc_id].append((chunk_index, result))

        # Process each document asynchronously
        async with self.async_connection_pool.get_connection_context() as conn:
            for doc_id, chunk_requests in doc_chunk_requests.items():
                # Calculate ranges needed for all chunks in this document
                ranges_needed = []
                for chunk_index, result in chunk_requests:
                    start_index = max(0, chunk_index - context_window)
                    end_index = chunk_index + context_window
                    ranges_needed.append((start_index, end_index, chunk_index, result))

                # Merge overlapping ranges to minimize data fetched
                merged_ranges = self._merge_overlapping_ranges(ranges_needed)

                # Fetch all needed chunks for this document in a single query
                all_chunk_indices = set()
                for start, end, _, _ in merged_ranges:
                    all_chunk_indices.update(range(start, end + 1))

                if not all_chunk_indices:
                    continue

                # Single async query to get all chunks we need for this document
                placeholders = ','.join(['?'] * len(all_chunk_indices))
                cursor = await conn.execute(f'''
                    SELECT chunk_index, content, start_pos, end_pos, start_line, start_col, end_line, end_col
                    FROM chunks 
                    WHERE document_id = ? AND chunk_index IN ({placeholders})
                    ORDER BY chunk_index
                ''', [doc_id] + list(all_chunk_indices))

                # Create lookup map
                chunks_by_index = {}
                async for row in cursor:
                    chunks_by_index[row['chunk_index']] = row

                # Process each original result for this document
                for chunk_index, result in chunk_requests:
                    # Get context chunks for this specific result
                    context_chunks = []
                    start_context = max(0, chunk_index - context_window)
                    end_context = chunk_index + context_window

                    for i in range(start_context, end_context + 1):
                        if i in chunks_by_index:
                            context_chunks.append(chunks_by_index[i])

                    if not context_chunks:
                        # No context found, use original result
                        context_results.append(result)
                        continue

                    # Combine chunks into single content (same logic as original)
                    combined_content = []
                    min_start_pos = float('inf')
                    max_end_pos = 0
                    min_start_line = float('inf')
                    min_start_col = float('inf')
                    max_end_line = 0
                    max_end_col = 0

                    for chunk_row in context_chunks:
                        combined_content.append(chunk_row['content'])

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
                    separator = "\n\n---\n\n"
                    combined_text = separator.join(combined_content)

                    context_result = QueryResult(
                        id=f"{doc_id}:context:{chunk_index}",
                        score=result.score,  # Keep original score
                        type="context",
                        content=combined_text,
                        metadata=result.metadata.copy(),
                        document_id=doc_id,
                        position=context_position
                    )

                    # Add metadata about context
                    context_result.metadata.update({
                        '_context_window': context_window,
                        '_original_chunk_index': chunk_index,
                        '_context_chunk_count': len(context_chunks),
                        '_context_start_index': start_context,
                        '_context_end_index': end_context
                    })

                    context_results.append(context_result)

        return context_results

    async def update_async(
            self,
            doc_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update a document's content and/or metadata asynchronously.

        Parameters
        ----------
        doc_id : str
            Document ID to update
        content : Optional[str]
            New content (if None, content is not updated)
        metadata : Optional[Dict[str, Any]]
            New metadata (merged with existing metadata)

        Returns
        -------
        bool
            True if document was updated, False if not found

        Examples
        --------
        Update content only::

            updated = await db.update_async("doc1", content="New content")

        Update metadata only::

            updated = await db.update_async("doc1", metadata={"status": "reviewed"})

        Update both content and metadata::

            updated = await db.update_async(
                "doc1",
                content="Updated content",
                metadata={"last_modified": datetime.now()}
            )

        Notes
        -----
        - If content is updated, the document will be re-chunked and re-embedded
        - Metadata updates are merged with existing metadata (not replaced)
        - Content changes trigger full document reprocessing for consistency
        - Uses async database operations for better performance
        """
        self._ensure_async_pool()

        # Get existing document asynchronously
        existing_doc = await self.get_async(doc_id)
        if not existing_doc:
            logger.debug(f"Document {doc_id} not found for update")
            return False

        # Track if any changes were made
        changes_made = False

        # Update content if provided
        if content is not None:
            # Check if content actually changed
            new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if new_hash != existing_doc.content_hash:
                # Content changed, use async upsert to handle chunking/embedding
                updated_metadata = existing_doc.metadata.copy()
                if metadata:
                    updated_metadata.update(metadata)

                logger.debug(f"Content changed for document {doc_id}, re-processing with upsert")
                await self.upsert_async([content], [updated_metadata], [doc_id])
                return True
            else:
                logger.debug(f"Content unchanged for document {doc_id} (same hash)")

        # Update metadata only if no content change or content was None
        if metadata:
            updated_metadata = existing_doc.metadata.copy()
            updated_metadata.update(metadata)

            # Validate metadata (reuse sync validation in executor)
            await self._validate_metadata_async(updated_metadata)

            # Check if any embedding-enabled metadata fields have changed
            changed_embedding_fields = self._get_changed_embedding_fields(
                existing_doc.metadata, updated_metadata
            )

            async with self.async_connection_pool.get_connection_context() as conn:
                await conn.execute('BEGIN')
                try:
                    # Remove old metadata embeddings if any fields changed
                    if changed_embedding_fields:
                        await self._remove_metadata_embeddings_async(conn, doc_id)
                        
                        # Generate new metadata embeddings for changed fields
                        new_field_embeddings = await self._generate_metadata_embeddings_async(
                            updated_metadata, changed_embedding_fields, batch_size=100
                        )
                        
                        # Store new metadata embeddings
                        if new_field_embeddings:
                            await self._store_metadata_embeddings_async(conn, doc_id, new_field_embeddings)
                            logger.debug(f"Updated embeddings for {len(new_field_embeddings)} metadata fields in document {doc_id}")

                    # Build UPDATE statement for metadata
                    set_clauses = ['updated_at = ?']
                    values = [datetime.now(UTC)]

                    for field_name, value in updated_metadata.items():
                        if field_name in self.metadata_schema:
                            set_clauses.append(f'{field_name} = ?')
                            values.append(value)

                    values.append(doc_id)  # For WHERE clause

                    update_sql = f"""
                        UPDATE documents 
                        SET {', '.join(set_clauses)}
                        WHERE id = ?
                    """

                    cursor = await conn.execute(update_sql, values)
                    affected_rows = cursor.rowcount
                    await conn.commit()

                    if affected_rows > 0:
                        changes_made = True
                        logger.debug(f"Updated metadata for document {doc_id}")
                    else:
                        logger.warning(f"No rows affected when updating document {doc_id}")
                        
                except Exception:
                    await conn.rollback()
                    raise

        if not changes_made:
            logger.debug(f"No changes made to document {doc_id}")

        return changes_made

    async def _validate_metadata_async(self, metadata: Dict[str, Any]) -> None:
        """
        Async wrapper for metadata validation.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Metadata dictionary to validate

        Raises
        ------
        ValueError
            If metadata validation fails
        """
        # Metadata validation is CPU-bound and doesn't need async,
        # but we provide async wrapper for consistency
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_metadata_batch, [metadata])

    async def get_chunk_embeddings_async(self, chunk_ids: str | List[str]) -> np.ndarray:
        """Returns embeddings for chunks given by `chunk_ids`"""
        single_id = isinstance(chunk_ids, str)
        if single_id:
            chunk_ids = [chunk_ids]

        chunk_list = []
        for cid in chunk_ids:
            doc_id, chunk_idx = self._split_chunk_id(cid)
            if chunk_idx == -1:
                raise ValueError(f"Expected chunk ids (e.g. doc_1:1), found: {cid}")
            chunk_list.append((doc_id, chunk_idx))

        placeholders = ",".join(["(?,?)"] * len(chunk_list))
        query_str = f"""SELECT faiss_id, document_id, chunk_index FROM chunks WHERE (document_id, chunk_index) IN ({placeholders})"""
        params = [item for pair in chunk_list for item in pair]
        with self._read_write_lock.read_lock():
            async with self.async_connection_pool.get_connection() as conn:
                cursor = conn.execute(query_str, params)
                rows = cursor.fetchall()
                faiss_ids = [row["faiss_id"] for row in rows]

            # Reconstruction of embeddings from faiss is cpu/gpu bound, so use the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None,  self._reconstruct_embeddings_batch, faiss_ids)

    async def query_multi_column_async(
            self,
            query: str,
            *,
            columns: Optional[List[str]] = None,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks'] = 'documents',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None
    ) -> List[QueryResult]:
        """
        Async query across multiple columns (main content + embedding-enabled metadata fields)
        
        Parameters
        ----------
        query : str
            Query text
        columns : Optional[List[str]]
            Specific columns to search. If None, searches all embedding-enabled fields
            plus main content. Use 'content' for main document content.
        search_type : Literal['vector', 'keyword', 'hybrid']
            Type of search to perform
        return_type : Literal['documents', 'chunks']
            Whether to return full documents or individual chunks
        k : int
            Maximum number of results to return
        score_threshold : float
            Minimum score threshold (0-1, higher=better)
        filters : Optional[Dict[str, Any]]
            Metadata filters to apply
        vector_weight : float
            Weight for vector search in hybrid mode (0-1)
        document_scoring_method : DocumentScoringMethod
            Method for aggregating chunk scores into document scores
        document_scoring_options : dict, optional
            Parameters for the scoring method
            
        Returns
        -------
        List[QueryResult]
            Search results with column attribution
        """
        self._ensure_async_pool()

        # Determine which columns to search
        embedding_enabled_fields = self._get_embedding_enabled_fields()

        if columns is None:
            # Search all embedding-enabled fields plus main content
            search_columns = ['content'] + list(embedding_enabled_fields.keys())
        else:
            # Validate requested columns
            search_columns = []
            for col in columns:
                if col == 'content':
                    search_columns.append(col)
                elif col in embedding_enabled_fields:
                    search_columns.append(col)
                else:
                    logger.warning(f"Column '{col}' is not embedding-enabled, skipping")

            if not search_columns:
                logger.warning("No valid columns specified for search")
                return []

        all_results = []

        # Search main content if requested
        if 'content' in search_columns:
            content_results = await self.query_async(
                query=query,
                search_type=search_type,
                return_type='chunks',  # Always get chunks for multi-column
                k=k * 2,  # Get more results to allow for proper ranking
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                document_scoring_method=document_scoring_method,
                document_scoring_options=document_scoring_options
            )

            # Add column attribution
            for result in content_results:
                result.metadata = result.metadata or {}
                result.metadata['_search_column'] = 'content'
                all_results.append(result)

        # Search metadata fields
        metadata_columns = [col for col in search_columns if col != 'content']
        if metadata_columns and search_type in ['vector', 'hybrid']:
            # Create tasks for concurrent metadata field searches
            metadata_search_tasks = []
            for field_name in metadata_columns:
                task = asyncio.create_task(
                    self._search_metadata_field_async(
                        query=query,
                        field_name=field_name,
                        k=k * 2,
                        score_threshold=score_threshold,
                        filters=filters
                    )
                )
                metadata_search_tasks.append((field_name, task))

            # Wait for all metadata searches to complete
            for field_name, task in metadata_search_tasks:
                field_results = await task

                # Add column attribution
                for result in field_results:
                    result.metadata = result.metadata or {}
                    result.metadata['_search_column'] = field_name
                    all_results.append(result)

        # Sort all results by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        limited_results = all_results[:k]

        if return_type == 'documents':
            # Aggregate chunks into documents
            return await self._aggregate_document_scores_with_method_async(
                limited_results,
                document_scoring_method,
                document_scoring_options
            )
        else:
            return limited_results

    async def _search_metadata_field_async(
            self,
            query: str,
            field_name: str,
            k: int,
            score_threshold: float,
            filters: Optional[Dict[str, Any]]
    ) -> List[QueryResult]:
        """
        Async search a specific metadata field's embeddings
        
        Parameters
        ----------
        query : str
            Query text
        field_name : str
            Name of metadata field to search
        k : int
            Maximum results to return
        score_threshold : float
            Minimum score threshold
        filters : Optional[Dict[str, Any]]
            Metadata filters
            
        Returns
        -------
        List[QueryResult]
            Search results for this field
        """
        # Generate query embedding asynchronously
        embeddings = await self.embedding_provider.embed_batch([query])
        query_embedding = embeddings[0]

        async with self.async_connection_pool.get_connection_context() as conn:
            # Get all metadata field embeddings
            cursor = await conn.execute("""
                SELECT ce.faiss_id, ce.document_id, ce.chunk_index, d.content, d.created_at, d.updated_at
                FROM column_embeddings ce
                JOIN documents d ON ce.document_id = d.id
                WHERE ce.field_name = ?
            """, (field_name,))

            field_embedding_data = await cursor.fetchall()

            if not field_embedding_data:
                return []

            # Extract FAISS IDs for this field
            faiss_ids = [row['faiss_id'] for row in field_embedding_data]

            if not faiss_ids:
                return []

            # Get embeddings for these FAISS IDs (run in executor as FAISS is not async)
            loop = asyncio.get_event_loop()
            field_embeddings = await loop.run_in_executor(
                None, self._reconstruct_embeddings_batch, faiss_ids
            )

            if field_embeddings.size == 0:
                return []

            # Compute similarities (run in executor for numpy operations)
            def compute_similarities():
                query_embedding_2d = query_embedding.reshape(1, -1)
                similarities = np.dot(field_embeddings, query_embedding_2d.T).flatten()
                # Convert to scores (higher is better)
                scores = (similarities + 1) / 2  # Normalize to 0-1
                return scores

            scores = await loop.run_in_executor(None, compute_similarities)

            # Filter by score threshold
            valid_indices = np.where(scores >= score_threshold)[0]

            if len(valid_indices) == 0:
                return []

            # Sort by score and limit
            sorted_indices = valid_indices[np.argsort(scores[valid_indices])[::-1]][:k]

            results = []

            # Get document metadata for all results in batch
            doc_ids = [field_embedding_data[idx]['document_id'] for idx in sorted_indices]
            doc_metadata_batch = await self._get_document_metadata_async(doc_ids)

            for idx in sorted_indices:
                row_data = field_embedding_data[idx]
                doc_metadata = doc_metadata_batch.get(row_data['document_id'], {})

                # Create result
                result = QueryResult(
                    id=f"{row_data['document_id']}:meta:{field_name}:{row_data['chunk_index']}",
                    content=str(doc_metadata.get(field_name, "")),
                    score=float(scores[idx]),
                    document_id=row_data['document_id'],
                    metadata=doc_metadata,
                    type='chunk'
                )
                results.append(result)

            # Apply metadata filters if provided
            if filters:
                loop = asyncio.get_event_loop()
                # Use the existing sync filter function in executor
                def apply_filters():
                    return [r for r in results if matches_metadata_filter(r.metadata, filters)]

                results = await loop.run_in_executor(None, apply_filters)

            return results

    async def __aenter__(self):
        self._ensure_async_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()
