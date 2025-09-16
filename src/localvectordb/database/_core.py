# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/database/_core.py

"""
Core initialization and infrastructure for LocalVectorDB.

This module contains the base LocalVectorDB class which is responsible for:
- Filesystem paths, connection pools and schema initialization
- Chunker and embedding provider setup
- FTS (SQLite FTS5) initialization
- FAISS index creation/loading, GPU transfer and persistence
- Shared properties and lifecycle management (save/close)

All domain-specific behavior (ingestion pipelines, search, metadata handling,
CRUD) is mixed in via other modules to compose the final LocalVectorDB class.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import uuid
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import faiss
import numpy as np

from localvectordb._filters import FilterQueryBuilder
from localvectordb._pools import AsyncConnectionPool, ConnectionPool, ReadWriteLock
from localvectordb._schema import DatabaseSchema, get_common_metadata_schemas
from localvectordb.chunking import ChunkerFactory
from localvectordb.core import Chunk, MetadataField
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.embeddings import EmbeddingProvider, EmbeddingRegistry
from localvectordb.exceptions import DatabaseError, DatabaseNotFoundError
from localvectordb.utils import get_system_version

logger = logging.getLogger(__name__)


class LocalVectorDBCore(LocalVectorDBBase, ABC):
    """
    Base class providing initialization, configuration, FAISS/FTS setup, and lifecycle.

    This class intentionally mirrors the original LocalVectorDB.__init__ and related
    helpers as closely as possible to preserve behavior.
    """

    def __init__(
            self, name: str, base_path: Union[str, Path] = ".lvdb", *, metadata_schema: Optional[Dict[str, Any]] = None,
            doc_id_pattern: str = "doc_{idx}", embedding_provider: str = "ollama",
            embedding_model: str = "nomic-embed-text", embedding_config: Optional[Dict[str, Any]] = None,
            chunking_method: Union[str, Any] = "sentences", chunk_size: int = 500, chunk_overlap: int = 1,
            faiss_index_type: Literal["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"] = "IndexFlatL2",
            faiss_index_hnsw_flat_neighbors: Optional[int] = None, faiss_index_lsh_bits: Optional[int] = None,
            enable_gpu: bool = False, enable_fts: bool = True, connection_pool_size: int = 10,
            create_if_not_exists: bool = True
            ):

        super().__init__(name, base_path, metadata_schema=metadata_schema, doc_id_pattern=doc_id_pattern,
                         embedding_provider=embedding_provider, embedding_model=embedding_model,
                         embedding_config=embedding_config, chunking_method=chunking_method, chunk_size=chunk_size,
                         chunk_overlap=chunk_overlap, faiss_index_type=faiss_index_type,
                         faiss_index_hnsw_flat_neighbors=faiss_index_hnsw_flat_neighbors,
                         faiss_index_lsh_bits=faiss_index_lsh_bits, enable_gpu=enable_gpu, enable_fts=enable_fts,
                         connection_pool_size=connection_pool_size, create_if_not_exists=create_if_not_exists)
        self.name = name
        self._original_memory_request = (name == ":memory:" or base_path == ":memory:")

        if self._original_memory_request:
            unique_id = str(uuid.uuid4()).replace("-", "")[:8]
            self.db_path: Union[str, Path] = f"file:memdb_{unique_id}?mode=memory&cache=shared"
            self.base_path = None
            self.index_path = None
            logger.info(f"Creating in-memory database with shared cache: {self.db_path}")
        else:
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.db_path: Union[str, Path] = self.base_path / f"{name}.sqlite"
            self.index_path = self.base_path / f"{name}.faiss"
            if not create_if_not_exists and not Path(self.db_path).exists():
                raise DatabaseNotFoundError(f"Database: {name} in {base_path} could not be found.")

        # Metadata schema
        if isinstance(metadata_schema, str):
            self._metadata_schema = get_common_metadata_schemas(metadata_schema)
        else:
            self._metadata_schema = metadata_schema or {}
        self.doc_id_pattern = doc_id_pattern

        # Chunker
        self._chunking_method = chunking_method
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.chunker = ChunkerFactory.create_chunker(chunking_method, chunk_size, chunk_overlap)

        # Embedding provider
        embedding_config = embedding_config or {}
        self._embedding_provider = EmbeddingRegistry.create_provider(
            embedding_provider, embedding_model, **embedding_config
        )
        if not self._embedding_provider.validate_model():
            raise ValueError(f"Embedding model '{embedding_model}' is not available")
        self._embedding_dimension = self._embedding_provider.get_dimension()

        # Threading
        self._read_write_lock = ReadWriteLock()
        self._faiss_lock = threading.Lock()  # Lock for FAISS operations to prevent race conditions

        # Database schema + pools
        self.schema = DatabaseSchema(self.db_path, self._read_write_lock)
        self.connection_pool = ConnectionPool(self.db_path, connection_pool_size)
        self.async_connection_pool: Optional[AsyncConnectionPool] = None
        self.async_max_connections = connection_pool_size or 10
        self._async_schema_initialized = False

        with self.connection_pool.get_connection() as conn:
            self.schema.initialize(self._metadata_schema, db_connection=conn)
            if not self.is_memory_only and Path(self.db_path).exists():
                existing_schema = self.schema.load_metadata_schema(db_connection=conn)
                self._metadata_schema.update(existing_schema)

        # FTS
        self._fts_enabled = False
        if enable_fts:
            self._init_fts()

        # FAISS
        self._init_faiss_index(enable_gpu, faiss_index_type, faiss_index_hnsw_flat_neighbors, faiss_index_lsh_bits)

        # How many items allowed on the processing queues.
        self.pipeline_queue_size: int = 3

        # State
        self._next_doc_id = self._load_next_doc_id()
        self._async_id_lock: Optional[asyncio.Lock] = None

        # Save config
        self._save_config()

    # Context managers
    def __enter__(self) -> "LocalVectorDBCore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # Properties
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
    def metadata_schema(self) ->  Dict[str, MetadataField]:
        return self._metadata_schema.copy()

    @property
    def closed(self) -> bool:
        return self.connection_pool.closed

    @property
    def is_memory_only(self) -> bool:
        return self._original_memory_request

    def ping(self) -> bool:
        return not self.closed

    # FTS helpers
    def _check_fts5_availability(self) -> bool:
        try:
            with self.connection_pool.get_connection() as conn:
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
        if not self._check_fts5_availability():
            self._fts_enabled = False
            return
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                        id,
                        content,
                        content='documents',
                        content_rowid='rowid'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        document_id,
                        content,
                        content='chunks',
                        content_rowid='id'
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                        INSERT INTO documents_fts(rowid, id, content)
                        VALUES (new.rowid, new.id, new.content);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                        DELETE FROM documents_fts WHERE rowid = old.rowid;
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                        DELETE FROM documents_fts WHERE rowid = old.rowid;
                        INSERT INTO documents_fts(rowid, id, content)
                        VALUES (new.rowid, new.id, new.content);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, document_id, content)
                        VALUES (new.id, new.document_id, new.content);
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.id;
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                        DELETE FROM chunks_fts WHERE rowid = old.id;
                        INSERT INTO chunks_fts(rowid, document_id, content)
                        VALUES (new.id, new.document_id, new.content);
                    END
                    """
                )
                conn.commit()
                self._fts_enabled = True
                logger.info("FTS5 initialized successfully")
        except Exception as e:
            logger.error(f"Error setting up FTS5: {e}")
            self._fts_enabled = False

    # FAISS helpers
    def _init_faiss_index(
        self,
        enable_gpu: bool,
        faiss_index_type,
        faiss_index_hnsw_flat_neighbors: Optional[int],
        faiss_index_lsh_bits: Optional[int],
    ):
        if self.index_path and self.index_path.exists():
            try:
                loaded_index = faiss.read_index(str(self.index_path))
            except RuntimeError as e:
                raise DatabaseError(f"Error loading faiss index: {str(e)}")
            if hasattr(loaded_index, 'id_map'):
                self.index = loaded_index
                logger.info(f"Loaded existing FAISS IndexIDMap2 with {self.index.ntotal} vectors")
            else:
                raise DatabaseError("Expected FAISS index to have `id_map` attribute. Invalid faiss index!")
        else:
            if faiss_index_type == "IndexFlatL2":
                base_index = faiss.IndexFlatL2(self.embedding_dimension)
            elif faiss_index_type == "IndexFlatIP":
                base_index = faiss.IndexFlatIP(self.embedding_dimension)
            elif faiss_index_type == "IndexHNSWFlat":
                base_index = faiss.IndexHNSWFlat(self.embedding_dimension, faiss_index_hnsw_flat_neighbors or 16)
            elif faiss_index_type == "IndexLSH":
                base_index = faiss.IndexLSH(self.embedding_dimension, faiss_index_lsh_bits or self.embedding_dimension * 2)
            else:
                raise ValueError("Invalid faiss index for LocalVectorDB. Must be one of: IndexFlatL2, IndexFlatIP, IndexHNSWFlat, IndexLSH")
            self.index = faiss.IndexIDMap2(base_index)
            logger.info(f"Created new FAISS IndexIDMap2 with dimension {self.embedding_dimension}")
        if enable_gpu and faiss.get_num_gpus() > 0:
            try:
                self.index = faiss.index_cpu_to_all_gpus(self.index)
                logger.info("Moved FAISS index to GPU")
            except Exception as e:
                logger.warning(f"Could not move IndexIDMap2 to GPU: {e}")
        elif enable_gpu:
            logger.warning("GPU requested but no GPUs available")

    def _add_vectors_to_faiss_bulk(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(embeddings) == 0:
            return
        with self._faiss_lock:
            start_faiss_id = self.index.ntotal
            new_faiss_ids = np.arange(start_faiss_id, start_faiss_id + len(embeddings), dtype=np.int64)
            self.index.add_with_ids(embeddings, new_faiss_ids)
            for i, chunk in enumerate(chunks):
                chunk.faiss_id = int(new_faiss_ids[i])

    def _remove_old_vectors_bulk(self, faiss_ids: List[int]) -> None:
        if not faiss_ids or not hasattr(self.index, 'remove_ids'):
            return
        try:
            with self._faiss_lock:
                ids_array = np.array(faiss_ids, dtype=np.int64)
                self.index.remove_ids(ids_array)
                logger.debug(f"Removed {len(faiss_ids)} vectors from FAISS index")
        except Exception as e:
            logger.warning(f"Failed to remove vectors from FAISS: {e}")

    def _reconstruct_embeddings_batch(self, faiss_ids: List[int]) -> np.ndarray:
        """
        Batch reconstruct embeddings with proper IndexIDMap2 handling and fallback strategies.
        
        For IndexIDMap/IndexIDMap2 indices, we need to map external FAISS IDs to internal 
        indices before calling reconstruct_batch on the base index.
        """
        if not faiss_ids:
            return np.array([]).reshape(0, self.embedding_dimension)

        # Check if we have an IndexIDMap/IndexIDMap2 wrapper
        if hasattr(self.index, 'id_map') and hasattr(self.index, 'index'):
            try:
                # Method 1: Use proper ID mapping for IndexIDMap2 (safest approach)
                return self._reconstruct_with_id_mapping(faiss_ids)
            except Exception as e:
                logger.warning(f"IndexIDMap reconstruction failed, falling back to individual calls: {e}")
                return self._reconstruct_individual_fallback(faiss_ids)

        # Method 2: Try reconstruct_batch if available (for non-wrapped indices)
        if hasattr(self.index, 'reconstruct_batch'):
            try:
                faiss_ids_array = np.array(faiss_ids, dtype=np.int64)
                embeddings = self.index.reconstruct_batch(faiss_ids_array)
                return embeddings
            except Exception as e:
                logger.warning(f"FAISS reconstruct_batch failed, falling back to individual calls: {e}")

        # Method 3: Fallback to individual reconstruct calls
        return self._reconstruct_individual_fallback(faiss_ids)

    def _get_faiss_ids_from_db(self, chunk_ids: List[int]) -> Dict[int, int]:
        """
        Get chunk_id -> faiss_id mapping from database efficiently.
        
        Parameters
        ----------
        chunk_ids : List[int]
            List of chunk IDs to look up
            
        Returns
        -------
        Dict[int, int]
            Mapping from chunk_id to faiss_id
        """
        if not chunk_ids:
            return {}

        try:
            with self.connection_pool.get_connection() as conn:
                # Use parameterized query with IN clause for efficient batch lookup
                placeholders = ','.join('?' * len(chunk_ids))
                sql = f"SELECT id, faiss_id FROM chunks WHERE id IN ({placeholders}) AND faiss_id IS NOT NULL"
                cursor = conn.execute(sql, chunk_ids)

                # Build mapping dict
                chunk_to_faiss = {}
                for row in cursor:
                    chunk_to_faiss[row['id']] = row['faiss_id']

                return chunk_to_faiss
        except Exception as e:
            logger.warning(f"Failed to get FAISS IDs from database: {e}")
            return {}

    def _reconstruct_with_id_mapping(self, faiss_ids: List[int]) -> np.ndarray:
        """Reconstruct embeddings for IndexIDMap2 using optimized mapping strategies."""
        if not faiss_ids:
            return np.array([]).reshape(0, self.embedding_dimension)

        faiss_ids_array = np.array(faiss_ids, dtype=np.int64)

        # Strategy 1: Try direct reconstruction on IndexIDMap2 first
        # For some index types, we can reconstruct directly using external IDs
        try:
            if hasattr(self.index, 'reconstruct'):
                embeddings = []
                for fid in faiss_ids_array:
                    try:
                        embedding = self.index.reconstruct(fid)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.debug(f"Direct reconstruct failed for FAISS ID {fid}: {e}")
                        break
                else:
                    # All direct reconstructs succeeded
                    result = np.array(embeddings)
                    logger.debug(f"Successfully reconstructed {len(embeddings)} embeddings using direct IndexIDMap2 reconstruct")
                    return result
        except Exception as e:
            logger.debug(f"Direct IndexIDMap2 reconstruction not available: {e}")

        # Strategy 2: Efficient mapping using internal FAISS methods (if available)
        try:
            # Check if FAISS provides an efficient way to get internal indices
            if hasattr(self.index, 'get_ids') and hasattr(self.index.index, 'reconstruct_batch'):
                # Get all current IDs efficiently
                current_ids = self.index.get_ids()
                id_to_internal = {ext_id: internal_idx for internal_idx, ext_id in enumerate(current_ids)}

                internal_indices = []
                for fid in faiss_ids_array:
                    if fid in id_to_internal:
                        internal_indices.append(id_to_internal[fid])
                    else:
                        logger.debug(f"FAISS ID {fid} not found in current index")

                if internal_indices:
                    internal_indices_array = np.array(internal_indices, dtype=np.int64)
                    embeddings = self.index.index.reconstruct_batch(internal_indices_array)
                    logger.debug(f"Successfully reconstructed {len(embeddings)} embeddings using efficient ID mapping")
                    return embeddings
        except Exception as e:
            logger.debug(f"Efficient ID mapping strategy failed: {e}")

        # Strategy 3: Fallback to optimized linear search (improved from O(n²) to O(n*m))
        # Build a reverse mapping dictionary first for better performance
        try:
            # Build id_map lookup once for all IDs
            id_map_lookup = {}
            for i in range(self.index.ntotal):
                external_id = self.index.id_map.at(i)
                id_map_lookup[external_id] = i

            internal_indices = []
            for fid in faiss_ids_array:
                if fid in id_map_lookup:
                    internal_indices.append(id_map_lookup[fid])
                else:
                    logger.debug(f"FAISS ID {fid} not found in id_map")

            if not internal_indices:
                logger.warning(f"No valid internal indices found for FAISS IDs: {faiss_ids}")
                return np.array([]).reshape(0, self.embedding_dimension)

            # Use the base index's reconstruct_batch method with internal indices
            internal_indices_array = np.array(internal_indices, dtype=np.int64)
            base_index = self.index.index

            if hasattr(base_index, 'reconstruct_batch'):
                embeddings = base_index.reconstruct_batch(internal_indices_array)
                logger.debug(f"Successfully reconstructed {len(embeddings)} embeddings using optimized ID mapping")
                return embeddings
            else:
                # Fallback: individual calls on base index
                embeddings = []
                for idx in internal_indices:
                    try:
                        embedding = base_index.reconstruct(idx)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.debug(f"Failed to reconstruct internal index {idx}: {e}")
                        continue

                return np.array(embeddings) if embeddings else np.array([]).reshape(0, self.embedding_dimension)

        except Exception as e:
            logger.warning(f"Optimized ID mapping failed: {e}")
            # Final fallback to original O(n²) method if all else fails
            return self._reconstruct_with_linear_search_fallback(faiss_ids_array)

    def _reconstruct_with_linear_search_fallback(self, faiss_ids_array: np.ndarray) -> np.ndarray:
        """Original O(n²) fallback method for critical compatibility."""
        internal_indices = []
        for fid in faiss_ids_array:
            internal_idx = -1
            # Original linear search through the id_map
            for i in range(self.index.ntotal):
                if self.index.id_map.at(i) == fid:
                    internal_idx = i
                    break
            if internal_idx != -1:
                internal_indices.append(internal_idx)
            else:
                logger.debug(f"FAISS ID {fid} not found in id_map (fallback)")

        if not internal_indices:
            logger.warning(f"No valid internal indices found for FAISS IDs (fallback): {faiss_ids_array.tolist()}")
            return np.array([]).reshape(0, self.embedding_dimension)

        # Use the base index's reconstruct_batch method with internal indices
        internal_indices_array = np.array(internal_indices, dtype=np.int64)
        base_index = self.index.index

        if hasattr(base_index, 'reconstruct_batch'):
            embeddings = base_index.reconstruct_batch(internal_indices_array)
            logger.debug(f"Successfully reconstructed {len(embeddings)} embeddings using fallback linear search")
            return embeddings
        else:
            # Individual calls on base index
            embeddings = []
            for idx in internal_indices:
                try:
                    embedding = base_index.reconstruct(idx)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.debug(f"Failed to reconstruct internal index {idx} (fallback): {e}")
                    continue

            return np.array(embeddings) if embeddings else np.array([]).reshape(0, self.embedding_dimension)

    def _reconstruct_individual_fallback(self, faiss_ids: List[int]) -> np.ndarray:
        """Fallback method using individual reconstruct calls."""
        embeddings = []
        for fid in faiss_ids:
            try:
                embedding = self.index.reconstruct(fid)
                embeddings.append(embedding)
            except Exception as e:
                logger.debug(f"Failed to reconstruct FAISS ID {fid}: {e}")
                continue

        result = np.array(embeddings) if embeddings else np.array([]).reshape(0, self.embedding_dimension)
        if len(embeddings) != len(faiss_ids):
            logger.warning(f"Only reconstructed {len(embeddings)}/{len(faiss_ids)} embeddings in fallback mode")

        return result

    # Config/state helpers
    def _load_next_doc_id(self) -> int:
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute('SELECT value FROM config WHERE key = ?', ('next_doc_id',))
            row = cursor.fetchone()
            return int(row['value']) if row else 1

    def _save_next_doc_id(self) -> None:
        with self.connection_pool.get_connection() as conn:
            conn.execute('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)', ('next_doc_id', str(self._next_doc_id)))
            conn.commit()

    def _save_config(self):
        config = {
            'embedding_provider': self.embedding_provider.provider_name,
            'embedding_model': self.embedding_provider.model,
            'embedding_dimension': self.embedding_dimension,
            'chunking_method': self.chunking_method,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'doc_id_pattern': self.doc_id_pattern,
            'fts_enabled': str(self.fts_enabled),
            'version': get_system_version(),
        }
        with self.connection_pool.get_connection() as conn:
            for key, value in config.items():
                conn.execute('INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)', (key, str(value)))
            conn.commit()

    def _get_faiss_metric_type(self) -> str:
        """Detect the metric type of the FAISS index.
        
        Returns:
            'L2' for L2 distance metrics
            'IP' for inner product metrics
        """
        if not hasattr(self, 'index') or self.index is None:
            return 'L2'  # Default

        # Check if it's an IndexIDMap wrapper
        if hasattr(self.index, 'index'):
            base_index = self.index.index
        else:
            base_index = self.index

        # Check the index type name
        index_type = str(type(base_index).__name__)
        if 'IP' in index_type or 'InnerProduct' in index_type:
            return 'IP'
        elif 'L2' in index_type:
            return 'L2'
        elif 'HNSW' in index_type:
            # HNSW can use different metrics, check if it's HNSW with IP
            if hasattr(base_index, 'metric_type'):
                if base_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    return 'IP'
            return 'L2'  # Default for HNSW
        else:
            # LSH and others default to L2-like behavior
            return 'L2'

    def _distance_to_similarity(self, distance: float, metric_type: Optional[str] = None) -> float:
        """Convert FAISS distance to similarity score based on metric type.

        Parameters
        ----------
        distance : float
            The distance value from FAISS
        metric_type : str
            'L2' or 'IP', if None will auto-detect

        Returns
        -------
        float
            Similarity measure
        """
        if metric_type is None:
            metric_type = self._get_faiss_metric_type()

        if metric_type == 'IP':
            # For inner product, the distance IS the similarity (if normalized)
            # Assuming normalized embeddings, IP ranges from -1 to 1
            # Convert to 0-1 range
            return max(0.0, min(1.0, (distance + 1.0) / 2.0))
        else:
            # L2 distance: convert using 1/(1+distance)
            return 1.0 / (1.0 + distance)

    def _similarity_to_distance(self, similarity: float, metric_type: str = None) -> float:
        """Convert similarity threshold to FAISS distance threshold.
        
        Args:
            similarity: Similarity threshold in [0, 1] range
            metric_type: 'L2' or 'IP', if None will auto-detect
        
        Returns:
            Distance threshold for FAISS
        """
        if metric_type is None:
            metric_type = self._get_faiss_metric_type()

        similarity = max(0.001, min(1.0, similarity))  # Clamp to valid range

        if metric_type == 'IP':
            # Convert from 0-1 to -1 to 1 range
            # Higher similarity means higher inner product
            return similarity * 2.0 - 1.0
        else:
            # L2 distance: invert the formula
            return (1.0 / similarity) - 1.0

    def _generate_doc_id(self) -> str:
        doc_id = self.doc_id_pattern.format(idx=self._next_doc_id)
        self._next_doc_id += 1
        return doc_id

    async def _generate_doc_id_async(self) -> str:
        """Async-safe version of _generate_doc_id with proper locking."""
        if self._async_id_lock is None:
            self._async_id_lock = asyncio.Lock()

        async with self._async_id_lock:
            doc_id = self.doc_id_pattern.format(idx=self._next_doc_id)
            self._next_doc_id += 1
            return doc_id

    # Persistence
    def _save_internal(self):
        if not self.is_memory_only and hasattr(self.index, 'ntotal') and self.index.ntotal > 0:
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'device'):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(self.index_path))
            else:
                faiss.write_index(self.index, str(self.index_path))

    def save(self):
        with self._read_write_lock.write_lock():
            self._save_internal()

    def close(self):
        """Close the database"""
        self.save()
        self.connection_pool.close_all()
        if hasattr(self, 'async_connection_pool') and self.async_connection_pool is not None:
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(self.close_async())
                else:
                    asyncio.run(self.close_async())
            except RuntimeError:
                asyncio.run(self.close_async())
            except Exception as e:
                logger.warning(f"Error closing async resources: {e}")

    # Stats
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics

        Returns
        -------
        dict[str, int|str|bool]
            A dict with the following keys:
            - documents
            - chunks
            - index_vectors
            - embedding_dimension
            - embedding_provider
            - embedding_model
            - chunking_method
            - chunk_size
            - chunk_overlap
            - fts_enabled
        """
        with self.connection_pool.get_connection() as conn:
            doc_count = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            chunk_count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
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
                'fts_enabled': self.fts_enabled,
            }

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """
        Count documents matching filter criteria

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

    # Async helpers
    def _ensure_async_pool(self) -> None:
        if self.async_connection_pool is None:
            self.async_connection_pool = AsyncConnectionPool(self.db_path, max_connections=self.async_max_connections)

    async def _ensure_async_schema_initialized(self) -> None:
        if not self.is_memory_only or self._async_schema_initialized:
            return
        try:
            async with self.async_connection_pool.get_connection_context() as conn:
                await conn.execute("SELECT 1 FROM documents LIMIT 1")
            self._async_schema_initialized = True
        except Exception:
            logger.info("Initializing database schema for async operations in in-memory database")
            async with self.async_connection_pool.get_connection_context() as conn:
                await self.schema.initialize_async(self._metadata_schema, db_connection=conn)
            self._async_schema_initialized = True

    async def close_async(self):
        """Close async resources"""
        if self.async_connection_pool:
            await self.async_connection_pool.close_all()
            self.async_connection_pool = None

    async def save_async(self):
        """Saves the database asynchronously"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.save)

    async def get_async_stats(self) -> Dict[str, Any]:
        """Get async-specific statistics"""
        stats = {}
        if self.async_connection_pool:
            stats['async_pool'] = self.async_connection_pool.stats
        else:
            stats['async_pool'] = {'status': 'not_initialized'}
        return stats

    async def __aenter__(self):
        self._ensure_async_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()
