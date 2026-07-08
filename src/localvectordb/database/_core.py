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
import json
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
from localvectordb._sqlite_retry import retry_on_locked
from localvectordb.chunking import ChunkerFactory
from localvectordb.core import Chunk, MetadataField
from localvectordb.database._faiss_utils import build_id_lookup
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.embeddings import EmbeddingProvider, EmbeddingRegistry
from localvectordb.exceptions import DatabaseError, DatabaseNotFoundError
from localvectordb.section_detection import SectionDetector
from localvectordb.section_metadata import SectionMetadataExtractor, resolve_extractors
from localvectordb.sqlite_tuning import SqliteProfile, get_sqlite_pragma_profile, is_valid_sqlite_pragma_profile
from localvectordb.utils import get_system_version

logger = logging.getLogger(__name__)


class LocalVectorDBCore(LocalVectorDBBase, ABC):
    """
    Base class providing initialization, configuration, FAISS/FTS setup, and lifecycle.

    This class intentionally mirrors the original LocalVectorDB.__init__ and related
    helpers as closely as possible to preserve behavior.
    """

    # Type declarations for attributes set during __init__
    connection_pool: ConnectionPool
    async_connection_pool: Optional[AsyncConnectionPool]
    schema: DatabaseSchema
    # Usually an IndexIDMap2, but may be a base faiss.Index (loaded via read_index)
    # or a GpuIndex (after index_cpu_to_all_gpus), so the broad base type is correct.
    index: Optional[faiss.Index]
    _read_write_lock: ReadWriteLock
    _faiss_lock: ReadWriteLock
    _metadata_schema: Dict[str, Any]
    _embedding_provider: EmbeddingProvider
    _embedding_dimension: int
    # Cache of (index_object, metric_type); invalidated when the index identity changes.
    _metric_type_cache: Optional[tuple[Any, str]] = None

    def __init__(
        self,
        name: str,
        base_path: Union[str, Path] = ".lvdb",
        *,
        metadata_schema: Optional[Dict[str, Any]] = None,
        doc_id_pattern: str = "doc_{idx}",
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text",
        embedding_config: Optional[Dict[str, Any]] = None,
        chunking_method: Union[str, Any] = "sentences",
        chunk_size: int = 500,
        chunk_overlap: int = 1,
        batch_size: int = 100,
        faiss_index_type: Literal["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"] = "IndexFlatL2",
        faiss_index_hnsw_flat_neighbors: Optional[int] = None,
        faiss_index_lsh_bits: Optional[int] = None,
        enable_gpu: bool = False,
        enable_fts: bool = True,
        connection_pool_size: int = 10,
        create_if_not_exists: bool = True,
        sqlite_profile: SqliteProfile = "balanced",
        sqlite_pragma_overrides: Optional[Dict[str, Any]] = None,
        pipeline_worker_timeout: float = 300.0,
        # Hierarchical embedding parameters
        hierarchical_embeddings: bool = False,
        section_pattern: str = r"^(#{1,6})\s+(.+)$",
        section_metadata_extractors: Optional[List[Union[str, SectionMetadataExtractor]]] = None,
    ):

        super().__init__(
            name,
            base_path,
            metadata_schema=metadata_schema,
            doc_id_pattern=doc_id_pattern,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_config=embedding_config,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=batch_size,
            faiss_index_type=faiss_index_type,
            faiss_index_hnsw_flat_neighbors=faiss_index_hnsw_flat_neighbors,
            faiss_index_lsh_bits=faiss_index_lsh_bits,
            enable_gpu=enable_gpu,
            enable_fts=enable_fts,
            connection_pool_size=connection_pool_size,
            create_if_not_exists=create_if_not_exists,
            sqlite_profile=sqlite_profile,
            sqlite_pragma_overrides=sqlite_pragma_overrides,
        )
        self.name = name
        # Hierarchical embeddings
        self._hierarchical_embeddings = hierarchical_embeddings
        self._section_pattern = section_pattern
        self._section_detector: Optional[SectionDetector] = None
        self._section_metadata_extractors: List[SectionMetadataExtractor] = []
        self.section_index: Optional[faiss.Index] = None
        self.document_index: Optional[faiss.Index] = None
        self.section_index_path: Optional[Path] = None
        self.document_index_path: Optional[Path] = None
        self._original_memory_request = name == ":memory:" or base_path == ":memory:"

        if self._original_memory_request:
            unique_id = str(uuid.uuid4()).replace("-", "")[:8]
            self.db_path: Union[str, Path] = f"file:memdb_{unique_id}?mode=memory&cache=shared"
            self.base_path = None
            self.index_path = None
            logger.info(f"Creating in-memory database with shared cache: {self.db_path}")
        else:
            self.base_path = Path(base_path)
            self.base_path.mkdir(parents=True, exist_ok=True)
            self.db_path = self.base_path / f"{name}.sqlite"
            self.index_path = self.base_path / f"{name}.faiss"
            if not create_if_not_exists and not Path(self.db_path).exists():
                raise DatabaseNotFoundError(f"Database: {name} in {base_path} could not be found.")

        # Set initial values from constructor
        self._chunking_method = chunking_method
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._batch_size = batch_size
        self.doc_id_pattern = doc_id_pattern

        # Metadata schema
        if isinstance(metadata_schema, str):
            self._metadata_schema = get_common_metadata_schemas(metadata_schema)
        else:
            self._metadata_schema = metadata_schema or {}

        # Chunker - create with initial values (might be overridden later)
        self.chunker = ChunkerFactory.create_chunker(self._chunking_method, self._chunk_size, self._chunk_overlap)

        # Embedding provider - create with initial values (might be overridden later)
        embedding_config = embedding_config or {}
        self._embedding_provider = EmbeddingRegistry.create_provider(
            embedding_provider, embedding_model, **embedding_config
        )
        if not self._embedding_provider.validate_model():
            raise ValueError(f"Embedding model '{embedding_model}' is not available")
        self._embedding_dimension = self._embedding_provider.get_dimension()

        # Threading
        self._read_write_lock: ReadWriteLock = ReadWriteLock()
        # ReadWrite lock for FAISS operations to allow concurrent reads
        self._faiss_lock: ReadWriteLock = ReadWriteLock()

        # Initialize SQLite tuning configuration
        profile = get_sqlite_pragma_profile(sqlite_profile, default="balanced")
        if profile is None:
            raise ValueError(f"Unknown SQLite pragma profile: {sqlite_profile!r}")
        pragmas = dict(profile.pragmas)
        if sqlite_pragma_overrides:
            pragmas.update(sqlite_pragma_overrides)

        self._sqlite_profile = sqlite_profile
        self._sqlite_pragma_overrides: dict = sqlite_pragma_overrides or {}
        self._sqlite_pragmas: dict = pragmas

        # Database schema + pools (with tuning pragmas)
        self.schema = DatabaseSchema(self.db_path, self._read_write_lock)
        self.connection_pool = ConnectionPool(self.db_path, connection_pool_size, pragmas=self._sqlite_pragmas)
        self.async_connection_pool: Optional[AsyncConnectionPool] = None
        self.async_max_connections = connection_pool_size or 10
        self._async_schema_initialized = False

        with self.connection_pool.get_connection() as conn:
            self.schema.initialize(self._metadata_schema, db_connection=conn)

            # Load configuration from existing database
            is_existing_db = not self.is_memory_only and Path(self.db_path).exists()
            if is_existing_db:
                existing_schema = self.schema.load_metadata_schema(db_connection=conn)
                self._metadata_schema.update(existing_schema)

                # Load and apply saved configuration
                loaded_config = self._load_config(conn)
                if loaded_config:
                    # Override constructor values with saved configuration
                    embedding_provider = loaded_config.get("embedding_provider", embedding_provider)
                    embedding_model = loaded_config.get("embedding_model", embedding_model)
                    self._chunking_method = loaded_config.get("chunking_method", self._chunking_method)
                    self._chunk_size = int(loaded_config.get("chunk_size", self._chunk_size))
                    self._chunk_overlap = int(loaded_config.get("chunk_overlap", self._chunk_overlap))
                    self._batch_size = int(loaded_config.get("batch_size", self._batch_size))
                    self.doc_id_pattern = loaded_config.get("doc_id_pattern", self.doc_id_pattern)

                    # Load SQLite tuning configuration
                    self._load_sqlite_tuning(loaded_config)
                    # Update connection pool with loaded pragma settings
                    self.connection_pool._pragmas = self._sqlite_pragmas

                    # Re-create chunker with loaded configuration
                    self.chunker = ChunkerFactory.create_chunker(
                        self._chunking_method, self._chunk_size, self._chunk_overlap
                    )

                    # Re-create embedding provider with loaded configuration
                    embedding_config = embedding_config or {}
                    self._embedding_provider = EmbeddingRegistry.create_provider(
                        embedding_provider, embedding_model, **embedding_config
                    )
                    if not self._embedding_provider.validate_model():
                        raise ValueError(f"Embedding model '{embedding_model}' is not available")
                    self._embedding_dimension = self._embedding_provider.get_dimension()

        # FTS
        self._fts_enabled = False
        if enable_fts:
            self._init_fts()

        # FAISS
        self._init_faiss_index(enable_gpu, faiss_index_type, faiss_index_hnsw_flat_neighbors, faiss_index_lsh_bits)

        # Hierarchical embeddings: section and document FAISS indices
        self._init_hierarchical(hierarchical_embeddings, section_pattern, section_metadata_extractors)

        # How many items allowed on the processing queues.
        self.pipeline_queue_size: int = 3

        # Timeout for joining pipeline worker threads (in seconds)
        self.pipeline_worker_timeout: float = pipeline_worker_timeout

        # State
        self._next_doc_id = self._load_next_doc_id()
        # Single lock for ID generation to prevent race conditions between sync and async paths.
        # Using threading.Lock for both since the critical section is minimal (integer increment).
        self._id_lock: threading.Lock = threading.Lock()

        # Save config including SQLite tuning
        self._save_config()
        self._save_sqlite_tuning()

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
        """Overlap between chunks, in the unit of ``chunking_method`` (not tokens
        unless the method is ``"tokens"``)."""
        return self._chunk_overlap

    @property
    def batch_size(self) -> int:
        return self._batch_size

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
    def metadata_schema(self) -> Dict[str, MetadataField]:
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
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                        id,
                        content,
                        content='documents',
                        content_rowid='rowid'
                    )
                    """)
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        document_id,
                        content,
                        content='chunks',
                        content_rowid='id'
                    )
                    """)
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
                raise DatabaseError(f"Error loading faiss index: {str(e)}") from e
            if hasattr(loaded_index, "id_map"):
                self.index = loaded_index
                logger.info(f"Loaded existing FAISS IndexIDMap2 with {self.index.ntotal} vectors")
            else:
                raise DatabaseError("Expected FAISS index to have `id_map` attribute. Invalid faiss index!")
        else:
            base_index: faiss.Index
            if faiss_index_type == "IndexFlatL2":
                base_index = faiss.IndexFlatL2(self.embedding_dimension)
            elif faiss_index_type == "IndexFlatIP":
                base_index = faiss.IndexFlatIP(self.embedding_dimension)
            elif faiss_index_type == "IndexHNSWFlat":
                base_index = faiss.IndexHNSWFlat(self.embedding_dimension, faiss_index_hnsw_flat_neighbors or 16)
            elif faiss_index_type == "IndexLSH":
                base_index = faiss.IndexLSH(
                    self.embedding_dimension, faiss_index_lsh_bits or self.embedding_dimension * 2
                )
            else:
                raise ValueError(
                    "Invalid faiss index for LocalVectorDB. "
                    "Must be one of: IndexFlatL2, IndexFlatIP, IndexHNSWFlat, IndexLSH"
                )
            self.index = faiss.IndexIDMap2(base_index)
            logger.info(f"Created new FAISS IndexIDMap2 with dimension {self.embedding_dimension}")
        if enable_gpu:
            try:
                # Check if GPU methods are available (guards against faiss-cpu builds)
                num_gpus = faiss.get_num_gpus()
                if num_gpus > 0 and self.index is not None:
                    try:
                        self.index = faiss.index_cpu_to_all_gpus(self.index)
                        logger.info("Moved FAISS index to GPU")
                    except Exception as e:
                        logger.warning(f"Could not move IndexIDMap2 to GPU: {e}")
                else:
                    logger.warning("GPU requested but no GPUs available")
            except AttributeError:
                logger.warning(
                    "GPU requested but FAISS was compiled without GPU support (faiss-cpu). "
                    "Install faiss-gpu for GPU acceleration or set enable_gpu=False."
                )
            except Exception as e:
                logger.warning(f"Failed to check GPU availability: {e}. Falling back to CPU.")

    # Hierarchical FAISS index initialization and management
    def _init_hierarchical(
        self,
        hierarchical_embeddings: bool,
        section_pattern: str,
        section_metadata_extractors: Optional[List] = None,
    ) -> None:
        """Initialize hierarchical embedding indices if enabled."""
        # Load from saved config if available
        with self.connection_pool.get_connection() as conn:
            loaded = self._load_config(conn)
            if loaded:
                saved_hier = loaded.get("hierarchical_embeddings", "")
                if saved_hier.lower() == "true":
                    self._hierarchical_embeddings = True
                    self._section_pattern = loaded.get("section_pattern", section_pattern)

        if not self._hierarchical_embeddings:
            return

        self._section_detector = SectionDetector(self._section_pattern)
        self._section_metadata_extractors = resolve_extractors(section_metadata_extractors)

        # Set up index paths
        if self.base_path is not None:
            self.section_index_path = self.base_path / f"{self.name}_sections.faiss"
            self.document_index_path = self.base_path / f"{self.name}_documents.faiss"

        # Load or create section index
        if self.section_index_path and self.section_index_path.exists():
            try:
                self.section_index = faiss.read_index(str(self.section_index_path))
                logger.info(f"Loaded section FAISS index with {self.section_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load section FAISS index: {e}, creating new")
                self.section_index = self._create_flat_index()
        else:
            self.section_index = self._create_flat_index()
            logger.info("Created new section FAISS index")

        # Load or create document index
        if self.document_index_path and self.document_index_path.exists():
            try:
                self.document_index = faiss.read_index(str(self.document_index_path))
                logger.info(f"Loaded document FAISS index with {self.document_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load document FAISS index: {e}, creating new")
                self.document_index = self._create_flat_index()
        else:
            self.document_index = self._create_flat_index()
            logger.info("Created new document FAISS index")

    def _create_flat_index(self) -> faiss.IndexIDMap2:
        """Create a new IndexFlatL2 wrapped in IndexIDMap2."""
        base = faiss.IndexFlatL2(self.embedding_dimension)
        return faiss.IndexIDMap2(base)

    @property
    def hierarchical_embeddings(self) -> bool:
        return self._hierarchical_embeddings

    def _add_vectors_to_section_index(self, embeddings: np.ndarray, faiss_ids: np.ndarray) -> None:
        """Add vectors to the section FAISS index."""
        if self.section_index is None or len(embeddings) == 0:
            return
        with self._faiss_lock.write_lock():
            self.section_index.add_with_ids(embeddings.astype(np.float32), faiss_ids.astype(np.int64))

    def _add_vectors_to_document_index(self, embeddings: np.ndarray, faiss_ids: np.ndarray) -> None:
        """Add vectors to the document FAISS index."""
        if self.document_index is None or len(embeddings) == 0:
            return
        with self._faiss_lock.write_lock():
            self.document_index.add_with_ids(embeddings.astype(np.float32), faiss_ids.astype(np.int64))

    def _remove_section_vectors(self, faiss_ids: List[int]) -> None:
        """Remove vectors from the section FAISS index."""
        if not faiss_ids or self.section_index is None or not hasattr(self.section_index, "remove_ids"):
            return
        try:
            with self._faiss_lock.write_lock():
                ids_array = np.array(faiss_ids, dtype=np.int64)
                # faiss accepts an ndarray of ids here (wrapped internally as an
                # IDSelectorBatch); the stub only types the IDSelector overload.
                self.section_index.remove_ids(ids_array)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Failed to remove section vectors from FAISS: {e}")

    def _remove_document_vectors(self, faiss_ids: List[int]) -> None:
        """Remove vectors from the document FAISS index."""
        if not faiss_ids or self.document_index is None or not hasattr(self.document_index, "remove_ids"):
            return
        try:
            with self._faiss_lock.write_lock():
                ids_array = np.array(faiss_ids, dtype=np.int64)
                # faiss accepts an ndarray of ids here (wrapped internally as an
                # IDSelectorBatch); the stub only types the IDSelector overload.
                self.document_index.remove_ids(ids_array)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Failed to remove document vectors from FAISS: {e}")

    def _add_vectors_to_faiss_bulk(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(embeddings) == 0:
            return
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        with self._faiss_lock.write_lock():
            start_faiss_id = self.index.ntotal
            new_faiss_ids = np.arange(start_faiss_id, start_faiss_id + len(embeddings), dtype=np.int64)
            self.index.add_with_ids(embeddings, new_faiss_ids)
            for i, chunk in enumerate(chunks):
                chunk.faiss_id = int(new_faiss_ids[i])

    def _remove_old_vectors_bulk(self, faiss_ids: List[int]) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        if not faiss_ids or not hasattr(self.index, "remove_ids"):
            return
        try:
            with self._faiss_lock.write_lock():
                ids_array = np.array(faiss_ids, dtype=np.int64)
                # faiss accepts an ndarray of ids here (wrapped internally as an
                # IDSelectorBatch); the stub only types the IDSelector overload.
                self.index.remove_ids(ids_array)  # type: ignore[arg-type]
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

        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        with self._faiss_lock.read_lock():
            # Method 1: Try reconstruct_batch if available (for non-wrapped indices)
            if hasattr(self.index, "reconstruct_batch"):
                try:
                    faiss_ids_array = np.array(faiss_ids, dtype=np.int64)
                    result: np.ndarray = self.index.reconstruct_batch(faiss_ids_array)
                    return result
                except Exception as e:
                    logger.warning(f"FAISS reconstruct_batch failed, falling back to individual calls: {e}")

            # Check if we have an IndexIDMap/IndexIDMap2 wrapper
            if hasattr(self.index, "id_map") and hasattr(self.index, "index"):
                try:
                    # Method 2: Use proper ID mapping for IndexIDMap2 (safest approach)
                    return self._reconstruct_with_id_mapping(faiss_ids)
                except Exception as e:
                    logger.warning(f"IndexIDMap reconstruction failed, falling back to individual calls: {e}")
                    return self._reconstruct_individual_fallback(faiss_ids)

            # Method 3: Fallback to individual reconstruct calls
            return self._reconstruct_individual_fallback(faiss_ids)

    def _reconstruct_with_id_mapping(self, faiss_ids: List[int]) -> np.ndarray:
        """Reconstruct embeddings for IndexIDMap2 using optimized mapping strategies."""
        if not faiss_ids:
            return np.array([]).reshape(0, self.embedding_dimension)

        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        faiss_ids_array = np.array(faiss_ids, dtype=np.int64)

        # Strategy 1: Try direct reconstruction on IndexIDMap2 first
        # For some index types, we can reconstruct directly using external IDs
        try:
            if hasattr(self.index, "reconstruct"):
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
                    logger.debug(
                        f"Successfully reconstructed {len(embeddings)} embeddings "
                        f"using direct IndexIDMap2 reconstruct"
                    )
                    return result
        except Exception as e:
            logger.debug(f"Direct IndexIDMap2 reconstruction not available: {e}")

        # Strategy 2: Efficient mapping using internal FAISS methods (if available)
        try:
            # Check if FAISS provides an efficient way to get internal indices
            # `.index`/`.id_map` are IndexIDMap2-specific attributes, guarded here by
            # hasattr/runtime checks; the widened faiss.Index base type does not expose them.
            if hasattr(self.index, "get_ids") and hasattr(self.index.index, "reconstruct_batch"):  # type: ignore[attr-defined]
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
                    batch_result: np.ndarray = self.index.index.reconstruct_batch(internal_indices_array)  # type: ignore[attr-defined]
                    logger.debug(
                        f"Successfully reconstructed {len(batch_result)} embeddings using efficient ID mapping"
                    )
                    return batch_result
        except Exception as e:
            logger.debug(f"Efficient ID mapping strategy failed: {e}")

        # Strategy 3: Fallback to standardized ID mapping using faiss.vector_to_array
        # Build a reverse mapping dictionary first for better performance

        # Build id_map lookup once for all IDs using centralized utilities
        try:
            id_map_lookup = build_id_lookup(self.index)
        except Exception as e:
            logger.debug(f"Failed to get ID mapping via centralized utilities: {e}")
            # Fallback to individual access if centralized method fails
            id_map_lookup = {}
            for i in range(self.index.ntotal):
                try:
                    external_id = self.index.id_map.at(i)  # type: ignore[attr-defined]
                    id_map_lookup[external_id] = i
                except Exception:
                    continue

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
        base_index = self.index.index  # type: ignore[attr-defined]

        if hasattr(base_index, "reconstruct_batch"):
            base_result: np.ndarray = base_index.reconstruct_batch(internal_indices_array)
            logger.debug(f"Successfully reconstructed {len(base_result)} embeddings using optimized ID mapping")
            return base_result
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

    def _reconstruct_individual_fallback(self, faiss_ids: List[int]) -> np.ndarray:
        """Fallback method using individual reconstruct calls."""
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
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
            cursor = conn.execute("SELECT value FROM config WHERE key = ?", ("next_doc_id",))
            row = cursor.fetchone()
            return int(row["value"]) if row else 1

    def _save_next_doc_id(self) -> None:
        # Idempotent single-statement write; retry on transient shared-cache locks
        # so concurrent async upserts don't fail after their document already committed.
        def _write() -> None:
            with self.connection_pool.get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    ("next_doc_id", str(self._next_doc_id)),
                )
                conn.commit()

        retry_on_locked(_write)

    def _save_config(self):
        config = {
            "embedding_provider": self.embedding_provider.provider_name,
            "embedding_model": self.embedding_provider.model,
            "embedding_dimension": self.embedding_dimension,
            "chunking_method": self.chunking_method,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "batch_size": self.batch_size,
            "doc_id_pattern": self.doc_id_pattern,
            "fts_enabled": str(self.fts_enabled),
            "version": get_system_version(),
            "hierarchical_embeddings": str(self._hierarchical_embeddings),
            "section_pattern": self._section_pattern,
        }
        with self.connection_pool.get_connection() as conn:
            for key, value in config.items():
                conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, str(value)))
            conn.commit()

    def _load_config(self, conn: sqlite3.Connection) -> Dict[str, str]:
        """
        Load configuration from the config table.

        Returns
        -------
        Dict[str, str]
            Dictionary of configuration key-value pairs
        """
        try:
            cursor = conn.execute("SELECT key, value FROM config")
            config = {row[0]: row[1] for row in cursor.fetchall()}
            return config
        except sqlite3.OperationalError:
            # Config table doesn't exist (shouldn't happen for properly initialized DB)
            logger.warning("Config table not found in existing database")
            return {}

    def _get_faiss_metric_type(self) -> str:
        """Detect the metric type of the FAISS index.

        The result is cached against the current index object: it cannot change
        for a given index, but a rebuild swaps in a new object (different
        identity), which transparently invalidates the cache. This matters
        because the method was previously called once per search candidate
        (tens of thousands of times per query in profiling).

        Returns:
            'L2' for L2 distance metrics
            'IP' for inner product metrics
        """
        idx = getattr(self, "index", None)
        cached = self._metric_type_cache
        if cached is not None and cached[0] is idx:
            return cached[1]

        result = self._detect_faiss_metric_type(idx)
        self._metric_type_cache = (idx, result)
        return result

    @staticmethod
    def _detect_faiss_metric_type(index: Any) -> str:
        """Inspect a FAISS index object and return its metric type ('L2'/'IP')."""
        if index is None:
            return "L2"  # Default

        # Check if it's an IndexIDMap wrapper
        if hasattr(index, "index"):
            base_index = index.index
        else:
            base_index = index

        # Check the index type name
        index_type = str(type(base_index).__name__)
        if "IP" in index_type or "InnerProduct" in index_type:
            return "IP"
        elif "L2" in index_type:
            return "L2"
        elif "HNSW" in index_type:
            # HNSW can use different metrics, check if it's HNSW with IP
            if hasattr(base_index, "metric_type"):
                if base_index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    return "IP"
            return "L2"  # Default for HNSW
        else:
            # LSH and others default to L2-like behavior
            return "L2"

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

        if metric_type == "IP":
            # For inner product, the distance IS the similarity (if normalized)
            # Assuming normalized embeddings, IP ranges from -1 to 1
            # Convert to 0-1 range
            return max(0.0, min(1.0, (distance + 1.0) / 2.0))
        else:
            # L2 distance: convert using 1/(1+distance)
            return 1.0 / (1.0 + distance)

    def _distances_to_similarities(self, distances: "np.ndarray", metric_type: Optional[str] = None) -> "np.ndarray":
        """Vectorized form of :meth:`_distance_to_similarity` over a distance array.

        Computing the conversion for a whole FAISS result row at once avoids a
        Python-level call per candidate (tens of thousands per query at scale).
        Returns a float array of the same shape as ``distances``.
        """
        if metric_type is None:
            metric_type = self._get_faiss_metric_type()
        if metric_type == "IP":
            return np.clip((distances + 1.0) / 2.0, 0.0, 1.0)
        return 1.0 / (1.0 + distances)

    def _similarity_to_distance(self, similarity: float, metric_type: Optional[str] = None) -> float:
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

        if metric_type == "IP":
            # Convert from 0-1 to -1 to 1 range
            # Higher similarity means higher inner product
            return similarity * 2.0 - 1.0
        else:
            # L2 distance: invert the formula
            return (1.0 / similarity) - 1.0

    def _generate_doc_id(self) -> str:
        """Generate a unique document ID with thread-safe locking."""
        with self._id_lock:
            doc_id = self.doc_id_pattern.format(idx=self._next_doc_id)
            self._next_doc_id += 1
            return doc_id

    async def _generate_doc_id_async(self) -> str:
        """
        Async version of _generate_doc_id.

        Uses the same threading.Lock as the sync version to prevent race conditions
        between concurrent sync and async ID generation. The lock is held only for
        an integer increment, so blocking the event loop is negligible.
        """
        with self._id_lock:
            doc_id = self.doc_id_pattern.format(idx=self._next_doc_id)
            self._next_doc_id += 1
            return doc_id

    # Persistence
    def _save_internal(self):
        if not self.is_memory_only and self.index is not None and self.index.ntotal > 0:
            if hasattr(self.index, "index") and hasattr(self.index.index, "device"):
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, str(self.index_path))
                except AttributeError:
                    logger.warning("GPU-to-CPU conversion failed - FAISS may not have GPU support")
                    faiss.write_index(self.index, str(self.index_path))
                except Exception as e:
                    logger.warning(f"Failed to convert GPU index to CPU for saving: {e}")
                    faiss.write_index(self.index, str(self.index_path))
            else:
                faiss.write_index(self.index, str(self.index_path))

        # Save hierarchical FAISS indices
        if self._hierarchical_embeddings and not self.is_memory_only:
            if self.section_index is not None and self.section_index.ntotal > 0 and self.section_index_path:
                faiss.write_index(self.section_index, str(self.section_index_path))
            if self.document_index is not None and self.document_index.ntotal > 0 and self.document_index_path:
                faiss.write_index(self.document_index, str(self.document_index_path))

    def save(self):
        with self._read_write_lock.write_lock():
            self._save_internal()

    def close(self):
        """Close the database"""
        self.save()
        self.connection_pool.close_all()
        if hasattr(self, "async_connection_pool") and self.async_connection_pool is not None:
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
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            section_count = conn.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            index_size = self.index.ntotal if self.index is not None else 0
            stats = {
                "documents": doc_count,
                "chunks": chunk_count,
                "sections": section_count,
                "index_vectors": index_size,
                "embedding_dimension": self.embedding_dimension,
                "embedding_provider": self.embedding_provider.provider_name,
                "embedding_model": self.embedding_provider.model,
                "chunking_method": self.chunking_method,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "fts_enabled": self.fts_enabled,
                "hierarchical_embeddings": self._hierarchical_embeddings,
            }
            if self._hierarchical_embeddings:
                stats["section_index_vectors"] = self.section_index.ntotal if self.section_index else 0
                stats["document_index_vectors"] = self.document_index.ntotal if self.document_index else 0
            return stats

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
            return row["count"] if row else 0

    # Async helpers
    def _ensure_async_pool(self) -> None:
        if self.async_connection_pool is None:
            self.async_connection_pool = AsyncConnectionPool(
                self.db_path, max_connections=self.async_max_connections, pragmas=self._sqlite_pragmas
            )

    async def _ensure_async_schema_initialized(self) -> None:
        if not self.is_memory_only or self._async_schema_initialized:
            return
        if self.async_connection_pool is None:
            raise RuntimeError("Async connection pool is not initialized")
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

    async def get_stats_async(self) -> Dict[str, Any]:
        """Async twin of :meth:`get_stats` (database statistics).

        Mirrors ``RemoteVectorDB.get_stats_async`` so ``await db.get_stats_async()``
        works against either backend. Statistics collection is lightweight sync
        SQLite work, so this delegates to the sync path.
        """
        return self.get_stats()

    async def get_async_pool_stats(self) -> Dict[str, Any]:
        """Get async connection-pool statistics.

        Renamed from ``get_async_stats`` for v0.1.0: the old name collided with
        (and read as) the async twin of ``get_stats`` while actually returning
        pool internals. This reports the async pool only.
        """
        stats = {}
        if self.async_connection_pool:
            stats["async_pool"] = self.async_connection_pool.stats
        else:
            stats["async_pool"] = {"status": "not_initialized"}
        return stats

    async def __aenter__(self):
        self._ensure_async_pool()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_async()

    # SQLite Tuning Methods
    def _save_sqlite_tuning(self) -> None:
        """Save SQLite tuning configuration to database."""
        with self.connection_pool.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", ("sqlite_profile", self._sqlite_profile)
            )
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                ("sqlite_pragma_overrides", json.dumps(self._sqlite_pragma_overrides)),
            )
            conn.commit()

    def _load_sqlite_tuning(self, config: Dict[str, str]) -> None:
        """Load SQLite tuning configuration from database config."""
        profile: str = config.get("sqlite_profile", "balanced")
        overrides_json = config.get("sqlite_pragma_overrides", "{}")

        try:
            overrides = json.loads(overrides_json) if overrides_json else {}
        except (json.JSONDecodeError, TypeError):
            overrides = {}

        from typing import cast

        profile_typed = cast(SqliteProfile, profile)
        if is_valid_sqlite_pragma_profile(profile_typed):
            profile_obj = get_sqlite_pragma_profile(profile_typed)
            if profile_obj is None:
                raise ValueError(f"Unknown SQLite pragma profile: {profile_typed!r}")
            pragmas = dict(profile_obj.pragmas)
            pragmas.update(overrides)

            self._sqlite_profile = profile_typed
            self._sqlite_pragma_overrides = overrides
            self._sqlite_pragmas = pragmas

            logger.debug(f"Loaded SQLite tuning profile '{profile}' with {len(overrides)} overrides")
        else:
            logger.warning(f"Unknown saved SQLite profile '{profile}', using balanced")
            self._sqlite_profile = "balanced"
            self._sqlite_pragma_overrides = {}
            balanced_profile = get_sqlite_pragma_profile("balanced")
            if balanced_profile is None:
                raise RuntimeError("Built-in 'balanced' SQLite pragma profile is unavailable")
            self._sqlite_pragmas = dict(balanced_profile.pragmas)
