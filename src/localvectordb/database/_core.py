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
import contextlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
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
from localvectordb.exceptions import (
    DatabaseError,
    DatabaseNotFoundError,
    IndexIntegrityError,
    UnsupportedIndexOperationError,
)
from localvectordb.section_detection import SectionDetector
from localvectordb.section_metadata import SectionMetadataExtractor, resolve_extractors
from localvectordb.sqlite_tuning import SqliteProfile, get_sqlite_pragma_profile, is_valid_sqlite_pragma_profile
from localvectordb.utils import get_system_version

logger = logging.getLogger(__name__)

# Bounded retry for os.replace when persisting the index. On Windows the rename can
# fail with PermissionError even when our own writers are correctly serialized, because
# an external process (virus scanner, search indexer) or a concurrent *reader* of the
# index file may hold a transient handle. See _replace_with_retry.
_REPLACE_MAX_RETRIES = 10
_REPLACE_BASE_DELAY = 0.02  # seconds
_REPLACE_MAX_DELAY = 0.5  # seconds (cap per backoff)


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
        mmap_index: bool = False,
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
        # Internal: bypass the on-open integrity check. Only ``repair`` sets this,
        # because it must open exactly the databases the check refuses.
        _skip_integrity_check: bool = False,
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
        # Guards the monotonic FAISS id counters only. Always acquired *before*
        # _faiss_lock so the two are taken in a consistent order everywhere.
        self._faiss_id_lock: threading.Lock = threading.Lock()
        self._faiss_id_counters: Dict[str, int] = {}

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

        # Read-only, memory-mapped index. An mmap'd FAISS index shares one copy of
        # the file across processes via the OS page cache (many read-only workers) and
        # cannot be mutated in place -- so a database opened this way refuses writes.
        # Route writes to a single writer process opened with mmap_index=False.
        self._mmap_index = mmap_index
        # Whether the in-RAM index has diverged from the on-disk file. A clean database
        # skips the rewrite (and its os.replace) on save()/close(), so idle-eviction or
        # shutdown of a read-only worker never touches -- or races on -- the file.
        self._index_dirty = False

        # FTS
        self._fts_enabled = False
        if enable_fts:
            self._init_fts()

        # FAISS
        self._init_faiss_index(enable_gpu, faiss_index_type, faiss_index_hnsw_flat_neighbors, faiss_index_lsh_bits)

        # Hierarchical embeddings: section and document FAISS indices
        self._init_hierarchical(hierarchical_embeddings, section_pattern, section_metadata_extractors)

        # FAISS id allocation and dual-store integrity. Seeding must precede the
        # integrity check: the counter floor is a max() and is safe to compute even
        # on a corrupt database, which means no *new* collisions can be issued from
        # this point on, whether or not the existing ones get repaired.
        with self.connection_pool.get_connection() as conn:
            self._seed_faiss_counters(conn)
            if not _skip_integrity_check:
                self._verify_integrity(conn)

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
                loaded_index = faiss.read_index(str(self.index_path), self._faiss_read_flags())
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
            # A freshly created (empty) index has no on-disk file yet; mark dirty so the
            # first save() persists it rather than short-circuiting on the clean flag.
            self._index_dirty = True
            logger.info(f"Created new FAISS IndexIDMap2 with dimension {self.embedding_dimension}")
        # Whether this index type can remove vectors. IndexHNSWFlat cannot: faiss
        # raises from remove_ids. Detected from the *concrete* base index so it is
        # correct for indices loaded from disk as well as freshly constructed ones.
        self.supports_deletion = self._index_supports_deletion(self.index)
        # Whether ``index.search`` accepts a ``SearchParameters(sel=...)`` id
        # selector, used to push a metadata filter into FAISS instead of
        # post-filtering in Python. IndexLSH rejects search params ("search
        # params not supported for this index"); the other three accept them.
        # Detected from the concrete base index so it is right for loaded indices.
        self.supports_id_selector = self._index_supports_id_selector(self.index)
        self.base_index_type = self._base_index_type_name(self.index)

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

    @staticmethod
    def _unwrap_base_index(index) -> Any:
        """Return the concrete index wrapped by an IndexIDMap/IndexIDMap2."""
        base = getattr(index, "index", index)
        # downcast_index is a SWIG shim that unwraps proxies by walking `obj.this`
        # until it reaches a SwigPyObject. Handed a non-Index it can loop forever
        # rather than raise, so the except below would never fire. Skip the check
        # when faiss itself is a test double -- then downcast_index is one too.
        index_cls = getattr(faiss, "Index", None)
        if isinstance(index_cls, type) and not isinstance(base, index_cls):
            return base
        try:
            return faiss.downcast_index(base)
        except Exception:  # pragma: no cover - downcast fails only on already-concrete types
            return base

    @classmethod
    def _index_supports_deletion(cls, index) -> bool:
        """
        Whether ``remove_ids`` will succeed on this index.

        Note ``hasattr(index, "remove_ids")`` is useless here: IndexIDMap2 always
        exposes the method, and delegates to a base index that may not implement it.
        """
        if index is None:
            return False
        return not isinstance(cls._unwrap_base_index(index), faiss.IndexHNSW)

    @classmethod
    def _index_supports_id_selector(cls, index) -> bool:
        """
        Whether ``index.search(..., params=SearchParameters(sel=...))`` will work.

        IndexLSH raises "search params not supported for this index" from its
        ``search``; IndexFlatL2, IndexFlatIP and IndexHNSWFlat all accept a
        selector through the IndexIDMap2 wrapper (verified against faiss 1.14.2).
        Databases whose index cannot take a selector fall back to over-fetching
        and post-filtering in Python.
        """
        if index is None:
            return False
        return not isinstance(cls._unwrap_base_index(index), faiss.IndexLSH)

    @classmethod
    def _base_index_type_name(cls, index) -> str:
        """Concrete base index class name, e.g. ``IndexFlatL2``. Used by ``repair``."""
        if index is None:
            return "IndexFlatL2"
        return type(cls._unwrap_base_index(index)).__name__

    def _faiss_read_flags(self) -> int:
        """
        IO flags for ``faiss.read_index``.

        Memory-map the file (``IO_FLAG_MMAP``) when ``mmap_index`` is set, so many
        read-only workers share one copy through the OS page cache instead of each
        loading a private, RAM-resident float32 copy. An mmap'd index is read-only.
        """
        return faiss.IO_FLAG_MMAP if self._mmap_index else 0

    def _require_writable(self, operation: str) -> None:
        """
        Refuse writes to a memory-mapped (read-only) database.

        A FAISS index opened with ``IO_FLAG_MMAP`` cannot be mutated in place, so
        every vector-mutating path funnels through this guard. Refusing here (rather
        than letting faiss fail unpredictably) leaves at worst an orphan vector, never
        a dangling row -- consistent with the dual-store governing rule.
        """
        if self._mmap_index:
            raise UnsupportedIndexOperationError(
                f"{operation} is not supported on a memory-mapped database. mmap_index=True "
                f"opens the FAISS index read-only for shared multi-worker reads, and an mmap'd "
                f"index cannot be mutated in place. Route writes to a single writer process "
                f"opened with mmap_index=False."
            )

    def _require_deletable(self, operation: str) -> None:
        """Refuse operations that would silently fail to remove vectors."""
        if not self.supports_deletion:
            raise UnsupportedIndexOperationError(
                f"{operation} requires removing vectors, which {self.base_index_type} does not support. "
                f"Vectors would be orphaned and the deletion silently lost. "
                f"Rebuild the database with faiss_index_type='IndexFlatL2' (or 'IndexFlatIP' / 'IndexLSH') "
                f"if you need to delete or replace documents; {self.base_index_type} is append-only."
            )

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
                self.section_index = faiss.read_index(str(self.section_index_path), self._faiss_read_flags())
                logger.info(f"Loaded section FAISS index with {self.section_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load section FAISS index: {e}, creating new")
                self.section_index = self._create_flat_index()
                self._index_dirty = True
        else:
            self.section_index = self._create_flat_index()
            self._index_dirty = True
            logger.info("Created new section FAISS index")

        # Load or create document index
        if self.document_index_path and self.document_index_path.exists():
            try:
                self.document_index = faiss.read_index(str(self.document_index_path), self._faiss_read_flags())
                logger.info(f"Loaded document FAISS index with {self.document_index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load document FAISS index: {e}, creating new")
                self.document_index = self._create_flat_index()
                self._index_dirty = True
        else:
            self.document_index = self._create_flat_index()
            self._index_dirty = True
            logger.info("Created new document FAISS index")

    def _create_flat_index(self) -> faiss.IndexIDMap2:
        """Create a new IndexFlatL2 wrapped in IndexIDMap2."""
        base = faiss.IndexFlatL2(self.embedding_dimension)
        return faiss.IndexIDMap2(base)

    @property
    def hierarchical_embeddings(self) -> bool:
        return self._hierarchical_embeddings

    def _add_vectors_to_section_index(self, embeddings: np.ndarray, faiss_ids: np.ndarray) -> None:
        """Add section centroid vectors to the section FAISS index."""
        if self.section_index is None or len(embeddings) == 0:
            return
        self._require_writable("Adding section vectors")
        embeddings = self._unit_normalize_centroids(embeddings)
        embeddings = self._normalize_for_index(embeddings, self.section_index)
        with self._faiss_lock.write_lock():
            self.section_index.add_with_ids(embeddings, faiss_ids.astype(np.int64))
        self._index_dirty = True

    def _add_vectors_to_document_index(self, embeddings: np.ndarray, faiss_ids: np.ndarray) -> None:
        """Add document centroid vectors to the document FAISS index."""
        if self.document_index is None or len(embeddings) == 0:
            return
        self._require_writable("Adding document vectors")
        embeddings = self._unit_normalize_centroids(embeddings)
        embeddings = self._normalize_for_index(embeddings, self.document_index)
        with self._faiss_lock.write_lock():
            self.document_index.add_with_ids(embeddings, faiss_ids.astype(np.int64))
        self._index_dirty = True

    def _remove_vectors(self, index, faiss_ids: List[int], what: str) -> None:
        """
        Remove vectors from an index, or raise.

        A swallowed failure here is how deletes came to silently not happen: the row
        disappears from SQLite, the vector survives, and the caller is told it worked.
        Callers gate on ``supports_deletion`` first, so reaching the raise means
        something genuinely unexpected happened.
        """
        if not faiss_ids or index is None:
            return
        self._require_writable(f"Removing {what} vectors")
        try:
            with self._faiss_lock.write_lock():
                ids_array = np.array(faiss_ids, dtype=np.int64)
                # faiss accepts an ndarray of ids here, wrapped internally as an
                # IDSelectorBatch.
                index.remove_ids(ids_array)
                logger.debug(f"Removed {len(faiss_ids)} {what} vectors from FAISS")
            self._index_dirty = True
        except Exception as e:
            raise IndexIntegrityError(
                f"Failed to remove {len(faiss_ids)} {what} vector(s) from the FAISS index: {e}. "
                f"SQLite and FAISS may now disagree; run `lvdb db {self.name} repair`."
            ) from e

    def _remove_section_vectors(self, faiss_ids: List[int]) -> None:
        """Remove vectors from the section FAISS index."""
        self._remove_vectors(self.section_index, faiss_ids, "section")

    def _remove_document_vectors(self, faiss_ids: List[int]) -> None:
        """Remove vectors from the document FAISS index."""
        self._remove_vectors(self.document_index, faiss_ids, "document")

    def _add_vectors_to_faiss_bulk(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if len(embeddings) == 0:
            return
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        self._require_writable("Adding documents")
        new_faiss_ids = self._allocate_faiss_ids("main", len(embeddings))
        embeddings = self._normalize_for_index(embeddings, self.index)
        with self._faiss_lock.write_lock():
            self.index.add_with_ids(embeddings, new_faiss_ids)
            for i, chunk in enumerate(chunks):
                chunk.faiss_id = int(new_faiss_ids[i])
        self._index_dirty = True

    def _remove_old_vectors_bulk(self, faiss_ids: List[int]) -> None:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        if not faiss_ids:
            return
        self._require_deletable("Replacing or deleting a document")
        self._remove_vectors(self.index, faiss_ids, "chunk")

    def _discard_faiss_ids_best_effort(self, faiss_ids: List[int]) -> None:
        """
        Undo an in-RAM FAISS add whose SQLite transaction rolled back.

        Ids are monotonic, so a failure here wastes ids and leaves orphan vectors --
        it can never cause a collision. Orphans cost recall and are swept by ``repair``,
        so this must not mask the original exception.
        """
        if not faiss_ids or self.index is None or not self.supports_deletion:
            return
        try:
            with self._faiss_lock.write_lock():
                self.index.remove_ids(np.array(faiss_ids, dtype=np.int64))  # type: ignore[arg-type]
            self._index_dirty = True
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Could not roll back {len(faiss_ids)} FAISS vector(s) after a failed transaction: {e}")

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

    # ------------------
    # Monotonic FAISS id allocation
    # ------------------
    #
    # FAISS ids were once derived from ``index.ntotal``. That is wrong: ``remove_ids``
    # *decrements* ntotal, so it is not a high-water mark, and any delete or replacing
    # upsert re-issued ids that were still live. A duplicated id makes the hydration
    # join return two chunk rows for one vector, so two documents get scored from a
    # single distance -- and the row inserted later wins, hiding the correct document
    # from its own query entirely.
    #
    # Ids now come from a per-index monotonic counter, seeded on open from a floor that
    # dominates every id live in *either* store.

    _FAISS_ID_CONFIG_KEYS = {
        "main": "next_faiss_id_main",
        "section": "next_faiss_id_section",
        "document": "next_faiss_id_document",
    }

    def _allocate_faiss_ids(self, name: str, count: int) -> np.ndarray:
        """
        Reserve ``count`` fresh FAISS ids for the named index (main/section/document).

        Never reads ``ntotal``. Acquire this before ``_faiss_lock``, never after, so
        the two locks are always taken in the same order.
        """
        if count <= 0:
            return np.array([], dtype=np.int64)
        with self._faiss_id_lock:
            start = self._faiss_id_counters.get(name, 0)
            self._faiss_id_counters[name] = start + count
            return np.arange(start, start + count, dtype=np.int64)

    @staticmethod
    def _live_faiss_ids(index) -> np.ndarray:
        """External ids currently present in an IndexIDMap/IndexIDMap2."""
        if index is None or index.ntotal == 0 or not hasattr(index, "id_map"):
            return np.array([], dtype=np.int64)
        return faiss.vector_to_array(index.id_map)

    @staticmethod
    def _scalar_or_none(conn, sql: str) -> Optional[int]:
        """First column of the first row, or None when the query yields nothing."""
        row = conn.execute(sql).fetchone()
        return None if row is None or row[0] is None else int(row[0])

    def _sqlite_max_faiss_id(self, conn, name: str) -> Optional[int]:
        """Largest faiss_id SQLite references for the named index."""
        if name == "main":
            # chunks and column_embeddings share the main index's id space.
            return self._scalar_or_none(
                conn,
                """
                SELECT MAX(m) FROM (
                    SELECT MAX(faiss_id) AS m FROM chunks
                    UNION ALL
                    SELECT MAX(faiss_id) AS m FROM column_embeddings
                )
                """,
            )
        if name == "section":
            return self._scalar_or_none(conn, "SELECT MAX(faiss_id) FROM sections")
        if name == "document":
            return self._scalar_or_none(conn, "SELECT MAX(doc_faiss_id) FROM documents")
        raise ValueError(f"unknown index name: {name!r}")

    def _compute_faiss_id_floor(self, conn, index, name: str) -> int:
        """
        The smallest id that is safe to hand out next.

        Dominates the persisted counter, every id SQLite references, and every id live
        in the index -- so it is correct against a stale counter, against orphan vectors
        left by a crash, and against a database written by an older version. Safe to
        compute even on a corrupt database, because it is a max().

        ``MAX(faiss_id) + 1`` alone would not do: a vector can outlive its row (a failed
        removal, a crash after commit), and reusing its id would collide.
        """
        persisted = self._load_int_config(conn, self._FAISS_ID_CONFIG_KEYS[name], default=0)
        sqlite_max = self._sqlite_max_faiss_id(conn, name)
        live = self._live_faiss_ids(index)

        floor = persisted
        if sqlite_max is not None:
            floor = max(floor, sqlite_max + 1)
        if live.size:
            floor = max(floor, int(live.max()) + 1)
        return floor

    def _seed_faiss_counters(self, conn) -> None:
        self._faiss_id_counters["main"] = self._compute_faiss_id_floor(conn, self.index, "main")
        if self._hierarchical_embeddings:
            self._faiss_id_counters["section"] = self._compute_faiss_id_floor(conn, self.section_index, "section")
            self._faiss_id_counters["document"] = self._compute_faiss_id_floor(conn, self.document_index, "document")

    def _save_faiss_counters(self) -> None:
        """
        Persist the counters best-effort, after commit.

        Correctness does not depend on this -- the open-time floor is authoritative.
        It matters for one case the floor cannot cover: when *every* document has been
        deleted, both SQLite and the index are empty and the floor collapses to 0, while
        incremental backups may still re-add vectors under their original ids.
        """
        if self.is_memory_only:
            return
        counters = dict(self._faiss_id_counters)
        if not counters:
            return

        def _write() -> None:
            with self.connection_pool.get_connection() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    [(self._FAISS_ID_CONFIG_KEYS[name], str(value)) for name, value in counters.items()],
                )
                conn.commit()

        retry_on_locked(_write)

    # ------------------
    # Dual-store integrity
    # ------------------

    def _duplicate_main_faiss_ids(self, conn) -> List[int]:
        rows = conn.execute("""
            SELECT faiss_id FROM (
                SELECT faiss_id FROM chunks WHERE faiss_id IS NOT NULL
                UNION ALL
                SELECT faiss_id FROM column_embeddings
            )
            GROUP BY faiss_id HAVING COUNT(*) > 1
            """).fetchall()
        return [int(r[0]) for r in rows]

    def _verify_integrity(self, conn) -> None:
        """
        Check the SQLite/FAISS agreement on open.

        Severity is split by whether a condition can produce *wrong results*:

        * Duplicate ``faiss_id`` -- one vector attributed to two documents. Queries
          return the wrong document, silently. Refuse to open.
        * Count mismatch -- orphan vectors or dangling rows. Costs recall, never
          correctness, and is the expected residue of a crash. Warn.
        """
        duplicates = self._duplicate_main_faiss_ids(conn)
        if duplicates:
            shown = ", ".join(str(d) for d in duplicates[:10])
            more = f" (and {len(duplicates) - 10} more)" if len(duplicates) > 10 else ""
            raise IndexIntegrityError(
                f"Database '{self.name}' has {len(duplicates)} duplicate FAISS id(s): {shown}{more}. "
                f"A duplicated id makes one vector hydrate two documents, so queries return the wrong "
                f"document. This database was written by a version that allocated ids from index.ntotal. "
                f"Run `lvdb db {self.name} repair` to rebuild the index."
            )

        if self.index is None:
            return

        n_chunks = self._scalar_or_none(conn, "SELECT COUNT(*) FROM chunks WHERE faiss_id IS NOT NULL")
        n_columns = self._scalar_or_none(conn, "SELECT COUNT(*) FROM column_embeddings")
        if n_chunks is None or n_columns is None:
            return
        expected = n_chunks + n_columns
        if self.index.ntotal != expected:
            logger.warning(
                f"Database '{self.name}': FAISS index holds {self.index.ntotal} vectors but SQLite references "
                f"{expected} ({n_chunks} chunks + {n_columns} metadata embeddings). This costs recall, not "
                f"correctness -- it is the expected residue of an interrupted write. "
                f"Run `lvdb db {self.name} repair` to reconcile."
            )

    # Config/state helpers
    def _load_int_config(self, conn, key: str, default: int = 0) -> int:
        row = conn.execute("SELECT value FROM config WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return default

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
        """Inspect a FAISS index object and return its metric type ('L2'/'IP').

        Read ``metric_type`` directly: the ``IndexIDMap``/``IndexIDMap2`` wrapper
        (and every base index) exposes it, and it is the only reliable signal.
        Unwrapping via ``index.index`` returns a *downcast-less* generic
        ``faiss.Index`` whose class name is just ``"Index"`` -- it matches neither
        "IP" nor "L2", so a name-based check silently fell through to the L2
        default and scored an ``IndexFlatIP`` with the L2 formula ``1/(1+ip)``,
        which inverts the ranking. The name check is kept only as a fallback for
        objects (e.g. test doubles) that do not expose ``metric_type``.
        """
        if index is None:
            return "L2"  # Default

        metric = getattr(index, "metric_type", None)
        if isinstance(metric, (int, np.integer)):
            return "IP" if metric == faiss.METRIC_INNER_PRODUCT else "L2"

        # Fallback: inspect the (possibly wrapped) index type name.
        base_index = index.index if hasattr(index, "index") else index
        index_type = str(type(base_index).__name__)
        if "IP" in index_type or "InnerProduct" in index_type:
            return "IP"
        return "L2"

    def _normalize_for_index(self, vectors: "np.ndarray", index: Any) -> "np.ndarray":
        """L2-normalize a copy of ``vectors`` when ``index`` uses the IP metric.

        Inner-product scoring assumes unit vectors: ``_distance_to_similarity``
        maps ``ip`` to ``(ip + 1) / 2`` and clamps to ``[0, 1]``. With
        unnormalized vectors the raw inner product exceeds 1, so a pile of
        unrelated documents all clamp to a tied ``1.0`` and ranking collapses.
        Normalizing at both the write and the query boundary makes IP scoring
        correct regardless of the embedding provider's ``normalize`` setting.

        This is a no-op for L2 (and every non-IP) index, so L2 rankings -- and
        the retrieval baseline, which runs on the default ``IndexFlatL2`` -- are
        byte-for-byte unchanged. Returns a fresh array on the IP path so the
        caller's buffer is never mutated in place by ``faiss.normalize_L2``.
        """
        if index is None or self._detect_faiss_metric_type(index) != "IP":
            return vectors
        normalized = np.array(vectors, dtype=np.float32, copy=True)
        faiss.normalize_L2(normalized)
        return normalized

    @staticmethod
    def _unit_normalize_centroids(vectors: "np.ndarray") -> "np.ndarray":
        """Scale each centroid row to unit L2 norm, leaving all-zero rows unchanged.

        Section and document vectors are means of their chunk embeddings.
        Averaging shrinks a centroid's norm below the scale of its constituents,
        so a raw centroid no longer sits on the unit sphere that normalized chunk
        and query vectors occupy -- ``_distance_to_similarity`` then reads its
        distance off a mismatched scale. Re-normalizing removes that averaging
        artifact so section/document scoring is well-behaved under both L2 and IP
        indices. This runs at every centroid write, independent of the index
        metric (unlike :meth:`_normalize_for_index`, which is IP-only), because
        the shrink is a property of averaging, not of the metric. Empty sections
        have an all-zero centroid with no direction and are left untouched.

        Returns a fresh ``float32`` array; the caller's buffer is never mutated.
        """
        arr = np.array(vectors, dtype=np.float32, copy=True)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        nonzero = norms[:, 0] > 0.0
        arr[nonzero] /= norms[nonzero]
        return arr

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
    @staticmethod
    def _replace_with_retry(src: Union[str, Path], dst: Union[str, Path]) -> None:
        """
        ``os.replace`` with a bounded retry, because on Windows it is not reliably
        available even when our own code is correctly serialized.

        A virus scanner, the search indexer, or any process that merely *read* the
        target (a backup copying the index, another worker opening it) can still hold a
        transient handle on it, and the rename then fails with
        ``PermissionError`` ``[WinError 5] Access is denied`` / ``[WinError 32]``.

        This is not cosmetic. The rename is the last step of persisting the index, so a
        failure propagates out of ``save()`` -> ``close()``: the index is never written,
        while SQLite has already committed its rows. Reopening then yields **dangling
        rows** -- exactly the residue the dual-store rule exists to prevent. Retrying a
        rename is safe: it is idempotent, and the temp file is still intact.
        """
        delay = _REPLACE_BASE_DELAY
        for attempt in range(_REPLACE_MAX_RETRIES):
            try:
                os.replace(str(src), str(dst))
                return
            except PermissionError:
                if attempt == _REPLACE_MAX_RETRIES - 1:
                    raise
                # jitter to spread retries, not a security/crypto use
                time.sleep(delay + random.uniform(0, delay))  # nosec B311
                delay = min(delay * 2, _REPLACE_MAX_DELAY)

    @staticmethod
    def _atomic_write_index(index, path: Union[str, Path]) -> None:
        """
        Serialize an index so that a crash can never leave a truncated file.

        Write to a temp file adjacent to the target (same volume, so ``os.replace`` is
        atomic on Windows as well as POSIX), fsync it, then rename over the target.
        The directory fsync is best-effort: Windows does not support it.
        """
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        try:
            faiss.write_index(index, str(tmp))

            # The handle must be writable: Windows rejects fsync on an O_RDONLY descriptor.
            with open(tmp, "rb+") as fh:
                os.fsync(fh.fileno())

            LocalVectorDBCore._replace_with_retry(tmp, path)
        except BaseException:
            # Leave no partial file behind; the previous index at `path` is untouched.
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            raise

        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, PermissionError):  # pragma: no cover - not supported on Windows
            pass

    def _write_index_to_disk(self, index, path) -> None:
        """Persist one index, converting off the GPU first if necessary."""
        if index is None or not path:
            return
        to_write = index
        if hasattr(index, "index") and hasattr(index.index, "device"):
            try:
                to_write = faiss.index_gpu_to_cpu(index)
            except AttributeError:
                logger.warning("GPU-to-CPU conversion failed - FAISS may not have GPU support")
            except Exception as e:
                logger.warning(f"Failed to convert GPU index to CPU for saving: {e}")
        self._atomic_write_index(to_write, path)

    def _save_internal(self):
        if self.is_memory_only:
            return

        # An emptied index must still be written: skipping the write when ntotal == 0
        # would leave the previous, populated file on disk, so reopening the database
        # would resurrect every deleted vector.
        with self._faiss_lock.read_lock():
            self._write_index_to_disk(self.index, self.index_path)
            if self._hierarchical_embeddings:
                self._write_index_to_disk(self.section_index, self.section_index_path)
                self._write_index_to_disk(self.document_index, self.document_index_path)

    def save(self):
        # Skip the rewrite (and its os.replace) when the in-RAM index matches disk.
        # This makes save()/close() a no-op for a database that only served reads, so
        # idle-eviction or shutdown of a read-only worker never rewrites -- or races
        # another worker on -- the shared index file. Mutations set _index_dirty; both
        # this method and any mutating op hold _read_write_lock.write_lock(), so the
        # flag transitions are serialized and no write can be lost.
        with self._read_write_lock.write_lock():
            if not self._index_dirty:
                return
            self._save_internal()
            self._index_dirty = False
        self._save_faiss_counters()

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
