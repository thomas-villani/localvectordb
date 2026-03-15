# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# localvectordb/database/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from localvectordb.query_builder import QueryBuilder

import aiosqlite
import numpy as np
from faiss import IndexIDMap2

from localvectordb._pools import AsyncConnectionPool, ConnectionPool, ReadWriteLock
from localvectordb._schema import DatabaseSchema
from localvectordb.chunking import PositionTrackingChunker
from localvectordb.core import Chunk, Document, DocumentScoringMethod, MetadataField, QueryResult
from localvectordb.embeddings import EmbeddingProvider
from localvectordb.sqlite_tuning import SqliteProfile

DEFAULT_QUEUE_SIZE = 3
DEFAULT_BATCH_SIZE = 100


class BaseVectorDB(ABC):
    """
    Abstract base class defining the interface for vector databases.

    This class defines the common interface that both LocalVectorDB and RemoteVectorDB
    must implement, allowing QueryBuilder and other components to work with either
    implementation seamlessly.
    """

    # Core database operations
    @abstractmethod
    def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None
    ) -> List[str]:
        """Insert or update documents in the database."""
        pass

    @abstractmethod
    def insert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise"
    ) -> List[str]:
        """Insert new documents into the database."""
        pass

    @abstractmethod
    def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document]]:
        """Retrieve documents by ID."""
        pass

    @abstractmethod
    def exists(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if documents exist."""
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count the number of documents in the database"""
        pass

    @abstractmethod
    def delete(self, ids: Union[str, List[str]]) -> int:
        """Delete documents."""
        pass

    @abstractmethod
    def update(
            self,
            doc_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document's content and/or metadata."""
        pass

    # Query operations
    @abstractmethod
    def query(
            self,
            query: str,
            *,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'vector',
            return_type: Literal['documents', 'chunks', 'sections', 'context', 'enriched'] = 'documents',
            search_level: Literal['chunks', 'sections', 'documents'] = 'chunks',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,
            context_window: int = 2,
            semantic_dedup_threshold: Optional[float] = None,
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: Optional[dict] = None,
            reranker: Optional[Any] = None,
            reranker_config: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Unified query interface for all search types."""
        pass

    @abstractmethod
    def filter(
            self,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List[Document]:
        """Filter documents using metadata filtering."""
        pass

    # Configuration and metadata properties
    @property
    @abstractmethod
    def embedding_model(self) -> str:
        """Return the embedding model name."""
        pass

    @property
    @abstractmethod
    def embedding_provider(self) -> EmbeddingProvider:
        """Return the embedding provider name or instance."""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Return the maximum tokens per chunk."""
        pass

    @property
    @abstractmethod
    def chunk_overlap(self) -> int:
        """Return the chunk overlap."""
        pass

    @property
    @abstractmethod
    def chunking_method(self) -> str:
        """Return the chunking method."""
        pass

    @property
    @abstractmethod
    def fts_enabled(self) -> bool:
        """Return whether full-text search is enabled."""
        pass

    @property
    @abstractmethod
    def metadata_schema(self) -> Dict[str, MetadataField]:
        """Return the metadata schema."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Return whether the database connection is closed."""
        pass

    # Schema management
    @abstractmethod
    def update_metadata_schema(
            self,
            new_schema: Union[str, Dict[str, MetadataField]],
            drop_columns: bool = False,
            column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        """Update the metadata schema."""
        pass

    @abstractmethod
    def get_metadata_schema_info(self) -> Dict[str, Any]:
        """Get detailed information about the current metadata schema."""
        pass

    # Database lifecycle
    @abstractmethod
    def save(self) -> None:
        """Save the database."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database."""
        pass

    # Context manager support
    def __enter__(self) -> "BaseVectorDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def query_builder(self) -> "QueryBuilder":
        """
        Create a new QueryBuilder for this database.

        Returns
        -------
        QueryBuilder
            A new QueryBuilder instance for building complex queries

        Examples
        --------
        Basic search::

            results = db.query_builder().search("machine learning").execute()

        Complex multi-field search with semantic filtering::

            results = (db.query_builder()
                .search_field("title", "neural networks", weight=0.3)
                .search_field("content", "deep learning", weight=0.7)
                .semantic_filter("methodology", "supervised learning", threshold=0.8)
                .filter("year", gte=2020)
                .hybrid(vector_weight=0.6)
                .limit(20)
                .execute())

        Async usage::

            results = await (db.query_builder()
                .search("machine learning")
                .semantic_filter("category", "research")
                .execute_async())
        """
        from localvectordb.query_builder import QueryBuilder
        return QueryBuilder(self)

    def ping(self) -> bool:
        """Check if the database is accessible. Override in subclasses."""
        return not self.closed

    @abstractmethod
    def upsert_from_chunks(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None
    ) -> List[str]:
        pass

    @abstractmethod
    def insert_from_chunks(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
    ) -> List[str]:
        pass

    @abstractmethod
    def upsert_from_file(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            extractor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def insert_from_file(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            extractor_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        pass

    # Core async database operations
    @abstractmethod
    async def upsert_async(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            **kwargs: Any
    ) -> List[str]:
        """Insert or update documents in the database asynchronously."""
        pass

    @abstractmethod
    async def insert_async(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            **kwargs: Any
    ) -> List[str]:
        """Insert new documents into the database asynchronously."""
        pass

    @abstractmethod
    async def upsert_from_chunks_async(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: int = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2
    ) -> List[str]:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def get_async(self, ids: Union[str, List[str]]) -> Union["Document", List["Document"]]:
        """Retrieve documents by ID asynchronously."""
        pass

    @abstractmethod
    async def exists_async(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Check if documents exist asynchronously."""
        pass

    @abstractmethod
    async def delete_async(self, ids: Union[str, List[str]]) -> int:
        """Delete documents asynchronously."""
        pass

    @abstractmethod
    async def count_async(
            self,
            filters: Optional[Dict[str, Any]] = None
    ) -> int:
        pass

    @abstractmethod
    async def update_async(
            self,
            doc_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a document's content and/or metadata asynchronously."""
        pass

    @abstractmethod
    async def query_async(
            self,
            query: str,
            *,
            search_type: Literal['vector', 'keyword', 'hybrid'] = 'hybrid',
            return_type: Literal['documents', 'chunks', 'sections', 'context', 'enriched'] = 'documents',
            search_level: Literal['chunks', 'sections', 'documents'] = 'chunks',
            k: int = 10,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
            vector_weight: float = 0.7,
            context_window: int = 2,
            semantic_dedup_threshold: Optional[float] = None,
            document_scoring_method: DocumentScoringMethod = "frequency_boost",
            document_scoring_options: dict = None,
            reranker: Optional[Any] = None,
            reranker_config: Optional[Dict[str, Any]] = None
    ) -> List["QueryResult"]:
        """Unified query interface for all search types asynchronously."""
        pass

    @abstractmethod
    async def filter_async(
            self,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List["Document"]:
        """Filter documents using metadata filtering asynchronously."""
        pass

    @abstractmethod
    async def save_async(self) -> None:
        """Save the database asynchronously."""
        pass

    @abstractmethod
    async def close_async(self) -> None:
        """Close the database asynchronously."""
        pass

    @abstractmethod
    async def update_metadata_schema_async(
            self,
            new_schema: Union[str, Dict[str, MetadataField]],
            drop_columns: bool = False,
            column_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Update metadata schema asynchronously."""
        pass

    @abstractmethod
    async def get_metadata_schema_info_async(self) -> Dict[str, Any]:
        """Get metadata schema information asynchronously."""
        pass

    # Async context manager support
    async def __aenter__(self) -> "BaseVectorDB":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close_async()


class LocalVectorDBBase(BaseVectorDB, ABC):
    """The abstract base class that defines the attributes and helper methods used by the various mixins."""

    def __init__(
            self,
            name: str,
            base_path: Union[str, Path] = ".lvdb",
            *,
            # Metadata schema
            metadata_schema: Optional[Dict[str, Any]] = None,
            # ID generation patterns
            doc_id_pattern: str = "doc_{idx}",
            # Embedding configuration
            embedding_provider: str = "ollama",
            embedding_model: str = "nomic-embed-text",
            embedding_config: Optional[Dict[str, Any]] = None,
            # Chunking configuration
            chunking_method: Union[str, Any] = "sentences",
            chunk_size: int = 500,
            chunk_overlap: int = 1,
            batch_size: int = 100,
            # Index type
            faiss_index_type: Literal["IndexFlatL2", "IndexFlatIP", "IndexHNSWFlat", "IndexLSH"] = "IndexFlatL2",
            faiss_index_hnsw_flat_neighbors: Optional[int] = None,
            faiss_index_lsh_bits: Optional[int] = None,
            # Performance settings
            enable_gpu: bool = False,
            enable_fts: bool = True,
            connection_pool_size: int = 10,
            # SQLite tuning settings
            sqlite_profile: SqliteProfile = "balanced",
            sqlite_pragma_overrides: Optional[Dict[str, Any]] = None,
            # Other
            create_if_not_exists: bool = True,
    ):
        super().__init__()
        self._read_write_lock: ReadWriteLock = None
        self.schema: DatabaseSchema = None
        self.chunker: PositionTrackingChunker = None
        self.connection_pool: ConnectionPool = None
        self.async_connection_pool: AsyncConnectionPool = None
        self._metadata_schema: Dict[str, MetadataField] = None
        self._embedding_provider: EmbeddingProvider = None
        self.index: IndexIDMap2 = None
        self.db_path: Path = None
        self.async_max_connections: int = None
        self.pipeline_queue_size: int = DEFAULT_QUEUE_SIZE
        self._batch_size: int = DEFAULT_BATCH_SIZE
        self._sqlite_profile: SqliteProfile = sqlite_profile
        self._sqlite_pragma_overrides = sqlite_pragma_overrides or {}
        self._sqlite_pragmas: dict = {}

    @abstractmethod
    def _generate_doc_id(self) -> str:
        pass

    @abstractmethod
    def _save_internal(self) -> None:
        pass

    @abstractmethod
    def _validate_metadata_batch(self, metadata_batch: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def _get_embedding_enabled_fields(self) -> Dict[str, 'MetadataField']:
        pass

    @abstractmethod
    def _generate_metadata_embeddings(
            self,
            metadata: Dict[str, Any],
            embedding_enabled_fields: Dict[str, 'MetadataField'],
            batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    def _remove_metadata_embeddings(self, conn, document_id: str) -> None:
        pass

    @abstractmethod
    def _store_metadata_embeddings(self, conn, document_id: str, field_embeddings: Dict[str, np.ndarray]) -> None:
        pass

    @abstractmethod
    def _add_vectors_to_faiss_bulk(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def _remove_old_vectors_bulk(self, faiss_ids: List[int]) -> None:
        pass

    @abstractmethod
    def _ensure_async_pool(self) -> None:
        pass

    @abstractmethod
    async def _ensure_async_schema_initialized(self) -> None:
        pass

    @abstractmethod
    async def _remove_metadata_embeddings_async(self, conn: aiosqlite.Connection, document_id: str) -> None:
        pass

    @abstractmethod
    async def _generate_metadata_embeddings_async(
            self,
            metadata: Dict[str, Any],
            embedding_enabled_fields: Dict[str, 'MetadataField'],
            batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        pass

    @abstractmethod
    async def _store_metadata_embeddings_async(
            self,
            conn: aiosqlite.Connection,
            document_id: str,
            field_embeddings: Dict[str, np.ndarray]
    ) -> None:
        pass

    @abstractmethod
    def _reconstruct_embeddings_batch(self, faiss_ids: List[int]) -> np.ndarray:
        pass

    @abstractmethod
    def _get_changed_embedding_fields(
            self,
            old_metadata: Dict[str, Any],
            new_metadata: Dict[str, Any]
    ) -> Dict[str, 'MetadataField']:
        pass

    @abstractmethod
    async def _validate_metadata_async(self, metadata: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _save_next_doc_id(self) -> None:
        pass

    @abstractmethod
    def is_memory_only(self) -> bool:
        pass
