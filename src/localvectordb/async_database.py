# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb/async_database.py
# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/async_database.py
"""
AsyncLocalVectorDB - Async wrapper for LocalVectorDB v1.0

This module provides async/await interface for LocalVectorDB using composition
and thread pool execution. CPU-bound operations (chunking, FAISS) and I/O
operations (SQLite) are executed in a thread pool, while embedding operations
use direct async calls when available for optimal performance.

Features:
- Drop-in async replacement for LocalVectorDB
- Optimized async embedding calls for HTTP-based providers
- Thread pool execution for CPU and I/O bound operations
- Async context manager support
- Lazy initialization to avoid blocking event loop
- Clean error handling and resource management

Examples
--------
Basic usage with context manager::

    async with AsyncLocalVectorDB("my_db") as db:
        # Insert documents
        doc_ids = await db.upsert([
            "This is a test document",
            "Another document for testing"
        ])

        # Query documents
        results = await db.query("test document", k=5)

        # Get documents
        docs = await db.get(doc_ids)

Factory function usage::

    db = await create_async_vectordb(
        "my_db",
        embedding_model="nomic-embed-text",
        chunk_size=500
    )
    try:
        stats = await db.get_stats()
        print(f"Database has {stats['documents']} documents")
    finally:
        await db.close()

Performance optimizations::

    # Use more workers for CPU-heavy workloads
    async with AsyncLocalVectorDB("my_db", max_workers=8) as db:
        # Batch operations benefit from more workers
        await db.upsert(large_document_list, batch_size=50)
"""

import asyncio
import functools
import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any, Tuple

import numpy as np

from localvectordb.core import Document, QueryResult, MetadataField, AsyncBaseVectorDB
from localvectordb.database import LocalVectorDB
from localvectordb.exceptions import DatabaseError, DuplicateDocumentIDError

logger = logging.getLogger(__name__)


class AsyncLocalVectorDB(AsyncBaseVectorDB):
    """
    Async wrapper for LocalVectorDB using composition and optimized async operations.

    This class provides the same document-focused interface as LocalVectorDB v1.0
    but with async/await support. It uses thread pool execution for CPU and I/O
    bound operations while leveraging direct async calls for embedding operations
    when available.

    Parameters
    ----------
    name : str
        Database name (used for file naming)
    base_path : Union[str, Path], optional
        Directory to store database files, by default ".lvdb"
    metadata_schema : Optional[Dict[str, MetadataField]], optional
        Schema definition for metadata fields
    doc_id_pattern : str, optional
        Pattern for auto-generating document IDs, by default "doc_{idx}"
    embedding_provider : str, optional
        Embedding provider name, by default "ollama"
    embedding_model : str, optional
        Embedding model name, by default "nomic-embed-text"
    embedding_config : Optional[Dict[str, Any]], optional
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
    connection_pool_size : int, optional
        Size of SQLite connection pool, by default 10
    create_if_not_exists : bool, optional
        Whether to create database if it doesn't exist, by default True
    max_workers : Optional[int], optional
        Maximum number of thread pool workers, by default None (auto-detect)
    executor : Optional[ThreadPoolExecutor], optional
        Custom thread pool executor, by default None
    """

    def __init__(
            self,
            name: str,
            base_path: Union[str, Path] = ".lvdb",
            *,
            # LocalVectorDB parameters
            metadata_schema: Optional[Dict[str, MetadataField]] = None,
            doc_id_pattern: str = "doc_{idx}",
            embedding_provider: str = "ollama",
            embedding_model: str = "nomic-embed-text",
            embedding_config: Optional[Dict[str, Any]] = None,
            chunking_method: str = "sentences",
            chunk_size: int = 500,
            chunk_overlap: int = 1,
            enable_gpu: bool = False,
            enable_fts: bool = True,
            connection_pool_size: int = 10,
            create_if_not_exists: bool = True,
            # Async-specific parameters
            max_workers: Optional[int] = None,
            executor: Optional[ThreadPoolExecutor] = None
    ):
        """Initialize AsyncLocalVectorDB with same parameters as LocalVectorDB"""

        # Store initialization parameters for lazy initialization
        self._init_params = {
            'name': name,
            'base_path': base_path,
            'metadata_schema': metadata_schema,
            'doc_id_pattern': doc_id_pattern,
            'embedding_provider': embedding_provider,
            'embedding_model': embedding_model,
            'embedding_config': embedding_config,
            'chunking_method': chunking_method,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'enable_gpu': enable_gpu,
            'enable_fts': enable_fts,
            'connection_pool_size': connection_pool_size,
            'create_if_not_exists': create_if_not_exists
        }

        # Async management
        self._executor = executor or ThreadPoolExecutor(
            max_workers=max_workers or min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="AsyncVectorDB"
        )
        self._owns_executor = executor is None

        # State management
        self._sync_db: Optional[LocalVectorDB] = None
        self._initialized = False
        self._closed = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self):
        """Lazy initialization of the synchronous database with proper locking"""
        if self._initialized and not self._closed:
            return

        async with self._init_lock:
            # Double-check pattern
            if self._initialized and not self._closed:
                return

            if self._closed:
                raise DatabaseError("Database has been closed")

            try:
                loop = asyncio.get_event_loop()
                # Initialize the sync database in thread pool to avoid blocking
                self._sync_db = await loop.run_in_executor(
                    self._executor,
                    self._create_sync_db
                )
                self._initialized = True
                # This parameter is initiated on __init__ of the class.
                self._init_params["embedding_dimension"] = self._sync_db.embedding_dimension
                self._init_params["fts_enabled"] = self._sync_db.fts_enabled
                logger.info(f"AsyncLocalVectorDB '{self.name}' initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AsyncLocalVectorDB: {e}")
                raise DatabaseError(f"Database initialization failed: {e}")

    def _create_sync_db(self) -> LocalVectorDB:
        """Create the underlying synchronous LocalVectorDB instance"""
        return LocalVectorDB(**self._init_params)

    # Async context manager support
    async def __aenter__(self):
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    # Property delegation (synchronous, safe to access)
    @property
    def name(self) -> str:
        return self._init_params['name']

    @property
    def embedding_model(self) -> str:
        return self._init_params['embedding_model']

    @property
    def embedding_provider(self) -> str:
        return self._init_params['embedding_provider']

    @property
    def chunk_size(self) -> int:
        return self._init_params['chunk_size']

    @property
    def chunk_overlap(self) -> int:
        return self._init_params['chunk_overlap']

    @property
    def chunking_method(self) -> str:
        return self._init_params['chunking_method']

    @property
    def closed(self) -> bool:
        """Check if database is closed"""
        return self._closed or self._sync_db is None or self._sync_db.closed

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension (requires initialization)"""
        if not self._initialized or self._sync_db is None:
            raise DatabaseError("Database not initialized. Use await db._ensure_initialized() first.")
        return self._init_params['embedding_dimension']

    @property
    def fts_enabled(self) -> bool:
        """Check if FTS is enabled (requires initialization)"""
        if not self._initialized or self._sync_db is None:
            raise DatabaseError("Database not initialized. Use await db._ensure_initialized() first.")
        return self._init_params['fts_enabled']

    @property
    def metadata_schema(self) -> Dict[str, MetadataField]:
        """Get metadata schema (requires initialization)"""
        if not self._initialized or self._sync_db is None:
            raise DatabaseError("Database not initialized. Use await db._ensure_initialized() first.")
        return self._sync_db.metadata_schema.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get database stats (requires initialization)"""
        if not self._initialized or self._sync_db is None:
            raise DatabaseError("Database not initialized. Use await db._ensure_initialized() first.")
        return self._sync_db.get_stats()

    # Optimized async embedding generation
    async def _generate_embeddings_async(
            self,
            texts: List[str],
            batch_size: int = 100
    ) -> np.ndarray:
        """
        Generate embeddings using async provider when available, with batching

        Parameters
        ----------
        texts : List[str]
            Text strings to embed
        batch_size : int
            Batch size for embedding generation

        Returns
        -------
        np.ndarray
            Array of embeddings
        """
        await self._ensure_initialized()

        if not texts:
            return np.array([]).reshape(0, self._sync_db.embedding_dimension)


        # Check if embedding provider supports async
        if hasattr(self._sync_db.embedding_provider, 'embed_async'):
            logger.debug(f"Using async embeddings for {len(texts)} texts")

            # Process in batches for memory efficiency
            if len(texts) <= batch_size:
                return await self._sync_db.embedding_provider.embed_async(texts)
            else:
                return await self._sync_db.embedding_provider.embed_batch(texts, batch_size=batch_size)
        else:
            # Fall back to sync version in thread pool
            logger.debug(f"Using sync embeddings in thread pool for {len(texts)} texts")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._sync_db._generate_embeddings_chunked,
                texts,
                batch_size
            )

    # Core async methods
    async def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None
    ) -> List[str]:
        """
        Async upsert documents with optimized embedding generation and chunk reuse

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
            Skip chunks that are too similar to existing chunks

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        await self._ensure_initialized()

        # Normalize inputs in thread pool
        loop = asyncio.get_event_loop()
        normalized_data = await loop.run_in_executor(
            self._executor,
            self._normalize_upsert_inputs,
            documents, metadata, ids
        )

        documents, metadata_list, ids_list = normalized_data

        # First, check which documents need updates based on content hash
        docs_to_update_indices = await self._identify_changed_documents(documents, ids_list)

        if not docs_to_update_indices:
            # No documents need updating
            return ids_list

        # Filter to only process documents that need updating
        update_documents = [documents[i] for i in docs_to_update_indices]
        update_metadata = [metadata_list[i] for i in docs_to_update_indices]
        update_ids = [ids_list[i] for i in docs_to_update_indices]

        # Process in batches if there are many documents to update
        if len(update_documents) > 10:
            result_ids = await self._batch_upsert_documents(
                update_documents, update_metadata, update_ids, batch_size, similarity_threshold
            )
        else:
            # For small batches, use the sync method in thread pool
            result_ids = await loop.run_in_executor(
                self._executor,
                functools.partial(
                    self._sync_db.upsert,
                    documents=update_documents,
                    metadata=update_metadata,
                    ids=update_ids,
                    batch_size=batch_size,
                    similarity_threshold=similarity_threshold
                )
            )

        # Save state
        await loop.run_in_executor(
            self._executor,
            self._sync_db._save_next_doc_id
        )

        await self.save()

        return ids_list

    async def _identify_changed_documents(
            self,
            documents: List[str],
            ids: List[str]
    ) -> List[int]:
        """
        Identify which documents have changed based on content hash comparison

        Parameters
        ----------
        documents : List[str]
            List of document texts
        ids : List[str]
            List of document IDs

        Returns
        -------
        List[int]
            Indices of documents that need updating
        """
        loop = asyncio.get_event_loop()

        # Calculate hashes for all documents
        doc_hashes = await loop.run_in_executor(
            self._executor,
            lambda: [hashlib.sha256(doc.encode('utf-8')).hexdigest() for doc in documents]
        )

        # Get existing document hashes from database
        existing_hashes = {}

        def fetch_existing_hashes():
            with self._sync_db.connection_pool.get_connection() as conn:
                placeholders = ','.join(['?'] * len(ids))
                if placeholders:
                    cursor = conn.execute(
                        f'SELECT id, content_hash FROM documents WHERE id IN ({placeholders})',
                        ids
                    )
                    return {row['id']: row['content_hash'] for row in cursor.fetchall()}
            return {}

        existing_hashes = await loop.run_in_executor(
            self._executor,
            fetch_existing_hashes
        )

        # Identify documents that need updating
        docs_to_update = []
        for i, (doc_id, doc_hash) in enumerate(zip(ids, doc_hashes)):
            if doc_id not in existing_hashes or existing_hashes[doc_id] != doc_hash:
                docs_to_update.append(i)

        return docs_to_update

    async def _batch_upsert_documents(
            self,
            documents: List[str],
            metadata_list: List[Dict[str, Any]],
            ids_list: List[str],
            batch_size: int,
            similarity_threshold: Optional[float]
    ) -> List[str]:
        """
        Process documents in batches with optimized async operations

        Parameters
        ----------
        documents : List[str]
            List of documents to process
        metadata_list : List[Dict[str, Any]]
            List of metadata dicts
        ids_list : List[str]
            List of document IDs
        batch_size : int
            Batch size for processing
        similarity_threshold : Optional[float]
            Similarity threshold for filtering

        Returns
        -------
        List[str]
            List of processed document IDs
        """
        loop = asyncio.get_event_loop()

        # Generate chunks for all documents in thread pool (CPU-bound)
        chunks_data = await loop.run_in_executor(
            self._executor,
            self._generate_chunks_with_mapping,
            documents
        )

        all_chunks, doc_chunk_mapping = chunks_data

        # Get chunk texts for embedding generation
        chunk_texts = [chunk.content for chunk in all_chunks]

        # Generate embeddings using async (I/O-bound for HTTP providers)
        embeddings = await self._generate_embeddings_async(chunk_texts, batch_size)

        # Database operations in thread pool (I/O-bound)
        result_ids = await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_db._upsert_with_precomputed_embeddings,
                documents=documents,
                metadata_list=metadata_list,
                ids_list=ids_list,
                chunks=all_chunks,
                embeddings=embeddings,
                doc_chunk_mapping=doc_chunk_mapping,
                similarity_threshold=similarity_threshold
            )
        )

        return result_ids

    def _generate_chunks_with_mapping(self, documents: List[str]) -> Tuple[List["Chunk"], List[int]]:
        """
        Generate chunks for documents with mapping information

        Parameters
        ----------
        documents : List[str]
            List of documents to chunk

        Returns
        -------
        Tuple[List[Chunk], List[int]]
            List of all chunks and mapping from chunk to document index
        """
        all_chunks = []
        doc_chunk_mapping = []

        for doc_idx, doc_text in enumerate(documents):
            chunks = self._sync_db.chunker.chunk(doc_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                doc_chunk_mapping.append(doc_idx)

        return all_chunks, doc_chunk_mapping

    def _normalize_upsert_inputs(self, documents, metadata, ids):
        """Helper to normalize inputs (runs in thread pool)"""
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
            ids = [self._sync_db._generate_doc_id() for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # If the user provided a list with some ids but None for others, generate for the Nones!
        for i, doc_id in enumerate(ids):
            if doc_id is None:
                ids[i] = self._sync_db._generate_doc_id()

        return documents, metadata, ids

    def _generate_chunks_for_documents(self, documents):
        """Helper to generate chunks (runs in thread pool)"""
        all_chunks = []
        chunk_texts = []
        doc_chunk_mapping = []

        for doc_idx, doc_text in enumerate(documents):
            chunks = self._sync_db.chunker.chunk(doc_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_texts.append(chunk.content)
                doc_chunk_mapping.append(doc_idx)

        return all_chunks, chunk_texts, doc_chunk_mapping

    async def insert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: int = 100,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise"
    ) -> List[str]:
        """
        Async insert new documents with optimized processing

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
            Skip chunks that are too similar to existing chunks
        errors : Literal["ignore", "raise"]
            How to handle document ID conflicts

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
        """
        await self._ensure_initialized()

        # Normalize inputs in thread pool
        loop = asyncio.get_event_loop()
        normalized_data = await loop.run_in_executor(
            self._executor,
            self._normalize_upsert_inputs,
            documents, metadata, ids
        )

        documents, metadata_list, ids_list = normalized_data

        # Check for existing document IDs before doing any processing
        existing_ids_indices = await self._check_existing_document_ids(ids_list)

        if existing_ids_indices:
            if errors == "raise":
                existing_ids = [ids_list[i] for i in existing_ids_indices]
                raise DuplicateDocumentIDError(
                    f"Document IDs already exist: {existing_ids}"
                )
            elif errors == "ignore":
                # Filter out documents with existing IDs
                valid_indices = [i for i in range(len(documents)) if i not in existing_ids_indices]

                if not valid_indices:
                    # All documents have existing IDs
                    return []

                # Keep only documents with non-existing IDs
                documents = [documents[i] for i in valid_indices]
                metadata_list = [metadata_list[i] for i in valid_indices]
                ids_list = [ids_list[i] for i in valid_indices]

        # Process in batches if there are many documents
        if len(documents) > 10:
            inserted_ids = await self._batch_insert_documents(
                documents, metadata_list, ids_list, batch_size, similarity_threshold
            )
        else:
            # For small batches, use the sync method in thread pool
            inserted_ids = await loop.run_in_executor(
                self._executor,
                functools.partial(
                    self._sync_db.insert,
                    documents=documents,
                    metadata=metadata_list,
                    ids=ids_list,
                    batch_size=batch_size,
                    similarity_threshold=similarity_threshold,
                    errors="raise"  # Already handled duplicates above
                )
            )

        # Save state
        await loop.run_in_executor(
            self._executor,
            self._sync_db._save_next_doc_id
        )

        await self.save()

        return inserted_ids

    async def _check_existing_document_ids(self, ids: List[str]) -> List[int]:
        """
        Check which document IDs already exist in the database

        Parameters
        ----------
        ids : List[str]
            List of document IDs to check

        Returns
        -------
        List[int]
            Indices of documents with IDs that already exist
        """
        loop = asyncio.get_event_loop()

        def fetch_existing_ids():
            existing_indices = []
            with self._sync_db.connection_pool.get_connection() as conn:
                for i, doc_id in enumerate(ids):
                    cursor = conn.execute(
                        'SELECT 1 FROM documents WHERE id = ? LIMIT 1',
                        (doc_id,)
                    )
                    if cursor.fetchone():
                        existing_indices.append(i)
            return existing_indices

        return await loop.run_in_executor(
            self._executor,
            fetch_existing_ids
        )

    async def _batch_insert_documents(
            self,
            documents: List[str],
            metadata_list: List[Dict[str, Any]],
            ids_list: List[str],
            batch_size: int,
            similarity_threshold: Optional[float]
    ) -> List[str]:
        """
        Process document insertion in batches with optimized async operations

        Parameters
        ----------
        documents : List[str]
            List of documents to process
        metadata_list : List[Dict[str, Any]]
            List of metadata dicts
        ids_list : List[str]
            List of document IDs
        batch_size : int
            Batch size for processing
        similarity_threshold : Optional[float]
            Similarity threshold for filtering

        Returns
        -------
        List[str]
            List of inserted document IDs
        """
        loop = asyncio.get_event_loop()

        # Generate chunks for all documents in thread pool (CPU-bound)
        chunks_data = await loop.run_in_executor(
            self._executor,
            self._generate_chunks_with_mapping,
            documents
        )

        all_chunks, doc_chunk_mapping = chunks_data

        # Get chunk texts for embedding generation
        chunk_texts = [chunk.content for chunk in all_chunks]

        # Generate embeddings using async (I/O-bound for HTTP providers)
        embeddings = await self._generate_embeddings_async(chunk_texts, batch_size)

        # Use the same mechanism as upsert but ensure we're only inserting new documents
        # We've already filtered out existing IDs based on the errors parameter
        result_ids = await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_db._upsert_with_precomputed_embeddings,
                documents=documents,
                metadata_list=metadata_list,
                ids_list=ids_list,
                chunks=all_chunks,
                embeddings=embeddings,
                doc_chunk_mapping=doc_chunk_mapping,
                similarity_threshold=similarity_threshold
            )
        )

        return result_ids

    async def get(self, ids: Union[str, List[str]]) -> Union[Document, List[Document], None]:
        """Async get documents by ID"""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_db.get,
            ids
        )

    async def exists(self, ids: Union[str, List[str]]) -> Union[bool, List[bool]]:
        """Async check if documents exist"""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_db.exists,
            ids
        )

    async def delete(self, ids: Union[str, List[str]]) -> int:
        """Async delete documents"""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_db.delete,
            ids
        )

    async def update(
            self,
            doc_id: str,
            content: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Async update document"""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_db.update,
                doc_id=doc_id,
                content=content,
                metadata=metadata
            )
        )

    async def query(
            self,
            query: str,
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
        Async unified query interface with optimized vector search

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
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()

        # For keyword search, no embedding is needed, so use the original method directly
        if search_type == 'keyword':
            return await loop.run_in_executor(
                self._executor,
                functools.partial(
                    self._sync_db.query,
                    query=query,
                    search_type='keyword',
                    return_type=return_type,
                    k=k,
                    score_threshold=score_threshold,
                    filters=filters,
                    vector_weight=vector_weight,
                    context_window=context_window,
                    semantic_dedup_threshold=semantic_dedup_threshold,
                    document_scoring_method=document_scoring_method
                )
            )

        # For vector and hybrid search, generate query embedding asynchronously
        query_embedding = await self._generate_embeddings_async([query])

        # Pass the precomputed embedding to the sync database for searching
        results = await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_db._search_with_embedding,
                query=query,
                query_embedding=query_embedding,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight,
                context_window=context_window,
                semantic_dedup_threshold=semantic_dedup_threshold,
                document_scoring_method=document_scoring_method
            )
        )

        return results

    async def filter(
            self,
            where: Optional[Dict[str, Any]] = None,
            order_by: Optional[str] = None,
            limit: Optional[int] = None,
            offset: int = 0
    ) -> List[Document]:
        """Async filter documents"""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_db.filter,
                where=where,
                order_by=order_by,
                limit=limit,
                offset=offset
            )
        )

    async def save(self):
        """Async save database"""
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._sync_db.save)

    async def close(self):
        # Non-blocking shutdown for better async behavior
        try:
            await asyncio.to_thread(self._executor.shutdown, wait=False)
        except AttributeError:
            # Fallback for older Python versions
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._executor.shutdown, False)

    async def update_metadata_schema(
            self,
            new_schema: Union[str, Dict[str, MetadataField]],
            drop_columns: bool = False,
            column_mapping: Optional[dict] = None
    ) -> Dict[str, Any]:
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._sync_db.update_metadata_schema,
                new_schema=new_schema,
                drop_columns=drop_columns,
                column_mapping=column_mapping
            )
        )

    async def get_metadata_schema_info(self) -> Dict[str, Any]:
        await self._ensure_initialized()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_db.get_metadata_schema_info,
        )


    def is_async_database(self) -> bool:
        """Mark this as an async database for QueryBuilder"""
        return True

    def supports_async_embeddings(self) -> bool:
        """Check if async embeddings are supported"""
        if not self._initialized:
            return False
        return hasattr(self._sync_db.embedding_provider, 'embed_async')


# Factory function for convenient creation
async def create_async_vectordb(
        name: str,
        base_path: Union[str, Path] = ".lvdb",
        **kwargs
) -> AsyncLocalVectorDB:
    """
    Factory function to create and initialize an AsyncLocalVectorDB.

    Parameters
    ----------
    name : str
        Database name
    base_path : Union[str, Path]
        Base path for database storage
    **kwargs
        All other parameters passed to AsyncLocalVectorDB constructor

    Returns
    -------
    AsyncLocalVectorDB
        Initialized async database instance
    """
    db = AsyncLocalVectorDB(name, base_path, **kwargs)
    await db._ensure_initialized()
    return db


# Export main classes and functions
__all__ = [
    'AsyncLocalVectorDB',
    'create_async_vectordb'
]