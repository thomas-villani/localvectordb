# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/database/ingest.py
"""
Document ingestion pipelines (sync and async), chunk operations, and bulk DB ops.

This module preserves the original logic while organizing ingestion-focused code
into a mixin used by the composed LocalVectorDB class.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import queue
import threading
from abc import ABC
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import aiosqlite
import numpy as np

from localvectordb.core import Chunk, ChunkPosition
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.exceptions import (
    DuplicateDocumentIDError,
)
from localvectordb.extractors import ExtractorRegistry

logger = logging.getLogger(__name__)


class PipelineMixin(LocalVectorDBBase, ABC):

    # Pure business logic helpers for DRY elimination
    def _build_documents_bulk_insert_sql(self, mode: Literal["insert", "replace"] = "replace") -> tuple[str, List[str]]:
        """Build SQL for bulk document insertion (pure business logic)"""
        base_columns = self.schema.BASE_COLUMNS.copy()
        metadata_columns = list(self.metadata_schema.keys())
        all_columns = base_columns + metadata_columns
        placeholders = ['?'] * len(all_columns)
        sql_verb = "INSERT OR REPLACE" if mode == "replace" else "INSERT"
        sql = f"{sql_verb} INTO documents ({', '.join(all_columns)}) VALUES ({', '.join(placeholders)})"
        return sql, all_columns

    def _prepare_documents_bulk_data(
            self, documents_data: List[Tuple[str, str, str, Dict[str, Any]]],
            conn=None, preserve_created_at: bool = True
            ) -> List[tuple]:
        """
        Prepare document data for bulk insertion (pure business logic).

        Parameters
        ----------
        documents_data : List[Tuple[str, str, str, Dict[str, Any]]]
            List of (doc_id, content, content_hash, metadata) tuples
        conn : sqlite3.Connection or aiosqlite.Connection, optional
            Database connection for fetching existing created_at values
        preserve_created_at : bool, default True
            If True, preserve existing created_at values for upserts

        Returns
        -------
        List[tuple]
            Bulk data ready for INSERT OR REPLACE
        """
        if not documents_data:
            return []

        _, all_columns = self._build_documents_bulk_insert_sql()
        metadata_columns = list(self.metadata_schema.keys())
        bulk_data = []
        current_time = datetime.now(UTC)

        # Fetch existing created_at values if connection provided and preservation enabled
        existing_created_at = {}
        if conn and preserve_created_at:
            doc_ids = [doc_id for doc_id, _, _, _ in documents_data]
            if doc_ids:
                placeholders = ','.join(['?' for _ in doc_ids])
                try:
                    cursor = conn.execute(f"SELECT id, created_at FROM documents WHERE id IN ({placeholders})", doc_ids)
                    for doc_id, created_at_str in cursor.fetchall():
                        if created_at_str:
                            # Parse the ISO format timestamp back to datetime
                            try:
                                if isinstance(created_at_str, str):
                                    existing_created_at[doc_id] = datetime.fromisoformat(
                                        created_at_str.replace('Z', '+00:00'))
                                else:
                                    # Already a datetime object
                                    existing_created_at[doc_id] = created_at_str
                            except (ValueError, AttributeError):
                                # Fallback to current time if parsing fails
                                existing_created_at[doc_id] = current_time
                except Exception as e:
                    # Log warning but continue - fallback to current time for all
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to fetch existing created_at values: {e}")

        for doc_id, content, content_hash, metadata in documents_data:
            # Use existing created_at if available, otherwise use current time
            created_at_value = existing_created_at.get(doc_id, current_time)
            updated_at_value = current_time

            row_data = [doc_id, content, content_hash, created_at_value, updated_at_value]
            for field_name in metadata_columns:
                value = metadata.get(field_name)
                row_data.append(value)
            bulk_data.append(tuple(row_data))

        return bulk_data

    async def _prepare_documents_bulk_data_async(
            self, documents_data: List[Tuple[str, str, str, Dict[str, Any]]],
            conn=None, preserve_created_at: bool = True
            ) -> List[tuple]:
        """
        Prepare document data for bulk insertion (async version).

        Parameters
        ----------
        documents_data : List[Tuple[str, str, str, Dict[str, Any]]]
            List of (doc_id, content, content_hash, metadata) tuples
        conn : aiosqlite.Connection, optional
            Database connection for fetching existing created_at values
        preserve_created_at : bool, default True
            If True, preserve existing created_at values for upserts

        Returns
        -------
        List[tuple]
            Bulk data ready for INSERT OR REPLACE
        """
        if not documents_data:
            return []

        _, all_columns = self._build_documents_bulk_insert_sql()
        metadata_columns = list(self.metadata_schema.keys())
        bulk_data = []
        current_time = datetime.now(UTC)

        # Fetch existing created_at values if connection provided and preservation enabled
        existing_created_at = {}
        if conn and preserve_created_at:
            doc_ids = [doc_id for doc_id, _, _, _ in documents_data]
            if doc_ids:
                placeholders = ','.join(['?' for _ in doc_ids])
                try:
                    cursor = await conn.execute(f"SELECT id, created_at FROM documents WHERE id IN ({placeholders})",
                                                doc_ids)
                    rows = await cursor.fetchall()
                    for doc_id, created_at_str in rows:
                        if created_at_str:
                            # Parse the ISO format timestamp back to datetime
                            try:
                                if isinstance(created_at_str, str):
                                    existing_created_at[doc_id] = datetime.fromisoformat(
                                        created_at_str.replace('Z', '+00:00'))
                                else:
                                    # Already a datetime object
                                    existing_created_at[doc_id] = created_at_str
                            except (ValueError, AttributeError):
                                # Fallback to current time if parsing fails
                                existing_created_at[doc_id] = current_time
                except Exception as e:
                    # Log warning but continue - fallback to current time for all
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to fetch existing created_at values: {e}")

        for doc_id, content, content_hash, metadata in documents_data:
            # Use existing created_at if available, otherwise use current time
            created_at_value = existing_created_at.get(doc_id, current_time)
            updated_at_value = current_time

            row_data = [doc_id, content, content_hash, created_at_value, updated_at_value]
            for field_name in metadata_columns:
                value = metadata.get(field_name)
                row_data.append(value)
            bulk_data.append(tuple(row_data))

        return bulk_data

    # -----------------
    # Public APIs (sync)
    # -----------------
    def upsert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
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

        Returns
        -------
        List[str]
            List of document IDs that were upserted
        """
        with self._read_write_lock.write_lock():
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
            self._validate_metadata_batch(metadata)
            batch_size = batch_size or self.batch_size
            result_ids = self._process_with_pipeline(
                documents, metadata, ids, batch_size, similarity_threshold, mode="upsert"
            )
            self._save_next_doc_id()
            self._save_internal()
            return result_ids

    def upsert_from_file(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            extractor_kwargs: Optional[Dict[str, Any]] = None,
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
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")
        documents = []
        merged_metadata = []
        final_ids = []
        extractor_kwargs = extractor_kwargs or {}
        for i, file_path in enumerate(file_paths):
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            file_content = file_path.read_bytes()
            filename = file_path.name
            extraction_result = ExtractorRegistry.extract_text(file_content, filename, **extractor_kwargs)
            if not extraction_result.success:
                raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")
            documents.append(extraction_result.text)
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                doc_id = file_path.stem
            final_ids.append(doc_id)
        batch_size = batch_size or self.batch_size
        return self.upsert(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
        )

    def upsert_from_chunks(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
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
            if not chunks_by_document:
                return []
            if metadata is None:
                metadata = {}
            metadata_batch = {doc_id: metadata.get(doc_id, {}) for doc_id in chunks_by_document.keys()}
            self._validate_metadata_batch(list(metadata_batch.values()))
            normalized_chunks_by_document = {}
            for doc_id, chunks in chunks_by_document.items():
                normalized_chunks = self._normalize_chunks(chunks, doc_id)
                if normalized_chunks:
                    normalized_chunks_by_document[doc_id] = normalized_chunks
            if not normalized_chunks_by_document:
                return []
            batch_size = batch_size or self.batch_size
            result_ids = self._process_from_chunks_pipeline(
                normalized_chunks_by_document,
                metadata_batch,
                batch_size,
                similarity_threshold,
                mode="upsert",
            )
            self._save_next_doc_id()
            self._save_internal()
            return result_ids

    def insert(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
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

        Returns
        -------
        List[str]
            List of document IDs that were actually inserted
        """
        with self._read_write_lock.write_lock():
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
            self._validate_metadata_batch(metadata)
            existing_ids = set()
            with self.connection_pool.get_connection() as conn:
                if ids:
                    placeholders = ','.join(['?'] * len(ids))
                    cursor = conn.execute(f'SELECT id FROM documents WHERE id IN ({placeholders})', ids)
                    existing_ids = {row['id'] for row in cursor.fetchall()}
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
                return []
            docs_to_process = [d[0] for d in docs_to_insert]
            meta_to_process = [d[1] for d in docs_to_insert]
            ids_to_process = [d[2] for d in docs_to_insert]
            batch_size = batch_size or self.batch_size
            result_ids = self._process_with_pipeline(
                docs_to_process, meta_to_process, ids_to_process, batch_size, similarity_threshold, mode="insert"
            )
            self._save_next_doc_id()
            self._save_internal()
            return result_ids

    def insert_from_file(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            extractor_kwargs: Optional[Dict[str, Any]] = None,
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
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")
        documents, merged_metadata, final_ids = [], [], []
        extractor_kwargs = extractor_kwargs or {}
        for i, file_path in enumerate(file_paths):
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            file_content = file_path.read_bytes()
            filename = file_path.name
            extraction_result = ExtractorRegistry.extract_text(file_content, filename, **extractor_kwargs)
            if not extraction_result.success:
                raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")
            documents.append(extraction_result.text)
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                doc_id = file_path.stem
            final_ids.append(doc_id)
        batch_size = batch_size or self.batch_size
        return self.insert(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            errors=errors,
        )

    def insert_from_chunks(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
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
            if not chunks_by_document:
                return []
            if metadata is None:
                metadata = {}
            doc_ids = list(chunks_by_document.keys())
            existing_ids = set()
            with self.connection_pool.get_connection() as conn:
                if doc_ids:
                    placeholders = ','.join(['?'] * len(doc_ids))
                    cursor = conn.execute(f'SELECT id FROM documents WHERE id IN ({placeholders})', doc_ids)
                    existing_ids = {row['id'] for row in cursor.fetchall()}
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
                return []
            self._validate_metadata_batch(list(metadata_to_insert.values()))
            normalized_chunks_by_document = {}
            for doc_id, chunks in chunks_to_insert.items():
                normalized_chunks = self._normalize_chunks(chunks, doc_id)
                if normalized_chunks:
                    normalized_chunks_by_document[doc_id] = normalized_chunks
            if not normalized_chunks_by_document:
                return []
            result_ids = self._process_from_chunks_pipeline(
                normalized_chunks_by_document,
                metadata_to_insert,
                batch_size,
                similarity_threshold,
                mode="insert",
            )
            self._save_next_doc_id()
            self._save_internal()
            return result_ids

    # ------------------
    # Chunk normalization
    # ------------------
    def _normalize_chunks(self, chunks: Union[List[Chunk], List[str]], doc_id: str) -> List[Chunk]:
        if not chunks:
            return []
        normalized_chunks = []
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, Chunk):
                if chunk.content_hash is None:
                    chunk.content_hash = chunk.calculate_content_hash()
                if chunk.index != i:
                    logger.warning(
                        f"Chunk index mismatch in document {doc_id}: expected {i}, got {chunk.index}. Correcting index."
                    )
                    chunk.index = i
                normalized_chunks.append(chunk)
            elif isinstance(chunk, str):
                position = ChunkPosition(
                    start=0,
                    end=len(chunk),
                    line=1,
                    column=1,
                    end_line=1,
                    end_column=len(chunk) + 1,
                )
                chunk_obj = Chunk(
                    content=chunk,
                    position=position,
                    tokens=self.chunker.count_tokens(chunk),
                    index=i,
                    faiss_id=None,
                    content_hash=None,
                )
                normalized_chunks.append(chunk_obj)
            else:
                raise ValueError(
                    f"Invalid chunk type in document {doc_id} at index {i}: expected Chunk or str, got {type(chunk)}"
                )
        return normalized_chunks

    # --------------------
    # Similarity filtering
    # --------------------
    def _filter_similar_chunks_vectorized(
            self, embeddings: np.ndarray, chunks: List[Chunk], doc_chunk_mapping: List[Tuple],
            similarity_threshold: float, existing_chunk_hashes: Optional[set] = None
    ):
        """
        Filter similar chunks based on content hash and vector similarity.

        Parameters:
            existing_chunk_hashes: Optional pre-computed set of chunk hashes to avoid repeated DB queries.
                                  If None, will query from database (expensive on large datasets).
        """
        if len(embeddings) == 0 or self.index.ntotal == 0:
            return chunks, embeddings, doc_chunk_mapping

        # Use provided hashes or query from database
        if existing_chunk_hashes is None:
            existing_chunk_hashes = set()
            with self.connection_pool.get_connection() as conn:
                cursor = conn.execute('SELECT DISTINCT content_hash FROM chunks')
                existing_chunk_hashes = {row['content_hash'] for row in cursor.fetchall()}

        hash_mask = np.array([chunk.content_hash not in existing_chunk_hashes for chunk in chunks])
        if not hash_mask.any():
            logger.debug("All chunks filtered out by content hash")
            return [], np.array([]).reshape(0, self.embedding_dimension), []
        filtered_chunks = [chunks[i] for i in range(len(chunks)) if hash_mask[i]]
        filtered_embeddings = embeddings[hash_mask]
        filtered_mappings = [doc_chunk_mapping[i] for i in range(len(doc_chunk_mapping)) if hash_mask[i]]
        if self.index.ntotal == 0 or similarity_threshold is None or similarity_threshold <= 0:
            return filtered_chunks, filtered_embeddings, filtered_mappings

        # Get the metric type and convert similarity to distance threshold
        metric_type = self._get_faiss_metric_type()
        if metric_type == 'IP':
            # For inner product, higher values mean more similar
            # We want to filter out chunks that are TOO similar (above threshold)
            distance_threshold = self._similarity_to_distance(similarity_threshold, metric_type)
            with self._faiss_lock.read_lock():
                distances, indices = self.index.search(filtered_embeddings, k=1)
            valid_matches = (indices[:, 0] != -1)
            too_similar = (distances[:, 0] > distance_threshold) & valid_matches
        else:
            # For L2, lower distances mean more similar
            # We want to filter out chunks that are TOO similar (below threshold)
            distance_threshold = self._similarity_to_distance(similarity_threshold, metric_type)
            with self._faiss_lock.read_lock():
                distances, indices = self.index.search(filtered_embeddings, k=1)
            valid_matches = (indices[:, 0] != -1)
            too_similar = (distances[:, 0] < distance_threshold) & valid_matches
        keep_mask = ~too_similar
        final_chunks = [filtered_chunks[i] for i in range(len(filtered_chunks)) if keep_mask[i]]
        final_embeddings = filtered_embeddings[keep_mask]
        final_mappings = [filtered_mappings[i] for i in range(len(filtered_mappings)) if keep_mask[i]]
        removed_count = len(chunks) - len(final_chunks)
        logger.debug(
            f"Similarity filtering: {len(chunks)} → {len(final_chunks)} chunks "
            f"(removed {removed_count} similar/duplicate)"
        )
        return final_chunks, final_embeddings, final_mappings

    # ------------------
    # Bulk DB operations
    # ------------------
    def _insert_documents_bulk(
            self, conn, documents_data: List[Tuple[str, str, str, Dict[str, Any]]],
            mode: Literal["insert", "replace"] = "replace"
    ) -> None:
        if not documents_data:
            return

        # Use shared business logic for SQL and data preparation
        sql, _ = self._build_documents_bulk_insert_sql(mode)
        # Pass connection to preserve created_at timestamps for upserts
        bulk_data = self._prepare_documents_bulk_data(documents_data, conn=conn,
                                                      preserve_created_at=(mode == "replace"))
        conn.executemany(sql, bulk_data)

    @staticmethod
    def _insert_chunks_bulk(conn, chunks_data: List[Tuple[str, Chunk]]) -> None:
        if not chunks_data:
            return
        bulk_data = []
        for doc_id, chunk in chunks_data:
            bulk_data.append(
                (
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
                    chunk.faiss_id,
                )
            )
        conn.executemany(
            '''
            INSERT INTO chunks
            (document_id, chunk_index, content, content_hash, start_pos, end_pos, start_line,
            start_col, end_line, end_col, tokens, faiss_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            bulk_data,
        )

    # -----------------
    # Pipelines (sync)
    # -----------------
    def _process_with_pipeline(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str], batch_size: int,
            similarity_threshold: Optional[float],
            # queue_size: int = 3,
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> List[str]:
        queue_size = self.pipeline_queue_size
        existing_chunks_by_doc = self._fetch_existing_chunks_batch(ids)
        chunk_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        embedding_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        result_queue: queue.Queue = queue.Queue()
        total_docs = len(documents)

        def chunking_worker():
            try:
                for i, (doc_text, metadata, doc_id) in enumerate(zip(documents, metadata_batch, ids, strict=False)):
                    content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
                    chunks = self.chunker.chunk(doc_text)
                    existing_chunks = existing_chunks_by_doc.get(doc_id, {})
                    unchanged_chunks, chunks_needing_embedding, chunk_texts_for_embedding = [], [], []
                    reused_chunk_indices = set()
                    for chunk in chunks:
                        existing_chunk = existing_chunks.get(chunk.index)
                        if (existing_chunk and existing_chunk['content_hash'] == chunk.content_hash and existing_chunk[
                            'faiss_id'] is not None):
                            chunk.faiss_id = existing_chunk['faiss_id']
                            unchanged_chunks.append(chunk)
                            reused_chunk_indices.add(chunk.index)
                        else:
                            chunks_needing_embedding.append(chunk)
                            chunk_texts_for_embedding.append(chunk.content)
                    chunk_indices_to_remove, faiss_ids_to_remove = [], []
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
                        'faiss_ids_to_remove': faiss_ids_to_remove,
                    }
                    chunk_queue.put(chunk_data)
                chunk_queue.put(None)
            except Exception as e:
                logger.error(f"Chunking worker error: {e}")
                chunk_queue.put(None)
                raise

        def embedding_worker():
            try:
                embedding_enabled_fields = self._get_embedding_enabled_fields()
                while True:
                    chunk_data = chunk_queue.get()
                    if chunk_data is None:
                        embedding_queue.put(None)
                        break
                    chunk_texts = chunk_data['chunk_texts_for_embedding']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    if chunk_texts:
                        embeddings = self.embedding_provider.embed_sync(chunk_texts, batch_size)
                        chunk_data['new_embeddings'] = embeddings
                        for chunk in chunks_needing_embedding:
                            chunk.faiss_id = None
                    else:
                        chunk_data['new_embeddings'] = np.array([]).reshape(0, self.embedding_dimension)
                    if embedding_enabled_fields:
                        metadata = chunk_data['metadata']
                        field_embeddings = self._generate_metadata_embeddings(metadata, embedding_enabled_fields,
                                                                              batch_size)
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
            try:
                while True:
                    chunk_data = embedding_queue.get()
                    if chunk_data is None:
                        result_queue.put(None)
                        break
                    unchanged_chunks = chunk_data['unchanged_chunks']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    new_embeddings = chunk_data['new_embeddings']
                    field_embeddings = chunk_data['field_embeddings']
                    if similarity_threshold is not None and len(chunks_needing_embedding) > 0:
                        doc_info = (chunk_data['doc_text'], chunk_data['metadata'], chunk_data['doc_id'],
                                    chunk_data['content_hash'])
                        doc_chunk_mapping = [doc_info] * len(chunks_needing_embedding)
                        filtered_chunks, filtered_embeddings, _ = self._filter_similar_chunks_vectorized(
                            new_embeddings, chunks_needing_embedding, doc_chunk_mapping, similarity_threshold
                        )
                        chunks_needing_embedding = filtered_chunks
                        new_embeddings = filtered_embeddings
                    all_chunks = unchanged_chunks + chunks_needing_embedding
                    if len(all_chunks) > 0 or mode == "upsert":
                        documents_data = [(chunk_data['doc_id'], chunk_data['doc_text'], chunk_data['content_hash'],
                                           chunk_data['metadata'])]
                        chunks_data = [(chunk_data['doc_id'], chunk) for chunk in all_chunks]
                        with self.connection_pool.get_connection() as conn:
                            conn.execute('BEGIN')
                            try:
                                if mode == "upsert":
                                    self._remove_metadata_embeddings(conn, chunk_data['doc_id'])
                                    self._remove_old_chunks_batch(
                                        conn, chunk_data['doc_id'], chunk_data['chunk_indices_to_remove'],
                                        chunk_data['faiss_ids_to_remove']
                                    )
                                self._insert_documents_bulk(conn, documents_data, mode=mode)
                                if new_embeddings.size > 0:
                                    self._add_vectors_to_faiss_bulk(new_embeddings, chunks_needing_embedding)
                                self._insert_chunks_bulk(conn, chunks_data)
                                if field_embeddings:
                                    self._store_metadata_embeddings(conn, chunk_data['doc_id'], field_embeddings)
                                conn.commit()
                            except Exception:
                                conn.rollback()
                                raise
                    result_queue.put(chunk_data['doc_id'])
                    embedding_queue.task_done()
            except Exception as e:
                logger.error(f"Database worker error: {e}")
                result_queue.put(None)
                raise

        workers = [
            threading.Thread(target=chunking_worker, name="ChunkingWorker"),
            threading.Thread(target=embedding_worker, name="EmbeddingWorker"),
            threading.Thread(target=database_worker, name="DatabaseWorker"),
        ]
        for w in workers:
            w.start()
        processed_ids = []
        try:
            while len(processed_ids) < total_docs:
                result = result_queue.get()
                if result is None:
                    break
                processed_ids.append(result)
        finally:
            # Join workers without timeout to allow completion of heavy ingestion
            for w in workers:
                w.join()
                if w.is_alive():
                    logger.warning(f"Worker {w.name} is still alive after join()")
        return processed_ids

    def _process_from_chunks_pipeline(
            self,
            chunks_by_document: Dict[str, List[Chunk]],
            metadata_batch: Dict[str, Dict[str, Any]],
            batch_size: int,
            similarity_threshold: Optional[float],
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> List[str]:

        queue_size = self.pipeline_queue_size
        doc_ids = list(chunks_by_document.keys())
        existing_chunks_by_doc = self._fetch_existing_chunks_batch(doc_ids)
        embedding_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        result_queue: queue.Queue = queue.Queue()
        total_docs = len(doc_ids)

        def chunk_comparison_worker():
            try:
                for doc_id, chunks in chunks_by_document.items():
                    metadata = metadata_batch.get(doc_id, {})
                    existing_chunks = existing_chunks_by_doc.get(doc_id, {})
                    unchanged_chunks, chunks_needing_embedding, chunk_texts_for_embedding = [], [], []
                    reused_chunk_indices = set()
                    for chunk in chunks:
                        existing_chunk = existing_chunks.get(chunk.index)
                        if (existing_chunk and existing_chunk['content_hash'] == chunk.content_hash and existing_chunk[
                            'faiss_id'] is not None):
                            chunk.faiss_id = existing_chunk['faiss_id']
                            unchanged_chunks.append(chunk)
                            reused_chunk_indices.add(chunk.index)
                        else:
                            chunks_needing_embedding.append(chunk)
                            chunk_texts_for_embedding.append(chunk.content)
                    chunk_indices_to_remove, faiss_ids_to_remove = [], []
                    for chunk_index, chunk_info in existing_chunks.items():
                        if chunk_index not in reused_chunk_indices:
                            chunk_indices_to_remove.append(chunk_index)
                            if chunk_info['faiss_id'] is not None:
                                faiss_ids_to_remove.append(chunk_info['faiss_id'])
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
                        'faiss_ids_to_remove': faiss_ids_to_remove,
                    }
                    embedding_queue.put(chunk_data)
                embedding_queue.put(None)
            except Exception as e:
                logger.error(f"Chunk comparison worker error: {e}")
                embedding_queue.put(None)
                raise

        def embedding_worker():
            try:
                embedding_enabled_fields = self._get_embedding_enabled_fields()
                while True:
                    chunk_data = embedding_queue.get()
                    if chunk_data is None:
                        result_queue.put(None)
                        break
                    chunk_texts = chunk_data['chunk_texts_for_embedding']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    if chunk_texts:
                        embeddings = self.embedding_provider.embed_sync(chunk_texts, batch_size)
                        chunk_data['new_embeddings'] = embeddings
                        for chunk in chunks_needing_embedding:
                            chunk.faiss_id = None
                    else:
                        chunk_data['new_embeddings'] = np.array([]).reshape(0, self.embedding_dimension)
                    if embedding_enabled_fields:
                        metadata = chunk_data['metadata']
                        field_embeddings = self._generate_metadata_embeddings(metadata, embedding_enabled_fields,
                                                                              batch_size)
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
            try:
                processed_ids = []
                while len(processed_ids) < total_docs:
                    chunk_data = result_queue.get()
                    if chunk_data is None:
                        break
                    unchanged_chunks = chunk_data['unchanged_chunks']
                    chunks_needing_embedding = chunk_data['chunks_needing_embedding']
                    new_embeddings = chunk_data['new_embeddings']
                    field_embeddings = chunk_data['field_embeddings']
                    if similarity_threshold is not None and len(chunks_needing_embedding) > 0:
                        doc_info = (chunk_data['doc_text'], chunk_data['metadata'], chunk_data['doc_id'],
                                    chunk_data['content_hash'])
                        doc_chunk_mapping = [doc_info] * len(chunks_needing_embedding)
                        filtered_chunks, filtered_embeddings, _ = self._filter_similar_chunks_vectorized(
                            new_embeddings, chunks_needing_embedding, doc_chunk_mapping, similarity_threshold
                        )
                        chunks_needing_embedding = filtered_chunks
                        new_embeddings = filtered_embeddings
                    all_chunks = unchanged_chunks + chunks_needing_embedding
                    if len(all_chunks) > 0 or mode == "upsert":
                        documents_data = [(chunk_data['doc_id'], chunk_data['doc_text'], chunk_data['content_hash'],
                                           chunk_data['metadata'])]
                        chunks_data = [(chunk_data['doc_id'], chunk) for chunk in all_chunks]
                        with self.connection_pool.get_connection() as conn:
                            conn.execute('BEGIN')
                            try:
                                if mode == "upsert":
                                    self._remove_metadata_embeddings(conn, chunk_data['doc_id'])
                                    self._remove_old_chunks_batch(
                                        conn, chunk_data['doc_id'], chunk_data['chunk_indices_to_remove'],
                                        chunk_data['faiss_ids_to_remove']
                                    )
                                self._insert_documents_bulk(conn, documents_data, mode=mode)
                                if new_embeddings.size > 0:
                                    self._add_vectors_to_faiss_bulk(new_embeddings, chunks_needing_embedding)
                                self._insert_chunks_bulk(conn, chunks_data)
                                if field_embeddings:
                                    self._store_metadata_embeddings(conn, chunk_data['doc_id'], field_embeddings)
                                conn.commit()
                            except Exception:
                                conn.rollback()
                                raise
                    processed_ids.append(chunk_data['doc_id'])
                    result_queue.task_done()
                return processed_ids
            except Exception as e:
                logger.error(f"Database worker error: {e}")
                raise

        workers = [
            threading.Thread(target=chunk_comparison_worker, name="ChunkComparisonWorker"),
            threading.Thread(target=embedding_worker, name="EmbeddingWorker"),
        ]
        for w in workers:
            w.start()
        try:
            processed_ids = database_worker()
        finally:
            # Join workers without timeout to allow completion of heavy ingestion
            for w in workers:
                w.join()
                if w.is_alive():
                    logger.warning(f"Worker {w.name} is still alive after join()")
        return processed_ids

    def _fetch_existing_chunks_batch(self, doc_ids: List[str]):
        if not doc_ids:
            return {}
        existing = {}
        with self.connection_pool.get_connection() as conn:
            placeholders = ','.join(['?'] * len(doc_ids))
            query = f'''SELECT document_id, chunk_index, content_hash, faiss_id FROM chunks
                        WHERE document_id IN ({placeholders})'''
            cursor = conn.execute(query, doc_ids)
            for row in cursor.fetchall():
                doc_id = row['document_id']
                if doc_id not in existing:
                    existing[doc_id] = {}
                existing[doc_id][row['chunk_index']] = {
                    'content_hash': row['content_hash'], 'faiss_id': row['faiss_id']
                }
        logger.debug(f"Fetched existing chunks for {len(existing)} documents")
        return existing

    def _remove_old_chunks_batch(
            self, conn, doc_id: str, chunk_indices_to_remove: List[int], faiss_ids_to_remove: List[int]
    ) -> None:
        if chunk_indices_to_remove:
            placeholders = ','.join(['?'] * len(chunk_indices_to_remove))
            conn.execute(
                f'DELETE FROM chunks WHERE document_id = ? AND chunk_index IN ({placeholders})',
                [doc_id] + chunk_indices_to_remove,
            )
        if faiss_ids_to_remove:
            self._remove_old_vectors_bulk(faiss_ids_to_remove)
            logger.debug(
                f"Removed {len(chunk_indices_to_remove)} old chunks and "
                f"{len(faiss_ids_to_remove)} FAISS vectors for {doc_id}"
            )

    # ------------------
    # Public APIs (async)
    # ------------------
    async def upsert_async(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
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
            ids = [await self._generate_doc_id_async() for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")
        new_ids = []
        for i in ids:
            if i is None:
                new_ids.append(await self._generate_doc_id_async())
            else:
                new_ids.append(i)
        ids = new_ids
        self._validate_metadata_batch(metadata)
        result_ids = await self._async_pipeline_process(
            documents, metadata, ids, batch_size, similarity_threshold, max_concurrent_chunks,
            max_concurrent_embeddings, mode="upsert"
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_next_doc_id)
        await loop.run_in_executor(None, self._save_internal)
        return result_ids

    async def upsert_from_file_async(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
            extractor_kwargs: Optional[Dict[str, Any]] = None,
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
        loop = asyncio.get_event_loop()
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        async def extract_file_text(file_path: Path, index: int):
            def _extract():
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                file_content = file_path.read_bytes()
                filename = file_path.name
                extraction_result = ExtractorRegistry.extract_text(file_content, filename, **(extractor_kwargs or {}))
                if not extraction_result.success:
                    raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")
                return extraction_result

            return await loop.run_in_executor(None, _extract)

        extraction_tasks = [extract_file_text(file_path, i) for i, file_path in enumerate(file_paths)]
        extraction_results = await asyncio.gather(*extraction_tasks)
        documents, merged_metadata, final_ids = [], [], []
        for i, (file_path, extraction_result) in enumerate(zip(file_paths, extraction_results, strict=False)):
            documents.append(extraction_result.text)
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                doc_id = file_path.stem
            final_ids.append(doc_id)
        batch_size = batch_size or self.batch_size
        return await self.upsert_async(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            max_concurrent_chunks=max_concurrent_chunks,
            max_concurrent_embeddings=max_concurrent_embeddings,
        )

    async def upsert_from_chunks_async(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
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
        await self._ensure_async_schema_initialized()
        if not chunks_by_document:
            return []
        if metadata is None:
            metadata = {}
        metadata_batch = {doc_id: metadata.get(doc_id, {}) for doc_id in chunks_by_document.keys()}
        self._validate_metadata_batch(list(metadata_batch.values()))
        normalized_chunks_by_document = {}
        for doc_id, chunks in chunks_by_document.items():
            normalized_chunks = self._normalize_chunks(chunks, doc_id)
            if normalized_chunks:
                normalized_chunks_by_document[doc_id] = normalized_chunks
        if not normalized_chunks_by_document:
            return []
        result_ids = await self._async_process_from_chunks_pipeline(
            normalized_chunks_by_document,
            metadata_batch,
            batch_size,
            similarity_threshold,
            max_concurrent_chunks,
            max_concurrent_embeddings,
            mode="upsert",
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_next_doc_id)
        await self.save_async()
        return result_ids

    async def insert_async(
            self,
            documents: Union[str, List[str]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
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
            ids = [await self._generate_doc_id_async() for _ in documents]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")
        self._validate_metadata_batch(metadata)
        existing_ids = await self._check_existing_ids_async(ids)
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
            return []
        docs_to_process = [item[0] for item in docs_to_insert]
        meta_to_process = [item[1] for item in docs_to_insert]
        ids_to_process = [item[2] for item in docs_to_insert]
        result_ids = await self._async_pipeline_process(
            docs_to_process, meta_to_process, ids_to_process, batch_size, similarity_threshold, max_concurrent_chunks,
            max_concurrent_embeddings, mode="insert"
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_next_doc_id)
        await loop.run_in_executor(None, self._save_internal)
        return result_ids

    async def insert_from_file_async(
            self,
            file_paths: Union[str, Path, List[Union[str, Path]]],
            metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            ids: Optional[Union[str, List[str]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
            extractor_kwargs: Optional[Dict[str, Any]] = None,
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
        loop = asyncio.get_event_loop()
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        file_paths = [Path(p) for p in file_paths]
        if isinstance(metadata, dict):
            metadata = [metadata]
        if isinstance(ids, str):
            ids = [ids]
        if metadata is not None and len(metadata) != len(file_paths):
            raise ValueError("Number of metadata entries must match number of files")
        if ids is not None and len(ids) != len(file_paths):
            raise ValueError("Number of IDs must match number of files")

        async def extract_file_text(file_path: Path, index: int):
            def _extract():
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                file_content = file_path.read_bytes()
                filename = file_path.name
                extraction_result = ExtractorRegistry.extract_text(file_content, filename, **(extractor_kwargs or {}))
                if not extraction_result.success:
                    raise ValueError(f"Failed to extract text from {file_path}: {extraction_result.error}")
                return extraction_result

            return await loop.run_in_executor(None, _extract)

        extraction_tasks = [extract_file_text(file_path, i) for i, file_path in enumerate(file_paths)]
        extraction_results = await asyncio.gather(*extraction_tasks)
        documents, merged_metadata, final_ids = [], [], []
        for i, (file_path, extraction_result) in enumerate(zip(file_paths, extraction_results, strict=False)):
            documents.append(extraction_result.text)
            doc_metadata = extraction_result.metadata.copy() if extraction_result.metadata else {}
            if metadata is not None and i < len(metadata):
                doc_metadata.update(metadata[i])
            merged_metadata.append(doc_metadata)
            if ids is not None and i < len(ids):
                doc_id = ids[i]
            else:
                doc_id = file_path.stem
            final_ids.append(doc_id)
        return await self.insert_async(
            documents=documents,
            metadata=merged_metadata,
            ids=final_ids,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
            errors=errors,
            max_concurrent_chunks=max_concurrent_chunks,
            max_concurrent_embeddings=max_concurrent_embeddings,
        )

    async def insert_from_chunks_async(
            self,
            chunks_by_document: Dict[str, Union[List[Chunk], List[str]]],
            metadata: Optional[Dict[str, Dict[str, Any]]] = None,
            batch_size: Optional[int] = None,
            similarity_threshold: Optional[float] = None,
            errors: Literal["ignore", "raise"] = "raise",
            max_concurrent_chunks: int = 3,
            max_concurrent_embeddings: int = 2,
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
        await self._ensure_async_schema_initialized()
        if not chunks_by_document:
            return []
        if metadata is None:
            metadata = {}
        doc_ids = list(chunks_by_document.keys())
        existing_ids = await self._check_existing_ids_async(doc_ids)
        chunks_to_insert, metadata_to_insert = {}, {}
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
            return []
        self._validate_metadata_batch(list(metadata_to_insert.values()))
        normalized_chunks_by_document = {}
        for doc_id, chunks in chunks_to_insert.items():
            normalized_chunks = self._normalize_chunks(chunks, doc_id)
            if normalized_chunks:
                normalized_chunks_by_document[doc_id] = normalized_chunks
        if not normalized_chunks_by_document:
            return []
        result_ids = await self._async_process_from_chunks_pipeline(
            normalized_chunks_by_document,
            metadata_to_insert,
            batch_size,
            similarity_threshold,
            max_concurrent_chunks,
            max_concurrent_embeddings,
            mode="insert",
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._save_next_doc_id)
        await self.save_async()
        return result_ids

    async def _check_existing_ids_async(self, ids: List[str]) -> set:
        if not ids:
            return set()
        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ','.join(['?'] * len(ids))
            cursor = await conn.execute(f'SELECT id FROM documents WHERE id IN ({placeholders})', ids)
            rows = await cursor.fetchall()
            return {row['id'] for row in rows}

    # -------------------
    # Pipelines (async)
    # -------------------
    async def _async_pipeline_process(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            batch_size: int,
            similarity_threshold: Optional[float],
            max_concurrent_chunks: int,
            max_concurrent_embeddings: int,
            mode: Literal["upsert", "insert"] = "upsert",
    ) -> List[str]:
        existing_chunks_by_doc = await self._fetch_existing_chunks_batch_async(ids)

        # Use asyncio.Queue for proper async pipeline communication
        chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=self.pipeline_queue_size)
        embedding_queue: asyncio.Queue = asyncio.Queue(maxsize=self.pipeline_queue_size)

        # Use asyncio.Semaphore for proper async concurrency control
        chunk_semaphore = asyncio.Semaphore(max_concurrent_chunks)
        embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)

        result_ids: List[str] = []
        total_docs = len(documents)

        # Create tasks for pipeline stages
        chunking_task = asyncio.create_task(
            self._chunking_stage(documents, metadata_batch, ids, existing_chunks_by_doc,
                               chunk_queue, chunk_semaphore)
        )

        embedding_task = asyncio.create_task(
            self._embedding_stage(chunk_queue, embedding_queue, batch_size, embedding_semaphore)
        )

        database_task = asyncio.create_task(
            self._database_stage(embedding_queue, similarity_threshold, mode, result_ids, total_docs)
        )

        # Run all pipeline stages concurrently
        await asyncio.gather(chunking_task, embedding_task, database_task)
        return result_ids

    async def _chunking_stage(
            self,
            documents: List[str],
            metadata_batch: List[Dict[str, Any]],
            ids: List[str],
            existing_chunks_by_doc: Dict[str, Dict[int, Dict[str, Any]]],
            chunk_queue: asyncio.Queue,
            chunk_semaphore: asyncio.Semaphore
    ) -> None:
        try:
            # Create concurrent chunking tasks
            tasks = []
            for i, (doc_text, metadata, doc_id) in enumerate(zip(documents, metadata_batch, ids, strict=False)):
                task = asyncio.create_task(
                    self._chunk_document_with_comparison_async(
                        i, doc_id, doc_text, metadata,
                        existing_chunks_by_doc.get(doc_id, {}),
                        chunk_semaphore
                    )
                )
                tasks.append(task)

            # Process completed tasks as they finish and send to next stage
            for task in asyncio.as_completed(tasks):
                chunk_data = await task
                await chunk_queue.put(chunk_data)

            # Signal completion to next stage
            await chunk_queue.put(None)
        except Exception as e:
            logger.error(f"Async chunking stage error: {e}")
            await chunk_queue.put(None)
            raise

    async def _embedding_stage(
            self,
            chunk_queue: asyncio.Queue,
            embedding_queue: asyncio.Queue,
            batch_size: int,
            embedding_semaphore: asyncio.Semaphore
    ) -> None:
        try:
            while True:
                chunk_data = await chunk_queue.get()
                if chunk_data is None:
                    await embedding_queue.put(None)
                    break

                # Process embeddings with proper async concurrency control
                async with embedding_semaphore:
                    chunk_texts_for_embedding = chunk_data['chunk_texts_for_embedding']
                    if chunk_texts_for_embedding:
                        new_embeddings = await self.embedding_provider.embed_batch(
                            chunk_texts_for_embedding, batch_size
                        )
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
                chunk_queue.task_done()
        except Exception as e:
            logger.error(f"Async embedding stage error: {e}")
            await embedding_queue.put(None)
            raise

    async def _database_stage(
            self,
            embedding_queue: asyncio.Queue,
            similarity_threshold: Optional[float],
            mode: Literal["upsert", "insert"],
            result_ids: List[str],
            total_docs: int
    ) -> None:
        try:
            processed_count = 0
            while processed_count < total_docs:
                chunk_data = await embedding_queue.get()
                if chunk_data is None:
                    break

                doc_id = await self._process_document_data_async(chunk_data, similarity_threshold, mode)
                if doc_id:
                    result_ids.append(doc_id)
                    processed_count += 1

                embedding_queue.task_done()
        except Exception as e:
            logger.error(f"Async database stage error: {e}")
            raise

    async def _async_process_from_chunks_pipeline(
            self,
            chunks_by_document: Dict[str, List[Chunk]],
            metadata_batch: Dict[str, Dict[str, Any]],
            batch_size: int,
            similarity_threshold: Optional[float],
            max_concurrent_chunks: int,
            max_concurrent_embeddings: int,
            mode: Literal["upsert", "insert"] = "upsert",
    ) -> List[str]:
        doc_ids = list(chunks_by_document.keys())
        existing_chunks_by_doc = await self._fetch_existing_chunks_batch_async(doc_ids)

        # Use asyncio.Queue for proper async pipeline communication
        processing_queue: asyncio.Queue = asyncio.Queue(maxsize=self.pipeline_queue_size)

        # Use asyncio.Semaphore for proper async concurrency control
        embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)

        result_ids: List[str] = []
        total_docs = len(doc_ids)

        # Create tasks for pipeline stages
        comparison_task = asyncio.create_task(
            self._chunk_comparison_stage(chunks_by_document, metadata_batch, existing_chunks_by_doc,
                                       processing_queue, batch_size, embedding_semaphore)
        )

        database_task = asyncio.create_task(
            self._chunk_database_stage(processing_queue, similarity_threshold, mode, result_ids, total_docs)
        )

        # Run pipeline stages concurrently
        await asyncio.gather(comparison_task, database_task)
        return result_ids

    async def _chunk_comparison_stage(
            self,
            chunks_by_document: Dict[str, List[Chunk]],
            metadata_batch: Dict[str, Dict[str, Any]],
            existing_chunks_by_doc: Dict[str, Dict[int, Dict[str, Any]]],
            processing_queue: asyncio.Queue,
            batch_size: int,
            embedding_semaphore: asyncio.Semaphore
    ) -> None:
        try:
            # Create concurrent chunk comparison and embedding tasks
            tasks = []
            for doc_id, chunks in chunks_by_document.items():
                metadata = metadata_batch.get(doc_id, {})
                task = asyncio.create_task(
                    self._compare_chunks_and_prepare_async(
                        doc_id, chunks, metadata,
                        existing_chunks_by_doc.get(doc_id, {}),
                        batch_size, embedding_semaphore
                    )
                )
                tasks.append(task)

            # Process completed tasks as they finish and send to next stage
            for task in asyncio.as_completed(tasks):
                chunk_data = await task
                if chunk_data:
                    await processing_queue.put(chunk_data)

            # Signal completion to next stage
            await processing_queue.put(None)
        except Exception as e:
            logger.error(f"Async chunk comparison stage error: {e}")
            await processing_queue.put(None)
            raise

    async def _chunk_database_stage(
            self,
            processing_queue: asyncio.Queue,
            similarity_threshold: Optional[float],
            mode: Literal["upsert", "insert"],
            result_ids: List[str],
            total_docs: int
    ) -> None:
        try:
            processed_count = 0
            while processed_count < total_docs:
                chunk_data = await processing_queue.get()
                if chunk_data is None:
                    break

                doc_id = await self._process_document_data_async(chunk_data, similarity_threshold, mode)
                if doc_id:
                    result_ids.append(doc_id)
                    processed_count += 1

                processing_queue.task_done()
        except Exception as e:
            logger.error(f"Async chunk database stage error: {e}")
            raise

    async def _compare_chunks_and_prepare_async(
            self, doc_id: str, chunks: List[Chunk], metadata: Dict[str, Any],
            existing_chunks: Dict[int, Dict[str, Any]], batch_size: int, embedding_semaphore: asyncio.Semaphore
    ):
        try:
            unchanged_chunks, chunks_needing_embedding, chunk_texts_for_embedding = [], [], []
            reused_chunk_indices = set()
            for chunk in chunks:
                existing_chunk = existing_chunks.get(chunk.index)
                if existing_chunk and existing_chunk['content_hash'] == chunk.content_hash and existing_chunk[
                    'faiss_id'] is not None:
                    chunk.faiss_id = existing_chunk['faiss_id']
                    unchanged_chunks.append(chunk)
                    reused_chunk_indices.add(chunk.index)
                else:
                    chunks_needing_embedding.append(chunk)
                    chunk_texts_for_embedding.append(chunk.content)
            chunk_indices_to_remove, faiss_ids_to_remove = [], []
            for chunk_index, chunk_info in existing_chunks.items():
                if chunk_index not in reused_chunk_indices:
                    chunk_indices_to_remove.append(chunk_index)
                    if chunk_info['faiss_id'] is not None:
                        faiss_ids_to_remove.append(chunk_info['faiss_id'])
            if chunk_texts_for_embedding:
                async with embedding_semaphore:
                    new_embeddings = await self.embedding_provider.embed_batch(chunk_texts_for_embedding, batch_size)
                    for chunk in chunks_needing_embedding:
                        chunk.faiss_id = None
            else:
                new_embeddings = np.array([]).reshape(0, self.embedding_dimension)
            embedding_enabled_fields = self._get_embedding_enabled_fields()
            field_embeddings = {}
            if embedding_enabled_fields:
                field_embeddings = await self._generate_metadata_embeddings_async(metadata, embedding_enabled_fields,
                                                                                  batch_size)
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
                'field_embeddings': field_embeddings,
            }
        except Exception as e:
            logger.error(f"Error comparing chunks for document {doc_id}: {e}")
            return None

    async def _fetch_existing_chunks_batch_async(self, doc_ids: List[str]):
        if not doc_ids:
            return {}
        existing = {}
        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ','.join(['?'] * len(doc_ids))
            query = f'''SELECT document_id, chunk_index, content_hash, faiss_id FROM chunks
                        WHERE document_id IN ({placeholders})'''
            cursor = await conn.execute(query, doc_ids)
            async for row in cursor:
                doc_id = row['document_id']
                if doc_id not in existing:
                    existing[doc_id] = {}
                existing[doc_id][row['chunk_index']] = {
                    'content_hash': row['content_hash'], 'faiss_id': row['faiss_id']
                }
        logger.debug(f"Fetched existing chunks for {len(existing)} documents")
        return existing

    async def _chunk_document_with_comparison_async(
            self, doc_index: int, doc_id: str, doc_text: str, metadata: Dict[str, Any],
            existing_chunks: Dict[int, Dict[str, Any]], semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            # Use asyncio.to_thread for CPU-bound chunking operation in Python 3.9+
            # Falls back to run_in_executor for compatibility
            try:
                chunks = await asyncio.to_thread(self.chunker.chunk, doc_text)
            except AttributeError:
                # Fallback for Python < 3.9
                loop = asyncio.get_event_loop()
                chunks = await loop.run_in_executor(None, self.chunker.chunk, doc_text)

            content_hash = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
            unchanged_chunks, chunks_needing_embedding, chunk_texts_for_embedding = [], [], []
            reused_chunk_indices = set()

            # Process chunks to determine which need embedding
            for chunk in chunks:
                existing_chunk = existing_chunks.get(chunk.index)
                if (existing_chunk and
                    existing_chunk['content_hash'] == chunk.content_hash and
                    existing_chunk['faiss_id'] is not None):
                    chunk.faiss_id = existing_chunk['faiss_id']
                    unchanged_chunks.append(chunk)
                    reused_chunk_indices.add(chunk.index)
                else:
                    chunks_needing_embedding.append(chunk)
                    chunk_texts_for_embedding.append(chunk.content)

            # Identify chunks to remove from previous version
            chunk_indices_to_remove, faiss_ids_to_remove = [], []
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
                'faiss_ids_to_remove': faiss_ids_to_remove,
            }

    async def _process_document_data_async(
            self, chunk_data: Dict[str, Any], similarity_threshold: Optional[float],
            mode: Literal["upsert", "insert"] = "upsert"
    ) -> Optional[str]:
        doc_id = chunk_data['doc_id']
        try:
            unchanged_chunks = chunk_data['unchanged_chunks']
            chunks_needing_embedding = chunk_data['chunks_needing_embedding']
            new_embeddings = chunk_data.get('new_embeddings', np.array([]).reshape(0, self.embedding_dimension))
            field_embeddings = chunk_data.get('field_embeddings', {})

            if similarity_threshold is not None and len(chunks_needing_embedding) > 0:
                # Use asyncio.to_thread for CPU-bound similarity filtering in Python 3.9+
                # Falls back to run_in_executor for compatibility
                doc_info = (chunk_data['doc_text'], chunk_data['metadata'], chunk_data['doc_id'],
                            chunk_data['content_hash'])
                doc_chunk_mapping = [doc_info] * len(chunks_needing_embedding)
                try:
                    filtered_chunks, filtered_embeddings, _ = await asyncio.to_thread(
                        self._filter_similar_chunks_vectorized, new_embeddings, chunks_needing_embedding,
                        doc_chunk_mapping, similarity_threshold
                    )
                except AttributeError:
                    # Fallback for Python < 3.9
                    loop = asyncio.get_event_loop()
                    filtered_chunks, filtered_embeddings, _ = await loop.run_in_executor(
                        None, self._filter_similar_chunks_vectorized, new_embeddings, chunks_needing_embedding,
                        doc_chunk_mapping, similarity_threshold
                    )
                chunks_needing_embedding = filtered_chunks
                new_embeddings = filtered_embeddings
            all_chunks = unchanged_chunks + chunks_needing_embedding
            if len(all_chunks) > 0 or mode == "upsert":
                documents_data = [
                    (chunk_data['doc_id'], chunk_data['doc_text'], chunk_data['content_hash'], chunk_data['metadata'])]
                chunks_data = [(chunk_data['doc_id'], chunk) for chunk in all_chunks]
                async with self.async_connection_pool.get_connection_context() as conn:
                    try:
                        await conn.execute('BEGIN')

                        # Remove old chunks and metadata embeddings on upsert
                        if mode == "upsert":
                            await self._remove_old_chunks_batch_async(
                                conn, chunk_data['doc_id'], chunk_data['chunk_indices_to_remove'],
                                chunk_data['faiss_ids_to_remove']
                            )
                            # Always remove metadata embeddings on upsert to prevent orphaned entries
                            await self._remove_metadata_embeddings_async(conn, doc_id)

                        await self._insert_documents_bulk_async(conn, documents_data, mode="replace")
                        if new_embeddings.size > 0:
                            # Use asyncio.to_thread for FAISS operations in Python 3.9+
                            # Falls back to run_in_executor for compatibility
                            try:
                                await asyncio.to_thread(self._add_vectors_to_faiss_bulk, new_embeddings,
                                                       chunks_needing_embedding)
                            except AttributeError:
                                # Fallback for Python < 3.9
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, self._add_vectors_to_faiss_bulk, new_embeddings,
                                                           chunks_needing_embedding)
                        await self._insert_chunks_bulk_async(conn, chunks_data)

                        # Store metadata embeddings if present
                        if field_embeddings:
                            await self._store_metadata_embeddings_async(conn, doc_id, field_embeddings)

                        await conn.commit()
                    except Exception:
                        await conn.rollback()
                        raise
            return doc_id
        except Exception as e:
            logger.error(f"Error processing document data for {doc_id}: {e}")
            return None

    async def _remove_old_chunks_batch_async(
            self, conn, doc_id: str, chunk_indices_to_remove: List[int], faiss_ids_to_remove: List[int]
    ) -> None:
        if faiss_ids_to_remove:
            # Use asyncio.to_thread for FAISS operations in Python 3.9+
            # Falls back to run_in_executor for compatibility
            try:
                await asyncio.to_thread(self._remove_old_vectors_bulk, faiss_ids_to_remove)
            except AttributeError:
                # Fallback for Python < 3.9
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids_to_remove)
        if chunk_indices_to_remove:
            placeholders = ','.join(['?'] * len(chunk_indices_to_remove))
            await conn.execute(
                f'DELETE FROM chunks WHERE document_id = ? AND chunk_index IN ({placeholders})',
                [doc_id] + chunk_indices_to_remove,
            )
        logger.debug(
            f"Removed {len(chunk_indices_to_remove)} old chunks and "
            f"{len(faiss_ids_to_remove)} FAISS vectors for document {doc_id}"
        )

    async def _insert_documents_bulk_async(
            self, conn: aiosqlite.Connection, documents_data: List[Tuple[str, str, str, Dict[str, Any]]],
            mode: Literal["insert", "replace"] = "replace"
    ) -> None:
        if not documents_data:
            return

        # Use shared business logic for SQL and data preparation
        sql, _ = self._build_documents_bulk_insert_sql(mode)
        # Use async version to properly preserve created_at timestamps for upserts
        bulk_data = await self._prepare_documents_bulk_data_async(documents_data, conn=conn,
                                                                  preserve_created_at=(mode == "replace"))
        await conn.executemany(sql, bulk_data)

    @staticmethod
    async def _insert_chunks_bulk_async(conn: aiosqlite.Connection, chunks_data: List[Tuple[str, Any]]) -> None:
        if not chunks_data:
            return
        bulk_data = []
        for doc_id, chunk in chunks_data:
            bulk_data.append(
                (
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
                    chunk.faiss_id,
                )
            )
        await conn.executemany(
            '''
            INSERT INTO chunks
            (document_id, chunk_index, content, content_hash, start_pos, end_pos, start_line,
            start_col, end_line, end_col, tokens, faiss_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            bulk_data,
        )

    async def _remove_old_document_data_async(self, doc_ids: List[str]) -> None:
        if not doc_ids:
            return
        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ','.join(['?'] * len(doc_ids))
            cursor = await conn.execute(
                f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                doc_ids,
            )
            faiss_ids = [row['faiss_id'] for row in await cursor.fetchall()]
            if faiss_ids:
                # Use asyncio.to_thread for FAISS operations in Python 3.9+
                # Falls back to run_in_executor for compatibility
                try:
                    await asyncio.to_thread(self._remove_old_vectors_bulk, faiss_ids)
                except AttributeError:
                    # Fallback for Python < 3.9
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids)
            await conn.execute(f'DELETE FROM chunks WHERE document_id IN ({placeholders})', doc_ids)
            await conn.execute(f'DELETE FROM documents WHERE id IN ({placeholders})', doc_ids)
