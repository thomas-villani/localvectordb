# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
#
# src/localvectordb/database/crud.py
"""
CRUD, filter, stats facades that use base/ingest/search/metadata helpers.

Public API is kept identical to the original LocalVectorDB methods.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

from localvectordb._filters import FilterQueryBuilder
from localvectordb.core import Document
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.exceptions import DocumentNotFoundError, MetadataFilterError
from localvectordb.query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class CrudMixin(LocalVectorDBBase, ABC):
    # -------------
    # Query builder
    # -------------
    def query_builder(self) -> QueryBuilder:
        """Returns a QueryBuilder for the database."""
        return QueryBuilder(self)

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
                metadata_columns = list(self.metadata_schema.keys())
                if metadata_columns:
                    columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
                else:
                    columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
                placeholders = ','.join(['?'] * len(requested_ids))
                sql = f"SELECT {', '.join(columns)} FROM documents WHERE id IN ({placeholders})"
                cursor = conn.execute(sql, requested_ids)
                rows = cursor.fetchall()
                found_ids = {row['id'] for row in rows}
                missing_ids = [doc_id for doc_id in requested_ids if doc_id not in found_ids]
                if missing_ids:
                    if single_id:
                        raise DocumentNotFoundError(f"Document not found: {missing_ids[0]}", missing_ids[0])
                    else:
                        raise DocumentNotFoundError(f"Documents not found: {', '.join(missing_ids)}", missing_ids)
                documents = []
                for row in rows:
                    metadata = {}
                    for col_name in metadata_columns:
                        if col_name in row.keys():
                            metadata[col_name] = row[col_name]
                    doc = Document(
                        id=row['id'],
                        content=row['content'],
                        metadata=metadata,
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        content_hash=row['content_hash'],
                    )
                    documents.append(doc)
            id_to_doc = {doc.id: doc for doc in documents}
            ordered_documents = [id_to_doc[doc_id] for doc_id in requested_ids]
            return ordered_documents[0] if single_id else ordered_documents

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
        return results[0] if single_id else results

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
            faiss_ids_to_remove: List[int] = []
            with self.connection_pool.get_connection() as conn:
                placeholders = ','.join(['?'] * len(ids))
                cursor = conn.execute(
                    f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                    ids,
                )
                faiss_ids_to_remove.extend([row['faiss_id'] for row in cursor.fetchall()])
                cursor = conn.execute(
                    f'SELECT faiss_id FROM column_embeddings WHERE document_id IN ({placeholders})',
                    ids,
                )
                faiss_ids_to_remove.extend([row['faiss_id'] for row in cursor.fetchall()])
                cursor = conn.execute(f'DELETE FROM documents WHERE id IN ({placeholders})', ids)
                deleted_count = cursor.rowcount
                conn.commit()
            if deleted_count == 0:
                if len(ids) == 1:
                    raise DocumentNotFoundError(f"Document with ID '{ids[0]}' not found")
                else:
                    raise DocumentNotFoundError(f"None of the {len(ids)} specified documents were found")
            if faiss_ids_to_remove and hasattr(self.index, 'remove_ids'):
                try:
                    import numpy as _np
                    ids_array = _np.array(faiss_ids_to_remove, dtype=_np.int64)
                    self.index.remove_ids(ids_array)
                    logger.info(f"Removed {len(faiss_ids_to_remove)} vectors from FAISS index")
                except Exception as e:
                    logger.error(f"Failed to remove vectors from FAISS index: {e}")
            elif faiss_ids_to_remove:
                logger.warning(f"FAISS index doesn't support removal, {len(faiss_ids_to_remove)} vectors orphaned")
            return deleted_count

    def update(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
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
            existing_doc = self.get(doc_id)
            if not existing_doc:
                return False
            if content is not None:
                new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                if new_hash != existing_doc.content_hash:
                    updated_metadata = existing_doc.metadata.copy()
                    if metadata:
                        updated_metadata.update(metadata)
                    self.upsert([content], [updated_metadata], [doc_id])
                    return True
            if metadata:
                updated_metadata = existing_doc.metadata.copy()
                updated_metadata.update(metadata)
                self._validate_metadata_batch([updated_metadata])
                changed_embedding_fields = self._get_changed_embedding_fields(existing_doc.metadata, updated_metadata)
                with self.connection_pool.get_connection() as conn:
                    conn.execute('BEGIN')
                    try:
                        if changed_embedding_fields:
                            self._remove_metadata_embeddings(conn, doc_id)
                            new_field_embeddings = self._generate_metadata_embeddings(updated_metadata, changed_embedding_fields, batch_size=100)
                            if new_field_embeddings:
                                self._store_metadata_embeddings(conn, doc_id, new_field_embeddings)
                                logger.debug(
                                    f"Updated embeddings for {len(new_field_embeddings)} metadata fields in document {doc_id}"
                                )
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

    # --------
    # Filter
    # --------
    def filter(self, where: Optional[Dict[str, Any]] = None, order_by: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
        """
        Filter documents using enhanced metadata filtering

        This method supports advanced MongoDB-style filtering with operators
        like $gt, $lt, $contains, $exists, etc.

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
        metadata_columns = list(self.metadata_schema.keys())
        if metadata_columns:
            columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
        else:
            columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
        query_parts = [f"SELECT {', '.join(columns)} FROM documents"]
        params: List[Any] = []
        filter_builder = None
        if where:
            try:
                filter_builder = FilterQueryBuilder(self.metadata_schema)
                where_clause, filter_params = filter_builder.build_where_clause(where)
                if where_clause:
                    query_parts.append(f"WHERE {where_clause}")
                    params.extend(filter_params)
            except Exception as e:
                raise MetadataFilterError(f"Error building filter query: {str(e)}")
        if order_by:
            try:
                if filter_builder is None:
                    filter_builder = FilterQueryBuilder(self.metadata_schema)
                valid_columns = set(columns)
                order_by_clause = filter_builder.build_order_by_clause(order_by, valid_columns)
                query_parts.append(order_by_clause)
            except Exception as e:
                raise MetadataFilterError(f"Error building ORDER BY clause: {str(e)}")
        if limit:
            if not isinstance(limit, int) or limit <= 0:
                raise MetadataFilterError("Limit must be a positive integer")
            query_parts.append(f"LIMIT {limit}")
        if offset:
            if not isinstance(offset, int) or offset < 0:
                raise MetadataFilterError("Offset must be a non-negative integer")
            query_parts.append(f"OFFSET {offset}")
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(' '.join(query_parts), params)
            rows = cursor.fetchall()
            documents: List[Document] = []
            for row in rows:
                metadata = {col_name: row[col_name] for col_name in metadata_columns}
                doc = Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=metadata,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    content_hash=row['content_hash'],
                )
                documents.append(doc)
        return documents

    # -------------
    # Async CRUD
    # -------------
    async def get_async(self, ids: Union[str, List[str]]) -> Union[Document, List[Document], None]:
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
        if isinstance(ids, str):
            single_id = True
            ids = [ids]
        else:
            single_id = False
        if not ids:
            return [] if not single_id else None
        async with self.async_connection_pool.get_connection_context() as conn:
            metadata_columns = list(self.metadata_schema.keys())
            if metadata_columns:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
            else:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
            placeholders = ','.join(['?' for _ in ids])
            sql = f"SELECT {', '.join(columns)} FROM documents WHERE id IN ({placeholders})"
            cursor = await conn.execute(sql, ids)
            rows = await cursor.fetchall()
        documents: List[Document] = []
        for row in rows:
            base_columns = self.schema.BASE_COLUMNS.copy()
            metadata = {}
            if hasattr(row, 'items'):
                row_items = row.items()
            else:
                row_items = [(key, row[key]) for key in row.keys()]
            for key, value in row_items:
                if key not in base_columns and value is not None:
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
                content_hash=row['content_hash'],
            )
            documents.append(document)
        if single_id:
            return documents[0] if documents else None
        else:
            return documents

    async def delete_async(self, ids: Union[str, List[str]]) -> int:
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
        if isinstance(ids, str):
            ids = [ids]
        if not ids:
            return 0
        deleted_count = 0
        async with self.async_connection_pool.get_connection_context() as conn:
            placeholders = ','.join(['?' for _ in ids])
            cursor = await conn.execute(
                f'SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL',
                ids,
            )
            faiss_ids = [row['faiss_id'] for row in await cursor.fetchall()]
            if faiss_ids:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids)
            await conn.execute('BEGIN')
            try:
                cursor = await conn.execute(f'DELETE FROM chunks WHERE document_id IN ({placeholders})', ids)
                cursor = await conn.execute(f'DELETE FROM documents WHERE id IN ({placeholders})', ids)
                deleted_count = cursor.rowcount or 0
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        if deleted_count == 0:
            if len(ids) == 1:
                raise DocumentNotFoundError(f"Document with ID '{ids[0]}' not found")
            else:
                raise DocumentNotFoundError(f"None of the {len(ids)} specified documents were found")
        logger.info(f"Deleted {deleted_count} documents")
        return deleted_count

    async def count_async(self, filters: Optional[Dict[str, Any]] = None) -> int:
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

    async def exists_async(self, ids: str) -> bool:
        """
        Async check if a document exists

        Parameters
        ----------
        ids : str
            Document ID to check

        Returns
        -------
        bool
            True if document exists, False otherwise
        """
        self._ensure_async_pool()
        single = isinstance(ids, str)
        ids_list = [ids] if single else list(ids)
        if not ids_list:
            return False if single else []

        placeholders = ','.join(['?'] * len(ids_list))
        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(f"SELECT id FROM documents WHERE id IN ({placeholders})", ids_list)
            rows = await cursor.fetchall()
            existing = {row['id'] for row in rows}

        result = [doc_id in existing for doc_id in ids_list]
        return result[0] if single else result

    async def filter_async(self, where: Optional[Dict[str, Any]] = None, order_by: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[Document]:
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
        filter_builder = FilterQueryBuilder(self.metadata_schema)
        where_clause, params = filter_builder.build_where_clause(where)
        order_clause = ""
        if order_by:
            try:
                if not isinstance(order_by, str):
                    raise ValueError("order_by must be a string")
                base_columns = self.schema.BASE_COLUMNS.copy()
                metadata_columns = set(self.metadata_schema.keys())
                valid_columns = set(base_columns).union(metadata_columns)
                order_by_clause = filter_builder.build_order_by_clause(order_by, valid_columns)
                order_clause = f" {order_by_clause}"
            except Exception as e:
                raise ValueError(f"Error building ORDER BY clause: {str(e)}")
        limit_clause = ""
        if limit is not None:
            limit_clause = f" LIMIT {limit}"
            if offset > 0:
                limit_clause += f" OFFSET {offset}"
        async with self.async_connection_pool.get_connection_context() as conn:
            metadata_columns = list(self.metadata_schema.keys())
            if metadata_columns:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at'] + metadata_columns
            else:
                columns = ['id', 'content', 'content_hash', 'created_at', 'updated_at']
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
        documents: List[Document] = []
        for row in rows:
            base_columns = self.schema.BASE_COLUMNS.copy()
            metadata = {}
            if hasattr(row, 'items'):
                row_items = row.items()
            else:
                row_items = [(key, row[key]) for key in row.keys()]
            for key, value in row_items:
                if key not in base_columns and value is not None:
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
                content_hash=row['content_hash'],
            )
            documents.append(document)
        return documents

    async def update_async(self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
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
        existing_doc = await self.get_async(doc_id)
        if not existing_doc:
            logger.debug(f"Document {doc_id} not found for update")
            return False
        changes_made = False
        if content is not None:
            new_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if new_hash != existing_doc.content_hash:
                updated_metadata = existing_doc.metadata.copy()
                if metadata:
                    updated_metadata.update(metadata)
                logger.debug(f"Content changed for document {doc_id}, re-processing with upsert")
                await self.upsert_async([content], [updated_metadata], [doc_id])
                return True
            else:
                logger.debug(f"Content unchanged for document {doc_id} (same hash)")
        if metadata:
            updated_metadata = existing_doc.metadata.copy()
            updated_metadata.update(metadata)
            await self._validate_metadata_async(updated_metadata)
            changed_embedding_fields = self._get_changed_embedding_fields(existing_doc.metadata, updated_metadata)
            async with self.async_connection_pool.get_connection_context() as conn:
                await conn.execute('BEGIN')
                try:
                    if changed_embedding_fields:
                        await self._remove_metadata_embeddings_async(conn, doc_id)
                        new_field_embeddings = await self._generate_metadata_embeddings_async(updated_metadata, changed_embedding_fields, batch_size=100)
                        if new_field_embeddings:
                            await self._store_metadata_embeddings_async(conn, doc_id, new_field_embeddings)
                            logger.debug(f"Updated embeddings for {len(new_field_embeddings)} metadata fields in document {doc_id}")
                    set_clauses = ['updated_at = ?']
                    values = [datetime.now(UTC)]
                    for field_name, value in updated_metadata.items():
                        if field_name in self.metadata_schema:
                            set_clauses.append(f'{field_name} = ?')
                            values.append(value)
                    values.append(doc_id)
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
