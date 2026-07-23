"""
CRUD, filter, stats facades that use base/ingest/search/metadata helpers.

Public API is kept identical to the original LocalVectorDB methods.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from abc import ABC
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Union

from localvectordb._filters import FilterQueryBuilder
from localvectordb.core import Chunk, ChunkPosition, Document, MetadataFieldType, PrefixEntry, PrefixListing
from localvectordb.database._ingest import _PendingFaissRemovals
from localvectordb.database._utils import AsyncDatabaseExecutor, SyncDatabaseExecutor, glob_escape
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.exceptions import DatabaseError, DocumentNotFoundError, MetadataFilterError, PatchConflictError
from localvectordb.patching import PatchResult, apply_ops
from localvectordb.query_builder import QueryBuilder

logger = logging.getLogger(__name__)

# Pattern for valid SQL identifiers (alphanumeric and underscores, starting with letter or underscore)
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_sql_identifier(name: str) -> None:
    """Validate that a string is a safe SQL identifier.

    This prevents SQL injection by ensuring field names only contain
    alphanumeric characters and underscores.

    Parameters
    ----------
    name : str
        The identifier to validate

    Raises
    ------
    DatabaseError
        If the identifier contains unsafe characters
    """
    if not _SAFE_IDENTIFIER_PATTERN.match(name):
        raise DatabaseError(
            f"Invalid SQL identifier '{name}': must contain only alphanumeric "
            "characters and underscores, and start with a letter or underscore"
        )


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier for safe use in queries.

    This function validates and quotes the identifier with double quotes,
    escaping any embedded double quotes (though validation should prevent them).

    Parameters
    ----------
    name : str
        The identifier to quote

    Returns
    -------
    str
        The safely quoted identifier

    Raises
    ------
    DatabaseError
        If the identifier contains unsafe characters
    """
    _validate_sql_identifier(name)
    # Escape any double quotes (shouldn't happen due to validation, but defense in depth)
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


class CrudMixin(LocalVectorDBBase, ABC):
    # Attributes from composed class
    _hierarchical_embeddings: bool
    _remove_section_vectors: Any
    _remove_document_vectors: Any

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._sync_executor = SyncDatabaseExecutor()
        self._async_executor = AsyncDatabaseExecutor()

    # -------------
    # Query builder
    # -------------
    def query_builder(self) -> QueryBuilder:
        """Returns a QueryBuilder for the database."""
        return QueryBuilder(self)

    # Pure business logic helpers
    def _build_document_columns_list(self) -> List[str]:
        """Build list of columns for document retrieval (pure business logic)"""
        metadata_columns = list(self.metadata_schema.keys())
        if metadata_columns:
            return ["id", "content", "content_hash", "created_at", "updated_at"] + metadata_columns
        else:
            return ["id", "content", "content_hash", "created_at", "updated_at"]

    def _build_get_documents_sql(self, requested_ids: List[str]) -> tuple[str, List[str]]:
        """Build SQL for retrieving documents by ID (pure business logic)"""
        columns = self._build_document_columns_list()
        placeholders = ",".join(["?"] * len(requested_ids))
        sql = f"SELECT {', '.join(columns)} FROM documents WHERE id IN ({placeholders})"
        return sql, requested_ids

    def _validate_missing_documents(self, requested_ids: List[str], found_ids: set) -> None:
        """Validate that all requested documents were found (pure business logic)"""
        missing_ids = [doc_id for doc_id in requested_ids if doc_id not in found_ids]
        if missing_ids:
            if len(requested_ids) == 1:
                raise DocumentNotFoundError(f"Document not found: {missing_ids[0]}", missing_ids[0])
            else:
                raise DocumentNotFoundError(f"Documents not found: {', '.join(missing_ids)}", missing_ids)

    def _construct_document_from_row(self, row) -> Document:
        """Construct Document object from database row (pure business logic)"""
        # Extract metadata (columns that are not base columns)
        metadata = {}
        base_columns = {"id", "content", "content_hash", "created_at", "updated_at"}

        if hasattr(row, "items"):
            row_items = row.items()
        else:
            row_items = [(key, row[key]) for key in row.keys()]

        for key, value in row_items:
            if key not in base_columns:
                # JSON metadata columns are declared TEXT in SQLite, so the
                # registered "json" converter never fires; deserialize here so
                # documents round-trip lists/dicts (search does the same).
                field_def = self.metadata_schema.get(key)
                if isinstance(value, str) and field_def is not None and field_def.type == MetadataFieldType.JSON:
                    try:
                        value = json.loads(value)
                    except (ValueError, TypeError):
                        pass  # malformed stored value: return raw string
                metadata[key] = value

        return Document(
            id=row["id"],
            content=row["content"],
            metadata=metadata,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            content_hash=row["content_hash"],
        )

    def _build_exists_sql(self, ids_list: List[str]) -> tuple[str, List[str]]:
        """Build SQL for checking document existence (pure business logic)"""
        placeholders = ",".join(["?"] * len(ids_list))
        sql = f"SELECT id FROM documents WHERE id IN ({placeholders})"
        return sql, ids_list

    def _process_exists_results(self, rows, ids_list: List[str]) -> List[bool]:
        """Process exists query results into boolean list (pure business logic)"""
        existing_ids = {row["id"] for row in rows}
        return [doc_id in existing_ids for doc_id in ids_list]

    def _build_filter_sql(
        self,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> tuple[str, List[Any]]:
        """Build SQL for filtering documents (pure business logic)"""
        columns = self._build_document_columns_list()
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
                raise MetadataFilterError(f"Error building filter query: {str(e)}") from e

        if order_by:
            try:
                if filter_builder is None:
                    filter_builder = FilterQueryBuilder(self.metadata_schema)
                valid_columns = set(columns)
                order_by_clause = filter_builder.build_order_by_clause(order_by, valid_columns)
                query_parts.append(order_by_clause)
            except Exception as e:
                raise MetadataFilterError(f"Error building ORDER BY clause: {str(e)}") from e

        if limit:
            if not isinstance(limit, int) or limit <= 0:
                raise MetadataFilterError("Limit must be a positive integer")
            query_parts.append(f"LIMIT {limit}")

        if offset:
            if not isinstance(offset, int) or offset < 0:
                raise MetadataFilterError("Offset must be a non-negative integer")
            query_parts.append(f"OFFSET {offset}")

        return " ".join(query_parts), params

    def _construct_documents_from_rows(self, rows) -> List[Document]:
        """Construct Document objects from database rows (pure business logic)"""
        documents: List[Document] = []
        for row in rows:
            doc = self._construct_document_from_row(row)
            documents.append(doc)
        return documents

    def _core_get_sync(self, conn, requested_ids: List[str]) -> List[Document]:
        """Core logic for retrieving documents by ID (sync version)"""
        # Use shared business logic helpers
        sql, params = self._build_get_documents_sql(requested_ids)
        cursor = self._sync_executor.execute(conn, sql, params)
        try:
            rows = self._sync_executor.fetchall(cursor)
        finally:
            cursor.close()

        # Validate all documents were found using shared logic
        found_ids = {row["id"] for row in rows}
        self._validate_missing_documents(requested_ids, found_ids)

        # Construct documents using shared logic
        documents = []
        for row in rows:
            doc = self._construct_document_from_row(row)
            documents.append(doc)
        return documents

    async def _core_get_async(self, conn, requested_ids: List[str]) -> List[Document]:
        """Core logic for retrieving documents by ID (async version)"""
        # Use shared business logic helpers
        sql, params = self._build_get_documents_sql(requested_ids)
        cursor = await self._async_executor.execute(conn, sql, params)
        rows = await self._async_executor.fetchall(cursor)

        # Validate all documents were found using shared logic
        found_ids = {row["id"] for row in rows}
        self._validate_missing_documents(requested_ids, found_ids)

        # Construct documents using shared logic
        documents = []
        for row in rows:
            document = self._construct_document_from_row(row)
            documents.append(document)
        return documents

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
        if isinstance(ids, str):
            single_id = True
            requested_ids: List[str] = [ids]
        else:
            single_id = False
            requested_ids = list(ids)
        with self._read_write_lock.read_lock():
            with self.connection_pool.get_connection() as conn:
                documents = self._core_get_sync(conn, requested_ids)
        id_to_doc = {doc.id: doc for doc in documents}
        ordered_documents = [id_to_doc[doc_id] for doc_id in requested_ids]
        return ordered_documents[0] if single_id else ordered_documents

    def _construct_chunk_from_row(self, row) -> Chunk:
        """Construct a Chunk (with full position) from a chunks-table row."""
        position = ChunkPosition(
            start=row["start_pos"],
            end=row["end_pos"],
            line=row["start_line"],
            column=row["start_col"],
            end_line=row["end_line"],
            end_column=row["end_col"],
        )
        return Chunk(
            content=row["content"],
            position=position,
            tokens=row["tokens"],
            index=row["chunk_index"],
            faiss_id=row["faiss_id"],
            content_hash=row["content_hash"],
        )

    def get_chunks(self, document_id: str, indices: Optional[List[int]] = None) -> List[Chunk]:
        """Retrieve the persisted chunks of a document, ordered by chunk index.

        Returns the chunks exactly as they were stored at ingest time (content
        plus full character/line position), unlike :meth:`get`, which only
        returns the whole-document content.

        Parameters
        ----------
        document_id : str
            The document whose chunks to retrieve.
        indices : list[int], optional
            If given, only chunks whose ``chunk_index`` appears in this list are
            returned; unknown indices are silently skipped. When ``None``
            (default), every chunk of the document is returned.

        Returns
        -------
        list[Chunk]
            Chunks with full position information, ordered by ``chunk_index``.
            Empty if the document has no chunks (or none match ``indices``).

        Notes
        -----
        Synchronous only; the CLI ``get`` command is the sole consumer. Add an
        async twin if a future async caller needs one.
        """
        sql = (
            "SELECT chunk_index, content, content_hash, start_pos, end_pos, "
            "start_line, start_col, end_line, end_col, tokens, faiss_id "
            "FROM chunks WHERE document_id = ?"
        )
        params: List[Any] = [document_id]
        if indices is not None:
            if not indices:
                return []
            placeholders = ",".join(["?"] * len(indices))
            sql += f" AND chunk_index IN ({placeholders})"
            params.extend(indices)
        sql += " ORDER BY chunk_index"

        with self._read_write_lock.read_lock():
            with self.connection_pool.get_connection() as conn:
                cursor = self._sync_executor.execute(conn, sql, params)
                try:
                    rows = self._sync_executor.fetchall(cursor)
                finally:
                    cursor.close()

        return [self._construct_chunk_from_row(row) for row in rows]

    def _core_exists_sync(self, conn, ids_list: List[str]) -> List[bool]:
        """Core logic for checking if documents exist (sync version)"""
        sql, params = self._build_exists_sql(ids_list)
        cursor = self._sync_executor.execute(conn, sql, params)
        try:
            rows = self._sync_executor.fetchall(cursor)
        finally:
            cursor.close()
        return self._process_exists_results(rows, ids_list)

    async def _core_exists_async(self, conn, ids_list: List[str]) -> List[bool]:
        """Core logic for checking if documents exist (async version)"""
        sql, params = self._build_exists_sql(ids_list)
        cursor = await self._async_executor.execute(conn, sql, params)
        rows = await self._async_executor.fetchall(cursor)
        return self._process_exists_results(rows, ids_list)

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
        if isinstance(ids, str):
            ids_list: List[str] = [ids]
            with self.connection_pool.get_connection() as conn:
                results = self._core_exists_sync(conn, ids_list)
            return results[0]
        else:
            ids_list = list(ids)
            with self.connection_pool.get_connection() as conn:
                results = self._core_exists_sync(conn, ids_list)
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
            self._require_writable("Deleting a document")
            if isinstance(ids, str):
                ids = [ids]
            # Refuse before touching SQLite: on an index that cannot remove vectors,
            # committing the row deletion would orphan them and report success.
            self._require_deletable("Deleting a document")
            faiss_ids_to_remove: List[int] = []
            with self.connection_pool.get_connection() as conn:
                placeholders = ",".join(["?"] * len(ids))

                # Collect chunk FAISS IDs
                cursor = conn.execute(
                    f"SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL",
                    ids,
                )
                try:
                    faiss_ids_to_remove.extend([row["faiss_id"] for row in cursor.fetchall()])
                finally:
                    cursor.close()

                # Also collect metadata embedding FAISS IDs
                # Note: column_embeddings rows are automatically deleted via ON DELETE CASCADE
                # when documents are deleted, but we need to collect their FAISS IDs first
                cursor = conn.execute(
                    f"SELECT faiss_id FROM column_embeddings WHERE document_id IN ({placeholders})",
                    ids,
                )
                try:
                    faiss_ids_to_remove.extend([row["faiss_id"] for row in cursor.fetchall()])
                finally:
                    cursor.close()

                # Collect section and document FAISS IDs for hierarchical indices
                section_faiss_ids_to_remove: list = []
                doc_faiss_ids_to_remove: list = []
                if self._hierarchical_embeddings:
                    cursor = conn.execute(
                        f"SELECT faiss_id FROM sections WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL",
                        ids,
                    )
                    try:
                        section_faiss_ids_to_remove.extend([row["faiss_id"] for row in cursor.fetchall()])
                    finally:
                        cursor.close()

                    cursor = conn.execute(
                        f"SELECT doc_faiss_id FROM documents WHERE id IN ({placeholders}) AND doc_faiss_id IS NOT NULL",
                        ids,
                    )
                    try:
                        doc_faiss_ids_to_remove.extend([row["doc_faiss_id"] for row in cursor.fetchall()])
                    finally:
                        cursor.close()

                # Delete documents (CASCADE deletes chunks, sections)
                cursor = conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)
                try:
                    deleted_count = cursor.rowcount
                finally:
                    cursor.close()
                conn.commit()
            if deleted_count == 0:
                if len(ids) == 1:
                    raise DocumentNotFoundError(f"Document with ID '{ids[0]}' not found")
                else:
                    raise DocumentNotFoundError(f"None of the {len(ids)} specified documents were found")
            # SQLite is committed; now drop the vectors. A crash here leaves orphan
            # vectors, which cost recall and are swept by `repair` -- never dangling
            # rows, which could only be fixed by re-embedding.
            if faiss_ids_to_remove:
                self._remove_vectors(self.index, faiss_ids_to_remove, "chunk")

            # Remove hierarchical FAISS vectors
            if self._hierarchical_embeddings:
                if section_faiss_ids_to_remove:
                    self._remove_section_vectors(section_faiss_ids_to_remove)
                if doc_faiss_ids_to_remove:
                    self._remove_document_vectors(doc_faiss_ids_to_remove)

            self._save_internal()
            self._save_faiss_counters()
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
            True if document was updated, False if no updates needed (`content` and `metadata` already match database)

        Raises
        ------
        DocumentNotFoundError
            Raised if `doc_id` does not exist.
        """
        with self._read_write_lock.write_lock():
            self._require_writable("Updating a document")
            existing_doc: Any = self.get(doc_id)
            if not existing_doc:
                raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found")
            if content is not None:
                new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
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
                # Deferred removals + added-vector tracking so a rollback leaves the
                # index consistent with the rolled-back rows (see _PendingFaissRemovals).
                pending_removals = _PendingFaissRemovals()
                added_faiss_ids: List[int] = []
                with self.connection_pool.get_connection() as conn:
                    conn.execute("BEGIN")
                    try:
                        if changed_embedding_fields:
                            self._remove_metadata_embeddings(conn, doc_id, pending=pending_removals)
                            new_field_embeddings = self._generate_metadata_embeddings(
                                updated_metadata, changed_embedding_fields, batch_size=100
                            )
                            if new_field_embeddings:
                                added_faiss_ids.extend(
                                    self._store_metadata_embeddings(conn, doc_id, new_field_embeddings)
                                )
                                logger.debug(
                                    f"Updated embeddings for {len(new_field_embeddings)} "
                                    f"metadata fields in document {doc_id}"
                                )
                        set_clauses = ['"updated_at" = ?']
                        values: list[Any] = [datetime.now(UTC)]
                        for field_name, value in updated_metadata.items():
                            if field_name in self.metadata_schema:
                                # Validate and quote field name to prevent SQL injection
                                quoted_field = _quote_identifier(field_name)
                                set_clauses.append(f"{quoted_field} = ?")
                                values.append(value)
                        values.append(doc_id)
                        sql = f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = ?"
                        conn.execute(sql, values)
                        conn.commit()
                        # Drop replaced metadata vectors only after the durable commit.
                        if pending_removals.main:
                            self._remove_old_vectors_bulk(pending_removals.main)
                        logger.debug(f"Updated metadata for document {doc_id}")
                    except Exception:
                        conn.rollback()
                        # Undo the vectors we added; leave the old (removed-deferred)
                        # vectors in place so the restored rows are not left dangling.
                        self._discard_faiss_ids_best_effort(added_faiss_ids)
                        raise
                if changed_embedding_fields:
                    # Re-embedding the metadata fields mutated the FAISS index in
                    # RAM. Persist it now (as delete()/upsert do) so a crash before
                    # the next save() cannot leave the just-committed rows pointing
                    # at metadata vectors that exist only in memory.
                    self._save_internal()
                    self._save_faiss_counters()
                return True
            return False

    def patch(
        self,
        doc_id: str,
        ops: List[Dict[str, Any]],
        *,
        expect_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatchResult:
        """
        Patch a document's content with find/replace or span-splice ops.

        Unlike :meth:`update` (which replaces the whole content string), ``patch``
        applies targeted edits resolved against the document's *current* content,
        so a caller need not re-send the untouched remainder. See
        :mod:`localvectordb.patching` for the op shapes.

        Parameters
        ----------
        doc_id : str
            Document ID to patch.
        ops : List[Dict[str, Any]]
            Patch ops (``splice`` / ``replace`` / ``append`` / ``prepend``),
            resolved against the original content, non-overlapping, applied
            atomically.
        expect_hash : Optional[str]
            If given and it does not equal the stored ``content_hash``, the patch
            fails with :class:`PatchConflictError` instead of clobbering a
            concurrent write.
        metadata : Optional[Dict[str, Any]]
            Metadata merged with existing (same semantics as :meth:`update`).

        Returns
        -------
        PatchResult
            ``updated`` is False only when the ops produced content identical to
            what is stored and no metadata changed.

        Raises
        ------
        DocumentNotFoundError
            If ``doc_id`` does not exist.
        PatchConflictError
            If ``expect_hash`` is given and does not match the stored hash.
        PatchError
            If an op is unmatched, ambiguous, overlapping, or out of range.
        """
        # write_lock is re-entrant for the same thread, so delegating to update()
        # (which re-locks and calls upsert) is safe and reuses its no-op detection,
        # metadata merge, and chunk-vector reuse.
        with self._read_write_lock.write_lock():
            self._require_writable("Patching a document")
            existing_doc: Any = self.get(doc_id)
            if not existing_doc:
                raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found")
            if expect_hash is not None and expect_hash != existing_doc.content_hash:
                raise PatchConflictError(
                    f"Patch precondition failed for document '{doc_id}': expected content_hash "
                    f"{expect_hash} but the stored content is {existing_doc.content_hash}; "
                    f"re-read the document and retry.",
                    expected=expect_hash,
                    actual=existing_doc.content_hash,
                )
            new_content = apply_ops(existing_doc.content, ops)
            new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            if new_hash != existing_doc.content_hash:
                # Surface an intelligible error before upsert tries (and fails) to
                # remove the changed chunks' vectors on an append-only index.
                self._require_deletable("Patching a document's content")
            updated = self.update(doc_id, content=new_content, metadata=metadata)
            return PatchResult(updated=updated, new_hash=new_hash, ops_applied=len(ops))

    # --------
    # Filter
    # --------
    def filter(
        self,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Document]:
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
        # Use shared business logic for SQL construction
        sql, params = self._build_filter_sql(where, order_by, limit, offset)

        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(sql, params)
            try:
                rows = cursor.fetchall()
            finally:
                cursor.close()
            return self._construct_documents_from_rows(rows)

    def list_prefixes(self, prefix: str = "", delimiter: str = "/") -> PrefixListing:
        """List the immediate children of a document-id prefix, S3-style.

        Treats ``delimiter`` as a virtual path separator over document ids and
        rolls documents up to their first segment beneath ``prefix``. A useful
        pattern is to use relative paths as manual document ids
        (``docs/reports/q1``) and then navigate them like folders -- there are no
        real directories, only ids that share a prefix.

        Parameters
        ----------
        prefix : str
            Literal id prefix to list beneath. Pass ``""`` (the default) to list
            the top level. For folder-like navigation, include the trailing
            delimiter (``"docs/"`` lists the children of ``docs/``, whereas
            ``"docs"`` lists children of every id beginning with ``docs``).
        delimiter : str
            Virtual path separator. Defaults to ``"/"``.

        Returns
        -------
        PrefixListing
            ``prefixes`` are the virtual sub-folders (common prefixes) with a
            recursive document ``count``; ``documents`` are the leaf documents
            that live directly at this level.

        Examples
        --------
        ::

            db.upsert(["..."], ids=["docs/reports/q1"])
            listing = db.list_prefixes("docs/")
            for folder in listing.prefixes:
                print(folder.path, folder.count)   # e.g. "docs/reports/" 1

        Notes
        -----
        - Matching is case-sensitive (SQLite ``GLOB`` / ``BINARY`` collation),
          which is the desired behaviour for path-like keys.
        - A document whose id equals ``prefix`` exactly is not reported as its own
          child.
        """
        if not delimiter:
            raise ValueError("delimiter must be a non-empty string")

        glob_pattern = glob_escape(prefix) + "*"
        prefix_len = len(prefix)

        # Strip the prefix in a subquery, then group each remaining id by its
        # first segment: ids containing the delimiter roll up into a virtual
        # folder (segment includes the trailing delimiter); ids without it are
        # leaf documents at this level. instr()/substr() are core SQLite scalar
        # functions, so the whole roll-up -- including counts -- happens in one
        # pass regardless of how many documents match.
        sql = """
            SELECT
                CASE WHEN instr(rest, :delim) > 0
                     THEN substr(rest, 1, instr(rest, :delim) - 1 + :dlen)
                     ELSE rest END AS segment,
                CASE WHEN instr(rest, :delim) > 0 THEN 1 ELSE 0 END AS is_prefix,
                COUNT(*) AS cnt
            FROM (
                SELECT substr(id, :restpos) AS rest
                FROM documents
                WHERE id GLOB :glob AND length(id) > :plen
            )
            GROUP BY segment, is_prefix
            ORDER BY is_prefix DESC, segment ASC
        """
        params = {
            "delim": delimiter,
            "dlen": len(delimiter),
            "restpos": prefix_len + 1,
            "glob": glob_pattern,
            "plen": prefix_len,
        }

        listing = PrefixListing(prefix=prefix, delimiter=delimiter)
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(sql, params)
            try:
                rows = cursor.fetchall()
            finally:
                cursor.close()

        for row in rows:
            segment, is_prefix, count = row["segment"], row["is_prefix"], row["cnt"]
            entry = PrefixEntry(
                name=segment,
                path=prefix + segment,
                is_prefix=bool(is_prefix),
                count=int(count),
            )
            (listing.prefixes if entry.is_prefix else listing.documents).append(entry)
        return listing

    # -------------
    # Async CRUD
    # -------------
    async def get_async(self, ids: Union[str, List[str]]) -> Union[Document, List[Document]]:
        """
        Async retrieve documents by ID

        Parameters
        ----------
        ids : Union[str, List[str]]
            Document ID(s) to retrieve

        Returns
        -------
        Union[Document, List[Document]]
            The requested document(s)

        Raises
        ------
        DocumentNotFoundError
            If any requested documents are not found
        """
        self._ensure_async_pool()
        assert self.async_connection_pool is not None
        await self._ensure_async_schema_initialized()
        if isinstance(ids, str):
            single_id = True
            requested_ids: List[str] = [ids]
        else:
            single_id = False
            requested_ids = list(ids)
        if not requested_ids:
            raise ValueError("`ids` must be provided.")
        async with self.async_connection_pool.get_connection_context() as conn:
            documents = await self._core_get_async(conn, requested_ids)
        id_to_doc = {doc.id: doc for doc in documents}
        ordered_documents = [id_to_doc[doc_id] for doc_id in requested_ids]
        return ordered_documents[0] if single_id else ordered_documents

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
        assert self.async_connection_pool is not None
        await self._ensure_async_schema_initialized()
        if isinstance(ids, str):
            ids = [ids]
        if not ids:
            return 0
        self._require_writable("Deleting a document")
        self._require_deletable("Deleting a document")
        deleted_count = 0
        # Hold the cross-thread write gate across the commit + index mutation so a
        # concurrent backup snapshot cannot capture the SQLite delete without the
        # matching FAISS removal (and vice versa). See _async_write_gate.
        async with self._async_write_gate():
            async with self.async_connection_pool.get_connection_context() as conn:
                placeholders = ",".join(["?" for _ in ids])
                cursor = await conn.execute(
                    f"SELECT faiss_id FROM chunks WHERE document_id IN ({placeholders}) AND faiss_id IS NOT NULL",
                    ids,
                )
                faiss_ids = [row["faiss_id"] for row in await cursor.fetchall()]

                # Also collect metadata embedding FAISS IDs
                # Note: column_embeddings rows are automatically deleted via ON DELETE CASCADE
                # when documents are deleted, but we need to collect their FAISS IDs first
                cursor = await conn.execute(
                    f"SELECT faiss_id FROM column_embeddings WHERE document_id IN ({placeholders})",
                    ids,
                )
                faiss_ids.extend([row["faiss_id"] for row in await cursor.fetchall()])

                # Commit SQLite first, then remove the vectors -- matching the sync path.
                # The reverse order (which this used to do) means a rollback leaves rows
                # whose vectors are already gone: dangling rows, recoverable only by
                # re-embedding. Committing first can only leave orphan vectors, which
                # `repair` sweeps for free.
                await conn.execute("BEGIN")
                try:
                    await conn.execute(f"DELETE FROM chunks WHERE document_id IN ({placeholders})", ids)
                    cursor = await conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)
                    deleted_count = cursor.rowcount or 0
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise

                if faiss_ids and deleted_count:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids)
            if deleted_count == 0:
                if len(ids) == 1:
                    raise DocumentNotFoundError(f"Document with ID '{ids[0]}' not found")
                else:
                    raise DocumentNotFoundError(f"None of the {len(ids)} specified documents were found")

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_internal)
            await loop.run_in_executor(None, self._save_faiss_counters)
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
        assert self.async_connection_pool is not None
        await self._ensure_async_schema_initialized()
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
            return row["count"] if row else 0

    async def exists_async(self, ids: Union[str, list[str]]) -> Union[bool, list[bool]]:
        """
        Async check if a document exists

        Parameters
        ----------
        ids : str
            Document ID to check

        Returns
        -------
        bool or list[bool]
            True if document exists, False otherwise
        """
        self._ensure_async_pool()
        assert self.async_connection_pool is not None
        await self._ensure_async_schema_initialized()
        single = isinstance(ids, str)
        ids_list: List[str] = [ids] if single else list(ids)  # type: ignore[list-item]
        if not ids_list:
            return False if single else []

        async with self.async_connection_pool.get_connection_context() as conn:
            results = await self._core_exists_async(conn, ids_list)
        return results[0] if single else results

    async def filter_async(
        self,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Document]:
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
        assert self.async_connection_pool is not None
        await self._ensure_async_schema_initialized()

        # Use shared business logic for SQL construction
        sql, params = self._build_filter_sql(where, order_by, limit, offset)

        async with self.async_connection_pool.get_connection_context() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
            return self._construct_documents_from_rows(rows)

    async def update_async(
        self, doc_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
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
            True if document was updated, False if no updates needed (`content` and
            `metadata` already match database)

        Raises
        ------
        DocumentNotFoundError
            Raised if `doc_id` does not exist.

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
        self._require_writable("Updating a document")
        existing_doc: Any = await self.get_async(doc_id)
        if not existing_doc:
            raise DocumentNotFoundError(f"Document {doc_id} not found for update")
        changes_made = False
        if isinstance(content, str):
            new_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
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
            assert self.async_connection_pool is not None
            # Deferred removals so a rollback cannot leave dangling rows (see
            # _PendingFaissRemovals). Added metadata vectors that orphan on a
            # rollback are the safe direction (repair sweeps them).
            pending_removals = _PendingFaissRemovals()
            # Gate the commit + metadata-embedding index mutation against a
            # concurrent backup snapshot (see _async_write_gate).
            async with self._async_write_gate():
                async with self.async_connection_pool.get_connection_context() as conn:
                    await conn.execute("BEGIN")
                    try:
                        if changed_embedding_fields:
                            await self._remove_metadata_embeddings_async(conn, doc_id, pending=pending_removals)
                            new_field_embeddings = await self._generate_metadata_embeddings_async(
                                updated_metadata, changed_embedding_fields, batch_size=100
                            )
                            if new_field_embeddings:
                                await self._store_metadata_embeddings_async(conn, doc_id, new_field_embeddings)
                                logger.debug(
                                    f"Updated embeddings for {len(new_field_embeddings)} "
                                    f"metadata fields in document {doc_id}"
                                )
                        set_clauses = ['"updated_at" = ?']
                        values: list[Any] = [datetime.now(UTC)]
                        for field_name, value in updated_metadata.items():
                            if field_name in self.metadata_schema:
                                # Validate and quote field name to prevent SQL injection
                                quoted_field = _quote_identifier(field_name)
                                set_clauses.append(f"{quoted_field} = ?")
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
                        # Drop replaced metadata vectors only after the durable commit.
                        if pending_removals.main:
                            await asyncio.to_thread(self._remove_old_vectors_bulk, pending_removals.main)
                        if affected_rows > 0:
                            changes_made = True
                            logger.debug(f"Updated metadata for document {doc_id}")
                        else:
                            logger.warning(f"No rows affected when updating document {doc_id}")
                    except Exception:
                        await conn.rollback()
                        raise
                if changed_embedding_fields:
                    # H2 (async twin): persist the FAISS index mutated by the
                    # metadata re-embedding so a crash before the next save() cannot
                    # leave the committed rows pointing at RAM-only vectors.
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._save_internal)
                    await loop.run_in_executor(None, self._save_faiss_counters)
        if not changes_made:
            logger.debug(f"No changes made to document {doc_id}")
        return changes_made

    async def patch_async(
        self,
        doc_id: str,
        ops: List[Dict[str, Any]],
        *,
        expect_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PatchResult:
        """Patch a document's content asynchronously. Same contract as :meth:`patch`."""
        self._ensure_async_pool()
        self._require_writable("Patching a document")
        existing_doc: Any = await self.get_async(doc_id)
        if not existing_doc:
            raise DocumentNotFoundError(f"Document {doc_id} not found for patch")
        if expect_hash is not None and expect_hash != existing_doc.content_hash:
            raise PatchConflictError(
                f"Patch precondition failed for document '{doc_id}': expected content_hash "
                f"{expect_hash} but the stored content is {existing_doc.content_hash}; "
                f"re-read the document and retry.",
                expected=expect_hash,
                actual=existing_doc.content_hash,
            )
        new_content = apply_ops(existing_doc.content, ops)
        new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        if new_hash != existing_doc.content_hash:
            self._require_deletable("Patching a document's content")
        updated = await self.update_async(doc_id, content=new_content, metadata=metadata)
        return PatchResult(updated=updated, new_hash=new_hash, ops_applied=len(ops))
