"""
Metadata validation, schema management, and metadata-embedding helpers.

Preserves original method implementations as much as possible.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import aiosqlite
import numpy as np

from localvectordb._pools import AsyncConnectionPool
from localvectordb._schema import get_common_metadata_schemas
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.database.base import LocalVectorDBBase
from localvectordb.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class MetadataMixin(LocalVectorDBBase, ABC):

    #############################
    # Metadata Validation/Info  #
    #############################

    # Pure business logic helpers for DRY elimination
    def _build_metadata_schema_info(self) -> Dict[str, Any]:
        """Build metadata schema information dictionary (pure business logic)"""
        info: Dict[str, Any] = {
            "fields": {},
            "field_count": len(self.metadata_schema),
            "indexed_fields": [],
            "required_fields": [],
            "field_types": {},
        }
        for field_name, field_def in self.metadata_schema.items():
            field_type_value = (
                field_def.type.value if isinstance(field_def.type, MetadataFieldType) else str(field_def.type)
            )
            info["fields"][field_name] = {
                "type": field_type_value,
                "indexed": field_def.indexed,
                "required": field_def.required,
                "default_value": field_def.default_value,
            }
            if field_def.indexed:
                info["indexed_fields"].append(field_name)
            if field_def.required:
                info["required_fields"].append(field_name)
            info["field_types"][field_type_value] = info["field_types"].get(field_type_value, 0) + 1
        return info

    def _add_embeddings_to_faiss(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Add metadata-field embeddings to the main FAISS index and return their ids.

        These share the main index -- and therefore its id space -- with chunk vectors,
        which is why the dual-store invariant counts ``column_embeddings`` alongside
        ``chunks``.
        """
        assert self.index is not None
        ids = self._allocate_faiss_ids("main", len(embeddings))
        with self._faiss_lock.write_lock():
            self.index.add_with_ids(embeddings, ids)
        return ids

    def _validate_metadata_batch(self, metadata_batch: List[Dict[str, Any]]):
        unknown_fields: set = set()
        for metadata in metadata_batch:
            for field_name, value in metadata.items():
                if field_name in self.metadata_schema:
                    field_def = self.metadata_schema[field_name]
                    if (
                        value is not None
                        and isinstance(field_def.type, MetadataFieldType)
                        and not isinstance(value, field_def.type.valid_types())
                    ):
                        raise ValueError(
                            f"Metadata field '{field_name}' is type {field_def.type.name}. Found: {type(value)}"
                        )
                else:
                    unknown_fields.add(field_name)
            for field_name, field_def in self.metadata_schema.items():
                if field_def.required and field_name not in metadata:
                    if field_def.default_value is not None:
                        metadata[field_name] = field_def.default_value
                    else:
                        raise ValueError(f"Required metadata field '{field_name}' is missing")
        if unknown_fields:
            logger.warning(
                "Metadata field(s) %s are not in the metadata schema and will not be stored. "
                "Add them with update_metadata_schema() to persist these values.",
                sorted(unknown_fields),
            )

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
        return self._build_metadata_schema_info()

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
        return self._build_metadata_schema_info()

    #############################
    # Schema Updates (sync/async)
    #############################
    def update_metadata_schema(
        self,
        new_schema,
        drop_columns: bool = False,
        column_mapping: Optional[dict] = None,
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
        if isinstance(new_schema, str):
            new_schema = get_common_metadata_schemas(new_schema)
        elif isinstance(new_schema, dict):
            normalized_schema = {}
            for field_name, field_def in new_schema.items():
                if isinstance(field_def, str):
                    normalized_schema[field_name] = MetadataField(MetadataFieldType(field_def))
                elif isinstance(field_def, tuple):
                    if len(field_def) == 2:
                        field_type, indexed = field_def
                        normalized_schema[field_name] = MetadataField(MetadataFieldType(field_type), indexed=indexed)
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
        for field_name, field_def in new_schema.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError("Metadata field names must be non-empty strings")
            if not isinstance(field_def, MetadataField):
                raise ValueError(f"Field definition for '{field_name}' must be a MetadataField instance")
        try:
            with self._read_write_lock.write_lock():
                with self.connection_pool.get_connection() as conn:
                    changes = self.schema.update_metadata_schema(new_schema, conn, drop_columns, column_mapping)
                self._metadata_schema = new_schema.copy()
            return changes
        except Exception as e:
            logger.error(f"Failed to update metadata schema: {e}")
            raise DatabaseError(f"Schema update failed: {str(e)}") from e

    async def update_metadata_schema_async(
        self,
        new_schema,
        drop_columns: bool = False,
        column_mapping: Optional[dict] = None,
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
        if isinstance(new_schema, str):
            new_schema = get_common_metadata_schemas(new_schema)
        elif isinstance(new_schema, dict):
            normalized_schema = {}
            for field_name, field_def in new_schema.items():
                if isinstance(field_def, str):
                    normalized_schema[field_name] = MetadataField(MetadataFieldType(field_def))
                elif isinstance(field_def, tuple):
                    if len(field_def) == 2:
                        field_type, indexed = field_def
                        normalized_schema[field_name] = MetadataField(MetadataFieldType(field_type), indexed=indexed)
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
        for field_name, field_def in new_schema.items():
            if not isinstance(field_name, str) or not field_name.strip():
                raise ValueError("Metadata field names must be non-empty strings")
            if not isinstance(field_def, MetadataField):
                raise ValueError(f"Field definition for '{field_name}' must be a MetadataField instance")
        try:
            if self.async_connection_pool is None:
                self.async_connection_pool = AsyncConnectionPool(self.db_path, self.async_max_connections)
            async with self.async_connection_pool.get_connection_context() as conn:
                changes = await self.schema.update_metadata_schema_async(new_schema, conn, drop_columns, column_mapping)
            self._metadata_schema = new_schema.copy()
            return changes
        except Exception as e:
            logger.error(f"Failed to update metadata schema (async): {e}")
            raise DatabaseError(f"Schema update failed: {str(e)}") from e

    #############################
    # Metadata Embeddings       #
    #############################
    def _get_embedding_enabled_fields(self) -> Dict[str, "MetadataField"]:
        return {
            field_name: field_def
            for field_name, field_def in self.metadata_schema.items()
            if field_def.embedding_enabled
        }

    def _get_changed_embedding_fields(
        self, old_metadata: Dict[str, Any], new_metadata: Dict[str, Any]
    ) -> Dict[str, "MetadataField"]:
        embedding_enabled_fields = self._get_embedding_enabled_fields()
        changed_fields = {}
        for field_name, field_def in embedding_enabled_fields.items():
            old_value = old_metadata.get(field_name)
            new_value = new_metadata.get(field_name)
            if old_value != new_value:
                if new_value is not None and str(new_value).strip():
                    changed_fields[field_name] = field_def
                elif old_value is not None and str(old_value).strip():
                    changed_fields[field_name] = field_def
        return changed_fields

    def _track_column_embedding(self, conn, document_id: str, field_name: str, chunk_index: int, faiss_id: int) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO column_embeddings
            (document_id, field_name, chunk_index, faiss_id)
            VALUES (?, ?, ?, ?)
            """,
            (document_id, field_name, chunk_index, faiss_id),
        )

    def _generate_metadata_embeddings(
        self, metadata: Dict[str, Any], embedding_enabled_fields: Dict[str, "MetadataField"], batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        field_embeddings = {}
        for field_name, field_def in embedding_enabled_fields.items():
            field_value = metadata.get(field_name)
            if field_value is None:
                continue
            if field_def.type == MetadataFieldType.JSON:
                text_value = json.dumps(field_value)
            else:
                text_value = str(field_value)
            # Use existing chunker
            field_chunks = self.chunker.chunk(text_value)
            if field_chunks:
                chunk_texts = [chunk.content for chunk in field_chunks]
                embeddings = self.embedding_provider.embed_sync(chunk_texts, batch_size)
                field_embeddings[field_name] = embeddings
        return field_embeddings

    async def _generate_metadata_embeddings_async(
        self, metadata: Dict[str, Any], embedding_enabled_fields: Dict[str, "MetadataField"], batch_size: int = 100
    ) -> Dict[str, np.ndarray]:
        field_embeddings = {}
        for field_name, field_def in embedding_enabled_fields.items():
            if field_name not in metadata:
                continue
            field_value = metadata[field_name]
            field_chunks = []
            if field_def.type == MetadataFieldType.TEXT:
                if isinstance(field_value, str) and field_value.strip():
                    field_chunks.extend(self.chunker.chunk(field_value))
            elif field_def.type == MetadataFieldType.JSON:
                if field_value:
                    text_value = json.dumps(field_value)
                    field_chunks.extend(self.chunker.chunk(text_value))
            if field_chunks:
                chunk_texts = [chunk.content for chunk in field_chunks]
                embeddings = await self.embedding_provider.embed_batch(chunk_texts, batch_size)
                field_embeddings[field_name] = embeddings
        return field_embeddings

    def _store_metadata_embeddings(self, conn, document_id: str, field_embeddings: Dict[str, np.ndarray]) -> List[int]:
        """Store metadata-field embeddings and return the FAISS ids allocated for them.

        The caller needs the ids so a rolled-back transaction can undo the in-RAM
        FAISS adds, which SQLite's rollback does not cover.
        """
        allocated: List[int] = []
        for field_name, embeddings in field_embeddings.items():
            if embeddings.size == 0:
                continue
            # Use shared business logic for FAISS operations
            actual_ids = self._add_embeddings_to_faiss(embeddings)
            allocated.extend(int(i) for i in actual_ids)
            # Track using actual IDs
            for chunk_index, faiss_id in enumerate(actual_ids):
                self._track_column_embedding(conn, document_id, field_name, chunk_index, int(faiss_id))
        return allocated

    async def _store_metadata_embeddings_async(
        self, conn: aiosqlite.Connection, document_id: str, field_embeddings: Dict[str, np.ndarray]
    ) -> None:
        for field_name, embeddings in field_embeddings.items():
            if embeddings.size == 0:
                continue
            # Use shared business logic for FAISS operations (run in executor)
            loop = asyncio.get_event_loop()
            actual_ids = await loop.run_in_executor(None, self._add_embeddings_to_faiss, embeddings)

            for chunk_index, faiss_id in enumerate(actual_ids):
                await conn.execute(
                    """
                    INSERT OR REPLACE INTO column_embeddings
                    (document_id, field_name, chunk_index, faiss_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    (document_id, field_name, chunk_index, int(faiss_id)),
                )

    def _remove_metadata_embeddings(self, conn, document_id: str) -> None:
        cursor = conn.execute(
            """
            SELECT faiss_id FROM column_embeddings
            WHERE document_id = ?
        """,
            (document_id,),
        )
        faiss_ids = [row["faiss_id"] for row in cursor.fetchall()]
        if faiss_ids:
            self._remove_old_vectors_bulk(faiss_ids)
            conn.execute(
                """
                DELETE FROM column_embeddings
                WHERE document_id = ?
            """,
                (document_id,),
            )

    async def _remove_metadata_embeddings_async(self, conn: aiosqlite.Connection, document_id: str) -> None:
        cursor = await conn.execute(
            """
            SELECT faiss_id FROM column_embeddings
            WHERE document_id = ?
        """,
            (document_id,),
        )
        rows = await cursor.fetchall()
        faiss_ids_to_remove = [row["faiss_id"] for row in rows]
        if faiss_ids_to_remove:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._remove_old_vectors_bulk, faiss_ids_to_remove)
        await conn.execute(
            """
            DELETE FROM column_embeddings
            WHERE document_id = ?
        """,
            (document_id,),
        )

    async def _validate_metadata_async(self, metadata: Dict[str, Any]) -> None:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_metadata_batch, [metadata])
