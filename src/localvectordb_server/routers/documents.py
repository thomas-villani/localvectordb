# src/localvectordb_server/routers/documents.py
"""Document CRUD routes (Pydantic request/response models + dependency injection)."""

import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, Query
from pydantic import AliasChoices, Field

from localvectordb.exceptions import DocumentNotFoundError
from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import APIError, ValidationError
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context, sanitize_log_value
from localvectordb_server._serializers import serialize_document
from localvectordb_server.config import Config
from localvectordb_server.routers._deps import get_config, get_db
from localvectordb_server.routers._models import (
    MAX_PAGE_LIMIT,
    BatchDeleteResponse,
    CountResponse,
    DeleteResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentsByIdResponse,
    ExistsResponse,
    FilterBody,
    PageInfo,
    StrictModel,
    UpdateResponse,
    WriteResponse,
)

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["documents"])

_MAX_BATCH = 1000


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class UpsertDocumentsBody(StrictModel):
    documents: Union[str, List[str]]
    metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None, validation_alias=AliasChoices("metadata", "metadatas")
    )
    ids: Optional[Union[str, List[str]]] = None
    batch_size: Optional[int] = Field(default=None, ge=1, le=1000)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InsertDocumentsBody(UpsertDocumentsBody):
    errors: str = "raise"  # "raise" or "ignore"


class UpsertChunksBody(StrictModel):
    chunks_by_document: Dict[str, List[Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=100, ge=1, le=1000)
    similarity_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class InsertChunksBody(UpsertChunksBody):
    errors: str = "raise"


class UpdateDocumentBody(StrictModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchDeleteBody(StrictModel):
    ids: List[str]


class ExistsBody(StrictModel):
    ids: Union[str, List[str]]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_list(value: Union[str, List[str], None]) -> Optional[List[str]]:
    if value is None:
        return None
    return [value] if isinstance(value, str) else value


def _normalize_chunks(chunks_by_document: Dict[str, List[Any]]) -> Dict[str, List[str]]:
    processed: Dict[str, List[str]] = {}
    for doc_id, chunks in chunks_by_document.items():
        if chunks and isinstance(chunks[0], dict):
            processed[doc_id] = [
                (chunk.get("content") or chunk.get("text", str(chunk))) if isinstance(chunk, dict) else chunk
                for chunk in chunks
            ]
        else:
            processed[doc_id] = chunks
    return processed


# --------------------------------------------------------------------------- #
# Write endpoints
# --------------------------------------------------------------------------- #


@router.post("/{db_name}/documents", response_model=WriteResponse, dependencies=[Depends(require_write_permission)])
@log_performance("upsert_documents")
async def upsert_documents(
    db_name: str, body: UpsertDocumentsBody, db=Depends(get_db), config: Config = Depends(get_config)
):
    """Upsert (insert or update) documents."""
    with request_context("upsert_documents"):
        documents = _as_list(body.documents)
        if not documents:
            raise ValidationError("Documents array cannot be empty", field="documents")
        for i, doc in enumerate(documents):
            if not isinstance(doc, str) or not doc.strip():
                raise ValidationError(f"Document at index {i} must be a non-empty string", field=f"documents[{i}]")

        metadata = body.metadata
        if isinstance(metadata, dict):
            metadata = [metadata]
        if metadata is not None and len(metadata) != len(documents):
            raise ValidationError(
                f"Number of metadata entries ({len(metadata)}) must match number of documents ({len(documents)})",
                field="metadata",
            )

        ids = _as_list(body.ids)
        if ids is not None and len(ids) != len(documents):
            raise ValidationError(
                f"Number of IDs ({len(ids)}) must match number of documents ({len(documents)})", field="ids"
            )

        batch_size = body.batch_size or (config.embedding.batch_size if config else 100)

        try:
            db_logger.log_query(
                "upsert_documents", database_name=db_name, document_count=len(documents), batch_size=batch_size
            )
            result_ids = db.upsert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                similarity_threshold=body.similarity_threshold,
            )
            return {"ids": result_ids, "message": f"Successfully processed {len(documents)} documents"}
        except Exception as e:
            db_logger.log_error("upsert_documents", e, database_name=db_name, document_count=len(documents))
            raise


@router.post(
    "/{db_name}/documents/insert", response_model=WriteResponse, dependencies=[Depends(require_write_permission)]
)
@log_performance("insert_documents")
async def insert_documents(
    db_name: str, body: InsertDocumentsBody, db=Depends(get_db), config: Config = Depends(get_config)
):
    """Insert new documents (fails if an ID already exists)."""
    with request_context("insert_documents"):
        if body.errors not in ("raise", "ignore"):
            raise ValidationError("errors parameter must be 'raise' or 'ignore'", field="errors", value=body.errors)

        documents = _as_list(body.documents)
        if not documents:
            raise ValidationError("Documents array cannot be empty", field="documents")

        metadata = body.metadata
        if isinstance(metadata, dict):
            metadata = [metadata]
        if metadata is not None and len(metadata) != len(documents):
            raise ValidationError(
                f"Number of metadata entries ({len(metadata)}) must match number of documents ({len(documents)})",
                field="metadata",
            )

        ids = _as_list(body.ids)
        if ids is not None and len(ids) != len(documents):
            raise ValidationError(
                f"Number of IDs ({len(ids)}) must match number of documents ({len(documents)})", field="ids"
            )

        batch_size = body.batch_size or (config.embedding.batch_size if config else 100)

        try:
            db_logger.log_query("insert_documents", database_name=db_name, document_count=len(documents))
            result_ids = db.insert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                errors=body.errors,
                similarity_threshold=body.similarity_threshold,
            )
            return {"ids": result_ids, "message": f"Successfully inserted {len(result_ids)} documents"}
        except Exception as e:
            db_logger.log_error("insert_documents", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/documents/chunks", response_model=WriteResponse, dependencies=[Depends(require_write_permission)]
)
@log_performance("upsert_from_chunks")
async def upsert_from_chunks(db_name: str, body: UpsertChunksBody, db=Depends(get_db)):
    """Upsert documents from pre-chunked data."""
    with request_context("upsert_from_chunks"):
        try:
            db_logger.log_query(
                "upsert_from_chunks", database_name=db_name, document_count=len(body.chunks_by_document)
            )
            result_ids = db.upsert_from_chunks(
                chunks_by_document=_normalize_chunks(body.chunks_by_document),
                metadata=body.metadata,
                batch_size=body.batch_size,
                similarity_threshold=body.similarity_threshold,
            )
            return {"ids": result_ids, "message": f"Successfully upserted {len(result_ids)} documents from chunks"}
        except Exception as e:
            db_logger.log_error("upsert_from_chunks", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/documents/chunks/insert",
    response_model=WriteResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("insert_from_chunks")
async def insert_from_chunks(db_name: str, body: InsertChunksBody, db=Depends(get_db)):
    """Insert documents from pre-chunked data with conflict handling."""
    with request_context("insert_from_chunks"):
        if body.errors not in ("raise", "ignore"):
            raise ValidationError("errors parameter must be 'raise' or 'ignore'", field="errors", value=body.errors)
        try:
            db_logger.log_query(
                "insert_from_chunks", database_name=db_name, document_count=len(body.chunks_by_document)
            )
            result_ids = db.insert_from_chunks(
                chunks_by_document=_normalize_chunks(body.chunks_by_document),
                metadata=body.metadata,
                batch_size=body.batch_size,
                similarity_threshold=body.similarity_threshold,
                errors=body.errors,
            )
            return {"ids": result_ids, "message": f"Successfully inserted {len(result_ids)} documents from chunks"}
        except Exception as e:
            db_logger.log_error("insert_from_chunks", e, database_name=db_name)
            raise


# --------------------------------------------------------------------------- #
# Read / single-document endpoints
# --------------------------------------------------------------------------- #


@router.get(
    "/{db_name}/documents/{doc_id}",
    response_model=DocumentResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_document")
def get_document(db_name: str, doc_id: str, db=Depends(get_db)):
    """Get a document by ID."""
    with request_context("get_document"):
        try:
            return serialize_document(db.get(doc_id))
        except DocumentNotFoundError as e:
            raise APIError(
                message=f"Document '{doc_id}' not found in database '{db_name}'",
                error_code="DOCUMENT_NOT_FOUND",
                status_code=404,
                recoverable=True,
            ) from e
        except Exception as e:
            db_logger.log_error("get_document", e, database_name=db_name, document_id=doc_id)
            raise


@router.patch(
    "/{db_name}/documents/{doc_id}",
    response_model=UpdateResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("update_document")
async def update_document(db_name: str, doc_id: str, body: UpdateDocumentBody, db=Depends(get_db)):
    """Partially update a document's content and/or metadata."""
    with request_context("update_document"):
        if body.content is None and body.metadata is None:
            raise ValidationError("Either content or metadata must be provided")

        try:
            db_logger.log_query(
                "update_document",
                database_name=db_name,
                document_id=doc_id,
                has_content=body.content is not None,
                has_metadata=body.metadata is not None,
            )
            was_updated = db.update(doc_id, content=body.content, metadata=body.metadata)
            if not was_updated:
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True,
                )
            return {"updated": True, "message": f"Successfully updated document {doc_id}"}
        except Exception as e:
            db_logger.log_error("update_document", e, database_name=db_name, document_id=doc_id)
            raise


@router.delete(
    "/{db_name}/documents/{doc_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("delete_document")
def delete_document(db_name: str, doc_id: str, db=Depends(get_db)):
    """Delete a document by ID."""
    with request_context("delete_document"):
        try:
            if not db.exists(doc_id):
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True,
                )
            deleted_count = db.delete(doc_id)
            db_logger.log_query(
                "delete_document_success", database_name=db_name, document_id=doc_id, deleted_count=deleted_count
            )
            return {"deleted_count": deleted_count, "message": f"Successfully deleted document {doc_id}"}
        except Exception as e:
            db_logger.log_error("delete_document", e, database_name=db_name, document_id=doc_id)
            raise


@router.post(
    "/{db_name}/documents/delete",
    response_model=BatchDeleteResponse,
    dependencies=[Depends(require_write_permission)],
)
@log_performance("delete_documents_batch")
async def delete_documents_batch(db_name: str, body: BatchDeleteBody, db=Depends(get_db)):
    """Delete multiple documents by ID."""
    with request_context("delete_documents_batch"):
        ids = body.ids
        if not ids:
            return {"deleted_count": 0, "failed_ids": [], "message": "No documents to delete"}

        unique_ids = list(dict.fromkeys(ids))
        if len(unique_ids) != len(ids):
            logger.warning("Duplicate IDs found in batch delete request, removing duplicates")
            ids = unique_ids
        if len(ids) > _MAX_BATCH:
            raise ValidationError(
                f"Batch size ({len(ids)}) exceeds maximum allowed ({_MAX_BATCH})", field="ids", value=len(ids)
            )

        try:
            deleted_count = 0
            failed_ids: List[str] = []
            for doc_id in ids:
                if not db.exists(doc_id):
                    failed_ids.append(doc_id)
                    continue
                try:
                    deleted_count += db.delete(doc_id)
                except Exception as e:  # noqa: BLE001 - per-id best effort
                    logger.warning(f"Failed to delete document {sanitize_log_value(doc_id)}: {sanitize_log_value(e)}")
                    failed_ids.append(doc_id)

            db_logger.log_query(
                "delete_documents_batch_success",
                database_name=db_name,
                total_requested=len(ids),
                deleted_count=deleted_count,
                failed_count=len(failed_ids),
            )
            return {
                "deleted_count": deleted_count,
                "failed_ids": failed_ids,
                "message": f"Batch delete completed. Deleted {deleted_count} documents, {len(failed_ids)} failed",
            }
        except Exception as e:
            db_logger.log_error("delete_documents_batch", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/documents/count",
    response_model=CountResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("count_documents")
async def count_documents(db_name: str, body: Optional[FilterBody] = None, db=Depends(get_db)):
    """Count documents matching optional metadata filters."""
    with request_context("count_documents"):
        filters = body.filters if body else None
        try:
            db_logger.log_query("count_documents", database_name=db_name, has_filters=filters is not None)
            count = db.count(filters=filters)
            db_logger.log_query("count_documents_success", database_name=db_name, result_count=count)
            return {"count": count}
        except Exception as e:
            db_logger.log_query("count_documents_error", database_name=db_name, error=str(e))
            raise


@router.post(
    "/{db_name}/documents/exists",
    response_model=ExistsResponse,
    dependencies=[Depends(require_read_permission)],
)
@log_performance("check_documents_exist")
async def check_documents_exist(db_name: str, body: ExistsBody, db=Depends(get_db)):
    """Check whether documents exist by ID."""
    with request_context("check_documents_exist"):
        ids = _as_list(body.ids) or []
        try:
            return {"exists": db.exists(ids), "ids": ids}
        except Exception as e:
            db_logger.log_error("check_documents_exist", e, database_name=db_name)
            raise


@router.get(
    "/{db_name}/documents",
    dependencies=[Depends(require_read_permission)],
)
@log_performance("list_documents")
def list_documents(
    db_name: str,
    db=Depends(get_db),
    ids: Optional[str] = Query(default=None, description="Comma-separated document IDs to fetch directly"),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE_LIMIT),
    offset: int = Query(default=0, ge=0),
):
    """List documents with ``limit``/``offset`` pagination, or fetch specific ``ids``."""
    with request_context("list_documents" if not ids else "get_documents"):
        try:
            if ids:
                id_list = [i.strip() for i in ids.split(",") if i.strip()]
                documents = db.get(id_list)
                returned_ids = [doc.id for doc in documents]
                return DocumentsByIdResponse(
                    documents=[DocumentResponse(**serialize_document(doc)) for doc in documents],
                    returned_ids=returned_ids,
                    missing_ids=[i for i in id_list if i not in returned_ids],
                )

            documents = db.filter(where=None, limit=limit, offset=offset)
            total = db.count(filters=None)
            return DocumentListResponse(
                documents=[DocumentResponse(**serialize_document(doc)) for doc in documents],
                pagination=PageInfo(limit=limit, offset=offset, total=total, has_more=offset + len(documents) < total),
            )
        except Exception as e:
            db_logger.log_error("list_documents", e, database_name=db_name)
            raise
