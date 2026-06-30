# src/localvectordb_server/routers/documents.py
"""Document CRUD routes."""

import json
import logging
from math import ceil

from fastapi import APIRouter, Depends, Request

from localvectordb.exceptions import DocumentNotFoundError
from localvectordb_server._auth import require_read_permission, require_write_permission
from localvectordb_server._error_handlers import (
    APIError,
    ValidationError,
    validate_field_type,
    validate_pagination_params,
    validate_required_fields,
)
from localvectordb_server._logcfg import DatabaseLogger, log_performance, request_context
from localvectordb_server._serializers import serialize_document
from localvectordb_server.routers._deps import get_db

logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
router = APIRouter(tags=["documents"])


@router.post("/{db_name}/documents", dependencies=[Depends(require_write_permission)])
@log_performance("upsert_documents")
async def upsert_documents(db_name: str, request: Request):
    """Upsert (insert or update) documents."""

    with request_context("upsert_documents"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ["documents"])

        documents = data["documents"]
        if not documents:
            raise ValidationError("Documents array cannot be empty", field="documents")

        # Convert to list if single document
        if isinstance(documents, str):
            documents = [documents]
        elif not isinstance(documents, list):
            raise ValidationError("Documents must be a string or array of strings", field="documents")

        # Validate document content
        for i, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise ValidationError(f"Document at index {i} must be a string", field=f"documents[{i}]")
            if not doc.strip():
                raise ValidationError(f"Document at index {i} cannot be empty", field=f"documents[{i}]")

        # Handle metadata
        metadata = data.get("metadata", data.get("metadatas"))
        if metadata is not None:
            if isinstance(metadata, dict):
                metadata = [metadata]
            elif not isinstance(metadata, list):
                raise ValidationError("Metadata must be an object or array of objects", field="metadata")

            if len(metadata) != len(documents):
                raise ValidationError(
                    f"Number of metadata entries ({len(metadata)}) must match number of documents ({len(documents)})",
                    field="metadata",
                )

        # Handle IDs
        ids = data.get("ids")
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            elif not isinstance(ids, list):
                raise ValidationError("IDs must be a string or array of strings", field="ids")

            if len(ids) != len(documents):
                raise ValidationError(
                    f"Number of IDs ({len(ids)}) must match number of documents ({len(documents)})",
                    field="ids",
                )

        # Validate batch size
        config = request.app.state.config
        default_batch_size = config.embedding.batch_size if config else 100
        batch_size = int(data.get("batch_size", default_batch_size))

        validate_field_type(data, "batch_size", int)
        if batch_size < 1 or batch_size > 1000:
            raise ValidationError(
                "Batch size must be between 1 and 1000",
                field="batch_size",
                value=batch_size,
            )

        similarity_threshold = data.get("similarity_threshold", None)
        if similarity_threshold is not None:
            validate_field_type(data, "similarity_threshold", (int, float))
            if similarity_threshold < 0 or similarity_threshold > 1:
                raise ValidationError(
                    "Similarity threshold must be between 0 and 1",
                    field="similarity_threshold",
                    value=similarity_threshold,
                )

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "upsert_documents",
                database_name=db_name,
                document_count=len(documents),
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
            )

            result_ids = db.upsert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
            )

            db_logger.log_query(
                "upsert_documents_success",
                database_name=db_name,
                result_count=len(result_ids),
            )

            return {
                "message": f"Successfully processed {len(documents)} documents",
                "ids": result_ids,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("upsert_documents", e, database_name=db_name, document_count=len(documents))
            raise


@router.post("/{db_name}/documents/insert", dependencies=[Depends(require_write_permission)])
@log_performance("insert_documents")
async def insert_documents(db_name: str, request: Request):
    """Insert new documents (fails if ID already exists)."""

    with request_context("insert_documents"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Similar validation as upsert_documents...
        validate_required_fields(data, ["documents"])

        documents = data["documents"]
        if isinstance(documents, str):
            documents = [documents]

        metadata = data.get("metadata", data.get("metadatas"))
        if isinstance(metadata, dict):
            metadata = [metadata]

        ids = data.get("ids")
        if isinstance(ids, str):
            ids = [ids]

        config = request.app.state.config
        default_batch_size = config.embedding.batch_size if config else 100
        batch_size = int(data.get("batch_size", default_batch_size))
        errors = data.get("errors", "raise")  # "raise" or "ignore"
        similarity_threshold = data.get("similarity_threshold")

        # Validate parameters
        if errors not in ["raise", "ignore"]:
            raise ValidationError(
                "Errors parameter must be 'raise' or 'ignore'",
                field="errors",
                value=errors,
            )

        if similarity_threshold is not None:
            validate_field_type(data, "similarity_threshold", (int, float))
            if similarity_threshold < 0 or similarity_threshold > 1:
                raise ValidationError(
                    "Similarity threshold must be between 0 and 1",
                    field="similarity_threshold",
                    value=similarity_threshold,
                )

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "insert_documents",
                database_name=db_name,
                document_count=len(documents),
                similarity_threshold=similarity_threshold,
            )

            result_ids = db.insert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                errors=errors,
                similarity_threshold=similarity_threshold,
            )

            return {
                "message": f"Successfully inserted {len(result_ids)} documents",
                "ids": result_ids,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("insert_documents", e, database_name=db_name)
            raise


@router.post("/{db_name}/documents/chunks", dependencies=[Depends(require_write_permission)])
@log_performance("upsert_from_chunks")
async def upsert_from_chunks(db_name: str, request: Request):
    """Upsert documents from pre-chunked data."""

    with request_context("upsert_from_chunks"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ["chunks_by_document"])

        chunks_by_document = data["chunks_by_document"]
        metadata = data.get("metadata", {})
        batch_size = data.get("batch_size", 100)
        similarity_threshold = data.get("similarity_threshold")

        # Convert incoming dict chunks to simple strings and let LocalVectorDB normalize
        processed_chunks = {}
        for doc_id, chunks in chunks_by_document.items():
            if chunks and isinstance(chunks[0], dict):
                processed_chunks[doc_id] = [
                    chunk.get("content") or chunk.get("text", str(chunk)) if isinstance(chunk, dict) else chunk
                    for chunk in chunks
                ]
            else:
                processed_chunks[doc_id] = chunks

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "upsert_from_chunks",
                database_name=db_name,
                document_count=len(chunks_by_document),
            )

            result_ids = db.upsert_from_chunks(
                chunks_by_document=processed_chunks,
                metadata=metadata,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
            )

            return {
                "message": f"Successfully upserted {len(result_ids)} documents from chunks",
                "ids": result_ids,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("upsert_from_chunks", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/documents/chunks/insert",
    dependencies=[Depends(require_write_permission)],
)
@log_performance("insert_from_chunks")
async def insert_from_chunks(db_name: str, request: Request):
    """Insert documents from pre-chunked data with conflict handling."""

    with request_context("insert_from_chunks"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ["chunks_by_document"])

        chunks_by_document = data["chunks_by_document"]
        metadata = data.get("metadata", {})
        batch_size = data.get("batch_size", 100)
        similarity_threshold = data.get("similarity_threshold")
        errors = data.get("errors", "raise")

        # Convert incoming dict chunks to simple strings and let LocalVectorDB normalize
        processed_chunks = {}
        for doc_id, chunks in chunks_by_document.items():
            if chunks and isinstance(chunks[0], dict):
                processed_chunks[doc_id] = [
                    chunk.get("content") or chunk.get("text", str(chunk)) if isinstance(chunk, dict) else chunk
                    for chunk in chunks
                ]
            else:
                processed_chunks[doc_id] = chunks

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "insert_from_chunks",
                database_name=db_name,
                document_count=len(chunks_by_document),
            )

            result_ids = db.insert_from_chunks(
                chunks_by_document=processed_chunks,
                metadata=metadata,
                batch_size=batch_size,
                similarity_threshold=similarity_threshold,
                errors=errors,
            )

            return {
                "message": f"Successfully inserted {len(result_ids)} documents from chunks",
                "ids": result_ids,
                "status": "success",
            }

        except Exception as e:
            db_logger.log_error("insert_from_chunks", e, database_name=db_name)
            raise


@router.get(
    "/{db_name}/documents/{doc_id}",
    dependencies=[Depends(require_read_permission)],
)
@log_performance("get_document")
def get_document(db_name: str, doc_id: str, request: Request):
    """Get a document by ID."""

    with request_context("get_document"):
        try:
            db = get_db(db_name, request)
            doc = db.get(doc_id)
            return serialize_document(doc)

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


@router.put(
    "/{db_name}/documents/{doc_id}",
    dependencies=[Depends(require_write_permission)],
)
@log_performance("update_document")
async def update_document(db_name: str, doc_id: str, request: Request):
    """Update a document's content and/or metadata."""

    with request_context("update_document"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        content = data.get("content")
        metadata = data.get("metadata")

        if not content and not metadata:
            raise ValidationError("Either content or metadata must be provided")

        if content is not None and not isinstance(content, str):
            raise ValidationError("Content must be a string", field="content")

        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError("Metadata must be an object", field="metadata")

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "update_document",
                database_name=db_name,
                document_id=doc_id,
                has_content=content is not None,
                has_metadata=metadata is not None,
            )

            was_updated = db.update(doc_id, content=content, metadata=metadata)

            if not was_updated:
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True,
                )

            return {
                "message": f"Successfully updated document {doc_id}",
                "status": "success",
                "updated": was_updated,
            }

        except Exception as e:
            db_logger.log_error("update_document", e, database_name=db_name, document_id=doc_id)
            raise


@router.delete(
    "/{db_name}/documents/{doc_id}",
    dependencies=[Depends(require_write_permission)],
)
@log_performance("delete_document")
def delete_document(db_name: str, doc_id: str, request: Request):
    """Delete a document by ID."""

    with request_context("delete_document"):
        try:
            db = get_db(db_name, request)

            if not db.exists(doc_id):
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True,
                )

            deleted_count = db.delete(doc_id)

            db_logger.log_query(
                "delete_document_success",
                database_name=db_name,
                document_id=doc_id,
                deleted_count=deleted_count,
            )

            return {
                "message": f"Successfully deleted document {doc_id}",
                "status": "success",
                "deleted_count": deleted_count,
            }

        except Exception as e:
            db_logger.log_error("delete_document", e, database_name=db_name, document_id=doc_id)
            raise


@router.post(
    "/{db_name}/documents/delete",
    dependencies=[Depends(require_write_permission)],
)
@log_performance("delete_documents_batch")
async def delete_documents_batch(db_name: str, request: Request):
    """Delete multiple documents by IDs."""

    with request_context("delete_documents_batch"):
        try:
            db = get_db(db_name, request)

            # Get request data
            data = await request.json()
            if not data or "ids" not in data:
                raise APIError(
                    message="Request must include 'ids' field with list of document IDs to delete",
                    error_code="MISSING_REQUIRED_FIELD",
                    status_code=400,
                    recoverable=True,
                )

            ids = data["ids"]
            if not isinstance(ids, list):
                raise APIError(
                    message="'ids' field must be a list of document IDs",
                    error_code="INVALID_FIELD_TYPE",
                    status_code=400,
                    recoverable=True,
                )

            if not ids:
                return {
                    "message": "No documents to delete",
                    "status": "success",
                    "deleted_count": 0,
                    "failed_ids": [],
                }

            # Validate all IDs are strings
            for i, doc_id in enumerate(ids):
                if not isinstance(doc_id, str):
                    raise APIError(
                        message=f"All document IDs must be strings, but item {i} is {type(doc_id).__name__}",
                        error_code="INVALID_DOCUMENT_ID_TYPE",
                        status_code=400,
                        recoverable=True,
                    )

            # Check for duplicates
            unique_ids = list(set(ids))
            if len(unique_ids) != len(ids):
                logger.warning("Duplicate IDs found in batch delete request, removing duplicates")
                ids = unique_ids

            # Limit batch size for safety
            max_batch_size = 1000
            if len(ids) > max_batch_size:
                raise APIError(
                    message=f"Batch size ({len(ids)}) exceeds maximum allowed ({max_batch_size})",
                    error_code="BATCH_SIZE_EXCEEDED",
                    status_code=400,
                    recoverable=True,
                )

            # Check which documents exist
            existing_ids = []
            non_existing_ids = []

            for doc_id in ids:
                if db.exists(doc_id):
                    existing_ids.append(doc_id)
                else:
                    non_existing_ids.append(doc_id)

            # Delete existing documents
            deleted_count = 0
            failed_ids = []

            for doc_id in existing_ids:
                try:
                    count = db.delete(doc_id)
                    deleted_count += count
                except Exception as e:
                    logger.warning(f"Failed to delete document {doc_id}: {e}")
                    failed_ids.append(doc_id)

            # Add non-existing IDs to failed list
            failed_ids.extend(non_existing_ids)

            db_logger.log_query(
                "delete_documents_batch_success",
                database_name=db_name,
                total_requested=len(ids),
                deleted_count=deleted_count,
                failed_count=len(failed_ids),
            )

            return {
                "message": f"Batch delete completed. Deleted {deleted_count} documents, {len(failed_ids)} failed",
                "status": "success",
                "deleted_count": deleted_count,
                "failed_ids": failed_ids,
            }

        except Exception as e:
            db_logger.log_error("delete_documents_batch", e, database_name=db_name)
            raise


@router.post(
    "/{db_name}/documents/count",
    dependencies=[Depends(require_read_permission)],
)
@log_performance("count_documents")
async def count_documents(db_name: str, request: Request):
    """Count documents matching filter criteria.

    Request body supports:
    {
        "filters": {
            // Optional MongoDB-style filters
            "author": "John Doe",
            "year": {"$gte": 2020},
            // ... any filter expression
        }
    }

    Returns:
    {
        "count": 42
    }
    """

    with request_context("count_documents"):
        # Allow both JSON body with filters or empty body/GET-like request
        filters = None

        try:
            data = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Optional body: empty or non-JSON request means "no filters".
            data = None

        if data:
            filters = data.get("filters")

            # Validate filters if provided
            if filters is not None:
                validate_field_type(data, "filters", dict)

        try:
            db = get_db(db_name, request)

            db_logger.log_query(
                "count_documents",
                database_name=db_name,
                has_filters=filters is not None,
                filter_complexity=len(str(filters)) if filters else 0,
            )

            count = db.count(where=filters)

            db_logger.log_query("count_documents_success", database_name=db_name, result_count=count)

            return {"count": count}

        except Exception as e:
            db_logger.log_query("count_documents_error", database_name=db_name, error=str(e))
            raise


@router.post(
    "/{db_name}/documents/exists",
    dependencies=[Depends(require_read_permission)],
)
@log_performance("check_documents_exist")
async def check_documents_exist(db_name: str, request: Request):
    """Check if documents exist by ID."""

    with request_context("check_documents_exist"):
        data = await request.json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ["ids"])

        ids = data["ids"]
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            raise ValidationError("IDs must be a string or array of strings", field="ids")

        try:
            db = get_db(db_name, request)
            exists = db.exists(ids)

            return {"exists": exists, "ids": ids}

        except Exception as e:
            db_logger.log_error("check_documents_exist", e, database_name=db_name)
            raise


@router.get(
    "/{db_name}/documents",
    dependencies=[Depends(require_read_permission)],
)
@log_performance("list_documents")
def list_documents(db_name: str, request: Request):
    """List documents with pagination and filtering."""
    ids = request.query_params.get("ids")

    with request_context("list_documents" if not ids else "get_documents"):
        try:
            db = get_db(db_name, request)

            if ids:
                id_list = [i.strip() for i in ids.split(",") if i.strip()]
                documents = db.get(id_list)
                serialized_docs = [serialize_document(doc) for doc in documents]
                returned_ids = [doc.id for doc in documents]
                missing_ids = [i for i in id_list if i not in returned_ids]

                return {
                    "documents": serialized_docs,
                    "returned_ids": returned_ids,
                    "missing_ids": missing_ids,
                }

            # Validate pagination parameters
            page, limit = validate_pagination_params(
                request.query_params.get("page"),
                request.query_params.get("limit"),
            )

            # Simple filtering parameters
            filters = {}
            for key, value in request.query_params.items():
                if key not in ["page", "limit", "ids"]:
                    filters[key] = value

            # Calculate offset
            offset = (page - 1) * limit

            # Get filtered documents
            documents = db.filter(where=filters if filters else None, limit=limit, offset=offset)

            # Get total count (this is inefficient, but works for now)
            total_count = db.count(where=filters if filters else None)
            total_pages = ceil(total_count / limit)

            # Serialize documents
            serialized_docs = [serialize_document(doc) for doc in documents]

            return {
                "documents": serialized_docs,
                "pagination": {
                    "current_page": page,
                    "total_pages": total_pages,
                    "page_size": limit,
                    "total_count": total_count,
                    "has_previous": page > 1,
                    "has_next": page < total_pages,
                },
            }

        except Exception as e:
            db_logger.log_error("list_documents", e, database_name=db_name)
            raise
