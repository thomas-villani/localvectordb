# Copyright (c) 2023-2025 Tom Villani, Ph.D.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# You may not use this file for commercial purposes without explicit permission.
#
# For more information, please visit: https://creativecommons.org/licenses/by-nc/4.0/
#
# Contact: thomas.villani@gmail.com
# 
# src/localvectordb_server/routes.py
"""
localvectordb_server/routes.py - LocalVectorDB v1.0 API Routes

Enhanced API routes with structured logging, comprehensive error handling,
input validation, and performance monitoring.
"""
import json
import logging
from datetime import datetime
from math import ceil
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from localvectordb.core import MetadataField, MetadataFieldType
from localvectordb.utils import get_system_version
from localvectordb_server._auth import require_api_key
from localvectordb_server._checkdeps import check_ollama_service
from localvectordb_server.config import DatabaseSettings, EmbeddingSettings
from localvectordb_server._cache import cache
from localvectordb_server._error_handlers import (
    handle_errors, APIError, ValidationError,
    validate_required_fields, validate_field_type, validate_pagination_params,
    validate_search_params, validate_database_creation_params
)
from localvectordb_server._logcfg import log_performance, request_context, DatabaseLogger
import mimetypes
from werkzeug.utils import secure_filename

# Add this import after the existing imports in routes.py
from localvectordb_server.extractors import get_extractor_registry, get_supported_formats
FILE_EXTRACTION_AVAILABLE = True



logger = logging.getLogger(__name__)
db_logger = DatabaseLogger()
api = Blueprint('api', __name__)


def serialize_document(doc) -> Dict[str, Any]:
    """Serialize a Document object for JSON response"""
    return {
        "id": doc.id,
        "content": doc.content,
        "metadata": doc.metadata,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
        "content_hash": doc.content_hash
    }


def serialize_query_result(result) -> Dict[str, Any]:
    """Serialize a QueryResult object for JSON response"""
    data = {
        "id": result.id,
        "score": result.score,
        "type": result.type,
        "content": result.content,
        "metadata": result.metadata
    }

    # Add chunk-specific fields if applicable
    if result.type == 'chunk' and result.document_id:
        data["document_id"] = result.document_id

    if result.position:
        data["position"] = result.position.to_dict()

    return data


def parse_metadata_schema(schema_data: Dict[str, Any]) -> Dict[str, MetadataField]:
    """Parse metadata schema from request data with validation"""
    if not schema_data:
        return {}

    if not isinstance(schema_data, dict):
        raise ValidationError("Metadata schema must be an object", field="metadata_schema")

    parsed_schema = {}
    for field_name, field_config in schema_data.items():
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValidationError(
                "Metadata field names must be non-empty strings",
                field=f"metadata_schema.{field_name}"
            )

        try:
            if isinstance(field_config, str):
                # Simple string type
                field_type = MetadataFieldType(field_config)
                parsed_schema[field_name] = MetadataField(type=field_type)
            elif isinstance(field_config, dict):
                # Full field configuration
                field_type = MetadataFieldType(field_config.get('type', 'text'))
                parsed_schema[field_name] = MetadataField(
                    type=field_type,
                    indexed=field_config.get('indexed', False),
                    required=field_config.get('required', False),
                    default_value=field_config.get('default_value')
                )
            else:
                raise ValidationError(
                    f"Invalid metadata field configuration for '{field_name}'. Must be string or object.",
                    field=f"metadata_schema.{field_name}"
                )
        except ValueError as e:
            raise ValidationError(
                f"Invalid metadata field type for '{field_name}': {str(e)}",
                field=f"metadata_schema.{field_name}"
            )

    return parsed_schema


# Database Management Routes
@api.route("/api/v1/databases", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("create_database")
def create_database():
    """Create a new vector database with optional metadata schema."""

    with request_context("create_database"):
        # Validate request has JSON body
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate input parameters
        data = validate_database_creation_params(data)

        name = data["name"]

        # Check if database already exists
        existing_dbs = current_app.db_manager.list_databases()
        if name in existing_dbs:
            raise APIError(
                message=f"Database '{name}' already exists",
                error_code="DATABASE_ALREADY_EXISTS",
                status_code=409,
                recoverable=True
            )

        # Get configuration with defaults
        if hasattr(current_app, "config_obj"):
            db_config = current_app.config_obj.database.copy()
            embedding_config = current_app.config_obj.embedding.copy()
        else:
            db_config = DatabaseSettings()
            embedding_config = EmbeddingSettings()

        # Parse metadata schema if provided
        metadata_schema = None
        if "metadata_schema" in data:
            metadata_schema = parse_metadata_schema(data["metadata_schema"])
        else:
            metadata_schema = db_config.default_metadata_schema

        # Update configurations from request
        if "database" in data:
            database_settings = data["database"]
            try:
                db_config.update_from_dict(database_settings)
            except Exception as e:
                raise ValidationError(f"Invalid database configuration: {str(e)}")

        if "embedding" in data:
            embedding_settings = data["embedding"]
            try:
                embedding_config.update_from_dict(embedding_settings)
            except Exception as e:
                raise ValidationError(f"Invalid embedding configuration: {str(e)}")

        # Create database
        db_logger.log_query("create_database", database_name=name)

        try:
            db = current_app.db_manager.create_db(name, metadata_schema, db_config, embedding_config)

            db_logger.log_query("create_database_success", database_name=name)

            return jsonify({
                "message": f"Successfully created database '{name}'",
                "status": "success",
                "config": {
                    "name": db.name,
                    "embedding_provider": db.embedding_provider.provider_name,
                    "embedding_model": db.embedding_provider.model,
                    "embedding_dimension": db.embedding_dimension,
                    "chunking_method": db.chunking_method,
                    "chunk_size": db.chunk_size,
                    "chunk_overlap": db.chunk_overlap,
                    "metadata_schema": {
                        field_name: {
                            "type": field.type.value,
                            "indexed": field.indexed,
                            "required": field.required,
                            "default_value": field.default_value
                        }
                        for field_name, field in (db.metadata_schema or {}).items()
                    },
                    "fts_enabled": db.fts_enabled
                }
            })

        except Exception as e:
            db_logger.log_error("create_database", e, database_name=name)
            raise


@api.route("/api/v1/databases", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=60)  # Cache for 1 minute
@log_performance("list_databases")
def list_databases():
    """List all available databases"""

    with request_context("list_databases"):
        try:
            databases = current_app.db_manager.list_databases()
            return jsonify({
                "databases": databases,
                "count": len(databases)
            })

        except Exception as e:
            db_logger.log_error("list_databases", e)
            raise


@api.route("/api/v1/<db_name>/info", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=300)  # Cache for 5 minutes
@log_performance("get_database_info")
def get_database_info(db_name):
    """Get information about a specific database"""

    with request_context("get_database_info"):
        try:
            db = current_app.db_manager.get_db(db_name)
            stats = db.stats

            return jsonify({
                "name": db.name,
                "stats": stats,
                "config": {
                    "embedding_provider": db.embedding_provider.provider_name,
                    "embedding_model": db.embedding_provider.model,
                    "embedding_dimension": db.embedding_dimension,
                    "chunking_method": db.chunking_method,
                    "chunk_size": db.chunk_size,
                    "chunk_overlap": db.chunk_overlap,
                    "metadata_schema": {
                        field_name: {
                            "type": field.type.value,
                            "indexed": field.indexed,
                            "required": field.required,
                            "default_value": field.default_value
                        }
                        for field_name, field in (db.metadata_schema or {}).items()
                    },
                    "fts_enabled": db.fts_enabled
                }
            })

        except Exception as e:
            db_logger.log_error("get_database_info", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>", methods=["DELETE"])
@require_api_key
@handle_errors
@log_performance("delete_database")
def delete_database(db_name):
    """Delete a database"""

    with request_context("delete_database"):

        success, errors = current_app.db_manager.delete_db(db_name)
        if not success:
            raise APIError(errors, "DATABASE_DELETE_ERROR", 500, {"db_name": db_name}, True)

        return jsonify({
            "message": f"Successfully deleted database '{db_name}'",
            "status": "success",
        })


# Document Management Routes
@api.route("/api/v1/<db_name>/documents", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("upsert_documents")
def upsert_documents(db_name):
    """Upsert (insert or update) documents"""

    with request_context("upsert_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ['documents'])

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
                    field="metadata"
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
                    field="ids"
                )

        # Validate batch size
        batch_size = data.get("batch_size", 100)
        validate_field_type(data, "batch_size", int)
        if batch_size < 1 or batch_size > 1000:
            raise ValidationError("Batch size must be between 1 and 1000", field="batch_size", value=batch_size)

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("upsert_documents",
                                database_name=db_name,
                                document_count=len(documents),
                                batch_size=batch_size)

            result_ids = db.upsert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size
            )

            db_logger.log_query("upsert_documents_success",
                                database_name=db_name,
                                result_count=len(result_ids))

            return jsonify({
                "message": f"Successfully processed {len(documents)} documents",
                "ids": result_ids,
                "status": "success"
            })

        except Exception as e:
            db_logger.log_error("upsert_documents", e, database_name=db_name, document_count=len(documents))
            raise


@api.route("/api/v1/<db_name>/documents/insert", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("insert_documents")
def insert_documents(db_name):
    """Insert new documents (fails if ID already exists)"""

    with request_context("insert_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Similar validation as upsert_documents...
        validate_required_fields(data, ['documents'])

        documents = data["documents"]
        if isinstance(documents, str):
            documents = [documents]

        metadata = data.get("metadata", data.get("metadatas"))
        if isinstance(metadata, dict):
            metadata = [metadata]

        ids = data.get("ids")
        if isinstance(ids, str):
            ids = [ids]

        batch_size = data.get("batch_size", 100)
        errors = data.get("errors", "raise")  # "raise" or "ignore"
        similarity_threshold = data.get("similarity_threshold")

        # Validate parameters
        if errors not in ["raise", "ignore"]:
            raise ValidationError("Errors parameter must be 'raise' or 'ignore'", field="errors", value=errors)

        if similarity_threshold is not None:
            validate_field_type(data, "similarity_threshold", (int, float))
            if similarity_threshold < 0 or similarity_threshold > 1:
                raise ValidationError(
                    "Similarity threshold must be between 0 and 1",
                    field="similarity_threshold",
                    value=similarity_threshold
                )

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("insert_documents",
                                database_name=db_name,
                                document_count=len(documents),
                                similarity_threshold=similarity_threshold)

            result_ids = db.insert(
                documents=documents,
                metadata=metadata,
                ids=ids,
                batch_size=batch_size,
                errors=errors,
                similarity_threshold=similarity_threshold
            )

            return jsonify({
                "message": f"Successfully inserted {len(result_ids)} documents",
                "ids": result_ids,
                "status": "success"
            })

        except Exception as e:
            db_logger.log_error("insert_documents", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=300)
@log_performance("get_document")
def get_document(db_name, doc_id):
    """Get a document by ID"""

    with request_context("get_document"):
        try:
            db = current_app.db_manager.get_db(db_name)
            doc = db.get(doc_id)

            if doc is None:
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            return jsonify(serialize_document(doc))

        except Exception as e:
            db_logger.log_error("get_document", e, database_name=db_name, document_id=doc_id)
            raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["PUT"])
@require_api_key
@handle_errors
@log_performance("update_document")
def update_document(db_name, doc_id):
    """Update a document's content and/or metadata"""

    with request_context("update_document"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
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
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("update_document",
                                database_name=db_name,
                                document_id=doc_id,
                                has_content=content is not None,
                                has_metadata=metadata is not None)

            was_updated = db.update(doc_id, content=content, metadata=metadata)

            if not was_updated:
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            return jsonify({
                "message": f"Successfully updated document {doc_id}",
                "status": "success",
                "updated": was_updated
            })

        except Exception as e:
            db_logger.log_error("update_document", e, database_name=db_name, document_id=doc_id)
            raise


@api.route("/api/v1/<db_name>/documents/<doc_id>", methods=["DELETE"])
@require_api_key
@handle_errors
@log_performance("delete_document")
def delete_document(db_name, doc_id):
    """Delete a document by ID"""

    with request_context("delete_document"):
        try:
            db = current_app.db_manager.get_db(db_name)

            if not db.exists(doc_id):
                raise APIError(
                    message=f"Document '{doc_id}' not found in database '{db_name}'",
                    error_code="DOCUMENT_NOT_FOUND",
                    status_code=404,
                    recoverable=True
                )

            deleted_count = db.delete(doc_id)

            db_logger.log_query("delete_document_success",
                                database_name=db_name,
                                document_id=doc_id,
                                deleted_count=deleted_count)

            return jsonify({
                "message": f"Successfully deleted document {doc_id}",
                "status": "success",
                "deleted_count": deleted_count
            })

        except Exception as e:
            db_logger.log_error("delete_document", e, database_name=db_name, document_id=doc_id)
            raise


@api.route("/api/v1/<db_name>/documents/exists", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("check_documents_exist")
def check_documents_exist(db_name):
    """Check if documents exist by ID"""

    with request_context("check_documents_exist"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ['ids'])

        ids = data["ids"]
        if isinstance(ids, str):
            ids = [ids]
        elif not isinstance(ids, list):
            raise ValidationError("IDs must be a string or array of strings", field="ids")

        try:
            db = current_app.db_manager.get_db(db_name)
            exists = db.exists(ids)

            return jsonify({
                "exists": exists,
                "ids": ids
            })

        except Exception as e:
            db_logger.log_error("check_documents_exist", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>/documents", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=60)
@log_performance("list_documents")
def list_documents(db_name):
    """List documents with pagination and filtering"""

    with request_context("list_documents"):
        try:
            # Validate pagination parameters
            page, limit = validate_pagination_params(
                request.args.get('page'),
                request.args.get('limit')
            )

            # Simple filtering parameters
            filters = {}
            for key, value in request.args.items():
                if key not in ['page', 'limit']:
                    filters[key] = value

            db = current_app.db_manager.get_db(db_name)

            # Calculate offset
            offset = (page - 1) * limit

            # Get filtered documents
            documents = db.filter(
                where=filters if filters else None,
                limit=limit,
                offset=offset
            )

            # Get total count (this is inefficient, but works for now)
            total_count = len(db.filter(where=filters if filters else None))
            total_pages = ceil(total_count / limit)

            # Serialize documents
            serialized_docs = [serialize_document(doc) for doc in documents]

            return jsonify({
                "documents": serialized_docs,
                "pagination": {
                    "current_page": page,
                    "total_pages": total_pages,
                    "page_size": limit,
                    "total_count": total_count,
                    "has_previous": page > 1,
                    "has_next": page < total_pages
                }
            })

        except Exception as e:
            db_logger.log_error("list_documents", e, database_name=db_name)
            raise


def search_handler(db_name, search_params):
    """Unified query interface for all search types with enhanced validation"""

    # Validate search parameters
    search_params = validate_search_params(search_params)

    query_text = search_params["query"]
    search_type = search_params.get("search_type", "vector")
    return_type = search_params.get("return_type", "documents")
    k = search_params.get("k", 10)
    score_threshold = search_params.get("score_threshold", 0.0)
    filters = search_params.get("filters", search_params.get("metadata_filters"))
    vector_weight = search_params.get("vector_weight", 0.7)

    try:
        db = current_app.db_manager.get_db(db_name)

        db_logger.log_query("search",
                            database_name=db_name,
                            search_type=search_type,
                            return_type=return_type,
                            k=k,
                            query_length=len(query_text))

        results = db.query(
            query=query_text,
            search_type=search_type,
            return_type=return_type,
            k=k,
            score_threshold=score_threshold,
            filters=filters,
            vector_weight=vector_weight
        )

        # Serialize results
        serialized_results = [serialize_query_result(result) for result in results]

        db_logger.log_query("search_success",
                            database_name=db_name,
                            search_type=search_type,
                            result_count=len(serialized_results))

        return jsonify({
            "results": serialized_results,
            "search_type": search_type,
            "return_type": return_type,
            "total_results": len(serialized_results)
        })

    except Exception as e:
        db_logger.log_error("search", e, database_name=db_name, search_type=search_type)
        raise


# Search Routes
@api.route("/api/v1/<db_name>/query", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("query_documents")
def query_documents(db_name):
    """Unified query interface for all search types"""

    with request_context("query_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/vector", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("vector_search")
def vector_search(db_name):
    """Vector similarity search (convenience endpoint)"""

    with request_context("vector_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Add search_type to data
        data["search_type"] = "vector"
        return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/keyword", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("keyword_search")
def keyword_search(db_name):
    """Keyword search (convenience endpoint)"""

    with request_context("keyword_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Add search_type to data
        data["search_type"] = "keyword"
        return search_handler(db_name, data)


@api.route("/api/v1/<db_name>/search/hybrid", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("hybrid_search")
def hybrid_search(db_name):
    """Hybrid search (convenience endpoint)"""

    with request_context("hybrid_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Add search_type to data
        data["search_type"] = "hybrid"
        return search_handler(db_name, data)


# Filter and Metadata Routes
@api.route("/api/v1/<db_name>/filter", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("filter_documents")
def filter_documents(db_name):
    """Filter documents by metadata and other criteria"""

    with request_context("filter_documents"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        where = data.get("where")
        sql = data.get("sql")
        order_by = data.get("order_by")
        limit = data.get("limit")
        offset = data.get("offset", 0)

        if not where and not sql:
            raise ValidationError("Either 'where' or 'sql' must be provided")

        # Validate types
        if where is not None:
            validate_field_type(data, "where", dict)
        if sql is not None:
            validate_field_type(data, "sql", str)
        if order_by is not None:
            validate_field_type(data, "order_by", str)
        if limit is not None:
            validate_field_type(data, "limit", int)
            if limit < 1 or limit > 10000:
                raise ValidationError("Limit must be between 1 and 10000", field="limit", value=limit)

        validate_field_type(data, "offset", int)
        if offset < 0:
            raise ValidationError("Offset must be >= 0", field="offset", value=offset)

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("filter_documents",
                                database_name=db_name,
                                has_where=where is not None,
                                has_sql=sql is not None,
                                limit=limit)

            documents = db.filter(
                where=where,
                sql=sql,
                order_by=order_by,
                limit=limit,
                offset=offset
            )

            # Serialize results
            serialized_docs = [serialize_document(doc) for doc in documents]

            return jsonify({
                "documents": serialized_docs,
                "count": len(serialized_docs)
            })

        except Exception as e:
            db_logger.log_error("filter_documents", e, database_name=db_name)
            raise


# Global Search Route
@api.route("/api/v1/search", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("global_search")
def global_search():
    """Search across multiple databases"""

    with request_context("global_search"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate search parameters
        data = validate_search_params(data)

        query = data["query"]
        search_type = data.get("search_type", "vector")
        return_type = data.get("return_type", "documents")
        k = data.get("k", 10)
        score_threshold = data.get("score_threshold", 0.0)
        filters = data.get("filters")
        databases = data.get("databases")  # Optional list of databases to search
        vector_weight = data.get("vector_weight", 0.7)

        try:

            results = current_app.db_manager.search_databases(
                query=query,
                database_names=databases,
                search_type=search_type,
                return_type=return_type,
                k=k,
                score_threshold=score_threshold,
                filters=filters,
                vector_weight=vector_weight
            )
            for db_name, db_results in results.items():
                results[db_name] = [serialize_query_result(result) for result in db_results]

            return jsonify({
                "results": results,
                "search_type": search_type,
                "return_type": return_type
            })

        except Exception as e:
            db_logger.log_error("global_search", e, search_type=search_type)
            raise


# Health and System Routes
@api.route("/api/v1/health", methods=["GET"])
@handle_errors
def health_check():
    """System health check endpoint"""

    try:
        status = {
            "status": "healthy",
            "version": get_system_version(),
            "ollama_available": check_ollama_service(),
            "timestamp": logger.manager.loggerDict.get('timestamp', 'unknown')
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503


# Embedding Routes
@api.route("/api/v1/<db_name>/embeddings", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("get_embeddings_for_db")
def get_embeddings_for_db(db_name):
    """Get embeddings using the database's embedding provider"""

    with request_context("get_embeddings_for_db"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ['texts'])

        texts = data["texts"]
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValidationError("`texts` must be a string or array of strings", field="texts")

        provider = data.get("provider")
        if not provider:
            raise ValidationError("`provider` must be provided", field="provider")
        elif not isinstance(provider, str):
            raise ValidationError("`provider` must be a string", field="provider")

        model = data.get("model")
        if not model:
            raise ValidationError("`model` must be provided", field="model")
        elif not isinstance(provider, str):
            raise ValidationError("`model` must be a string", field="model")

        from localvectordb.embeddings import EmbeddingRegistry
        if provider not in EmbeddingRegistry.list():
            raise ValidationError(f"`provider` must be one of: {EmbeddingRegistry.list()}", field="provider")

        try:
            embeddings = current_app.db_manager.get_embeddings_for_model(texts, provider, model)

            return jsonify({"embeddings": embeddings})

        except Exception as e:
            db_logger.log_error("get_embeddings_for_db", e, database_name=db_name)
            raise


@api.route("/api/v1/embeddings", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("get_embeddings")
def get_embeddings():
    """Get embeddings from specified provider and model"""

    with request_context("get_embeddings"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        validate_required_fields(data, ['texts', 'provider', 'model'])

        texts = data["texts"]
        provider = data["provider"]
        model = data["model"]

        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list):
            raise ValidationError("Texts must be a string or array of strings", field="texts")

        from localvectordb.embeddings import EmbeddingRegistry
        if provider not in EmbeddingRegistry.list():
            raise ValidationError(f"Provider must be one of: {', '.join(EmbeddingRegistry.list())}", field="provider", value=provider)

        try:
            embedding_provider = EmbeddingRegistry.create_provider(provider, model)
            embeddings = embedding_provider.embed_sync(texts)

            return jsonify({"embeddings": embeddings.tolist()})

        except Exception as e:
            logger.error(f"Error getting embeddings with {provider}/{model}: {e}")
            raise APIError(
                message=f"Failed to get embeddings: {str(e)}",
                error_code="EMBEDDING_GENERATION_FAILED",
                status_code=503,
                recoverable=True,
                details={"provider": provider, "model": model}
            )


@api.route("/api/v1/<db_name>/upload", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("upload_files")
def upload_files(db_name):
    """
    Upload files to the database with automatic text extraction

    Supports both single and multiple file uploads. Files are processed to extract
    text content using appropriate extractors based on file type.

    Form data:
    - files: File(s) to upload (required)
    - metadata: JSON string with metadata for files (optional)
    - batch_size: Batch size for processing (default: 100)
    - ids: JSON string with the list of document ids for files (optional)
    - use_filename_as_id : boolean, ignored if `ids` is provided (optional)

    Returns JSON with uploaded file IDs and extraction details.
    """

    with request_context("upload_files"):
        # Check if server uploads are enabled
        if not current_app.config_obj.server.file_upload_enabled:
            raise APIError(
                message="File extraction route is not enabled",
                error_code="EXTRACTION_NOT_AVAILABLE",
                status_code=503
            )
        # Check if files are present
        if 'files' not in request.files:
            raise ValidationError("No files provided", field="files")

        files = request.files.getlist('files')
        if not files or all(f.filename == '' for f in files):
            raise ValidationError("No files selected", field="files")

        file_ids = None
        ids_param = request.form.get('ids')
        use_filenames_as_ids = request.form.get("use_filename_as_id", "false").lower() == "true"

        if ids_param:
            try:
                # Try JSON array first
                file_ids = json.loads(ids_param)
                if not isinstance(file_ids, list):
                    raise ValueError("IDs must be an array")
            except json.JSONDecodeError:
                # Fallback to comma-separated string
                file_ids = [id_str.strip() for id_str in ids_param.split(',') if id_str.strip()]

        # Validate IDs if provided
        if file_ids is not None:
            if len(file_ids) != len(files):
                raise ValidationError(
                    f"Number of IDs ({len(file_ids)}) must match number of files ({len(files)})",
                    field="ids"
                )

        # Get form parameters
        extract_text = request.form.get('extract_text', 'true').lower() == 'true'
        batch_size = int(request.form.get('batch_size', '100'))

        # Parse metadata if provided
        metadata_json = request.form.get('metadata')
        base_metadata = {}
        if metadata_json:
            try:
                base_metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON in metadata field", field="metadata")

        # Validate batch size
        if batch_size < 1 or batch_size > 1000:
            raise ValidationError("Batch size must be between 1 and 1000", field="batch_size", value=batch_size)

        try:
            db = current_app.db_manager.get_db(db_name)

            documents = []
            metadata_list = []
            extraction_results = []
            document_ids = []

            extractor_registry = get_extractor_registry()

            db_logger.log_query("upload_files",
                                database_name=db_name,
                                file_count=len(files)
                                )

            for file_idx, file in enumerate(files):
                if file.filename == '':
                    continue

                if file_ids is not None:
                    file_id = file_ids[file_idx]
                elif use_filenames_as_ids:
                    file_id = file.filename
                else:
                    file_id = None

                # Secure the filename
                filename = secure_filename(file.filename)
                if not filename:
                    filename = "uploaded_file"

                # Read file content
                file_content = file.read()
                if len(file_content) == 0:
                    logger.warning(f"Empty file uploaded: {filename}")
                    continue

                # Get mimetype
                mimetype = file.content_type or mimetypes.guess_type(filename)[0]

                # Prepare file metadata - only include fields that exist in the database schema
                file_metadata = base_metadata.copy()

                # Standard file upload metadata (only add if in schema)
                standard_metadata = {
                    'source': 'file_upload',
                    'original_filename': file.filename,
                    'secure_filename': filename,
                    'file_size_bytes': len(file_content),
                    'mimetype': mimetype,
                    'upload_timestamp': datetime.now().isoformat()
                }

                # Only add metadata fields that exist in the database schema
                for key, value in standard_metadata.items():
                    if key in db.metadata_schema:
                        file_metadata[key] = value

                # Extract text content

                try:
                    extraction_result = extractor_registry.extract_text(
                        file_content, filename, mimetype
                    )

                    if extraction_result.success:
                        documents.append(extraction_result.text)
                        document_ids.append(file_id)
                        # Add extraction metadata - only fields that exist in schema
                        extraction_metadata = {
                            # 'extraction_success': True,
                            'extraction_method': extraction_result.method,
                            'text_length': len(extraction_result.text),
                            **extraction_result.metadata  # Include all extraction metadata
                        }

                        # Filter extraction metadata to only include schema fields
                        for key, value in extraction_metadata.items():
                            if key in db.metadata_schema:
                                file_metadata[key] = value

                        extraction_results.append({
                            'filename': filename,
                            'extraction_success': True,
                            'extraction_method': extraction_result.method,
                            'text_length': len(extraction_result.text) if extraction_result.text else 0,
                            'error': None,
                            'metadata_fields_used': [key for key in extraction_result.metadata.keys() if
                                                     key in db.metadata_schema],
                            'metadata_fields_ignored': [key for key in extraction_result.metadata.keys() if
                                                        key not in db.metadata_schema]
                        })
                    else:
                        extraction_results.append({
                            'filename': filename,
                            'extraction_success': False,
                            'error': extraction_result.error if not extraction_result.success else None,
                            'metadata_fields_used': [],
                            'metadata_fields_ignored': []
                        })

                except Exception as e:
                    logger.error(f"Error extracting text from {filename}: {e}")

                    extraction_results.append({
                        'filename': filename,
                        'extraction_success': False,
                        'error': str(e),
                        'metadata_fields_used': [],
                        'metadata_fields_ignored': []
                    })

                metadata_list.append(file_metadata)

            if not documents:
                raise ValidationError("No valid files to process")

            # Upsert documents to database
            result_ids = db.upsert(
                documents=documents,
                metadata=metadata_list,
                batch_size=batch_size
            )

            db_logger.log_query("upload_files_success",
                                database_name=db_name,
                                processed_files=len(documents),
                                result_count=len(result_ids))

            # Prepare response
            response_data = {
                "message": f"Successfully processed {len(documents)} file(s)",
                "files_processed": len(documents),
                "document_ids": result_ids,
                "extraction_results": extraction_results,
                "status": "success"
            }

            # Add extraction summary
            if extract_text and FILE_EXTRACTION_AVAILABLE:
                successful_extractions = sum(1 for r in extraction_results if r['extraction_success'])
                response_data["extraction_summary"] = {
                    "total_files": len(extraction_results),
                    "successful_extractions": successful_extractions,
                    "failed_extractions": len(extraction_results) - successful_extractions,
                    "supported_formats": get_supported_formats()
                }

            return jsonify(response_data)

        except Exception as e:
            db_logger.log_error("upload_files", e, database_name=db_name)
            raise


@api.route("/api/v1/upload/supported-formats", methods=["GET"])
@require_api_key
@handle_errors
def get_upload_supported_formats():
    """
    Get information about supported file formats for upload

    Returns information about which file formats can be processed
    and what extraction methods are available.
    """

    if not current_app.config_obj.server.file_upload_enabled:
        raise APIError(
            message="File extraction route is not enabled",
            error_code="EXTRACTION_NOT_AVAILABLE",
            status_code=503
        )

    supported = get_supported_formats()

    # Convert to the expected format for API response
    format_details = {}
    for format_key, format_info in supported.items():
        format_details[format_key] = {
            "extensions": format_info.get('extensions', []),
            "mimetypes": format_info.get('mimetypes', []),
            "description": f"{format_key.upper()} files",
            "extractors": format_info.get('extractors', []),
            "supported": format_info.get('available', False)
        }

    response = {
        "extraction_available": True,
        "supported_formats": format_details,
        "basic_text_support": True,
        "text_file_extensions": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".xml", ".csv"],
    }

    if current_app.config["ENVIRONMENT"] == "development":
        response["installation_hints"] = {
            "pdf": "pip install pdfplumber or pip install PyPDF2",
            "docx": "pip install python-docx",
            "pptx": "pip install python-pptx",
            "xlsx": "pip install openpyxl",
            "rtf": "pip install striprtf"
        }

    return jsonify(response)


@api.route("/api/v1/upload/extract-preview", methods=["POST"])
@require_api_key
@handle_errors
@log_performance("extract_preview")
def extract_preview():
    """
    Preview text extraction from uploaded files without adding to database

    This endpoint allows testing text extraction on files before committing
    them to the database. Useful for validating extraction quality.

    Form data:
    - file: Single file to preview (required)

    Returns extracted text and extraction metadata.
    """

    with request_context("extract_preview"):
        if not current_app.config_obj.server.file_upload_enabled:
            raise APIError(
                message="File extraction is not enabled",
                error_code="EXTRACTION_NOT_AVAILABLE",
                status_code=503
            )

        if 'file' not in request.files:
            raise ValidationError("No file provided", field="file")

        file = request.files['file']
        if file.filename == '':
            raise ValidationError("No file selected", field="file")

        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            if not filename:
                filename = "preview_file"

            # Read file content
            file_content = file.read()
            mimetype = file.content_type or mimetypes.guess_type(filename)[0]

            # Extract text
            extractor_registry = get_extractor_registry()
            extraction_result = extractor_registry.extract_text(
                file_content, filename, mimetype
            )

            # Prepare response
            response_data = {
                "filename": filename,
                "original_filename": file.filename,
                "file_size_bytes": len(file_content),
                "mimetype": mimetype,
                "extraction_success": extraction_result.success,
                "extraction_method": extraction_result.method,
                "extraction_metadata": extraction_result.metadata,
                "extracted_text": extraction_result.text,
                "text_length": len(extraction_result.text),
                "text_preview": extraction_result.text[:500] + "..." if len(
                    extraction_result.text) > 500 else extraction_result.text
            }

            if not extraction_result.success:
                response_data["extraction_error"] = extraction_result.error

            return jsonify(response_data)

        except Exception as e:
            logger.error(f"Error during extraction preview: {e}")
            raise APIError(
                message=f"Preview extraction failed: {str(e)}",
                error_code="EXTRACTION_PREVIEW_FAILED",
                status_code=500
            )


# Add this route to routes.py after the other database management routes

@api.route("/api/v1/<db_name>/schema", methods=["PUT"])
@require_api_key
@handle_errors
@log_performance("update_metadata_schema")
def update_metadata_schema(db_name):
    """Update the metadata schema for a database"""

    with request_context("update_metadata_schema"):
        if not request.is_json:
            raise ValidationError("Request must contain JSON data")

        data = request.get_json()
        if not data:
            raise ValidationError("Request body cannot be empty")

        # Validate required fields
        validate_required_fields(data, ['metadata_schema'])

        # Parse metadata schema
        try:
            new_schema = parse_metadata_schema(data['metadata_schema'])
        except Exception as e:
            raise ValidationError(f"Invalid metadata schema: {str(e)}", field="metadata_schema")

        if not new_schema:
            raise ValidationError("Metadata schema cannot be empty", field="metadata_schema")

        # Get optional parameters
        drop_columns = data.get('drop_columns', False)
        if not isinstance(drop_columns, bool):
            raise ValidationError("drop_columns must be a boolean", field="drop_columns", value=drop_columns)

        try:
            db = current_app.db_manager.get_db(db_name)

            db_logger.log_query("update_metadata_schema",
                                database_name=db_name,
                                field_count=len(new_schema),
                                drop_columns=drop_columns)

            # Apply schema update
            changes = db.update_metadata_schema(new_schema, drop_columns=drop_columns)

            db_logger.log_query("update_metadata_schema_success",
                                database_name=db_name,
                                added_fields=len(changes.get('added_fields', [])),
                                removed_fields=len(changes.get('removed_fields', [])),
                                modified_fields=len(changes.get('modified_fields', [])),
                                populated_defaults=len(changes.get('populated_defaults', [])))

            # Prepare response
            response_data = {
                "message": f"Successfully updated metadata schema for database '{db_name}'",
                "status": "success",
                "changes": changes,
                "new_schema": {
                    field_name: {
                        "type": field.type.value,
                        "indexed": field.indexed,
                        "required": field.required,
                        "default_value": field.default_value
                    }
                    for field_name, field in new_schema.items()
                }
            }

            return jsonify(response_data)

        except Exception as e:
            db_logger.log_error("update_metadata_schema", e, database_name=db_name)
            raise


@api.route("/api/v1/<db_name>/schema", methods=["GET"])
@require_api_key
@handle_errors
@cache.cached(timeout=300)  # Cache for 5 minutes
@log_performance("get_metadata_schema_info")
def get_metadata_schema_info(db_name):
    """Get detailed information about the current metadata schema"""

    with request_context("get_metadata_schema_info"):
        try:
            db = current_app.db_manager.get_db(db_name)
            schema_info = db.get_metadata_schema_info()

            return jsonify({
                "database": db_name,
                "schema_info": schema_info,
                "status": "success"
            })

        except Exception as e:
            db_logger.log_error("get_metadata_schema_info", e, database_name=db_name)
            raise